"""Optional v3 supervision, disabled behavior and complete-stage recovery."""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from test_dimo_update import case,deterministic_cpu
from src.dimo import train
from src.dimo.gt_ce import region_token_ce,encode_target
from src.dimo.losses import OptionalLosses,build_losses,validate_losses
from src.dimo.checkpoint import save_checkpoint,load_checkpoint,training_state,state_digest,rng_state,restore_rng


def prepare_gt(case,monkeypatch,weight=.05):
    c=case;c.config.update(training_mode='v3_gt_ce',gt_ce_weight=weight,teacher_cfg=6.,noise_emb_perturb=.3,
        student_initial_mask_ratio=1.,max_steps=4)
    vq=torch.nn.Linear(1,1).eval().requires_grad_(False);vq.config=SimpleNamespace(num_vq_embeddings=31)
    extension=OptionalLosses(c.config,vq,c.teacher,c.acc.device)
    monkeypatch.setattr('src.dimo.losses.encode_target',lambda raw,vq,source,device:raw['target'])
    original=c.batch_fn
    def batch():
        prepared=original();raw={'target':(prepared[0]+1)%31};extension.bind_batch(raw,prepared);return prepared
    return extension,batch


def test_gt_ce_original_M_only_sample_normalization_and_halo_gradient():
    logits=torch.randn(2,4,5,9,requires_grad=True);target=torch.randint(9,(2,4,5));m=torch.zeros_like(target,dtype=torch.bool)
    m[0,0,0]=True;m[1,1:,:]=True
    loss=region_token_ce(logits,target,m)
    expected=torch.stack([torch.nn.functional.cross_entropy(z[mask],t[mask]) for z,t,mask in zip(logits,target,m)]).mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    grad=torch.autograd.grad(loss,logits)[0]
    assert grad[m].abs().sum()>0 and grad[~m].eq(0).all()
    bad=target.clone();bad[~m]=-1000
    torch.testing.assert_close(loss,region_token_ce(logits,bad,m),rtol=0,atol=0)


def test_target_encoder_alignment_frozen_eval_and_vocabulary(monkeypatch):
    vq=torch.nn.Linear(1,1).eval().requires_grad_(False);vq.config=SimpleNamespace(num_vq_embeddings=7)
    source=torch.zeros(2,3,4,dtype=torch.long);image=torch.randn(2,3,48,64)
    batch=dict(image=image,reference_image=image.clone())
    monkeypatch.setattr('src.v2_utils.prepare_cond_token',lambda *args:torch.ones(2,12,dtype=torch.long))
    assert torch.equal(encode_target(batch,vq,source,'cpu'),torch.ones_like(source))
    with pytest.raises(ValueError):encode_target(dict(batch,image=image[:1]),vq,source,'cpu')
    vq.train()
    with pytest.raises(ValueError):encode_target(batch,vq,source,'cpu')
    vq.eval();monkeypatch.setattr('src.v2_utils.prepare_cond_token',lambda *args:torch.full((2,12),7,dtype=torch.long))
    with pytest.raises(ValueError):encode_target(batch,vq,source,'cpu')


@pytest.mark.parametrize('key',[None,0.])
def test_off_has_no_encoding_or_random_draw_and_same_update(case,monkeypatch,key):
    c=case;c.config.update(teacher_cfg=6.,noise_emb_perturb=.3)
    initial=copy.deepcopy(training_state(c.acc,c.models,c.optimizers,c.schedulers));before=rng_state()
    expected=train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,c.batch_fn,c.sampler,c.config,True,global_step=0)
    expected_hash=state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers));expected_rng=state_digest(rng_state())
    for role,m in c.models.items():m.load_state_dict(initial['models'][role])
    for role,m in c.optimizers.items():m.load_state_dict(initial['optimizers'][role])
    for role,m in c.schedulers.items():m.load_state_dict(initial['schedulers'][role])
    c.positions.clear();restore_rng(before)
    if key is not None:c.config.update(gt_ce_weight=0.,training_mode='v3_baseline',spatial_gan={'enabled':False})
    monkeypatch.setattr('src.dimo.losses.encode_target',lambda *args:pytest.fail('disabled GT encoded'))
    assert build_losses(c.config,None,c.teacher,c.acc.device) is None
    actual=train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,c.batch_fn,c.sampler,c.config,True,global_step=0,extensions=None)
    assert expected==actual
    assert expected_hash==state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers))
    assert expected_rng==state_digest(rng_state())


def test_gt_single_shared_student_forward_and_auxiliary_fresh_batch(case,monkeypatch):
    c=case;extension,batch=prepare_gt(c,monkeypatch)
    events=[];handle=c.models['student'].register_forward_pre_hook(lambda *args:events.append(torch.is_grad_enabled()))
    result=train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=0,extensions=extension)
    handle.remove();assert events==[True,True,False,False]
    assert c.positions==[0,1,2,3] and len(result['gt_ce'])==2 and extension.phase_step==1
    assert all(p.grad is None for model in [c.teacher,c.models['auxiliary'],c.models['ema'],extension.vq] for p in model.parameters())


def test_gt_checkpoint_exact_next_update_and_stage_weight(case,monkeypatch,tmp_path):
    c=case;extension,batch=prepare_gt(c,monkeypatch)
    def update(step):return train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=step,extensions=extension)
    update(0);path=save_checkpoint(c.acc,tmp_path,1,4,c.config,c.models,c.optimizers,c.schedulers,extensions=extension)
    expected=update(1);digest=state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers,extension));rnd=state_digest(rng_state())
    p=load_checkpoint(c.acc,path,c.config,c.models,c.optimizers,c.schedulers,extensions=extension)
    c.positions[:]=c.positions[:p['consumed_batches']]
    assert extension.phase_step==1 and extension.weight==.05
    assert update(1)==expected
    assert digest==state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers,extension)) and rnd==state_digest(rng_state())


def test_mutually_exclusive_modes_and_bad_weights():
    with pytest.raises(ValueError,match='cannot be enabled together'):
        validate_losses(dict(training_mode='v3_gt_ce',gt_ce_weight=.1,spatial_gan={'enabled':True}))
    for weight in [-1,True,float('nan'),float('inf'),'bad']:
        with pytest.raises(ValueError):validate_losses(dict(training_mode='v3_gt_ce',gt_ce_weight=weight))


def test_auto_calibration_no_update_rng_preserved(case,monkeypatch):
    from src.dimo.calibration import calibrate
    c=case;extension,batch=prepare_gt(c,monkeypatch,weight='auto')
    before=state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers));random=state_digest(rng_state())
    result=calibrate(c.acc,c.teacher,c.models,batch,c.sampler,c.config,extension)
    assert result['weight']>0 and result['weighted_parameter_ratio']==pytest.approx(.15)
    assert state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers))==before
    assert state_digest(rng_state())==random and extension.phase_step==0
