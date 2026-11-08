"""Latest spatial ROI head, v3 frozen/noisy Student and stage checkpoint."""
import copy
from types import SimpleNamespace
import pytest
import torch
from test_dimo_update import case as base_case,deterministic_cpu
from src.dimo import train
from src.dimo.losses import build_losses
from src.dimo.spatial_gan import (TeacherFeatures,RegionalDiscriminator,GANRandom,restore_spatial,
    restore_and_average,expected_embedding,real_embedding,mask_embedding,adversarial_generator)
from src.sparse_utils import prepare_sparse_target_layout
from src.dimo.optimization import configure_trainability
from src.dimo.checkpoint import save_checkpoint,load_checkpoint,training_state,state_digest,rng_state,restore_rng


@pytest.fixture
def case(base_case):
    from src.transformer import Transformer2DModel
    from src.dimo.core import clone_models,generation_mask
    c=base_case
    teacher,student,auxiliary,ema=clone_models(Transformer2DModel(num_layers=4,num_single_layers=1,
        attention_head_dim=12,num_attention_heads=2,joint_attention_dim=24,pooled_projection_dim=8,
        axes_dims_rope=(4,4,4),vocab_size=32,codebook_size=31,downsample=True,upsample=True,
        sparse_target=True,sparse_target_halo=1,supervise_target_halo=True,
        time_conditioning='mask_ratio',cache_conditioning=True))
    c.teacher=teacher;c.models=dict(student=student,auxiliary=auxiliary,ema=ema)
    student.enable_gradient_checkpointing();auxiliary.enable_gradient_checkpointing()
    c.optimizers={name:torch.optim.AdamW(c.models[name].parameters(),lr=1e-3) for name in ['student','auxiliary']}
    c.schedulers={name:torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.) for name,opt in c.optimizers.items()}
    original=c.batch_fn
    def batch():
        source,m,g,cond,uncond=original()
        return source,m,generation_mask(m,teacher),cond,uncond
    c.batch_fn=batch
    return c


def prepare_gan(c,monkeypatch,weight='auto'):
    c.config.update(training_mode='v3_spatial_gan',gt_ce_weight=0.,teacher_cfg=6.,student_cfg=1.,auxiliary_cfg=1.,
        noise_emb_perturb=.3,ema_warmup=True,max_steps=500,
        freeze_student_embedding=True,freeze_student_mlm_head=True,freeze_auxiliary_embedding=True,freeze_auxiliary_mlm_head=True,
        spatial_gan=dict(enabled=True,feature_layers=[0,1,2,3],head_hidden=16,weight=weight))
    configure_trainability(dict(teacher=c.teacher,**c.models),c.config)
    vq=torch.nn.Linear(1,1).eval().requires_grad_(False);vq.config=SimpleNamespace(num_vq_embeddings=31)
    extension=build_losses(c.config,vq,c.teacher,c.acc.device)
    extension.attach(c.acc,c.models,c.optimizers,c.schedulers)
    monkeypatch.setattr('src.dimo.losses.encode_target',lambda raw,vq,source,device:raw['target'])
    original=c.batch_fn
    def batch():
        prepared=original();source,m,g,cond,uncond=prepared;source=source%31
        cond=dict(cond,reference_image_hidden_states=source);uncond=dict(uncond,reference_image_hidden_states=source)
        prepared=(source,m,g,cond,uncond)
        extension.bind_batch({'target':(source+1)%31},prepared)
        return prepared
    return extension,batch


def test_sparse_padding_holes_and_spatial_order(case):
    c=case;source,m,g,cond,_=c.batch_fn()
    layout=prepare_sparse_target_layout(m,(6,8),(3,4),2,source.device,1)
    values=torch.randn(2,layout.indices.shape[1],24,requires_grad=True)
    dense,valid=restore_spatial(values,layout)
    assert dense.shape==(2,24,3,4) and int(valid.sum())==int(layout.valid.sum())
    roi=restore_and_average(values,layout);assert roi.shape==(2,25,16,16)
    bad=values.detach().clone();bad[~layout.valid]=1e6
    torch.testing.assert_close(roi.detach(),restore_and_average(bad,layout),rtol=0,atol=0)
    roi.sum().backward();assert values.grad[~layout.valid].eq(0).all()
    full=prepare_sparse_target_layout(torch.ones(1,4,4,dtype=torch.bool),(4,4),(4,4),1,source.device,0)
    original=torch.randn(1,16,24);flipped=original.reshape(1,4,4,24).flip(2).reshape(1,16,24)
    torch.testing.assert_close(original.mean(1),flipped.mean(1))
    a,b=restore_and_average(original,full),restore_and_average(flipped,full)
    d=RegionalDiscriminator(24,16);ratio=torch.tensor([.2])
    assert not torch.allclose(d((a,)*4,ratio),d((b,)*4,ratio),atol=1e-6,rtol=0)


@pytest.mark.parametrize('checkpointing',[False,True])
def test_real_teacher_input_gradient_g_lock_and_hook_cleanup(case,checkpointing):
    c=case;source,m,g,cond,_=c.batch_fn();teacher=c.teacher
    backbone=TeacherFeatures(teacher,[0,1,2,3],checkpointing)
    d=RegionalDiscriminator(24,16)
    logits=torch.randn(*source.shape,31,requires_grad=True)
    loss,_,ratio,noise=adversarial_generator(logits,source,m,g,cond,teacher,backbone,d,GANRandom('cpu',22))
    loss.backward()
    assert logits.grad[g&~noise].abs().sum()>0 and logits.grad[~g].eq(0).all() and logits.grad[noise].eq(0).all()
    assert logits.grad[g&~m&~noise].abs().sum()>0
    assert all(p.grad is None for model in [teacher,d] for p in model.parameters())
    assert not teacher.embed.embeddings._forward_hooks
    assert all(not block._forward_hooks for block in teacher.transformer_blocks)


def test_real_fake_shared_condition_and_mask_identical_input_scores(case):
    c=case;source,m,g,cond,_=c.batch_fn();teacher=c.teacher
    a=real_embedding(source,source,g,teacher);b=teacher.embed.embeddings(source)
    assert torch.equal(a,b)
    ratio,noise=GANRandom('cpu',3).sample(g);backbone=TeacherFeatures(teacher,[0,1,2,3],False);d=RegionalDiscriminator(24,16).eval()
    torch.testing.assert_close(d(backbone(mask_embedding(a,noise,teacher),source,m,cond,ratio),ratio),
        d(backbone(mask_embedding(b,noise,teacher),source,m,cond,ratio),ratio),rtol=0,atol=0)


def test_warmup_uses_stage_counter_and_exact_first_boundary(case,monkeypatch):
    c=case;extension,batch=prepare_gan(c,monkeypatch)
    before=state_digest(c.models['discriminator'].state_dict())
    result=train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=0,extensions=extension)
    assert not extension.student_active() and not extension.needs_calibration()
    assert 'gan_g' not in result and 'gan_d' in result
    assert before!=state_digest(c.models['discriminator'].state_dict())
    assert extension.phase_step==1 and c.positions==[0,1,2,3]
    extension.phase_step=199;assert not extension.student_active()
    extension.phase_step=200;assert extension.student_active() and extension.needs_calibration()
    with pytest.raises(ValueError,match='phase counter'):
        train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=8000,extensions=extension)


def test_gan_warmup_preserves_original_v3_update_data_and_sampling_rng(case,monkeypatch):
    c=case;extension,batch=prepare_gan(c,monkeypatch)
    initial=copy.deepcopy(training_state(c.acc,c.models,c.optimizers,c.schedulers));random=rng_state()
    def shared_state():
        models={k:c.models[k] for k in ['student','auxiliary','ema']}
        opts={k:c.optimizers[k] for k in ['student','auxiliary']}
        sched={k:c.schedulers[k] for k in ['student','auxiliary']}
        return state_digest(training_state(c.acc,models,opts,sched))
    expected=train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=0)
    state=shared_state();rng=state_digest(rng_state());positions=list(c.positions)
    for role,model in c.models.items():model.load_state_dict(initial['models'][role])
    for role,opt in c.optimizers.items():opt.load_state_dict(initial['optimizers'][role])
    for role,sched in c.schedulers.items():sched.load_state_dict(initial['schedulers'][role])
    c.positions.clear();restore_rng(random)
    actual=train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=0,extensions=extension)
    assert expected['student']==actual['student'] and expected['auxiliary']==actual['auxiliary']
    assert shared_state()==state and state_digest(rng_state())==rng and c.positions==positions


@pytest.mark.parametrize('joint',[False,True])
def test_gan_complete_checkpoint_exact_next_update_including_rank_rng(case,monkeypatch,tmp_path,joint):
    c=case;extension,batch=prepare_gan(c,monkeypatch,weight=.001 if joint else 'auto')
    start=200 if joint else 0
    extension.phase_step=start;c.positions[:]=[0]*(start*4)
    def update(step):return train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=step,extensions=extension)
    update(start)
    path=save_checkpoint(c.acc,tmp_path,start+1,len(c.positions),c.config,c.models,c.optimizers,c.schedulers,extension)
    expected=update(start+1)
    state=state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers,extension));random=state_digest((rng_state(),extension.rank_state()))
    p=load_checkpoint(c.acc,path,c.config,c.models,c.optimizers,c.schedulers,extension)
    c.positions[:]=c.positions[:p['consumed_batches']]
    actual=update(start+1)
    assert expected==actual
    assert state==state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers,extension))
    assert random==state_digest((rng_state(),extension.rank_state()))
    assert all(not p.requires_grad for p in c.models['student'].embed.parameters())
    assert all(not p.requires_grad for p in c.models['student'].mlm_layer.parameters())


def test_hooks_clean_up_if_teacher_forward_fails(case,monkeypatch):
    c=case;source,m,g,cond,_=c.batch_fn();teacher=c.teacher
    backbone=TeacherFeatures(teacher,[0,1,2,3],False)
    monkeypatch.setattr(teacher,'forward',lambda **kwargs:(_ for _ in ()).throw(RuntimeError('intentional failure')))
    with pytest.raises(RuntimeError,match='intentional'):
        backbone(teacher.embed.embeddings(source),source,m,cond,torch.tensor([.2,.2]))
    assert not teacher.embed.embeddings._forward_hooks
    assert all(not block._forward_hooks for block in teacher.transformer_blocks)


def test_raw_ema_inference_loads_gan_checkpoint_without_discriminator(case,monkeypatch,tmp_path):
    from src.dimo.infer import load_role
    from src.dimo.origin import sha256
    import json
    c=case;extension,batch=prepare_gan(c,monkeypatch)
    train.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,batch,c.sampler,c.config,True,global_step=0,extensions=extension)
    c.teacher.save_pretrained(tmp_path/'teacher',safe_serialization=True)
    (tmp_path/'teacher_origin.json').write_text(json.dumps(dict(pinned_teacher_checksums={'config.json':sha256(tmp_path/'teacher/config.json')})))
    path=save_checkpoint(c.acc,tmp_path,1,4,c.config,c.models,c.optimizers,c.schedulers,extension)
    monkeypatch.setattr('src.dimo.gan_runtime.SpatialLosses',lambda *args:pytest.fail('Inference created D'))
    for role in ['student','ema']:
        model,metadata=load_role(path,role)
        assert state_digest(model.state_dict())==state_digest(c.models[role].state_dict())
        assert metadata['config']['training_mode']=='v3_spatial_gan'
