"""Tiny synthetic v3 loss/DDP smoke, NOT production training or quality evidence."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs,InitProcessGroupKwargs,set_seed
from diffusers import VQModel
from src.transformer import Transformer2DModel
from src.dimo.core import clone_models,generation_mask
from src.dimo.optimization import configure_trainability,build_optimizers,build_schedulers,FREEZE_DEFAULTS
from src.dimo.losses import build_losses
from src.dimo.train import run_update
from src.dimo.checkpoint import (save_checkpoint,load_checkpoint,check_roundtrip,training_state,
    state_digest,rng_state,microbatches_per_update)
from src.dimo.calibration import calibrate
from src.dimo.device import initialize_local_cuda_device
from src.timestep_sampling import MaskRatioSampler
from src.v2_utils import prepare_cond_token


def smoke(out,mode,steps,device='cpu'):
    torch.set_num_threads(1)
    if device=='cuda':initialize_local_cuda_device()
    handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        handlers.append(InitProcessGroupKwargs(backend='nccl' if device=='cuda' else 'gloo',timeout=timedelta(seconds=300)))
    acc=Accelerator(cpu=device=='cpu',mixed_precision='bf16' if device=='cuda' else 'no',kwargs_handlers=handlers)
    set_seed(108)
    model=Transformer2DModel(num_layers=4,num_single_layers=1,attention_head_dim=12,num_attention_heads=2,
        joint_attention_dim=24,pooled_projection_dim=8,axes_dims_rope=(4,4,4),vocab_size=32,codebook_size=31,
        downsample=True,upsample=True,sparse_target=True,sparse_target_halo=1,supervise_target_halo=True,
        time_conditioning='mask_ratio',cache_conditioning=True)
    teacher,student,auxiliary,ema=clone_models(model);del model
    c=dict(training_mode=mode,initialization='sft',validation_only=True,max_steps=max(steps+1,205),
        micro_batch=1,gradient_accumulation=1,fake_rounds=1,teacher_cfg=6.,student_cfg=1.,auxiliary_cfg=1.,
        noise_emb_perturb=.3,student_initial_mask_ratio=1.,auxiliary_label_smoothing=.1,ema_decay=.9999,ema_warmup=True,
        student_lr=1e-6,auxiliary_lr=1e-6,adam_betas=[.9,.999],adam_epsilon=1e-8,weight_decay=.01,
        lr_scheduler='constant_with_warmup',lr_warmup_steps=100,mixed_precision='bf16' if device=='cuda' else 'no',
        max_grad_norm=1.,gt_ce_weight='auto' if mode=='v3_gt_ce' else 0.,gt_ce_target_gradient_ratio=.15,
        spatial_gan=dict(enabled=mode=='v3_spatial_gan',feature_layers=[0,1,2,3],head_hidden=16),
        **{k:True for k in FREEZE_DEFAULTS})
    report=configure_trainability(dict(teacher=teacher,student=student,auxiliary=auxiliary,ema=ema),c)
    student.enable_gradient_checkpointing();auxiliary.enable_gradient_checkpointing()
    teacher.to(acc.device);ema.to(acc.device)
    opts=build_optimizers(dict(student=student,auxiliary=auxiliary),c);schedulers=build_schedulers(opts,c)
    student,auxiliary,s,a=acc.prepare(student,auxiliary,opts['student'],opts['auxiliary'])
    models=dict(student=student,auxiliary=auxiliary,ema=ema);opts=dict(student=s,auxiliary=a)
    vq=VQModel(block_out_channels=(16,),latent_channels=4,vq_embed_dim=4,num_vq_embeddings=31,
        norm_num_groups=4,lookup_from_codebook=True).eval().requires_grad_(False).to(acc.device)
    extension=build_losses(c,vq,teacher,acc.device)
    if extension is not None and hasattr(extension,'attach'):extension.attach(acc,models,opts,schedulers)
    set_seed(109+acc.process_index)
    cursor=[0]
    def batch():
        index=cursor[0];cursor[0]+=1
        generator=torch.Generator().manual_seed(7000+index*acc.num_processes+acc.process_index)
        raw=dict(reference_image=torch.rand(1,3,6,8,generator=generator),image=torch.rand(1,3,6,8,generator=generator),
                 edit_mask=torch.zeros(1,1,6,8))
        raw['edit_mask'][0,0,1,3]=1.
        with torch.no_grad():source=prepare_cond_token(None,raw['reference_image'].to(acc.device),vq).reshape(1,6,8)
        m=raw['edit_mask'][:,0].to(acc.device).bool();g=generation_mask(m,teacher)
        y,x=torch.meshgrid(torch.arange(3,device=acc.device),torch.arange(4,device=acc.device),indexing='ij')
        ids=torch.stack((torch.zeros_like(x),y,x),-1).reshape(-1,3)
        cond=dict(encoder_hidden_states=torch.randn(1,3,24,generator=generator).to(acc.device),
            pooled_projections=torch.randn(1,8,generator=generator).to(acc.device),micro_conds=torch.zeros(1,5,device=acc.device),
            img_ids=ids,reference_image_ids=ids,txt_ids=torch.zeros(3,3,device=acc.device),target_edit_mask=m,
            reference_image_hidden_states=source,lora_part_enable=True)
        uncond=dict(cond,encoder_hidden_states=torch.zeros_like(cond['encoder_hidden_states']),pooled_projections=torch.zeros_like(cond['pooled_projections']))
        prepared=(source,m,g,cond,uncond)
        if extension is not None:extension.bind_batch(raw,prepared)
        return prepared
    sampler=MaskRatioSampler();out=Path(out)
    if acc.is_main_process:
        if out.exists() and any(out.iterdir()):raise FileExistsError('Smoke refuses existing nonempty output')
        out.mkdir(parents=True,exist_ok=True)
    acc.wait_for_everyone()
    initial=state_digest(acc.unwrap_model(student).state_dict());frozen=state_digest(teacher.state_dict())
    vq_frozen=state_digest(vq.state_dict());calibrations=[]
    def update(step):
        if extension is not None and extension.needs_calibration():
            position=cursor[0]
            result=calibrate(acc,teacher,models,batch,sampler,c,extension)
            cursor[0]=position;calibrations.append(result)
        return run_update(acc,teacher,models,opts,schedulers,batch,sampler,c,True,global_step=step,extensions=extension)
    for step in range(steps):
        losses=update(step)
        if acc.is_main_process and (step+1)%25==0:print(f'{mode} synthetic update{step+1}/{steps}',flush=True)
    checkpoint=save_checkpoint(acc,out,steps,cursor[0],c,models,opts,schedulers,extension)
    progress=check_roundtrip(acc,checkpoint,c,models,opts,schedulers,extension)
    update(steps)
    expected=state_digest(training_state(acc,models,opts,schedulers,extension))
    random=state_digest((rng_state(),extension.rank_state() if extension is not None else None))
    load_checkpoint(acc,checkpoint,c,models,opts,schedulers,extension);cursor[0]=progress['consumed_batches']
    update(steps)
    assert expected==state_digest(training_state(acc,models,opts,schedulers,extension)),'Next GPU/CPU update differs after restore'
    assert random==state_digest((rng_state(),extension.rank_state() if extension is not None else None)),'RNG resume differs'
    assert initial!=state_digest(acc.unwrap_model(student).state_dict()),'Student did not update'
    assert frozen==state_digest(teacher.state_dict()) and vq_frozen==state_digest(vq.state_dict())
    for name in ['embedding','mlm_head']:
        module='embed' if name=='embedding' else 'mlm_layer'
        assert all(not p.requires_grad for p in getattr(acc.unwrap_model(student),module).parameters())
    assert cursor[0]==(steps+1)*microbatches_per_update(c)
    common=state_digest(training_state(acc,models,opts,schedulers,extension))
    if acc.num_processes>1:
        hashes=[None]*acc.num_processes;torch.distributed.all_gather_object(hashes,common)
        assert len(set(hashes))==1,'DDP common states differ'
    result=dict(status='PASS',scope='tiny random 4-layer EditMGT + frozen tiny real VQModel + synthetic images; NOT production1024/quality validation',
        mode=mode,rank=acc.process_index,world_size=acc.num_processes,device=str(acc.device),completed_updates=steps,
        additional_next_update_resume_test=True,exact_resume=True,DDP_state_identical=True,
        teacher_and_vq_unchanged=True,student_updated=True,parameter_report=report,
        inherited_original_v3_recipe=True,calibrations=calibrations,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        loss_state=extension.state_dict() if extension is not None else None)
    (out/f'result-rank{acc.process_index}.json').write_text(json.dumps(result,indent=2)+'\n')
    acc.wait_for_everyone();acc.end_training()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['v3_baseline','v3_gt_ce','v3_spatial_gan'],required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    if args.steps<2:p.error('Use at least2 updates: the original v3 first warmup LR is0')
    smoke(args.output,args.mode,args.steps,args.device)


if __name__=='__main__':main()
