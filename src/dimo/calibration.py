"""Full-accumulation/all-rank parameter-gradient calibration, no updates."""
import torch
import torch.distributed as dist
from src.dimo.core import distillation_losses
from src.dimo.losses import capture_logits
from src.dimo.checkpoint import rng_state,restore_rng


def calibrate(acc,teacher,models,batch_fn,sampler,c,extension,update_weight=True):
    student=acc.unwrap_model(models['student']);auxiliary=acc.unwrap_model(models['auxiliary'])
    params=tuple(p for p in student.parameters() if p.requires_grad)
    fbuffer=[torch.zeros_like(p) for p in params];ebuffer=[torch.zeros_like(p) for p in params]
    before=rng_state();rank_before=extension.rank_state();samples=[]
    try:
        for micro in range(c['gradient_accumulation']):
            prepared=batch_fn();source,m,g,cond,uncond=prepared
            ratio=sampler.sample(len(source),acc.device)
            with acc.autocast():
                with capture_logits(student) as captured:
                    fkl,_=distillation_losses(teacher,student,auxiliary,source,m,g,
                        teacher.config.vocab_size-1,cond,uncond,ratio,r_init=c['student_initial_mask_ratio'],
                        teacher_cfg=c['teacher_cfg'],noise_emb_perturb=c['noise_emb_perturb'])
                raw=captured.pop();logits=raw.movedim(1,-1)
                extra=extension.unweighted_loss(logits,prepared)
            fg=torch.autograd.grad(fkl,raw,retain_graph=True)[0].float()
            eg=torch.autograd.grad(extra,raw,retain_graph=True)[0].float()
            for index,(f,e,mask,region) in enumerate(zip(fg,eg,m,g)):
                if e.movedim(0,-1)[~(region if extension.mode=='v3_spatial_gan' else mask)].count_nonzero():
                    raise RuntimeError('Optional loss gradient escaped its allowed mask')
                fraction=float(extension.raw['edit_mask'][index].float().mean()) if extension.raw is not None and 'edit_mask' in extension.raw else float(mask.float().mean())
                samples.append(dict(rank=acc.process_index,microbatch=micro,sample=index,FKL_logits_l2=float(f.norm()),
                    extra_logits_l2=float(e.norm()),logits_cosine=float(torch.nn.functional.cosine_similarity(f.flatten()[None],e.flatten()[None])),
                    M_tokens=int(mask.sum()),G_tokens=int(region.sum()),mask_fraction=fraction,
                    mask_group='small' if fraction<.03 else 'large' if fraction>=.25 else 'medium'))
            for loss,buffers,retain in [(fkl,fbuffer,True),(extra,ebuffer,False)]:
                grads=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
                for grad,buffer in zip(grads,buffers):
                    if grad is not None:
                        if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite calibration parameter gradient')
                        buffer.add_(grad.detach()/c['gradient_accumulation'])
                del grads
            del fkl,extra,logits,raw,fg,eg,captured
        for buffers in [fbuffer,ebuffer]:
            for value in buffers:
                if acc.num_processes>1:dist.all_reduce(value);value.div_(acc.num_processes)
        f2=sum(v.double().square().sum() for v in fbuffer);e2=sum(v.double().square().sum() for v in ebuffer)
        dot=sum((a.double()*b.double()).sum() for a,b in zip(fbuffer,ebuffer))
        if not torch.isfinite(f2+e2+dot) or f2<=0 or e2<=0:raise RuntimeError('Invalid calibrated parameter norm')
        target=extension.options['target_gradient_ratio'] if hasattr(extension,'options') else c.get('gt_ce_target_gradient_ratio',.15)
        weight=target*float(f2.sqrt()/e2.sqrt()) if update_weight else extension.weight
        all_samples=samples
        if acc.num_processes>1:
            gathered=[None]*acc.num_processes;dist.all_gather_object(gathered,samples)
            all_samples=[row for group in gathered for row in group]
        proposed=weight
        cosine=float(dot/(f2*e2).sqrt())
        if hasattr(extension,'options'):
            options=extension.options
            if update_weight:
                caps=[options['per_sample_logits_ratio_cap']*s['FKL_logits_l2']/max(s['extra_logits_l2'],1e-30) for s in all_samples]
                for group in ['small','medium','large']:
                    selected=[s for s in all_samples if s['mask_group']==group]
                    if selected:caps.append(options['mask_group_logits_ratio_cap']*sum(s['FKL_logits_l2'] for s in selected)/max(sum(s['extra_logits_l2'] for s in selected),1e-30))
                weight=min(weight,*caps)
            if cosine<options['conflict_stop_cosine']:raise RuntimeError(f'Strong GAN/FKL parameter gradient conflict: {cosine}')
        if weight is None or not weight>0:raise RuntimeError('No positive fixed loss weight')
        result=dict(version=1,mode=extension.mode,phase_step=extension.phase_step,weight=weight,
            FKL_parameter_gradient_l2=float(f2.sqrt()),extra_parameter_gradient_l2=float(e2.sqrt()),
            weighted_parameter_ratio=weight*float(e2.sqrt()/f2.sqrt()),target_parameter_ratio=target,
            proposed_weight=proposed,parameter_cosine=cosine,weight_recalibrated=update_weight,
            complete_accumulation_microbatches=c['gradient_accumulation'],DDP_ranks=acc.num_processes,
            student_optimizer_updates=0,source='training shadow stream; no DEV/Test tuning',samples=all_samples)
        for row in result['samples']:
            row['weighted_logits_ratio']=weight*row['extra_logits_l2']/max(row['FKL_logits_l2'],1e-30)
    finally:
        restore_rng(before)
        extension.restore_rank_state(rank_before) if hasattr(extension,'restore_rank_state') else None
    if update_weight:extension.weight=weight;extension.calibration=result
    if hasattr(extension,'last_diagnostic_phase'):
        extension.last_diagnostic_phase=extension.phase_step;extension.diagnostic_history.append(result)
    return result
