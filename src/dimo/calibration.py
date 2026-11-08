"""Full-accumulation/all-rank parameter-gradient calibration, no updates."""
import torch
import torch.distributed as dist
from src.dimo.core import distillation_losses
from src.dimo.losses import capture_logits
from src.dimo.checkpoint import rng_state,restore_rng


def calibrate(acc,teacher,models,batch_fn,sampler,c,extension):
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
            samples.append(dict(rank=acc.process_index,microbatch=micro,FKL_logits_l2=float(fg.norm()),
                extra_logits_l2=float(eg.norm()),logits_cosine=float(torch.nn.functional.cosine_similarity(fg.flatten()[None],eg.flatten()[None])),
                M_tokens=int(m.sum()),G_tokens=int(g.sum())))
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
        target=c.get('gt_ce_target_gradient_ratio',.15)
        weight=target*float(f2.sqrt()/e2.sqrt())
        result=dict(version=1,mode=extension.mode,phase_step=extension.phase_step,weight=weight,
            FKL_parameter_gradient_l2=float(f2.sqrt()),extra_parameter_gradient_l2=float(e2.sqrt()),
            weighted_parameter_ratio=target,parameter_cosine=float(dot/(f2*e2).sqrt()),
            complete_accumulation_microbatches=c['gradient_accumulation'],DDP_ranks=acc.num_processes,
            student_optimizer_updates=0,source='training shadow stream; no DEV/Test tuning',samples=samples)
        if acc.num_processes>1:
            gathered=[None]*acc.num_processes;dist.all_gather_object(gathered,samples)
            result['samples']=[row for group in gathered for row in group]
        for row in result['samples']:
            row['weighted_logits_ratio']=weight*row['extra_logits_l2']/max(row['FKL_logits_l2'],1e-30)
    finally:
        restore_rng(before)
        extension.restore_rank_state(rank_before) if hasattr(extension,'restore_rank_state') else None
    extension.weight=weight;extension.calibration=result
    return result
