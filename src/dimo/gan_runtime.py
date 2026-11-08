"""Spatial Teacher-feature GAN stage, independent of the v3 FKL/Auxiliary."""
import math
import torch
import torch.distributed as dist
from src.dimo.losses import OptionalLosses
from src.dimo.spatial_gan import (TeacherFeatures,RegionalDiscriminator,GANRandom,
    adversarial_generator,expected_embedding,real_embedding,mask_embedding,discriminator_loss,phase_enabled)

GAN_DEFAULTS=dict(warmup_updates=200,feature_layers=[4,7,10,13],roi_size=16,head_hidden=128,
    lr=1e-6,betas=[0.,.999],eps=1e-8,weight='auto',max_grad_norm=1.,
    target_gradient_ratio=.15,mask_group_logits_ratio_cap=.5,per_sample_logits_ratio_cap=1.5,
    conflict_stop_cosine=-.5,parameter_diagnostics_every=50,seed=707421,init_seed=707420,
    activation_checkpointing=True)


def gan_options(config):
    values={**GAN_DEFAULTS,**config.get('spatial_gan',{})};values.pop('enabled',None)
    if values['warmup_updates']!=200:raise ValueError('Spatial GAN requires a new-stage 200-update warmup')
    if values['lr']!=1e-6 or values['betas']!=[0.,.999]:raise ValueError('Keep independent D Adam1e-6 betas(0,0.999)')
    if not isinstance(values['feature_layers'],(list,tuple)) or len(values['feature_layers'])!=4 or any(type(i) is not int or i<0 for i in values['feature_layers']) or len(set(values['feature_layers']))!=4:
        raise ValueError('Require four distinct Teacher feature indices')
    for key in ['roi_size','head_hidden','parameter_diagnostics_every']:
        if type(values[key]) is not int or values[key]<1:raise ValueError(f'Invalid spatial_gan.{key}')
    weight=values['weight']
    if weight!='auto' and (isinstance(weight,bool) or not isinstance(weight,(float,int)) or not math.isfinite(weight) or weight<=0):
        raise ValueError('GAN weight must be positive finite or auto')
    target=values['target_gradient_ratio']
    if isinstance(target,bool) or not isinstance(target,(int,float)) or not math.isfinite(target) or not .1<=target<=.2:
        raise ValueError('Initial GAN gradient target must be within0.1..0.2')
    if type(values['activation_checkpointing']) is not bool:raise ValueError('activation_checkpointing must be boolean')
    cosine=values['conflict_stop_cosine']
    if isinstance(cosine,bool) or not isinstance(cosine,(int,float)) or not math.isfinite(cosine) or not -1<=cosine<=1:
        raise ValueError('Invalid GAN conflict cosine threshold')
    for key in ['mask_group_logits_ratio_cap','per_sample_logits_ratio_cap','max_grad_norm','eps']:
        if isinstance(values[key],bool) or not isinstance(values[key],(float,int)) or not math.isfinite(values[key]) or values[key]<=0:
            raise ValueError(f'Invalid spatial_gan.{key}')
    return values


class SpatialLosses(OptionalLosses):
    def __init__(self,c,vq,teacher,device):
        super().__init__(c,vq,teacher,device)
        self.options=gan_options(c)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.options['init_seed'])
            self.discriminator=RegionalDiscriminator(teacher.inner_dim,self.options['head_hidden']).to(device)
        self.backbone=TeacherFeatures(teacher,self.options['feature_layers'],self.options['activation_checkpointing'],self.options['roi_size'])
        rank=dist.get_rank() if dist.is_initialized() else 0
        self.random=GANRandom(device,self.options['seed']+rank)
        self.weight=None if self.options['weight']=='auto' else float(self.options['weight'])
        self.pending=[];self.last_diagnostic_phase=None;self.diagnostic_history=[]

    def attach(self,acc,models,optimizers,schedulers):
        d=self.discriminator
        opt=torch.optim.Adam(d.parameters(),lr=self.options['lr'],betas=tuple(self.options['betas']),
                             eps=self.options['eps'],weight_decay=0.,foreach=False)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        d,opt=acc.prepare(d,opt)
        self.discriminator=d;models['discriminator']=d;optimizers['discriminator']=opt;schedulers['discriminator']=sched

    def student_active(self):return phase_enabled(self.phase_step,self.options['warmup_updates'])
    def needs_calibration(self):return self.student_active() and self.weight is None
    def needs_diagnostic(self):
        return self.student_active() and self.weight is not None and self.phase_step%self.options['parameter_diagnostics_every']==0 and self.last_diagnostic_phase!=self.phase_step

    def unweighted_loss(self,logits,prepared):
        source,m,g,cond,_=prepared
        loss,_,_,_=adversarial_generator(logits,source,m,g,cond,self.teacher,self.backbone,self.discriminator,self.random)
        return loss

    def student_loss(self,fkl,logits,prepared,raw_logits=None):
        if not self.student_active() or self.weight is None:raise RuntimeError('GAN warmup/calibration is incomplete')
        gloss=self.unweighted_loss(logits,prepared)
        self.records.setdefault('fkl',[]).append(fkl.detach());self.records.setdefault('gan_g',[]).append(gloss.detach())
        self.records.setdefault('gan_weight',[]).append(self.weight)
        if (self.phase_step+1)%self.options['parameter_diagnostics_every']==0:
            if raw_logits is None:raise RuntimeError('Original Student output node required for logits diagnostics')
            fg=torch.autograd.grad(fkl,raw_logits,retain_graph=True)[0].float().movedim(1,-1)
            gg=torch.autograd.grad(gloss,raw_logits,retain_graph=True)[0].float().movedim(1,-1)
            g=prepared[2]
            if gg[~g].count_nonzero() or not torch.isfinite(gg).all():raise RuntimeError('GAN gradient outside G or nonfinite')
            self.records.setdefault('gan_fkl_logits_ratio',[]).append((self.weight*gg.norm()/fg.norm().clamp_min(1e-30)).detach())
            self.records.setdefault('gan_fkl_logits_cosine',[]).append(torch.nn.functional.cosine_similarity(fg.flatten()[None],gg.flatten()[None]).detach().squeeze())
        return fkl+self.weight*gloss

    def remember_auxiliary(self,logits,sampled,prepared):
        source,_,g,_,_=prepared
        # This is the already UPDATED Student forward used for original Aux CE.
        with torch.no_grad():embedding=expected_embedding(logits.detach(),source,g,self.teacher).detach()
        self.pending.append((prepared,self.raw,embedding,sampled.detach()))

    def discriminator_update(self,acc,models,optimizers,schedulers,preflight):
        if not self.pending:raise RuntimeError('D requires fresh Auxiliary Batch B, no extra data draw')
        real_scores=[];fake_scores=[];hard_scores=[]
        hard_due=(self.phase_step+1)%self.options['parameter_diagnostics_every']==0
        for prepared,raw,embedding,sampled in self.pending:
            source,m,g,cond,_=prepared;self.bind_batch(raw,prepared)
            ratio,noise=self.random.sample(g)
            with acc.autocast():
                with torch.no_grad():
                    target=self.target_tokens(source)
                    real=mask_embedding(real_embedding(target,source,g,self.teacher),noise,self.teacher)
                    fake=mask_embedding(embedding,noise,self.teacher)
                    features=[self.backbone(real,source,m,cond,ratio),self.backbone(fake,source,m,cond,ratio)]
                    if hard_due:
                        hard=mask_embedding(real_embedding(sampled,source,g,self.teacher),noise,self.teacher)
                        features.append(self.backbone(hard,source,m,cond,ratio))
                    combined=tuple(torch.cat([f[i] for f in features],0).detach() for i in range(4))
                # A single DDP forward avoids duplicate-ready reducers.
                chunks=self.discriminator(combined,torch.cat([ratio]*len(features))).chunk(len(features))
                loss=discriminator_loss(chunks[0],chunks[1])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite D loss')
            acc.backward(loss/len(self.pending))
            self.records.setdefault('gan_d',[]).append(loss.detach())
            real_scores.append(chunks[0].detach().float());fake_scores.append(chunks[1].detach().float())
            if hard_due:hard_scores.append(chunks[2].detach().float())
        norm=acc.clip_grad_norm_(self.discriminator.parameters(),self.options['max_grad_norm'])
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite D gradient norm')
        if preflight:
            for role in ['student','auxiliary','ema']:
                if any(p.grad is not None for p in models[role].parameters()):raise RuntimeError(f'D leaked gradients into {role}')
            for model in [self.teacher,self.vq]:
                if any(p.grad is not None for p in model.parameters()):raise RuntimeError('D leaked gradients into frozen backbone/VQ')
        optimizers['discriminator'].step();optimizers['discriminator'].zero_grad(set_to_none=True);schedulers['discriminator'].step()
        def gather(scores):
            scores=torch.cat(scores)
            if acc.num_processes>1:
                rows=[torch.empty_like(scores) for _ in range(acc.num_processes)];dist.all_gather(rows,scores);scores=torch.cat(rows)
            return scores
        real,fake=gather(real_scores),gather(fake_scores)
        for key,value in [('d_real_logit',real.mean()),('d_fake_logit',fake.mean()),('d_mean_logit_gap',real.mean()-fake.mean()),
            ('d_accuracy',torch.cat((real>0,fake<0)).float().mean()),
            ('d_auc',((real[:,None]>fake[None]).float()+.5*(real[:,None]==fake[None]).float()).mean())]:
            self.records[key]=[value.detach()]
        if hard_due:
            hard=gather(hard_scores)
            self.records['d_hard_fake_logit']=[hard.mean().detach()]
            self.records['d_soft_hard_gap']=[(fake.mean()-hard.mean()).detach()]
            self.records['d_hard_auc']=[((real[:,None]>hard[None]).float()+.5*(real[:,None]==hard[None]).float()).mean().detach()]

    def finish_update(self):
        self.pending.clear();super().finish_update()

    def state_dict(self):
        return dict(super().state_dict(),gan_options=self.options,last_diagnostic_phase=self.last_diagnostic_phase,
            diagnostic_history=self.diagnostic_history,update_rule='student_then_auxiliary_then_discriminator_then_ema')

    def rank_state(self):return {'gan_rng':self.random.state_dict()}
    def restore_rank_state(self,state):self.random.load_state_dict(state['gan_rng'])

    def validate_state(self,state,rank,step):
        if state.get('version')!=1 or state.get('mode')!=self.mode or state.get('phase_step')!=step or state.get('gan_options')!=self.options:
            raise ValueError('Incompatible GAN stage/options')
        weight=state.get('weight')
        if weight is None:
            if step>self.options['warmup_updates']:raise ValueError('Joint stage requires fixed calibrated GAN weight')
        elif isinstance(weight,bool) or not isinstance(weight,(float,int)) or not math.isfinite(weight) or weight<=0:
            raise ValueError('Invalid saved GAN weight')
        saved=rank.get('gan_rng',{}).get('generator')
        current=self.random.state_dict()['generator']
        if not torch.is_tensor(saved) or saved.shape!=current.shape or saved.dtype!=current.dtype:raise ValueError('Missing or incompatible rank GAN RNG')

    def load_state(self,state,rank,step):
        self.validate_state(state,rank,step)
        self.phase_step=step;self.weight=state['weight'];self.calibration=state['calibration']
        self.last_diagnostic_phase=state['last_diagnostic_phase'];self.diagnostic_history=state['diagnostic_history']
        self.restore_rank_state(rank)
