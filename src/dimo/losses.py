"""Optional v3 Student supervision; a disabled extension has no runtime work."""
from contextlib import contextmanager
import math
import torch
from src.dimo.gt_ce import encode_target, region_token_ce


def validate_losses(c):
    mode=c.get('training_mode','v3_baseline')
    if mode not in ('v3_baseline','v3_gt_ce','v3_spatial_gan'):
        raise ValueError('Unknown v3 training_mode')
    weight=c.get('gt_ce_weight',0.)
    if weight!='auto' and (isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight<0):
        raise ValueError('gt_ce_weight must be nonnegative finite or auto')
    gan=c.get('spatial_gan',{})
    if not isinstance(gan,dict) or type(gan.get('enabled',False)) is not bool:
        raise ValueError('spatial_gan.enabled must be boolean')
    gt=weight=='auto' or weight>0
    if gt and gan.get('enabled',False):raise ValueError('GT CE and Spatial GAN cannot be enabled together')
    if gt and mode!='v3_gt_ce':raise ValueError('GT CE requires training_mode=v3_gt_ce')
    if gan.get('enabled',False) and mode!='v3_spatial_gan':raise ValueError('GAN requires training_mode=v3_spatial_gan')
    if gan.get('enabled',False):
        from src.dimo.gan_runtime import gan_options
        gan_options(c)
    initialization=c.get('initialization','sft')
    if initialization not in ('sft','finetune'):raise ValueError('initialization must be sft or finetune; strict resume uses --resume')
    if bool(c.get('parent_checkpoint'))!=(initialization=='finetune'):
        raise ValueError('Only finetune initialization requires parent_checkpoint')
    if gt:
        v=c.get('gt_ce_target_gradient_ratio',.15)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=1:
            raise ValueError('Invalid GT gradient calibration target')


@contextmanager
def capture_logits(student):
    """Keep the original output node, not a sibling movedim view of FKL."""
    captured=[]
    bare=getattr(student,'module',student)
    handle=bare.register_forward_hook(lambda module,args,output:captured.append(output))
    try:yield captured
    finally:handle.remove()


class OptionalLosses:
    def __init__(self,config,vq,teacher,device):
        validate_losses(config)
        self.config=config;self.vq=vq;self.teacher=teacher;self.device=device
        self.mode=config['training_mode'];self.phase_step=0;self.calibration=None
        value=config.get('gt_ce_weight',0.)
        self.weight=None if value=='auto' else float(value)
        self.raw=None;self.target=None;self.records={}

    def bind_batch(self,raw,prepared):
        self.raw=raw;self.target=None

    def target_tokens(self,source):
        if self.target is None:
            if self.raw is None:raise RuntimeError('Optional loss requires its aligned raw training batch')
            self.target=encode_target(self.raw,self.vq,source,self.device)
            if self.vq.config.num_vq_embeddings!=self.teacher.config.codebook_size:
                raise ValueError('GT VQ and Student vocabularies differ')
        return self.target

    def needs_calibration(self):return self.weight is None
    def student_active(self):return True

    def unweighted_loss(self,logits,prepared):
        source,original,region,conditions,_=prepared
        if not torch.equal(conditions['target_edit_mask'],original):raise ValueError('Transformer must receive M, never expanded G')
        return region_token_ce(logits,self.target_tokens(source),original)

    def student_loss(self,fkl,logits,prepared,raw_logits=None):
        if self.weight is None:raise RuntimeError('Calibrate GT weight before the first optimizer update')
        gt=self.unweighted_loss(logits,prepared)
        self.records.setdefault('fkl',[]).append(fkl.detach())
        self.records.setdefault('gt_ce',[]).append(gt.detach())
        return fkl+self.weight*gt

    def state_dict(self):
        return dict(version=1,mode=self.mode,phase_step=self.phase_step,weight=self.weight,calibration=self.calibration)

    def rank_state(self):return {}

    def validate_state(self,state,rank,step):
        if state.get('version')!=1 or state.get('mode')!=self.mode or state.get('phase_step')!=step:
            raise ValueError('Incompatible optional-loss stage state')
        weight=state.get('weight')
        if weight is None or not isinstance(weight,(float,int)) or not math.isfinite(weight) or weight<=0:
            raise ValueError('Missing calibrated optional-loss weight')

    def load_state(self,state,rank,step):
        self.validate_state(state,rank,step)
        self.phase_step=state['phase_step'];self.weight=state['weight'];self.calibration=state['calibration']

    def finish_update(self):self.phase_step+=1


def build_losses(config,vq,teacher,device):
    validate_losses(config)
    if config.get('spatial_gan',{}).get('enabled',False):
        from src.dimo.gan_runtime import SpatialLosses
        return SpatialLosses(config,vq,teacher,device)
    value=config.get('gt_ce_weight',0.)
    if value=='auto' or value>0:return OptionalLosses(config,vq,teacher,device)
    return None
