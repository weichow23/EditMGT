"""v3 weights-only finetuning is explicitly distinct from full resume."""
import json
from pathlib import Path
import torch
from src.dimo.checkpoint import CHECKPOINT_KIND,verify_checkpoint,read_progress
from src.dimo.origin import load_strict,sha256


def parent_metadata(c):
    if c.get('initialization','sft')!='finetune':return None
    path=Path(c['parent_checkpoint']).resolve()
    header=json.loads((path/'COMPLETE').read_text())
    if header.get('kind')!=CHECKPOINT_KIND:raise ValueError('Finetuning requires a genuine v3 Parent; v2 is not supported')
    progress=read_progress(path,0);recipe=progress['config']
    if recipe.get('validation_only'):raise ValueError('A synthetic smoke checkpoint is not a production v3 Parent')
    if progress.get('kind')!=CHECKPOINT_KIND or progress['global_step']!=header['global_step']:
        raise ValueError('Inconsistent v3 Parent progress')
    required=dict(teacher_cfg=6.,student_cfg=1.,auxiliary_cfg=1.,noise_emb_perturb=.3,
        lr_scheduler='constant_with_warmup',lr_warmup_steps=100,
        freeze_student_embedding=True,freeze_student_mlm_head=True,
        freeze_auxiliary_embedding=True,freeze_auxiliary_mlm_head=True,
        training_sparse_target_halo=1,training_supervise_target_halo=True)
    if any(recipe.get(k)!=v for k,v in required.items()):raise ValueError('Parent does not use the required v3 recipe')
    c['parent_metadata']=dict(checkpoint=str(path),training_sha256=header['checksums']['training.pt'],
        parent_global_step=header['global_step'],ema_age_base=recipe.get('parent_metadata',{}).get('ema_age_base',0)+header['global_step'],
        initialization='weights_only; fresh optimizers/schedulers/data stage; inherited EMA age')
    return c['parent_metadata']


def parent_teacher(c):
    path=Path(c['parent_checkpoint'])
    verify_checkpoint(path)
    origin=json.loads((path.parent/'teacher_origin.json').read_text())
    for name,digest in origin['pinned_teacher_checksums'].items():
        if sha256(path.parent/'teacher'/name)!=digest:raise ValueError('Parent frozen SFT Teacher changed')
    return load_strict(path.parent/'teacher'),dict(origin,parent_checkpoint=str(path.resolve()))


def load_parent_weights(c,models):
    path=Path(c['parent_checkpoint']);verify_checkpoint(path)
    state=torch.load(path/'training.pt',map_location='cpu',weights_only=False,mmap=True)
    if state.get('kind')!=CHECKPOINT_KIND:raise ValueError('Require v3 weights')
    if not {'student','auxiliary','ema'}<=set(state['models']):raise ValueError('Incomplete v3 Parent weights')
    for role in ('student','auxiliary','ema'):
        getattr(models[role],'module',models[role]).load_state_dict(state['models'][role],strict=True)
    # Intentionally no optimizer, scheduler, RNG, data or discriminator load.
