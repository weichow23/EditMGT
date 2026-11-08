"""Explicit private paths, historical defaults and public v3 presets."""
import json
import math
import os
from pathlib import Path
from src.dimo.optimization import FREEZE_DEFAULTS
from src.dimo.origin import git_revision_or_none, sha256
from src.dimo.telemetry import validate_monitoring
from src.mask_utils import MAGICBRUSH_MASK_DECODING

HISTORICAL_DEFAULTS = dict(FREEZE_DEFAULTS, noise_emb_perturb=0.0, lr_scheduler='constant',
                          lr_warmup_steps=0, teacher_cfg=4.0, student_cfg=1.0,
                          auxiliary_cfg=1.0, ema_warmup=False, fake_rounds=1,
                          auxiliary_label_smoothing=0.1, lora_part_enable=True,
                          checkpoints_total_limit=5, checkpoint_milestones=[], preview_every=500)
PATH_ENV = {'SFT_RUN_DIR':'sft_run_dir', 'DIMO_TEACHER_DIR':'teacher_transformer_dir',
            'BASE_MODEL_DIR':'base_model_dir', 'DIMO_OUTPUT_DIR':'output_dir',
            'DIMO_MANIFEST':'data_manifest', 'INTER_EDIT_ROOT':'inter_edit_root',
            'MAGICBRUSH_ROOT':'magicbrush_root', 'DIMO_PARENT_CHECKPOINT':'parent_checkpoint'}


def resolve_config(path, environ=None):
    environ = os.environ if environ is None else environ
    c = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key,value in HISTORICAL_DEFAULTS.items(): c.setdefault(key,value)
    # An explicitly set environment variable overrides its matching config
    # field. It never selects a different source kind or another dataset.
    for env,key in PATH_ENV.items():
        if env in environ: c[key]=environ[env]
    for key in ('base_model_dir','output_dir','sft_run_dir','teacher_transformer_dir',
                'data_manifest','inter_edit_root','magicbrush_root','cache_dir','parent_checkpoint'):
        if c.get(key): c[key]=str(Path(c[key]).expanduser().resolve())
    if c.get('data_roots'):
        c['data_roots']={k:str(Path(v).expanduser().resolve()) if v else '' for k,v in c['data_roots'].items()}
    return c


def validate_config(c):
    from src.dimo.checkpoint import microbatches_per_update
    from src.dimo.losses import validate_losses
    validate_losses(c)
    if c.get('gt_ce_weight',0) or c.get('spatial_gan',{}).get('enabled',False):
        required=dict(teacher_cfg=6.,student_cfg=1.,auxiliary_cfg=1.,student_lr=1e-6,auxiliary_lr=1e-6,
            noise_emb_perturb=.3,lr_scheduler='constant_with_warmup',lr_warmup_steps=100,
            freeze_student_embedding=True,freeze_student_mlm_head=True,
            freeze_auxiliary_embedding=True,freeze_auxiliary_mlm_head=True,
            training_sparse_target_halo=1,training_supervise_target_halo=True)
        if any(c.get(k)!=v for k,v in required.items()):raise ValueError('Optional losses require the current v3 recipe')
    validate_monitoring(c)
    for key in ('micro_batch','gradient_accumulation','max_steps','resolution','save_every','preflight_steps'):
        if type(c[key]) is not int or c[key]<1:raise ValueError(f'{key} must be a positive integer')
    if c['resolution']%16:raise ValueError('resolution must be divisible by 16')
    if c['preflight_steps']>c['max_steps']:raise ValueError('preflight_steps exceeds max_steps')
    if c['mixed_precision'] not in ('no','bf16'):raise ValueError('mixed_precision must be no or bf16')
    if c['objective']!='FKL' or c.get('student_cfg',1)!=1 or c['auxiliary_cfg']!=1:
        raise ValueError('Require FKL and Student/Auxiliary CFG=1')
    for key,lo,hi,open_lo,open_hi in [('teacher_cfg',1,float('inf'),False,True),
            ('noise_emb_perturb',0,1,False,True),('student_initial_mask_ratio',0,1,True,False),
            ('ema_decay',0,1,False,True),('student_lr',0,float('inf'),True,True),
            ('auxiliary_lr',0,float('inf'),True,True),('auxiliary_label_smoothing',0,1,False,False)]:
        v=c.get(key,HISTORICAL_DEFAULTS.get(key))
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or (v<=lo if open_lo else v<lo) or (v>=hi if open_hi else v>hi):
            raise ValueError(f'Invalid {key}: {v}')
    for key in [*FREEZE_DEFAULTS,'ema_warmup','lora_part_enable','training_supervise_target_halo','gradient_checkpointing']:
        if key in c and type(c[key]) is not bool:raise ValueError(f'{key} must be boolean')
    if c['optimizer']!='AdamW' or c['lr_scheduler'] not in ('constant','constant_with_warmup'):
        raise ValueError('Require AdamW and a constant / warmup schedule')
    if c.get('weight_decay')!=0.01 or c.get('adam_betas')!=[0.9,0.999]:
        raise ValueError('Require weight_decay=0.01 and adam_betas=[0.9,0.999]')
    warmup=c.get('lr_warmup_steps',0)
    if type(warmup) is not int or warmup<0 or (c['lr_scheduler']=='constant_with_warmup' and warmup<1):
        raise ValueError('Invalid lr_warmup_steps')
    halo=c.get('training_sparse_target_halo',0)
    if type(halo) is not int or halo<0:raise ValueError('Invalid training_sparse_target_halo')
    if c['dataset_split']!='train' or c['mask_loss_reduction']!='sample' or c['mask_decoding']!=MAGICBRUSH_MASK_DECODING:
        raise ValueError('Require train split, per-sample Q mean and corrected mask decoding')
    if not c.get('expected_profile',{}).get('sparse_target'):raise ValueError('Require explicit sparse Teacher profile')
    for key in ('base_model_dir','output_dir'):
        if not c.get(key):raise ValueError(f'Explicit {key} is required')
    sources=[k for k in ('sft_run_dir','teacher_transformer_dir','teacher_hf','parent_checkpoint') if c.get(k)]
    if len(sources)!=1:raise ValueError('Select exactly one explicit Teacher source')
    base=Path(c['base_model_dir']).resolve()
    if not base.is_dir():raise FileNotFoundError(base)
    out=Path(c['output_dir']).resolve()
    protected=[base]
    if c.get('parent_checkpoint'):protected.append(Path(c['parent_checkpoint']).resolve().parent)
    if c.get('sft_run_dir'): protected.append(Path(c['sft_run_dir']).resolve())
    if c.get('teacher_transformer_dir'): protected.append(Path(c['teacher_transformer_dir']).resolve())
    if c.get('teacher_hf'):
        import re
        spec=c['teacher_hf']
        if not spec.get('repo_id') or not re.fullmatch(r'[0-9a-f]{40}',spec.get('revision','')):
            raise ValueError('teacher_hf requires repo_id and pinned 40-character revision')
        if spec.get('cache_dir'):protected.append(Path(spec['cache_dir']).resolve())
    for path in protected:
        if out==path or path in out.parents or out in path.parents:raise ValueError('output_dir must be disjoint from Teacher/base')
    microbatches_per_update(c)
    if c.get('data_manifest'):
        p=Path(c['data_manifest'])
        if not p.is_file():raise FileNotFoundError(p)
        if not c.get('data_manifest_sha256') or sha256(p)!=c['data_manifest_sha256']:
            raise ValueError('Fixed training manifest changed or SHA256 missing')
        if not c.get('data_roots') or not c.get('expected_dataset_counts'):raise ValueError('Manifest roots/counts required')
        for path in c['data_roots'].values():
            if not path:raise ValueError('Each data root must be explicitly configured')
            if not Path(path).is_dir():raise FileNotFoundError(path)
    elif c.get('data_mode')=='local_stream':
        if c.get('language') not in ('en','all','zh'):raise ValueError('Explicit stream language required')
        probability=c.get('magicbrush_probability')
        if isinstance(probability,bool) or not isinstance(probability,(float,int)) or not 0<probability<1:
            raise ValueError('Local mixed stream requires 0<magicbrush_probability<1')
        for key in ('inter_edit_root','magicbrush_root','cache_dir'):
            if not c.get(key):raise ValueError(f'Local stream requires {key}')
        for key in ('inter_edit_root','magicbrush_root'):
            if not Path(c[key]).is_dir():raise FileNotFoundError(c[key])
    else:raise ValueError('Explicit data_manifest or data_mode=local_stream required; no fallback')
    if type(c.get('checkpoints_total_limit',5)) is not int or c.get('checkpoints_total_limit',5)<1:
        raise ValueError('checkpoints_total_limit must be positive')
    if any(type(x) is not int or x<1 for x in c.get('checkpoint_milestones',[])):
        raise ValueError('checkpoint_milestones must contain positive steps')
    if type(c.get('preview_every',500)) is not int or c.get('preview_every',500)<1:raise ValueError('preview_every must be positive')
    if c.get('output_compositing','none')!='none':raise ValueError('Training previews use direct VQ decode')


def source_metadata():
    import platform,torch
    root=Path(__file__).resolve().parents[2]
    return dict(git_sha=git_revision_or_none(), implementation_checksums={str(p.relative_to(root)):sha256(p)
                for p in sorted((root/'src').rglob('*.py'))}, python=platform.python_version(),
                torch=str(torch.__version__),cuda=torch.version.cuda)
