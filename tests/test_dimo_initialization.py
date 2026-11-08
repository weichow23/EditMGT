"""SFT, weights-only v3 parent and strict-stage restoration stay distinct."""
import copy
import json
from pathlib import Path
import pytest
import torch
from test_dimo_update import case,deterministic_cpu
from src.dimo.initialization import parent_metadata,load_parent_weights,parent_teacher
from src.dimo.checkpoint import save_checkpoint,training_state,state_digest


def v3_recipe(c):
    c.update(teacher_cfg=6.,student_cfg=1.,auxiliary_cfg=1.,noise_emb_perturb=.3,
        lr_scheduler='constant_with_warmup',lr_warmup_steps=100,
        freeze_student_embedding=True,freeze_student_mlm_head=True,
        freeze_auxiliary_embedding=True,freeze_auxiliary_mlm_head=True,
        training_sparse_target_halo=1,training_supervise_target_halo=True,max_steps=8)


def test_sft_initialization_has_no_parent_metadata():
    c={};assert parent_metadata(c) is None and c=={}


def test_v3_parent_loads_only_weights_and_inherits_ema_age(case,tmp_path):
    c=case;v3_recipe(c.config)
    with torch.no_grad():
        for model in c.models.values():
            for p in model.parameters():p.add_(.01)
    path=save_checkpoint(c.acc,tmp_path,2,8,c.config,c.models,c.optimizers,c.schedulers)
    source={name:state_digest(m.state_dict()) for name,m in c.models.items()}
    old_opts=state_digest({k:v.state_dict() for k,v in c.optimizers.items()})
    with torch.no_grad():
        for m in c.models.values():
            for p in m.parameters():p.zero_()
    new=dict(c.config,initialization='finetune',parent_checkpoint=str(path))
    metadata=parent_metadata(new);assert metadata['parent_global_step']==2 and metadata['ema_age_base']==2
    load_parent_weights(new,c.models)
    assert source=={name:state_digest(m.state_dict()) for name,m in c.models.items()}
    assert old_opts==state_digest({k:v.state_dict() for k,v in c.optimizers.items()})
    assert c.schedulers['student'].last_epoch==0


def test_parent_v2_label_and_non_v3_recipe_rejected(case,tmp_path):
    c=case;c.config['max_steps']=8
    path=save_checkpoint(c.acc,tmp_path,0,0,c.config,c.models,c.optimizers,c.schedulers)
    config=dict(initialization='finetune',parent_checkpoint=str(path))
    with pytest.raises(ValueError,match='v3 recipe'):parent_metadata(config)
    file=path/'COMPLETE';header=json.loads(file.read_text());header['kind']='region-dimo-v2';file.write_text(json.dumps(header))
    with pytest.raises(ValueError,match='genuine v3'):parent_metadata(config)


def test_parent_teacher_is_pinned_sft_not_student_and_checksums_checked(case,tmp_path):
    from src.dimo.origin import sha256
    c=case;v3_recipe(c.config)
    teacher_dir=tmp_path/'teacher';c.teacher.save_pretrained(teacher_dir,safe_serialization=True)
    hashes={p.name:sha256(p) for p in teacher_dir.iterdir() if p.is_file()}
    (tmp_path/'teacher_origin.json').write_text(json.dumps(dict(pinned_teacher_checksums=hashes)))
    with torch.no_grad():
        for p in c.models['student'].parameters():p.add_(.5)
    path=save_checkpoint(c.acc,tmp_path,0,0,c.config,c.models,c.optimizers,c.schedulers)
    parent=dict(initialization='finetune',parent_checkpoint=str(path))
    loaded,origin=parent_teacher(parent)
    assert state_digest(loaded.state_dict())==state_digest(c.teacher.state_dict())
    assert state_digest(loaded.state_dict())!=state_digest(c.models['student'].state_dict())
    (teacher_dir/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='Teacher changed'):parent_teacher(parent)
