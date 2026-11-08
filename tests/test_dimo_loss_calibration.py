import pytest
import torch
from test_dimo_spatial_gan import case,prepare_gan,deterministic_cpu,base_case
from src.dimo.calibration import calibrate
from src.dimo.checkpoint import training_state,state_digest,rng_state


def test_gan_parameter_calibration_fixed_diagnostics_preserve_all_rng(case,monkeypatch):
    c=case;extension,batch=prepare_gan(c,monkeypatch)
    extension.phase_step=200
    before=state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers))
    random=state_digest((rng_state(),extension.rank_state()))
    result=calibrate(c.acc,c.teacher,c.models,batch,c.sampler,c.config,extension)
    assert result['weight']>0 and 0<result['weighted_parameter_ratio']<=.15+1e-8
    assert result['DDP_ranks']==1 and len(result['samples'])==4
    assert state_digest((rng_state(),extension.rank_state()))==random
    assert state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers))==before
    weight=extension.weight
    result=calibrate(c.acc,c.teacher,c.models,batch,c.sampler,c.config,extension,update_weight=False)
    assert extension.weight==weight and result['weight_recalibrated'] is False
    assert state_digest((rng_state(),extension.rank_state()))==random


def test_pending_calibration_cannot_resume_beyond_warmup(case,monkeypatch):
    c=case;extension,_=prepare_gan(c,monkeypatch)
    extension.phase_step=200
    extension.validate_state(extension.state_dict(),extension.rank_state(),200)
    extension.phase_step=201
    with pytest.raises(ValueError,match='calibrated'):
        extension.validate_state(extension.state_dict(),extension.rank_state(),201)
