import json
from pathlib import Path
import pytest
from test_dimo_interfaces import origin_dirs
from src.dimo.config import validate_config
from src.dimo.losses import validate_losses


@pytest.mark.parametrize('mode',['v3_baseline','v3_gt_ce','v3_spatial_gan'])
def test_examples_preserve_entire_v3_preset_and_validate(mode,origin_dirs,tmp_path):
    root=Path(__file__).resolve().parents[1];run,base=origin_dirs
    original=json.loads((root/'configs/dimo/region_dimo_v3.json').read_text())
    c=json.loads((root/f'configs/dimo/{mode}.example.json').read_text())
    assert all(c[key]==value for key,value in original.items())
    inter=tmp_path/'inter';magic=tmp_path/'magic';inter.mkdir();magic.mkdir()
    c.update(sft_run_dir=str(run),base_model_dir=str(base),output_dir=str(tmp_path/'output'),
        data_mode='local_stream',inter_edit_root=str(inter),magicbrush_root=str(magic))
    validate_config(c)


@pytest.mark.parametrize('changes',[{'teacher_cfg':4.},{'noise_emb_perturb':0.},{'lr_warmup_steps':0},
    {'freeze_student_embedding':False},{'freeze_auxiliary_mlm_head':False},{'training_sparse_target_halo':0}])
def test_enabled_losses_cannot_silently_use_v2_recipe(changes):
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/dimo/v3_gt_ce.example.json').read_text());c.update(changes)
    with pytest.raises(ValueError,match='v3 recipe'):validate_config(c)


def test_both_enabled_rejected_before_asset_loading():
    with pytest.raises(ValueError,match='cannot be enabled together'):
        validate_losses(dict(training_mode='v3_gt_ce',gt_ce_weight='auto',spatial_gan={'enabled':True}))
