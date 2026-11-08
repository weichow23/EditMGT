# Optional v3 training losses

The existing `region_dimo_v3.json` stays unchanged. Optional losses use the same
Teacher CFG6, Student/Auxiliary CFG1, frozen Embedding/MLM, perturbation0.3,
AdamW1e-6,100-update warmup, FKL, fresh Auxiliary batches and EMA.

Copy an example to a local configuration and fill the same explicit asset,
data and output paths as the baseline. GPU selection remains a launcher setting.
No model weights, datasets or server-specific paths are part of this feature.

```bash
python -m src.dimo.preflight --config local.json
CUDA_VISIBLE_DEVICES=0,1 DIMO_NUM_PROCESSES=2 \
  torchrun --standalone --nproc_per_node=2 -m src.dimo.train --config local.json
```

| Configuration | Student loss | Additional training state |
|---|---|---|
| `v3_baseline.example.json` | Original FKL | None |
| `v3_gt_ce.example.json` | FKL + original-M GT token CE | Calibrated weight and stage |
| `v3_spatial_gan.example.json` | FKL, then FKL + Spatial GAN | D, Adam, constant scheduler, independent RNG and calibrated weight |

`v3_baseline.example.json` disables both additions. `v3_gt_ce.example.json`
uses `gt_ce_weight: "auto"`: before the first update a training-only shadow
batch covering all accumulation microbatches and DDP ranks calibrates the
GT/FKL parameter-gradient norm ratio to0.15. No optimizer update occurs during
calibration; RNG is restored and the main data stream is untouched. The chosen
weight/norms/cosine are written to `calibration-step0.json` and checkpoint state.
The first actual optimizer update includes GT CE. An explicit positive weight
may instead be supplied after separate calibration; no old v2 weight is copied.
`gt_ce_weight: 0` performs no GT VQ encoding or Student output capture.

GT Token CE uses the frozen/eval VQ encoder and aligned preprocessed Target.
It averages CE over original mask M within each sample, then averages samples.
G includes the existing Halo; Q is the existing FKL/Auxiliary submask of G.
GT CE never uses Q or the Halo outside M. Neither GT tokens nor GT images enter
the Student input or replace Auxiliary labels.

## Initialization and recovery

- **SFT:** `initialization: "sft"` (default), original initialization flow.
- **Weights-only v3 finetune:** set `initialization: "finetune"` and
  `parent_checkpoint`, clear the other Teacher source fields. The pinned SFT
  Teacher is verified and loaded from that parent run. Student/Auxiliary/EMA
  weights are loaded; optimizers,100-update LR warmup, data stage and local
  step start fresh. Parent provenance and inherited EMA age are recorded.
  This is not strict resume. v2 checkpoints are rejected.
- **Strict resume:** `python -m src.dimo.train --config local.json --resume latest`.
  Use the same run/config/world size. Models, optimizers, schedulers, RNG,
  consumed data position, stage counter and calibrated weight are restored.
  Do not enable a new objective through strict resume; use weights-only
  finetuning for a new stage. Existing source-fingerprint checks remain strict.

Inference remains the original one-step Raw/EMA Student with direct VQ decode.

## Spatial Teacher-feature GAN

Set `training_mode: "v3_spatial_gan"`, `gt_ce_weight: 0` and
`spatial_gan.enabled: true`. Enabling both new losses is a configuration error.
The default base v3 configuration enables neither loss. GAN independently
encodes paired GT only for its Real branch; disabling GT CE alone does not
disable the GT tokens required by an explicitly enabled GAN.

Real uses GT VQ tokens inside G and Source tokens outside G. Fake uses the
expectation of the frozen **Teacher token embedding**, at softmax temperature1,
inside G and the same Source embedding outside. Both use identical conditions,
GAN noise mask and ratio. The independent checkpointed GAN RNG draws ratios
Uniform(0,0.5); it never consumes the FKL RNG. No VQ decoder/image reward is
used in the differentiable Generator path.

Frozen Teacher joint-block indices4/7/10/13 are captured using scoped hooks.
Existing Sparse Layout scatter restores original 2D coordinates and excludes
ragged padding. A validity-normalized G bounding-box ROI is resized to16x16;
its coverage mask is an additional channel. Four timestep-conditioned
Conv2d/GroupNorm/SiLU heads fuse into a raw real/fake logit. The historical
helper name `restore_and_average` now returns spatial ROIs; it performs no
region mean before the convolution. Teacher attention/cache are unchanged.

Original M is always passed as `target_edit_mask`; the existing Transformer
projects/dilates it once. G is obtained from the original `generation_mask`,
and Q remains the original FKL/Auxiliary subset of G. The GAN uses G and its
Teacher/Halo context, with exact zero Generator logits gradient outside G.

The Generator uses `softplus(-D(fake))`. Frozen Teacher and D parameters still
propagate gradients to their inputs. D uses half the sum of
`softplus(-D(real))` and `softplus(D(fake))`; Student embeddings are detached
in this phase. D consumes the already updated Student outputs on fresh
Auxiliary Batch B, without another Student forward or another data batch.
Order remains Student, fresh-batch Auxiliary, then added D, then original EMA.

New-stage updates1–200 use FKL-only Student, normal Auxiliary/EMA and D
training. After200 completed Student updates, training-only shadow calibration
covers the entire accumulation and all ranks. Its initial parameter-gradient
target is0.15 of FKL, with mean logits ratio caps0.5 per observed mask group and
1.5 per observed sample. The actual ratio can be lower when a cap is active;
the report records the proposed/final weight, norms, cosine and observed mask
areas. A parameter cosine below the configured conflict threshold stops the
run. No DEV/Test image or score selects the weight. An explicit positive weight
is accepted only as a user-supplied separately calibrated value; the examples
use `auto` and no historical GAN coefficient is reused.

From update201, the fixed weighted GAN loss is added. Every50 stage updates,
training-only shadow diagnostics remeasure parameter-gradient norms/cosine
without changing the weight or advancing the main stream/RNG. Logs also include
logits gradient ratio/cosine, D logits, accuracy/AUC, periodic Soft/Hard Fake
scores and original Auxiliary CE. D uses separate Adam1e-6, betas(0,0.999),
weight decay0 and a constant optimizer-update-counted scheduler. This does not
replace the original Student/Auxiliary100-update LR warmup.

Strict resume retains D, all optimizers/schedulers, both RNG streams, data
position, new-stage update counter and fixed calibration. A checkpoint at200
without calibration resumes into calibration, never a restarted warmup or an
uncalibrated joint update. Weights-only finetuning always creates a fresh D and
starts its warmup at stage0 regardless of the parent's cumulative step.

## Bounded validation

```bash
python -m src.dimo.loss_smoke --mode v3_gt_ce --steps 2 --device cpu --output ../smoke-local
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m src.dimo.loss_smoke --mode v3_spatial_gan --steps 202 --device cuda --output ../smoke-gpu
```

This harness uses a tiny four-layer random EditMGT and a real frozen tiny
VQModel on synthetic paired images. It exercises the original v3 recipe and
complete checkpoint continuation, including the full200-update GAN boundary;
it does not validate production1024-memory use or editing quality. See
[validation results](dimo_losses_validation.md) for actual execution and limits.

For an effects study, set all three local configurations to `finetune`, the
same genuine trained v3 Parent, identical data/order/seed and500 Student
updates. Evaluate fixed MagicBrush DEV128, seeds0/1, Raw/EMA direct VQ decode;
compare primarily with the matched FKL-only continuation, report original-M
area groups including<3%, and inspect Hard images. Loss reductions or an
absence of crashes do not establish improved editing. No automatic extension
or production effects run is part of the validation harness.
