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
