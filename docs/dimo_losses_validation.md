# v3 optional-loss validation

Base: `Shi-qingyu/EditMGT`, `feature/region-dimo-clean`, commit
`9fe0cddd4a703fe59854fcc1175313160f8eaaf8`.

The existing v3 preset, FKL/core functions, Transformer, Sparse Layout and
reference-cache source files are unchanged. Disabled-extension tests compare
complete updates, model/optimizer/scheduler states and RNG with the same v3
baseline. There is no extra target encoding, D initialization or Student
output capture when both losses are disabled.

## Executed checks

- Original and new tests cover M/G/Q, per-sample CE, aligned frozen VQ token
  encoding, disabled behavior, shared Student forward, fresh Auxiliary batches,
  sparse scatter/padding/ROI validity, spatial-order sensitivity, shared
  Real/Fake conditions, Generator gradients and Teacher/D/Student isolation.
- SFT initialization, v3 weights-only parent loading, inherited EMA age,
  fresh optimizer/scheduler semantics, rejection of v2, and pinned parent SFT
  Teacher provenance are covered. Weights-only finetuning is not strict resume.
- GT and GAN checkpoints replay the exact next CPU update, including the
  Generator/D stage boundary, independent GAN RNG and calibrated weight.
  Loading Raw/EMA inference from a GAN checkpoint requires no D runtime.
- Example configurations preserve every original v3 preset field. Combined
  GT/GAN activation and enabled-loss v2 recipes are rejected.
- Python compilation and whitespace checks pass. Existing launcher shell
  syntax is checked without changing the launcher or its GPU configuration.

## Actual two-rank GPU smoke

Executed on two A800 GPUs, BF16/NCCL, gradient checkpointing enabled, with
frozen Student/Auxiliary Embedding/MLM, perturbation0.3, CFG6/1/1, original
AdamW1e-6 and100-update warmup. These are **tiny four-layer random EditMGT
models with a real frozen tiny VQModel and synthetic paired images**.

| Mode | Completed prefix | Result |
|---|---|---|
| FKL-only | 2 Student/Auxiliary updates | PASS |
| GT CE | Training-only calibration, then2 updates | PASS |
| Spatial GAN | 200 FKL+D warmup updates, calibration,2 joint updates | PASS |

Each mode additionally executes a next update, restores the full checkpoint,
replays that update and verifies bitwise identical common state and RNG.
Common states agree across ranks. Student parameters update; Teacher/VQ
parameters remain unchanged. Maximum PyTorch allocated memory per rank was
about0.071GiB for these tiny models; it excludes CUDA/NCCL context memory and
is not a production-memory estimate. A standalone CPU GT smoke also passed.

The GAN smoke's parameter-gradient ratio was calibrated to0.15 with positive
parameter cosine. Its coefficient belongs only to this synthetic test and is
not a production setting. Public examples retain `weight: "auto"`.

## Full-suite issues already present on the base

The complete suite was executed. It has two failures and six setup errors in
this environment, reproduced in the unchanged base worktree:

- Cache-step plotting test: optional `matplotlib` is not installed.
- Inter-Edit analysis adapter test: existing `scripts.test_zero_shot` is absent.
- Six old resume-launcher tests: existing
  `scripts/train_en_cache_magicbrush_no_dilation_resume.sh` is absent.

These unrelated upstream files/tests were not restored, removed or skipped
to make this feature appear green. The complete suite is **not reported as
all passed**. Focused v3 and new loss tests pass; detailed machine-local logs
and counts are listed in the delivery report.

## Not executed

- Full production1024-model GPU smoke. Existing unrelated GPU allocations
  leave insufficient memory for the unchanged complete training setup.
- Three matched500-update production continuations and DEV128 quality
  evaluation. No mature trained v3 Parent was found. An old one-step v3
  migration-preflight checkpoint with accumulation1 was identified and
  explicitly excluded as the production effects-study Parent.
- Production weight calibration or a conclusion about improved Hard editing.

Thus the evidence supports optional-loss wiring, stage/recovery semantics and
tiny distributed updates. It does not establish production-scale memory use,
editing gains, small-mask quality or absence of reward exploitation. Formal
evaluation requires the same trained v3 Parent/data/seeds for all three groups
and the matched FKL-only continuation as the primary control.
