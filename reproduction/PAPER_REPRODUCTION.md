# Paper reproduction — ModelNet10 / ModelNet40

Script: `train_a100.py`, stock defaults, `--epochs 300`, single seed.
Hardware: NVIDIA H100 NVL (shared), torch 2.11.0+cu128, BF16 + `torch.compile`.
Data: `modelnet40_normal_resampled` restaged as `<class>/<split>/*.txt`;
canonical splits (MN10 3991/908, MN40 9843/2468).

Paper reference: `AAAI_2027_ASP_SNN/AnonymousSubmission2027.tex` Table, and
`a100_ckpts/final_results.json` (the authors' own run output, which matches
the paper table exactly).

## Results

### ModelNet10

| metric | this run | paper | Δ |
|---|---|---|---|
| Teacher (PointTransformer) | 90.75% | 91.19% | −0.44 |
| SPM (fixed-order spiking) | 91.63% | 92.51% | −0.88 |
| **ASP** | **92.07%** | **93.28%** | **−1.21** |
| ASP − SPM | +0.44 pp | +0.77 pp | |
| avg chunks | 3.96 / 4 | 3.89 / 4 | |
| firing rate | 27.11% | 26.27% | |

### ModelNet40

| metric | this run | paper | Δ |
|---|---|---|---|
| Teacher (PointTransformer) | 88.82% | 89.14% | −0.32 |
| SPM (fixed-order spiking) | 87.68% | 89.51% | −1.83 |
| **ASP** | **86.95%** | **89.10%** | **−2.15** |
| ASP − SPM | −0.73 pp | −0.41 pp | |
| avg chunks | 3.84 / 4 | 3.84 / 4 | exact |
| firing rate | 24.22% | 24.45% | |

## Reading

**MN10 clears 90%** at 92.07%, 1.21 pp under the paper. **MN40 lands at 86.95%,
2.15 pp under** the paper's 89.10%.

Both qualitative claims reproduce:

- **MN10: ASP > SPM** (+0.44 pp here, +0.77 pp in the paper) — the learned
  policy beats fixed-order traversal.
- **MN40: ASP < SPM** (−0.73 pp here, −0.41 pp in the paper) — the paper is
  candid that ASP loses on MN40; that sign reproduces.
- **Late exit**: 3.96/4 and 3.84/4 chunks, matching the paper's statement that
  savings come from spike sparsity rather than aggressive early exit. MN40's
  3.8444/4 matches to every reported digit.
- Firing rates land within 0.9 pp on MN10 and 0.2 pp on MN40.

Teachers reproduce closely (−0.44 / −0.32 pp), so the pipeline and data staging
are sound. The residual gap is concentrated in the spiking students, largest on
MN40's SPM (−1.83). Plausible causes, in order:

1. **Single seed.** The paper states "each configuration is a single run;
   reported values are best validation accuracies from the saved run history."
   With no error bars on either side, a 1–2 pp spread is not separable from
   seed noise.
2. **Data source.** The paper used `balraj98/modelnet{10,40}-princeton`, raw
   `.off` meshes surface-sampled by trimesh. I substituted
   `modelnet40_normal_resampled` `.txt` (10k pre-sampled points, subsampled to
   1024) because the server could not reach the Kaggle-auth'd mirror. Same
   splits and point count, different sampling provenance.
3. **`torch.compile` / BF16 numerics** on H100 vs the authors' A100.

## Cost of using the wrong script first

I initially ran `kaggle_asp_mn10_train.py`, which is *not* the paper pipeline:

| | kaggle_asp_mn10 | train_a100 (paper) |
|---|---|---|
| dim / depth / groups | 256 / 8 / 64 | **384 / 12 / 128** |
| batch | 32 × 2 accum | 64 |
| weight decay | 0.05 | 0.1 |
| drop_path | 0.2 | 0.3 |
| label smooth | 0.15 | 0.2 |
| warmup | 20 | 30 |
| SPM baseline | not trained | trained |
| result (MN10 ASP) | **88.22%** | **92.07%** |

Same data, same seed, 3.85 pp apart — the model size and regularization account
for it. The identifying evidence was `a100_ckpts/final_results.json`, whose
numbers match the paper table exactly and which `train_a100.py` writes.

## Artifacts

```
paper_a100/ModelNet10/     teacher_/spm_/asp_ModelNet10_best.pth
paper_a100/ModelNet40/     teacher_/spm_/asp_ModelNet40_best.pth
                           final_results.json, histories.json (300 ep × spm+asp)
paper_a100/logs/           a100_mn10.log, a100_mn40.log
kaggle_script_mn10/        the superseded 88.22% run, for the comparison above
```

`*_latest.pt` (optimizer/resume state, ~220 MB each) were left on the server at
`~/asp_a100_run/ckpts_mn{10,40}/` — pull them only if you want to resume rather
than re-run.

## Caveats

- **Single seed (no error bars).** Same as the paper. Do not read the 1–2 pp
  gaps as a reproduction failure or success without repeats.
- `histories.json` here holds SPM and ASP curves for 300 epochs each, same
  schema as `a100_ckpts/histories.json`, so the two are directly diffable.
- No slice-level dumps for these runs — this pipeline uses M=4 chunks, not the
  K=16-slice suite, so the earlier `slices/` artifact schema does not apply.
