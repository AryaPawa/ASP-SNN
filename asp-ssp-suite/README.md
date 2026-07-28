# ASP/SSP Rigor Suite — Ablations, Strengthening Experiments, Evidence-Backed Theory

Complete evaluation suite for **Active Spiking Perception (ASP)** and its
**Slice Selection Policy (SSP)**: five core ablations (A1–A5), six
strengthening experiments (S1–S6), seven theorem-verification scripts
(V1–V7), and a paper-ready LaTeX theory section in which **every theorem
names the exact executable test that backs it**.

Pure PyTorch (no SNN-library dependency): custom LIF with surrogate
gradients, Gumbel-softmax straight-through selection, margin early exit.

## Layout
```
asp/                 core library (LIF, FPS slicing + 6-D descriptors, SSP,
                     model with early exit, metrics, train/eval)
asp/datasets/        synthetic primitives · ModelNet10/40 · ScanObjectNN
                     (PB-T50-RS) · CIFAR-10/100 patch-slice adapters ·
                     ModelNet40-C-style corruptions
configs/base/        one config per benchmark (incl. cifar10/cifar100 adapted
                     to the SNN pipeline: 4x4 grid of 8x8 patches = K=16
                     slices with a 6-D descriptor analog)
configs/ablations/   A1 theta sweep · A2 masking · A3 geometry components ·
                     A4 d_ssp · A5 membrane vs random/fixed/geometry-only
configs/strengthening/ S1 occlusion · S2 density · S3 adversarial forced
                     start · S4 frozen-policy transfer · S5 calibration ·
                     S6 exit-time audit
experiments/         unified runner + aggregation (95% CIs, Welch t-tests)
verification/        V1–V7: one executable test per theorem (see paper/)
paper/               theory_section.tex (theorems + proofs + verification
                     protocols + claim–evidence map)
tests/               implementation rechecks (run first)
```

## Quickstart
```bash
pip install -r requirements.txt
python tests/test_suite.py                 # recheck implementation
python verification/run_all.py             # execute all theorem checks (CPU, minutes)

# Core ablations on any benchmark (swap the base config freely):
python -m experiments.run --base configs/base/modelnet40.yaml --exp configs/ablations/A1_theta.yaml
python -m experiments.run --base configs/base/cifar100.yaml   --exp configs/ablations/A5_policy.yaml
python -m experiments.run --base configs/base/scanobjectnn.yaml --exp configs/strengthening/S1_occlusion.yaml
python -m experiments.aggregate --dir results/A5_policy/modelnet40 --baseline ssp
```
Every experiment runs on **all** benchmarks by construction: the model is
modality-agnostic (regions + 6-D descriptors), so one config swap covers
ModelNet10/40, ScanObjectNN, CIFAR-10/100, and the synthetic control set.
Use `--seeds 0 1 2 3 4` (default) for the 95% CI protocol.

## Datasets
- **synthetic** — runs immediately (CPU); controlled ground truth for theory checks.
- **modelnet40 / modelnet10** — download `modelnet40_normal_resampled.zip`
  (link in `asp/datasets/modelnet.py`), extract under `./data/`.
- **scanobjectnn** — PB-T50-RS h5 files under `./data/scanobjectnn/main_split/`
  (registration link in the loader).
- **cifar10 / cifar100** — auto-download via torchvision; adapted to the SNN
  pipeline as 16 patch-slices (position, distance-to-center, intensity
  spread, edge density, mean intensity → the 6-D descriptor slot).

## The theory ↔ evidence contract
| Theorem (paper/theory_section.tex) | Verified by | Scale experiment |
|---|---|---|
| T1 selective-risk bound (C−1)(1−θ)/C | V2 | A1, S5 |
| T2 stopping time E[T] ≤ 1+(θ+c−E[M₁])/δ | V1 | A1, S6 |
| T3 submodularity + greedy (1−1/e) | V3 | A5 anytime curves |
| T4 membrane sufficiency (DPI + Pinsker) | V6 | A5, S3 |
| T5 bilinear rank cap ⇒ 2K params exact | V4 | A4 |
| T6 mask coverage / no-mask collapse | V5 | A2 |
| T7 Pareto dominance via margin dominance | A5 KS tests | A1, S1, S2 |
| P8 Gumbel-max consistency identity | V7 | — |
| P9 spread = max-entropy prior | A3 | S2 |

## Notes for the paper
- **2K-parameter claim**: at D=128, d_ssp=64, plain Wk+Wq = 8,576 params.
  Theorem 5 (rank cap ≤ 6) makes the rank-8 factorization (`ssp_rank: 8`)
  **exactly** expressive-equivalent at **1,920 params** — state the claim
  through the factorized variant (config flag already set up in A4).
- The proposal PDF's 7-D descriptor/66K-param SSP variant is a different
  operating point; this suite implements the 6-D/2K spec of the ASP/SSP
  papers and covers the proposal's d_hidden sweep inside A4.
- Verification results at synthetic scale are *mechanism* evidence; the
  paper's benchmark tables must come from GPU runs of the same configs.
