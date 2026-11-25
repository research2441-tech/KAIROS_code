# KAIROS — velocity-proportional frame and token allocation for long-form action recognition

Code for the manuscript *"Velocity-Proportional Frame and Token Allocation with Coverage Guarantees for Recognizing Transient Actions in Long Videos"*.

A camera-compensated pixel probe decides, before any deep encoding, which frames of a long video are encoded (coverage-floored velocity-quantile sampling). The patch tokens of those frames then compete for one global budget (motion + novelty + learned score, mass-weighted context tokens, proportional attention), and every token carries an encoding of its continuous timestamp, sampling interval and feature velocity.

## Files (flat layout)

| File | Purpose |
|---|---|
| `common.py` | frozen DINOv2 ViT-S/14 encoder, camera-compensated pixel probe, PCA token store |
| `synth.py` | TRACE-10 generator (transient-event long videos with look-alike distractors and camera bumps) |
| `prep.py` | decoding, probe on all raw frames, lattice encoding: `python prep.py {trace,hmdb,ssv2}` |
| `kairos.py` | samplers, token reducers (KAIROS, STTS-style, ToMe, EViT, RLT-style, random), temporal transformer, training loop |
| `plans.py` | every compared method, ablation and sensitivity configuration |
| `run_exp.py` | resumable job runner with stress tests: `python run_exp.py main 0 2` (worker 0 of 2) |
| `analyze.py` | aggregation, calibration, burst hit rates, bootstrap/McNemar/Holm, Friedman–Nemenyi, Wilcoxon |
| `flops.py` | measured GFLOPs (FlopCounterMode) and CPU latency |
| `fig_arch.py`, `fig_sampler.py`, `fig_tokens.py`, `figs_results.py`, `fig_extra.py` | all manuscript figures (grayscale) |
| `merge_chunks.py` | optional: merges chunked extraction (`prep.py hmdb --lo 0 --hi 1000`, ...) on low-memory machines |
| `KAIROS_Colab.ipynb` | end-to-end notebook for a free Google Colab instance |

## Quick start

```bash
pip install torch timm av opencv-python-headless pyarrow scipy matplotlib
# data: HMDB51 split 1 and the SSv2 evaluation subset from the Hugging Face Hub (mteb/HMDB51, mteb/SomethingSomethingV2)
python prep.py trace && python prep.py hmdb && python prep.py ssv2
for p in main abl sens; do python run_exp.py $p 0 2 & python run_exp.py $p 1 2; wait; done
python flops.py && python analyze.py
python fig_arch.py && python fig_sampler.py && python fig_tokens.py && python figs_results.py && python fig_extra.py
```

Raw parquet files are expected in `raw/hmdb/` and `raw/ssv2/` (the notebook downloads them). Results are written to `results/`, figures to `figs/`.

## Data and licences

HMDB51 and Something-Something V2 remain under their original licences; this repository redistributes no video. TRACE-10 is generated deterministically by `synth.py` (seeded per video).

## Reproducibility

Seeds 0, 1, 2 for the main study and the factorial ablation; seed 0 for sensitivity sweeps. Encoder features are computed under bfloat16 autocast; the PCA used to reduce token width is fitted on training clips only.
