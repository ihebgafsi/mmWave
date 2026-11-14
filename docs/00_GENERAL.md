# General Plan (read first, 3 min)

**Goal:** predict the best of 64 mmWave beams (60 GHz) from sensors; measure beams saved vs exhaustive sweep.
**Data:** DeepSense 6G **scenario 33**: 3,981 samples, 18 sequences, camera / radar / LiDAR / GPS. Label = `unit1_beam`.

## Team = one modality each
| Member | Modality | Doc |
|---|---|---|
| M1 | Camera (+ owns `common/` & splits) | 01_M1_camera.md |
| M2 | Radar | 02_M2_radar.md |
| M3 | LiDAR | 03_M3_lidar.md |
| M4 | GPS | 04_M4_gps.md |

## Phases (no one starts the next before ALL finish the current one)
| Phase | What | Output |
|---|---|---|
| **P0** | M1 builds cache + splits, shares it. Others read literature meanwhile. | `splits.json`, `ds6g-cache` |
| **P1** | Unimodal: **3 experiments each**, simplest -> strongest (E1, E2, E3) | 12 experiments |
| **P2** | Multimodal: **1 fusion method each**, on all 4 modalities | 4 fusion methods |
| **P3** | **1 bonus each** | 4 bonuses |
| **P4** | Paper: methodology + one master table | |

## Rules (shared comparability = the whole point)
1. Only `common/` loaders. Never your own split. Never edit `common/`.
2. Seeds 0,1,2 for every final experiment -> mean ± std (18 sequences = noisy, one seed proves nothing).
3. Tune on **val only**. Test is evaluated once per final run through `evaluate()`.
4. Same hparam budget per experiment: max ~6 configs tried, write them down.
5. After **every** experiment: `save_results(...)` (json + csv), checkpoint, and `notes.md` (5 lines: hypothesis / change / result / failure / next).
6. **For fusion:** each best unimodal run must also save `val_logits`, `test_logits` and penultimate `embeddings` (shuffle=False).
7. EXP IDs: `CAM-E1..3`, `RAD-E1..3`, `LID-E1..3`, `GPS-E1..3`, `FUS-F1..4`, `BON-*`.

## Objective
Be **close to exhaustive sweep (64 beams) while testing far fewer beams**. Success = small **K90 / K95** (beams needed for Top-K >= 90 % / 95 %), not just high Top-1.
Latency is *deduced*, never measured in seconds: a beam test lasts T -> exhaustive = 64T, guided = KT -> gain = 1 - K/64, whatever T is.

## Metrics (all on TEST, same rows, same order for all modalities, mean ± std over seeds 0,1,2)
| Metric | Definition | Computation |
|---|---|---|
| Top-1 / 3 / 5 | true beam ranked <= K | `topk(scores, y, k)` |
| **K90 / K95** | smallest K with Top-K >= 90 % / 95 % | `k_for_target(scores, y, 0.90)` |
| Overhead | K/64 | |
| Gain (= latency saved) | 1 - K/64 | |
| Reference | exhaustive (K=64), majority beam, random | `baselines()` |
Optional (not required): power loss (dB), inference time. All implemented in `common/metrics.py`; `python -m common.aggregate` makes the mean±std table.
Reference to beat: **GPS-E3 test: Top-1 55.6 %, Top-3 85.2 %, Top-5 92.8 %, K90 = 5 (7.8 %, gain 92.2 %), K95 ~ 6.7**. Majority baseline Top-1 = 36.6 %.
The master table must show, for every row: Top-1/3/5, K90, K95, gain, params.

## Facts from the EDA every member must know
- **All camera frames are night** -> no day/night study; camera is the hard modality here.
- 704 indices missing in 3 blocks; **18 sequences** -> splits by sequence only (consecutive frames are near-duplicates).
- GPS bearing correlates with beam at 0.91 (Spearman) -> **GPS is a strong baseline**; fusion must beat it, not just the camera.
- Power file has 62 NaN; ~1% of CSV labels != argmax(power). We keep the CSV label (documented).
- Label never uses beam 1 (63 classes seen) but we keep 64 outputs.

## P2 fusion: which modalities to combine (complementarity, not "throw everything in")
| Modality | Brings | Weak at |
|---|---|---|
| GPS | precise angle/position of the user -> strongest single cue | no scene info, GPS error, interpolated rows |
| LiDAR | precise geometry/range, works at night | no identity/motion, sparse far away |
| Radar | range + velocity (moving target), weather robust | coarse angular resolution |
| Camera | semantics | **night-only data -> expected weakest** |
Hypotheses to test (fixed set for ALL fusion methods, so the 4 methods are comparable):
`GPS+LiDAR` (coarse bearing + precise geometry) | `GPS+Radar` (angle + range/velocity) | `LiDAR+Radar` (sensing only, no GPS dependency) | `GPS+LiDAR+Radar` | `all 4` | `all 4 minus camera`.
Key question: **does camera add anything at night?** and **can a GPS-free combo (LiDAR+Radar[+Camera]) approach GPS?** (deployable without a user-reported position).
Before fusing, M1 runs `python -m common.aggregate --complementarity GPS-E3 LID-E3` for all pairs of best models: share of rows where one is right and the other wrong. Large = complementary = worth fusing.
Each member also adds ONE extra combination of their own choice, justified in `notes.md`.
Primary fusion comparison metric: **K90 and K95**, then Top-1.

| Member | Method | Idea |
|---|---|---|
| M1 | **F1 Late fusion** | weighted/avg of per-modality softmax; all 15 subsets are cheap -> full subset table |
| M2 | **F2 Feature concat** | concat embeddings -> MLP (lit. baseline) |
| M3 | **F3 Gated fusion** | learned per-modality gate (plot which modality the gate trusts) |
| M4 | **F4 Cross-attention** | transformer over modality tokens (Charan et al.) |

## P3 bonus assignment (run on best fusion model)
| Member | Bonus |
|---|---|
| M1 | **Generalization:** leave-sequence-out / 5-fold by sequence (`sequence_kfold`); cross-scenario if another scenario is added (verify format first) |
| M2 | **Robustness:** missing / degraded sensor, training with `mod_drop` |
| M3 | **Future beam:** predict t+1, t+3, t+5 (`horizon`), with/without history |
| M4 | **Temporal tracking:** history K frames, beam-switch jitter; + **adaptive sweep** (sweep K chosen by model confidence -> avg beams swept) |
Spare ideas if someone finishes early: blockage detection from power drop, latency/FLOPs table, calibration.

## Literature (read BEFORE coding, each writes 5 bullets: input repr., model, reported Top-K, what to replicate, what to improve)
| Read | Who |
|---|---|
| Alkhateeb 2023, DeepSense 6G | all |
| Multi-Modal Beam Prediction Challenge 2022 | M1, M4 |
| Demirhan & Alkhateeb 2022 (radar) | M2 |
| Charan et al. 2022 (LiDAR) | M3 |
| Morais 2023 (DeepSense-V2V; GPS) | M4 |
| Blockage prediction paper, ViWi | spare/bonus |

## Repo layout
```
common/  splits/  results/<EXPID>/  exp/<modality>/<EXPID>/{train notebook, notes.md}
```
## Paper outline
Intro > Dataset & preprocessing > Unimodal > Fusion > Bonus studies > Overhead analysis > Limits.
