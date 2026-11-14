# M2: Radar

Read Demirhan & Alkhateeb 2022 first.

## P1 Radar
| ID | Modality key | Experiment |
|---|---|---|
| RAD-E1 | `radar_ra` (1x256x64) | Small CNN from scratch on range-angle map (simplest) |
| RAD-E2 | `radar` (4x256x250) | ResNet18, 4-channel range-Doppler input (lit. setup) |
| RAD-E3 | best of above | Improved: input ablation (RA vs RD vs both as 2 branches), regularization, tuned head |
Note: radar is a *moving-target* sensor: check what happens on static frames (speed 0). Analyse errors vs `speed_kmph`.
Save logits + embeddings of the best one.

## P2: F2 Feature concatenation
Concat penultimate embeddings of all 4 best unimodal models -> MLP head. Compare frozen vs end-to-end fine-tune (only if time). Report all-4, drop-one, best pair.

### P2 combos to test (F2)
Fixed set: `GPS+LiDAR`, `GPS+Radar`, `LiDAR+Radar`, `GPS+LiDAR+Radar`, `all 4`, `all 4 minus camera`. Your focus: does radar's range/velocity fix GPS failures (e.g. high speed rows)? Extra combo: `Radar+Camera`.

## P3 bonus: Robustness
On the best fusion model: (a) drop each sensor at test, (b) degrade (`transforms=` hook: camera darkening/occlusion patch, LiDAR point dropout, radar noise, GPS noise), (c) retrain with `mod_drop` 0.2/0.5 and show recovery. One robustness plot.

## Deliver
`results/RAD-E*`, F2 results, robustness table/plot, `notes.md`.

**Metrics reminder:** report Top-1/3/5, **K90, K95**, gain = 1-K/64 on test (mean±std, 3 seeds). Goal: K90 below GPS-E3's 5.
