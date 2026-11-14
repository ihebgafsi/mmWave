# M3: LiDAR

Read Charan et al. 2022 first.

## P1 LiDAR
| ID | Modality key | Experiment |
|---|---|---|
| LID-E1 | `lidar` (3x128x128 BEV) | Small CNN, density channel only (simplest) |
| LID-E2 | `lidar` | ResNet18 on 3-channel BEV (density, max z, mean z) |
| LID-E3 | `lidar_pts` (4096x3) | PointNet-style on raw points (or BEV range/resolution ablation) |
Check first: does BEV range 64 m / 128 px suit this scene? If not, tell M1 to rebuild (`config.BEV_RANGE`, `BEV_SIZE`), don't hack locally.
Save logits + embeddings of the best one.

## P2: F3 Gated fusion
Learned per-modality gate (scalar or vector, input-dependent) over the 4 embeddings. Plot average gate per modality. Report all-4, drop-one, best pair.

### P2 combos to test (F3)
Fixed set: `GPS+LiDAR`, `GPS+Radar`, `LiDAR+Radar`, `GPS+LiDAR+Radar`, `all 4`, `all 4 minus camera`. Your focus: gates should show LiDAR taking over when GPS is interpolated/noisy. Extra combo: `LiDAR+Camera`.

## P3 bonus: Future beam prediction
Best fusion (or best LiDAR if fusion too heavy). `BeamDataset(horizon=h)` for h = 1, 3, 5, with `history=0` vs `history=3`. Plot Top-K vs horizon.

## Deliver
`results/LID-E*`, F3 results, future-beam table/plot, `notes.md`.

**Metrics reminder:** report Top-1/3/5, **K90, K95**, gain = 1-K/64 on test (mean±std, 3 seeds). Goal: K90 below GPS-E3's 5.
