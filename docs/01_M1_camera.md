# M1: Camera (+ data owner)

## P0 (you block the team: do first, ~1-2 days)
- [ ] Run `python -m common.build_cache --stage all` (see README)
- [ ] Check BEV range (`config.BEV_RANGE`) against EDA x-y density plot before building LiDAR cache; confirm with M3
- [ ] Commit `splits/splits.json`, publish `ds6g-cache`, run `common.sanity`, tell team
- [ ] Check split quality: test sequences cover enough beams? Print test beam histogram.

## P1 Camera (`modalities=["camera"]`, input 3x224x400)
| ID | Experiment | Why |
|---|---|---|
| CAM-E1 | ResNet18 ImageNet, **frozen**, linear head | cheapest baseline |
| CAM-E2 | ResNet50, full fine-tune (literature setup) | lit. reference |
| CAM-E3 | EfficientNet-B0 / MobileNetV3 + stronger augmentation | efficient + best |
Rules: no horizontal flip. Report training curve. Expect weak results (night images): analyse failures by beam range.
Save logits + embeddings of the best one.

## P2: F1 Late fusion
Per-modality softmax of best CAM/RAD/LID/GPS models (their saved logits) -> average and weighted (weights fit on val). Report all-4, drop-one, best pair.

### P2 combos to test (F1)
All 15 subsets (cheap, only logits). Highlight: `GPS+LiDAR`, `GPS+Radar`, `LiDAR+Radar`, `GPS+LiDAR+Radar`, `all 4`, `all 4 minus camera`. Run the complementarity script on every pair first and put the matrix in the paper. Report K90/K95 for each.

## P3 bonus: Generalization
Leave-sequence-out / 5-fold by sequence for best fusion + best unimodal. Report mean ± std over folds. Cross-scenario only if another scenario is added.

## Deliver
`results/CAM-E*`, F1 results, generalization table, `notes.md` per experiment.

**Metrics reminder:** report Top-1/3/5, **K90, K95**, gain = 1-K/64 on test (mean±std, 3 seeds). Goal: K90 below GPS-E3's 5.
