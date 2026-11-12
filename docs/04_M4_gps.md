# M4: GPS (+ final evaluation owner)

Read Morais 2023 and the Challenge 2022 paper first.

## P1 GPS (`modalities=["gps"]`, 5 features: x, y, dist, sin/cos bearing)
| ID | Experiment |
|---|---|
| GPS-E1 | kNN / logistic regression on bearing only (sklearn; still save via `save_results`) |
| GPS-E2 | MLP on all 5 features |
| GPS-E3 | MLP/GRU on `history=K` frames (K = 3, 5, 10) |
Expect GPS to be **very strong** (bearing vs beam rho = 0.91). This is your key finding: every fusion must be compared to it. Check effect of `interpolated_gps` rows.
Save logits + embeddings of the best one.

## P2: F4 Cross-attention fusion
Modality tokens (one per embedding) -> small transformer encoder or cross-attention -> head. Report all-4, drop-one, best pair.

## P3 bonus: Temporal tracking + adaptive sweep
(a) Best fusion with `history=K`: Top-K, and beam-switch jitter (switches/s) vs frame-by-frame.
(b) Adaptive sweep: choose K per sample from model confidence (e.g. smallest K with cumulative prob >= tau) -> avg beams swept vs accuracy curve.

## Also (you): assemble the master results table (`results/all_results.csv` -> mean±std) and the overhead-vs-accuracy plot with baselines.

## Deliver
`results/GPS-E*`, F4 results, tracking + adaptive-sweep results, master table, `notes.md`.
