"""python -m common.aggregate  -> results/summary.csv : mean ± std over seeds, one row per experiment (TEST metrics).
Also: python -m common.aggregate --complementarity A B  -> error-overlap between two experiments' saved test logits."""
import argparse, json, glob
import numpy as np, pandas as pd
from . import config as C
from .metrics import ranks_of_true

COLS = ["top1", "top3", "top5", "K90", "K95", "overhead_K90", "gain_K90", "gain_K95", "params"]


def summary():
    rows = []
    for exp in sorted(p.name for p in C.RESULTS.iterdir() if p.is_dir()):
        recs = [json.load(open(f)) for f in glob.glob(str(C.RESULTS / exp / "seed*.json"))]
        df = pd.DataFrame(recs)
        row = {"exp_id": exp, "n_seeds": len(df)}
        for c in COLS:
            if c in df:
                v = df[c].astype(float)
                row[c] = f"{v.mean():.3f} ± {v.std(ddof=0):.3f}" if c not in ("params",) else int(v.mean())
        rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(C.RESULTS / "summary.csv", index=False); print(out.to_string(index=False))


def complementarity(a, b, seed=0):
    """Share of test rows where A is right (Top-1) and B wrong, and vice versa. High off-diagonal = complementary."""
    y = np.load(C.WORK / "cache" / "test_labels.npy") if (C.WORK / "cache" / "test_labels.npy").exists() else None
    from .dataset import BeamDataset
    ds = BeamDataset("test", ["gps"]); y = ds.beam0[ds.rows]
    La = np.load(C.RESULTS / a / f"seed{seed}_test_logits.npy"); Lb = np.load(C.RESULTS / b / f"seed{seed}_test_logits.npy")
    ra, rb = ranks_of_true(La, y).numpy() == 1, ranks_of_true(Lb, y).numpy() == 1
    print(f"{a} only right: {(ra & ~rb).mean():.3f} | {b} only right: {(~ra & rb).mean():.3f} | both: {(ra & rb).mean():.3f} | none: {(~ra & ~rb).mean():.3f}")
    print(f"oracle Top-1 of union: {(ra | rb).mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--complementarity", nargs=2)
    a = ap.parse_args()
    complementarity(*a.complementarity) if a.complementarity else summary()
