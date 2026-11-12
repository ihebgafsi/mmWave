"""Leakage-safe splits: whole sequences go to ONE split (consecutive frames are near-duplicates)."""
import json
import numpy as np
import pandas as pd


def make_splits(m: pd.DataFrame, seed=0, val_frac=0.15, test_frac=0.15, n_trials=5000):
    N = len(m)
    sizes = m.groupby("seq").size()
    beams_of = m.groupby("seq")["beam0"].apply(set).to_dict()
    rng = np.random.default_rng(seed)
    best, best_score = None, 1e9
    for _ in range(n_trials):
        order = rng.permutation(sizes.index.values)
        te, va, tr, nt, nv = [], [], [], 0, 0
        for s in order:
            if nt < test_frac * N:
                te.append(s); nt += sizes[s]
            elif nv < val_frac * N:
                va.append(s); nv += sizes[s]
            else:
                tr.append(s)
        seen = set().union(*[beams_of[s] for s in tr])
        unseen = sum((m[m.seq == s].beam0.map(lambda b: b not in seen)).sum() for s in te + va) / (nt + nv)
        score = abs(nt / N - test_frac) + abs(nv / N - val_frac) + 2 * unseen
        if score < best_score:
            best_score, best = score, (tr, va, te)
    out = {"seed": seed, "score": float(best_score)}
    for name, seqs in zip(("train", "val", "test"), best):
        out[f"{name}_seqs"] = [int(s) for s in sorted(seqs)]
        out[name] = [int(i) for i in m[m.seq.isin(seqs)].idx]
    return out


def save_splits(sp, path):
    json.dump(sp, open(path, "w"))


def sequence_kfold(m: pd.DataFrame, k=5, seed=0):
    """Bonus/robustness: k folds over sequences. Returns list of (train_idx, test_idx)."""
    seqs = np.random.default_rng(seed).permutation(m.seq.unique())
    folds = np.array_split(seqs, k)
    return [(m[~m.seq.isin(f)].idx.tolist(), m[m.seq.isin(f)].idx.tolist()) for f in folds]


def leave_one_sequence_out(m: pd.DataFrame):
    return [(m[m.seq != s].idx.tolist(), m[m.seq == s].idx.tolist()) for s in sorted(m.seq.unique())]
