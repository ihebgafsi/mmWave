import json, random
import numpy as np, pandas as pd, torch
from . import config as C

KS = (1, 3, 5)


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ranks_of_true(scores, y):
    """1 = true beam ranked first. scores: (N,64) tensor/array, y: (N,) int. Ties count against the model."""
    scores, y = torch.as_tensor(scores), torch.as_tensor(y)
    true = scores.gather(1, y[:, None])
    return (scores > true).sum(1) + 1


def topk(scores, y, k):
    return (ranks_of_true(scores, y) <= k).float().mean().item()


def k_for_target(scores, y, target=0.90):
    """Smallest K with Top-K >= target (K90 / K95). Returns 64 if never reached."""
    r = ranks_of_true(scores, y)
    for k in range(1, C.N_BEAMS + 1):
        if (r <= k).float().mean().item() >= target:
            return k
    return C.N_BEAMS


def metrics_from_logits(L, Y, P=None, ks=KS):
    """Required metrics (same rows, same order for every modality):
    top1/3/5, K90, K95, overhead = K/64, gain = 1-K/64 (= signalling latency saved, independent of T).
    Optional: power loss (dB) if P (power matrix) is given."""
    L, Y = torch.as_tensor(L), torch.as_tensor(Y)
    r = ranks_of_true(L, Y)
    res = {"n": int(len(Y))}
    for k in ks:
        res[f"top{k}"] = (r <= k).float().mean().item()
        res[f"overhead_top{k}"] = k / C.N_BEAMS
        res[f"gain_top{k}"] = 1 - k / C.N_BEAMS
    for t in (90, 95):
        K = k_for_target(L, Y, t / 100)
        res[f"K{t}"] = K; res[f"overhead_K{t}"] = K / C.N_BEAMS; res[f"gain_K{t}"] = 1 - K / C.N_BEAMS
    if P is not None:                                  # optional (not required by the professor)
        top, pm = L.topk(max(ks), dim=1).indices, P.max(1).values
        valid = pm > 0
        for k in ks:
            pk = P.gather(1, top[:, :k]).max(1).values.clamp(min=1e-3)
            res[f"pwr_loss_db_top{k}"] = (10 * torch.log10(pm[valid] / pk[valid])).mean().item()
    return res


@torch.no_grad()
def evaluate(model, loader, device="cuda", keys=None, forward=None):
    """keys=['camera'] -> model(batch['camera']); or pass forward(model,batch,device)->logits.
    Returns (metrics dict, logits tensor). Save the logits: late fusion needs them."""
    model.eval(); Ls, Ys, Ps = [], [], []
    for b in loader:
        out = forward(model, b, device) if forward else model(*[b[k].to(device) for k in keys])
        Ls.append(out.float().cpu()); Ys.append(b["label"]); Ps.append(b["power"])
    L, Y, P = torch.cat(Ls), torch.cat(Ys), torch.cat(Ps)
    return metrics_from_logits(L, Y, P), L


def save_results(exp_id, seed, metrics, config, ckpt_path=None, params=None, train_time_s=None,
                 val_logits=None, test_logits=None, embeddings=None):
    """exp_id e.g. 'CAM-E2'. Writes results/<exp_id>/seed<seed>.json (+ logits/embeddings), appends all_results.csv."""
    d = C.RESULTS / exp_id; d.mkdir(parents=True, exist_ok=True)
    rec = {"exp_id": exp_id, "seed": seed, **metrics, "params": params, "train_time_s": train_time_s,
           "ckpt": str(ckpt_path), "config": config}
    json.dump(rec, open(d / f"seed{seed}.json", "w"), indent=1, default=str)
    for name, arr in (("val_logits", val_logits), ("test_logits", test_logits), ("embeddings", embeddings)):
        if arr is not None:
            np.save(d / f"seed{seed}_{name}.npy", arr.cpu().numpy() if torch.is_tensor(arr) else arr)
    row = pd.DataFrame([{k: v for k, v in rec.items() if k != "config"}])
    f = C.RESULTS / "all_results.csv"
    row.to_csv(f, mode="a", header=not f.exists(), index=False)


def baselines(split="test"):
    """Reference rows for the paper table: random, majority-beam, exhaustive sweep."""
    m = pd.read_csv(C.WORK / "master.csv"); sp = json.load(open(C.SPLIT_FILE))
    tr, te = m[m.idx.isin(sp["train"])], m[m.idx.isin(sp[split])]
    freq = tr.beam0.value_counts().index.tolist()
    return {"random": {f"top{k}": k / 64 for k in KS},
            "majority": {f"top{k}": float(te.beam0.isin(freq[:k]).mean()) for k in KS},
            "exhaustive(64 beams)": {f"top{k}": 1.0 for k in KS} | {"K90": 64, "K95": 64, "overhead": 1.0, "gain": 0.0}}
