"""One Dataset for everybody.
modalities: any of 'camera','radar','radar_ra','lidar','lidar_pts','gps'
Returns dict: {<modality>: tensor, 'mask': (n_mod,), 'label': long in [0,63], 'power': (64,), 'idx': int}
Rows are ordered by sample idx -> logits saved with shuffle=False are aligned across teammates.
history=K -> each modality tensor gets a leading time dim (K+1, ...) oldest->current (clamped at sequence start).
horizon=h -> label/power are those of frame t+h (future beam); frames without a valid future are dropped.
"""
import json
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from . import config as C

GPS_COLS = ["x_m", "y_m", "dist_m", "sin_b", "cos_b"]
_IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FILES = {"camera": "camera.npy", "radar": "radar.npy", "radar_ra": "radar_ra.npy",
         "lidar": f"lidar_bev_R{int(C.BEV_RANGE)}_S{C.BEV_SIZE}.npy", "lidar_pts": f"lidar_pts_{C.N_PTS}.npy"}


class BeamDataset(Dataset):
    def __init__(self, split="train", modalities=("camera",), history=0, horizon=0, augment=None,
                 transforms=None, mod_drop=0.0, indices=None):
        self.m = pd.read_csv(C.WORK / "master.csv")
        self.mods, self.K, self.h = list(modalities), history, horizon
        self.augment = (split == "train") if augment is None else augment
        self.transforms, self.mod_drop = transforms or {}, mod_drop
        self.stats = json.load(open(C.CACHE / "stats.json"))
        self.power = np.nan_to_num(np.load(C.CACHE / "power.npy"), nan=0.0)
        self.gps = self.m[GPS_COLS].to_numpy(np.float32).copy()
        mu, sd = np.array(self.stats["gps_mean"], np.float32), np.array(self.stats["gps_std"], np.float32)
        self.gps[:, :3] = (self.gps[:, :3] - mu) / sd
        self.seq, self.pos, self.beam0 = self.m.seq.to_numpy(), self.m.pos.to_numpy(), self.m.beam0.to_numpy()
        ids = set(json.load(open(C.SPLIT_FILE))[split]) if indices is None else set(indices)
        rows = self.m.index[self.m.idx.isin(ids)].to_numpy()
        if horizon > 0:
            rows = np.array([r for r in rows if r + horizon < len(self.m) and self.seq[r + horizon] == self.seq[r]])
        self.rows, self._mm = rows, {}

    def __len__(self):
        return len(self.rows)

    def _arr(self, name):
        if name not in self._mm:
            self._mm[name] = np.load(C.CACHE / FILES[name], mmap_mode="r")
        return self._mm[name]

    def _load(self, name, r, crop):
        if name == "gps":
            return torch.from_numpy(self.gps[r])
        a = np.asarray(self._arr(name)[r])
        if name == "camera":
            x = torch.from_numpy(a).permute(2, 0, 1).float() / 255
            ch, cw = C.IMG_CROP; i, j = crop[0], crop[1]
            x = x[:, i:i + ch, j:j + cw]
            if self.augment:                                   # NO horizontal flip: it mirrors the beam label
                x = ((x - 0.5) * crop[2] + 0.5 + crop[3]).clamp(0, 1)
            return (x - _IMN_M) / _IMN_S
        x = torch.from_numpy(a.astype(np.float32))
        if name in ("radar", "radar_ra"):
            mu = torch.tensor(self.stats[f"{name}_mean"]).view(-1, 1, 1)
            sd = torch.tensor(self.stats[f"{name}_std"]).view(-1, 1, 1)
            return (x - mu) / sd
        if name == "lidar_pts":
            return x.clamp(-C.BEV_RANGE, C.BEV_RANGE) / C.BEV_RANGE
        return x                                               # lidar BEV already scaled

    def __getitem__(self, k):
        r = int(self.rows[k])
        H, W = C.IMG_HW; ch, cw = C.IMG_CROP
        if self.augment:
            crop = (np.random.randint(0, H - ch + 1), np.random.randint(0, W - cw + 1),
                    np.random.uniform(0.7, 1.3), np.random.uniform(-0.1, 0.1))
        else:
            crop = ((H - ch) // 2, (W - cw) // 2, 1.0, 0.0)
        start = r - int(self.pos[r])
        frames = [max(r - d, start) for d in range(self.K, -1, -1)]
        out, mask = {}, []
        for name in self.mods:
            xs = [self._load(name, f, crop) for f in frames]
            x = torch.stack(xs) if self.K > 0 else xs[0]
            if name in self.transforms:
                x = self.transforms[name](x)                   # degradation hook (robustness tests)
            keep = not (self.augment and self.mod_drop > 0 and np.random.rand() < self.mod_drop)
            out[name] = x if keep else torch.zeros_like(x)
            mask.append(float(keep))
        t = r + self.h
        out.update(mask=torch.tensor(mask), label=torch.tensor(int(self.beam0[t])),
                   power=torch.from_numpy(self.power[t]), idx=int(self.m.idx.iloc[t]))
        return out


def get_loaders(modalities, batch_size=32, num_workers=2, **kw):
    mk = lambda s, sh: DataLoader(BeamDataset(s, modalities, **kw), batch_size=batch_size, shuffle=sh,
                                  num_workers=num_workers, pin_memory=True, drop_last=False)
    return mk("train", True), mk("val", False), mk("test", False)   # val/test: shuffle=False (alignment!)
