# DeepSense 6G scenario 33: beam prediction


## Preliminary Notes
- Since we wanted to do everything on Kaggle, and since the DeepSense Scenarios need multi-step authentication. Haythem engineered a pipeline that uploaded all the scenarios (+400GB) to Kaggle, accessible via the [Catalog](https://www.kaggle.com/datasets/haythemkrid/deepsense6g-catalog).
- All of our weights, embeddings, and results are found in this [Drive](https://drive.google.com/drive/folders/1m_6KqQr4QVNpg3bj_Tn9IkFqKOc8MPbM)

## Setup (every teammate, Kaggle)
1. Upload this folder as a Kaggle Dataset (or `git clone`). Add the scenario-33 dataset as input.
2. **M1 only, once:** build everything, then publish `/kaggle/working/ds6g` as Kaggle dataset `ds6g-cache` and commit `splits/splits.json`.
   ```
   !pip -q install plyfile tqdm
   %cd /kaggle/working/project
   !python -m common.build_cache --stage all
   ```
3. **Everyone else:** attach `ds6g-cache` and set
   ```
   import os; os.environ["DS6G_WORK"] = "/kaggle/input/ds6g-cache"   # master.csv + cache/ live here
   ```
   (do this BEFORE importing `common`). Then `!python -m common.sanity` must print without error.
4. Never edit `common/` or `splits/`. Need a change -> ask M1.

## Usage
```python
from common.dataset import get_loaders
from common.metrics import seed_everything, evaluate, save_results
tr, va, te = get_loaders(["camera"], batch_size=32)
# ... train ...
m, logits = evaluate(model, te, keys=["camera"])
save_results("CAM-E1", seed, m, config_dict, ckpt_path, params, secs, test_logits=logits)
```
Modalities: `camera` (3,224,400) | `radar` (4,256,250) | `radar_ra` (1,256,64) | `lidar` (3,128,128) | `lidar_pts` (4096,3) | `gps` (5,)

## Metrics / aggregation
`evaluate()` returns top1/3/5, K90, K95, overhead, gain. After runs: `python -m common.aggregate` -> `results/summary.csv` (mean ± std over seeds).


