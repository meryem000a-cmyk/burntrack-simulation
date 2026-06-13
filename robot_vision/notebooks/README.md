# notebooks/

Colab-compatible notebooks for training YOLO classifiers.
These assume the VSCode Colab extension syncs the workspace — local files, remote GPU compute.

## Files

| Notebook | Purpose | Device | Batch | Data path |
|----------|---------|--------|-------|-----------|
| `train_yolo11m_colab.ipynb` | Train YOLO11m-cls on 17 species | Colab T4 (GPU) | 256 | `../datasets/final_17species/` |

## Common conventions

- **Data path**: `../datasets/final_17species/` relative to this directory
- **Cache**: `../.cache/cache_17species.mmap` (1-time pre-load, 3.7 GB)
- **Output**: `../models/yolo/runs/species_yolo11m/weights/` — `best.pt`, `last.pt`, `epoch_{N}.pt`
- **Plots**: `../models/yolo/runs/species_yolo11m/` — confusion matrix, results CSV, training curves
- **Kernel**: Colab runtime (T4 GPU) via VSCode Colab extension
- **Ultralytics**: Installed within the notebook if missing (`pip install -q ultralytics`)
- **Checkpoint tail**: `KEEP_LAST_N = 15` — only the last 15 epoch checkpoints are retained, old ones are deleted automatically
