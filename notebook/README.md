# Notebook Run Order

Run the notebooks in this order — each step depends on the previous.

## 1. Clustering (pair identification)

| Notebook | Output |
|----------|--------|
| `k_mean_clustering.ipynb` | `dataset/stock_clusters.csv` |
| `dbscan.ipynb` | cluster labels + t-SNE plot |

## 2. Spread Models (trading strategy)

| Notebook | Output |
|----------|--------|
| `spread_models/linear_regression.ipynb` | main fixed results (XOM/CVX) |
| `spread_models/kalman_filtering.ipynb` | dynamic hedge ratio (XOM/CVX) |

## Prerequisites

Make sure `xom_with_log.csv` and `cvx_with_log.csv` exist in `spread_models/` before running step 2:

```bash
ls /Users/wesleylu/Desktop/CS4641/spread_models/*.csv
```

> **Note:** Data files (`.csv`, `.txt`) remain in `spread_models/`. The notebooks in `spread_models/` reference those files by relative path.
