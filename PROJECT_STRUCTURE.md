# Project Structure

| Path | Responsibility |
|---|---|
| `plan.md` | Milestones, implementation decisions, verification journal and open Kaggle steps |
| `data/` | Original Kaggle files; unchanged |
| `src/data.py` | CSV schema validation, test IDs, fixed holdout |
| `src/eda.py` | Descriptive statistics and EDA figures |
| `src/preprocess.py` | Training-only `ColumnTransformer`, serialization, inference transform |
| `src/dataset.py` | `float32` tensors and DataLoaders |
| `src/model.py` | Configurable PyTorch MLP |
| `src/train.py` | Epoch loop, early stopping, best checkpoint, experiments, final run |
| `src/evaluate.py` | Log RMSE, batch prediction and four evaluation plots |
| `src/predict.py` | Final artifact loading, price inversion, submission validation |
| `src/config.py`, `src/utils.py` | YAML config, paths, seed and device selection |
| `configs/baseline.yaml` | Shared settings and E0–E4 hyperparameters |
| `scripts/run_train.py`, `scripts/run_predict.py` | Thin command-line entry points |
| `notebooks/01_eda.ipynb` | Data Understanding displays and plots |
| `notebooks/02_preprocessing.ipynb` | Split/transform dimensions and sanity checks |
| `notebooks/03_mlp_experiments.ipynb` | Real experiment table and selected plots |
| `docs/01`–`06_*.md` | Business Understanding through Deployment |
| `tests/test_pipeline.py` | Shape, leakage, serialization and submission checks |
| `models/experiments/` | Per-experiment best checkpoints and fitted preprocessors |
| `models/final/` | Selected final best checkpoint and fitted preprocessor |
| `outputs/figures/` | EDA and evaluation PNG figures |
| `outputs/metrics/` | Epoch histories and experiment comparison CSV |
| `outputs/submission.csv` | Locally validated Kaggle upload file |

Run the exact commands in `README.md`. Reusable operations live in `src/`; notebooks and scripts import them.
