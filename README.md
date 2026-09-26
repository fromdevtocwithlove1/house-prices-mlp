# House Prices MLP (PyTorch)

Dự đoán `SalePrice` trong [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) bằng MLP PyTorch theo 6 phase CRISP-DM. Model học `log1p(SalePrice)`; metric chọn model là validation RMSE trên log-price. `Id` không vào model và được giữ để tạo submission.

## Cấu trúc

- `data/`: `train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt` gốc.
- `src/`: logic đọc dữ liệu, EDA, preprocessing, MLP, training, evaluation và prediction tái sử dụng.
- `configs/baseline.yaml`: split, hyperparameters và năm cấu hình E0–E4.
- `scripts/`: CLI gọi logic trong `src/`.
- `notebooks/`: EDA, preprocessing và trình bày experiments; không chứa pipeline riêng.
- `docs/`: báo cáo sáu phase CRISP-DM.
- `models/`: fitted preprocessor và best checkpoints.
- `outputs/figures/`, `outputs/metrics/`, `outputs/submission.csv`: kết quả.

Chi tiết vai trò file ở [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md); thứ tự triển khai và trạng thái ở [plan.md](plan.md).

## Setup

Yêu cầu Python 3.11. Đặt bốn file Kaggle vào `data/` với đúng tên như trên. Từ thư mục project trên Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` dùng PyTorch CPU wheel để cài đặt ổn định trên máy đã kiểm chứng. Code hỗ trợ CUDA nếu thay bằng wheel CUDA tương thích; `device=auto` chỉ chọn GPU sau một phép thử forward/backward. Môi trường đã kiểm chứng: Python 3.11.15, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, PyTorch 2.14.0+cpu, matplotlib 3.11.2, seaborn 0.13.2.

## Chạy

```powershell
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
python scripts/run_train.py                   # E1 baseline
python scripts/run_train.py --all-experiments # E0–E4, cùng split
jupyter notebook notebooks/03_mlp_experiments.ipynb
python scripts/run_train.py --final           # chọn cấu hình thắng, train lại trên cùng holdout
python scripts/run_predict.py                 # outputs/submission.csv
python -m pytest -q
```

Thứ tự quan trọng: `--all-experiments` tạo bảng so sánh trước `--final`; prediction cần final checkpoint và fitted preprocessor. Các script resolve đường dẫn từ project root nên có thể gọi ở thư mục khác.

## Kết quả đã kiểm chứng

Holdout cố định 80/20, seed 42. Năm thí nghiệm được lưu tại `outputs/metrics/experiments.csv`; E0 thắng với validation RMSE log-price **0.133316** ở epoch 3. Model cuối ở `models/final/best_mlp.pt`, fitted pipeline ở `models/final/preprocessor.joblib`; best E1 baseline cũng có ở `models/best_mlp.pt`. Epoch histories ở `outputs/metrics/*_history.csv` (final dùng `final_history.csv`), các biểu đồ EDA/evaluation ở `outputs/figures/`.

`outputs/submission.csv` có 1.459 dòng, đúng `Id,SalePrice`, Id khớp thứ tự `test.csv`, mọi giá hữu hạn và không âm. Người dùng đã báo **Kaggle score 0.13543** cho submission; score này chưa được xác minh độc lập trong workspace. Giá trị **0.133316** là validation RMSE cục bộ.
