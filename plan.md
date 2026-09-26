# PLAN — House Prices: MLP bằng PyTorch theo CRISP-DM

## 0. Khảo sát ban đầu và quyết định triển khai — 2026-09-26

`plan.md` là nguồn kế hoạch chính. Giữ nguyên thứ tự **Milestone 1 → 2 → 3 → 4 → 5** và kiến trúc baseline; bổ sung các chi tiết dưới đây để triển khai nhất quán.

### Hiện trạng trước triển khai (để đối chiếu với nhật ký ở cuối file)

| Thành phần | Trạng thái |
|---|---|
| `plan.md` | Đã đọc toàn bộ; chưa milestone nào được nghiệm thu |
| `data/` | Có `train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt` |
| `README.md`, `PROJECT_STRUCTURE.md` | Chưa tồn tại |
| `src/`, `notebooks/`, `configs/`, `scripts/`, `tests/`, `docs/` | Chưa tồn tại; cần bổ sung scaffold và triển khai |
| `requirements.txt`, `.gitignore` | Chưa tồn tại |
| `models/`, `outputs/` | Chưa có model, metrics, biểu đồ hoặc submission của dự án |

Kiểm tra sơ bộ bằng Python standard library trên dữ liệu thật: train có **1.460 dòng × 81 cột**, test có **1.459 dòng × 80 cột**, không có dòng trùng; Id duy nhất và thứ tự test Id khớp sample submission. Có 79 features sau khi bỏ Id và SalePrice. Skewness SalePrice khoảng **1,883**, giảm còn **0,121** sau `log1p`. Id **524** và **1299** là ứng viên outlier cần xem trong EDA, chưa bị loại bỏ.

Đây là kết quả khảo sát ban đầu, chưa thay thế notebook EDA và tài liệu đã chạy/kiểm chứng. Chưa có validation RMSE. Python 3.11 và `uv` có sẵn; môi trường Python đang dùng thiếu các thư viện ML/Jupyter/testing chính. Máy có GTX 1060 6GB nhưng chưa xác minh khả năng chạy PyTorch CUDA.

### Quyết định đã thống nhất

- Giữ dữ liệu tại `data/` và submission chính tại `outputs/submission.csv`, theo kế hoạch gốc; không chuyển sang `data/raw/` hoặc `outputs/submissions/` như scaffold trong prompt.
- Chạy đủ **5 cấu hình E0–E4** theo Milestone 4; Definition of Done được thống nhất với yêu cầu này.
- **Final model vẫn giữ holdout 80/20, seed 42**, dùng cùng split với experiments. Preprocessing chỉ fit trên phần training; không refit trên validation, test hoặc toàn bộ train.csv.
- **Người dùng tự nộp Kaggle**. Agent tạo và kiểm định submission, hướng dẫn nộp; người dùng đã báo score **0.13543** sau khi nộp. Milestone 5 được nghiệm thu theo báo cáo này.
- Chưa áp dụng feature engineering, cross-validation hoặc scheduler trong lượt baseline E0–E4; không tự động xóa outlier.
- Tạo môi trường `.venv` riêng, ghi dependency và phiên bản đã kiểm chứng. Hỗ trợ CPU và CUDA; `device=auto` chỉ chọn CUDA sau kiểm tra forward/backward thực tế, nếu không tương thích thì dùng CPU và ghi rõ thiết bị/lý do.
- Notebook dùng cho EDA, visualization, thử nghiệm và trình bày; logic tái sử dụng ở `src/`. Script và notebook gọi cùng logic, không copy-paste pipeline.
- Dùng `pathlib`, đường dẫn tương đối resolve theo project root và seed cố định cho Python, NumPy, PyTorch/DataLoader. Ghi phiên bản môi trường và thiết bị để đối chiếu reproducibility.

---

## 1. Mục tiêu dự án

Xây dựng một mô hình **Multi-Layer Perceptron (MLP) bằng PyTorch** để dự đoán `SalePrice` cho bài toán Kaggle **House Prices: Advanced Regression Techniques**.

### Mục tiêu chính
- Xây dựng pipeline end-to-end từ dữ liệu thô đến `submission.csv`.
- Tổ chức quy trình theo **CRISP-DM**.
- Huấn luyện MLP trên biến mục tiêu `log1p(SalePrice)`.
- Đánh giá bằng **RMSE trên log-price**.
- Kiểm soát overfitting bằng validation, dropout, weight decay và early stopping.
- So sánh nhiều cấu hình MLP trước khi chọn mô hình cuối.

---

# 2. Quy trình CRISP-DM

## Phase 1 — Business Understanding

### Việc cần làm
- [x] Xác định bài toán là **Supervised Regression**.
- [x] Xác định biến mục tiêu: `SalePrice`.
- [x] Xác định đầu vào: các thuộc tính mô tả căn nhà.
- [x] Xác định metric chính: RMSE trên log-price.
- [x] Xác định đầu ra cuối cùng: file `submission.csv` gồm:
  - `Id`
  - `SalePrice`

### Câu hỏi cần trả lời
- Mô hình đang dự đoán gì?
- Tại sao đây là bài toán hồi quy?
- Tại sao dùng `log1p(SalePrice)`?
- Tiêu chí nào dùng để chọn mô hình tốt nhất?

### Deliverable
- `docs/01_business_understanding.md`

---

## Phase 2 — Data Understanding

### Việc cần làm

#### 2.1. Đọc dữ liệu
- [x] Load `train.csv`.
- [x] Load `test.csv`.
- [x] Kiểm tra:
  - shape
  - tên cột
  - dtype
  - duplicated rows
  - missing values

#### 2.2. Phân loại biến
- [x] Xác định numerical features.
- [x] Xác định categorical features.
- [x] Tách:
  - `Id`
  - `SalePrice`
  - feature columns

#### 2.3. EDA cho target
- [x] Histogram của `SalePrice`.
- [x] Histogram của `log1p(SalePrice)`.
- [x] Tính:
  - mean
  - median
  - std
  - skewness

#### 2.4. EDA cho features
- [x] Top cột có nhiều missing values.
- [x] Correlation giữa numeric features và `SalePrice`.
- [x] Scatter:
  - `GrLivArea` vs `SalePrice`
  - `TotalBsmtSF` vs `SalePrice`
  - `GarageArea` vs `SalePrice`
  - `YearBuilt` vs `SalePrice`
- [x] Boxplot:
  - `OverallQual` vs `SalePrice`
  - `Neighborhood` vs `SalePrice` nếu cần

#### 2.5. Outlier
- [x] Kiểm tra các điểm bất thường rõ ràng.
- [x] Không xóa outlier ngay nếu chưa có lý do.
- [x] Ghi lại quyết định nếu loại dữ liệu.

### Deliverable
- `notebooks/01_eda.ipynb`
- `docs/02_data_understanding.md`

---

## Phase 3 — Data Preparation

> Đây là phase quan trọng nhất đối với MLP.

### 3.1. Tách target và ID

```python
test_ids = test["Id"].copy()

X = train.drop(columns=["Id", "SalePrice"])
y = train["SalePrice"]
```

- [x] Không dùng `SalePrice` làm feature.
- [x] Không dùng `Id` làm feature cho MLP.
- [x] Giữ `test_ids` để tạo submission.

---

### 3.2. Biến đổi target

```python
y_log = np.log1p(y)
```

- [x] MLP dự đoán `log1p(SalePrice)`.
- [x] Khi submission:

```python
pred_price = np.expm1(pred_log)
```

---

### 3.3. Chia train / validation

Baseline:

```text
80% train
20% validation
```

- [x] Dùng `random_state` cố định.
- [x] Chia dữ liệu **trước khi fit preprocessing**.
- [x] Tránh data leakage.

Ví dụ:

```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)
```

---

### 3.4. Xử lý numerical features

Pipeline đề xuất:

```text
Numerical
   ↓
Median Imputation
   ↓
StandardScaler
```

- [x] Missing numeric → median.
- [x] Scale numeric → `StandardScaler`.

Lý do: MLP hoạt động tốt hơn khi các feature có scale tương đối đồng nhất.

---

### 3.5. Xử lý categorical features

Pipeline đề xuất:

```text
Categorical
   ↓
Most Frequent Imputation
   ↓
OneHotEncoder
```

- [x] Missing categorical → most frequent.
- [x] `OneHotEncoder(handle_unknown="ignore")`.

Khi đọc CSV, giữ chuỗi `"None"` hợp lệ của MasVnrType bằng quy tắc missing tường minh, ví dụ `keep_default_na=False, na_values=["", "NA"]`. Baseline xử lý `"NA"` như missing theo pipeline trên; tài liệu EDA phải giải thích rằng nhiều trường hợp mang nghĩa không có tiện ích, và most-frequent imputation làm mất thông tin này.

---

### 3.6. ColumnTransformer

```python
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])
```

Nguyên tắc:

```text
fit       → chỉ X_train
transform → X_train
transform → X_val
transform → X_test
```

---

### 3.7. Chuyển sang Tensor

```python
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32
).view(-1, 1)
```

- [x] Đảm bảo dtype là `float32`.
- [x] Feature shape là `(N, num_features)` và không có NaN/inf sau transform.
- [x] Target shape là `(N, 1)`.

Dùng đầu ra dense cho bộ dữ liệu nhỏ này để đưa vào MLP; ghi kích thước và dung lượng thực tế sau one-hot encoding. Xác định nhóm cột từ training, lưu schema/thứ tự feature và fitted preprocessor để inference dùng lại đúng pipeline.

---

### 3.8. DataLoader

Baseline:

```python
batch_size = 32
```

- [x] `shuffle=True` cho train.
- [x] `shuffle=False` cho validation.
- [x] `shuffle=False` cho test.

Nếu batch train cuối chỉ có một mẫu, bỏ riêng batch đó để tránh lỗi BatchNorm. Khi evaluation, luôn đánh giá toàn bộ train/validation, kể cả mẫu không tham gia batch train cuối.

### Deliverable
- `src/preprocess.py`
- Pipeline preprocessing có thể tái sử dụng.

---

# 4. Phase 4 — Modeling

## 4.1. Baseline MLP

Kiến trúc khởi đầu:

```text
Input
  ↓
Linear(input_dim → 128)
BatchNorm
ReLU
Dropout(0.15)
  ↓
Linear(128 → 64)
BatchNorm
ReLU
Dropout(0.10)
  ↓
Linear(64 → 32)
ReLU
  ↓
Linear(32 → 1)
```

### PyTorch skeleton

```python
class HousePriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

---

## 4.2. Loss function

```python
criterion = nn.MSELoss()
```

Vì model dự đoán log-price:

```text
MSE Loss
   ↓
sqrt
   ↓
RMSE log-price
```

---

## 4.3. Optimizer

Baseline:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

---

## 4.4. Training config baseline

```yaml
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
max_epochs: 500
early_stopping_patience: 30
seed: 42
```

- [x] Xác minh CUDA tương thích; dùng GPU khi chạy được, nếu không dùng CPU và ghi rõ lý do.
- [x] Lưu best checkpoint theo validation RMSE.
- [x] Không lấy model ở epoch cuối nếu epoch trước tốt hơn.

---

## 4.5. Training loop cần có

Mỗi epoch:

```text
TRAIN
  ↓
forward
  ↓
loss
  ↓
backward
  ↓
optimizer.step()

VALIDATION
  ↓
model.eval()
  ↓
no_grad()
  ↓
validation RMSE
```

Cần lưu:

```python
history = {
    "train_loss": [],
    "val_loss": [],
    "train_rmse": [],
    "val_rmse": []
}
```

Model nhận `hidden_dims` và `dropout` từ config. BatchNorm và dropout áp dụng cho các hidden layer trước hidden layer cuối; hidden layer cuối dùng Linear → ReLU, output không có activation. Dropout dạng scalar được áp dụng cho tất cả vị trí dropout; dạng list phải khớp số vị trí.

Mỗi epoch, đo train/validation loss và RMSE trên toàn bộ từng tập bằng `model.eval()` và `torch.no_grad()`; tổng hợp MSE theo số mẫu, không lấy trung bình không trọng số giữa các batch. Early stopping theo validation RMSE, patience 30; reload best checkpoint trước đánh giá cuối.

Checkpoint lưu weights, kiến trúc, input dimension, config, best epoch/RMSE và tham chiếu fitted preprocessor tương ứng. Lưu history CSV trong `outputs/metrics/`; mỗi experiment có artifact riêng để không ghi đè kết quả khác.

### Deliverable
- `src/model.py`
- `src/train.py`
- `models/best_mlp.pt`

---

# 5. Phase 5 — Evaluation

## 5.1. Metric chính

```text
RMSE trên log1p(SalePrice)
```

```python
rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
```

---

## 5.2. Biểu đồ bắt buộc

- [x] Train loss vs Validation loss.
- [x] Train RMSE vs Validation RMSE.
- [x] Actual log-price vs Predicted log-price.
- [x] Residual plot.

---

## 5.3. Theo dõi overfitting

Dấu hiệu:

```text
Train loss ↓
Validation loss ↑
```

Nếu xảy ra:
- tăng dropout
- tăng weight decay
- giảm số hidden units
- early stopping sớm hơn

---

# 6. Kế hoạch thí nghiệm

Không kết luận dựa trên một model duy nhất.

## Experiment 0 — Naive baseline

Mục tiêu:
- kiểm tra toàn bộ pipeline chạy được.

```yaml
architecture: [64, 32]
dropout: 0
lr: 0.001
weight_decay: 0
```

---

## Experiment 1 — MLP baseline

```yaml
architecture: [128, 64, 32]
dropout: [0.15, 0.10]
lr: 0.001
weight_decay: 0.0001
```

---

## Experiment 2 — Mạng lớn hơn

```yaml
architecture: [256, 128, 64]
dropout: [0.20, 0.15]
lr: 0.001
weight_decay: 0.0001
```

---

## Experiment 3 — Learning rate thấp hơn

```yaml
architecture: [128, 64, 32]
dropout: [0.15, 0.10]
lr: 0.0005
weight_decay: 0.0001
```

---

## Experiment 4 — Regularization mạnh hơn

```yaml
architecture: [128, 64, 32]
dropout: [0.25, 0.15]
lr: 0.001
weight_decay: 0.001
```

---

## Bảng theo dõi experiment

| Exp | Architecture | Dropout | LR | Weight Decay | Best Val RMSE | Best Epoch |
|---|---|---:|---:|---:|---:|---:|
| E0 | 64-32 | 0 | 1e-3 | 0 | **0.133316** | 3 |
| E1 | 128-64-32 | [0.15, 0.10] | 1e-3 | 1e-4 | 0.141518 | 10 |
| E2 | 256-128-64 | [0.20, 0.15] | 1e-3 | 1e-4 | 0.136446 | 4 |
| E3 | 128-64-32 | [0.15, 0.10] | 5e-4 | 1e-4 | 0.142573 | 11 |
| E4 | 128-64-32 | [0.25, 0.15] | 1e-3 | 1e-3 | 0.138196 | 24 |

Dùng cùng split, preprocessing và seed cho E0–E4. Có thể dùng lại kết quả E1 từ Milestone 3 nếu config/split không thay đổi và artifact đã kiểm chứng. Chọn cấu hình theo validation RMSE thấp nhất; nếu bằng nhau, ưu tiên mã experiment nhỏ hơn. Không chọn theo train loss. Ghi hạn chế của việc chọn model trên một holdout.

Xuất bảng đầy đủ tại `outputs/metrics/experiments.csv`, gồm experiment, architecture, dropout, learning_rate, weight_decay, best_epoch và best_val_rmse. Notebook experiments gọi logic training trong `src/`, đọc và trình bày bảng kết quả thật.

---

# 7. Phase 6 — Deployment / Kaggle Submission

Sau khi chọn best configuration:

Train lại cấu hình thắng với seed 42 trên cùng phần training 80%, giữ validation 20% và early stopping. Lưu bộ checkpoint/preprocessor final riêng; không fit preprocessing trên holdout và không refit toàn bộ train.csv. Predict dùng best checkpoint của lượt final, không dùng epoch cuối mặc định.

```text
Best preprocessing
       +
Best hyperparameters
       ↓
Train final model
       ↓
Predict test.csv
       ↓
expm1(prediction)
       ↓
submission.csv
       ↓
Kaggle
```

## Tạo submission

```python
model.eval()

with torch.no_grad():
    pred_log = model(X_test_tensor).cpu().numpy().ravel()

pred_price = np.expm1(pred_log)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred_price
})

submission_path = Path("outputs") / "submission.csv"  # Path được import từ pathlib.
submission_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_path, index=False)
```

Đoạn trên minh họa định dạng; implementation phải resolve đường dẫn theo project root, kiểm tra schema và tính hữu hạn của pred_log trước `expm1`, kiểm tra lại sau inverse transform, chặn giá âm về 0 và ghi số trường hợp bị chặn. Chỉ ghi file sau khi validation đạt.

Kiểm tra trước khi submit:

- [x] Đúng 2 cột `Id`, `SalePrice`.
- [x] Không có NaN.
- [x] Không có inf.
- [x] Không có giá âm.
- [x] Số row bằng `test.csv`.
- [x] `Id` giữ nguyên thứ tự của test data.

### Deliverable
- `outputs/submission.csv`

---

# 8. Project Structure triển khai

```text
house-prices-mlp/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_mlp_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── configs/
│   └── baseline.yaml
│
├── scripts/
│   ├── run_train.py
│   └── run_predict.py
│
├── tests/
│
├── models/
│   ├── best_mlp.pt
│   ├── experiments/
│   └── final/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   │   └── experiments.csv
│   └── submission.csv
│
├── docs/
│   ├── 01_business_understanding.md
│   ├── 02_data_understanding.md
│   ├── 03_data_preparation.md
│   ├── 04_modeling.md
│   ├── 05_evaluation.md
│   └── 06_deployment.md
│
├── plan.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── .gitignore
└── README.md
```

`data` phụ trách đọc/kiểm tra/split; `eda` chứa helper EDA; `preprocess` xây dựng/fit/transform pipeline; `dataset` quản lý tensor/DataLoader; `model`, `train`, `evaluate`, `predict` giữ đúng trách nhiệm tương ứng; `config` và `utils` quản lý cấu hình, path và seed. `src/eda.py` bổ sung để tránh đặt logic EDA tái sử dụng trong notebook.

Giao diện CLI dự kiến:

```text
python scripts/run_train.py
python scripts/run_train.py --all-experiments
python scripts/run_train.py --final
python scripts/run_predict.py
python -m pytest
```

Lệnh mặc định train E1; `--all-experiments` chạy E0–E4; `--final` dùng cấu hình thắng đã được xác minh. Prediction mặc định nạp bộ artifact final. Script chỉ parse tham số và gọi logic trong `src/`.

---

# 9. Thứ tự triển khai

### Quy tắc nghiệm thu và cập nhật

- Không bỏ qua milestone. Tests, README và tài liệu CRISP-DM được cập nhật cùng từng milestone, không đợi đến cuối.
- Sau mỗi milestone: ghi file tạo/sửa, lý do, lệnh kiểm chứng và kết quả, metric thật nếu có, vấn đề còn tồn tại; chỉ chuyển bước khi phần hiện tại nhất quán.
- Chỉ đổi `[ ]` thành `[x]` khi deliverable tương ứng đã chạy hoặc được kiểm chứng; không đánh dấu hoàn thành chỉ vì đã viết code.
- Notebook phải chạy từ kernel sạch. Không tạo dữ liệu giả để thay thế kết quả EDA/training/experiments/submission.
- Nếu thiếu dữ liệu: vẫn hoàn thiện code theo thứ tự, ghi rõ bước bị chặn và để checkbox thực thi chưa hoàn thành; không tuyên bố milestone đã đạt.

## Milestone 1 — Hiểu dữ liệu
- [x] Load train/test.
- [x] Kiểm tra shape.
- [x] Kiểm tra missing.
- [x] Phân tích target.
- [x] Hoàn thành EDA cơ bản.

**Done khi:** hiểu được dữ liệu đầu vào và các vấn đề chính cần xử lý.

---

## Milestone 2 — Preprocessing pipeline
- [x] Train/validation split.
- [x] Numeric imputation.
- [x] Categorical imputation.
- [x] StandardScaler.
- [x] OneHotEncoder.
- [x] Transform train/val/test.
- [x] Tensor conversion.

**Done khi:** dữ liệu có thể đưa trực tiếp vào PyTorch MLP.

---

## Milestone 3 — MLP baseline
- [x] Viết `HousePriceMLP`.
- [x] Viết training loop.
- [x] Viết validation loop.
- [x] Tính RMSE.
- [x] Lưu best model.
- [x] Early stopping.

**Done khi:** có một validation RMSE hợp lệ và pipeline train không lỗi.

---

## Milestone 4 — Experiment
- [x] Chạy E0.
- [x] Chạy E1.
- [x] Chạy E2.
- [x] Chạy E3.
- [x] Chạy E4.
- [x] Lưu kết quả vào bảng experiment.

**Done khi:** xác định được cấu hình tốt nhất dựa trên validation RMSE.

---

## Milestone 5 — Kaggle submission
- [x] Train final model.
- [x] Predict test.
- [x] `np.expm1`.
- [x] Tạo `submission.csv`.
- [x] Validate submission.
- [x] Nộp Kaggle.
- [x] Ghi lại Kaggle score: **0.13543** (người dùng báo, 2026-09-26).

**Done khi:** có submission hợp lệ và score đầu tiên trên Kaggle.

Người dùng đã thực hiện bước nộp bài và báo Kaggle score **0.13543**; Milestone 5 đã được nghiệm thu theo báo cáo này. Artifact cục bộ được kiểm định độc lập như nhật ký bên dưới.

---

# 10. Sau baseline: hướng cải thiện

Chỉ thực hiện sau khi baseline MLP chạy ổn.

## Priority 1
- [ ] 5-Fold Cross Validation.
- [ ] Learning-rate scheduler.
- [ ] Tune dropout.
- [ ] Tune hidden layer sizes.
- [ ] Tune weight decay.

## Priority 2
- [ ] Feature engineering:
  - `TotalSF`
  - `TotalBathrooms`
  - `HouseAge`
  - `RemodAge`
  - `TotalPorchSF`
- [ ] Kiểm tra skewed numeric features.
- [ ] Log-transform một số feature lệch mạnh.

## Priority 3
- [ ] So sánh MLP với baseline truyền thống:
  - Linear/Ridge
  - Random Forest
  - Gradient Boosting / XGBoost nếu phạm vi bài cho phép

Mục đích của bước này là xác định MLP thực sự mang lại lợi ích gì trên tabular data.

---

# 11. Definition of Done

Dự án được xem là hoàn thành khi:

- [x] Có EDA rõ ràng.
- [x] Không có data leakage trong preprocessing.
- [x] MLP được xây dựng hoàn toàn bằng PyTorch.
- [x] Có train/validation metrics.
- [x] Có early stopping.
- [x] Đã chạy và so sánh đủ 5 cấu hình MLP E0–E4 trên cùng split.
- [x] Chọn model dựa trên validation RMSE.
- [x] Có biểu đồ training history.
- [x] Có `submission.csv` hợp lệ.
- [x] Có Kaggle score: **0.13543** (người dùng báo).
- [x] Có báo cáo theo 6 phase CRISP-DM.
- [x] Có best checkpoint, fitted preprocessor và training history được kiểm chứng.
- [x] Có đủ bốn biểu đồ evaluation và bảng experiment CSV.
- [x] Các tests tối thiểu đạt; script chạy độc lập với notebook.
- [x] Ba notebook chạy từ kernel sạch và không duplicate logic tái sử dụng trong `src/`.
- [x] README và PROJECT_STRUCTURE mô tả đầy đủ setup, dữ liệu, EDA, training, prediction và vị trí artifact.

### Kiểm thử nghiệm thu tối thiểu

- Model output `(N, 1)`, kể cả inference một mẫu.
- Preprocessing xử lý missing/category chưa thấy, cho ra float32 hữu hạn; không có Id hoặc target trong features.
- Thay đổi validation/test không làm thay đổi thống kê preprocessing đã fit trên training.
- Lưu/đọc lại pipeline và checkpoint giữ nguyên dự đoán trong sai số số học cho phép.
- Early stopping lưu đúng best checkpoint; RMSE khớp phép tính độc lập, kể cả khi batch cuối nhỏ hơn.
- Prediction đúng số lượng/thứ tự; validation submission phát hiện sai cột, sai Id, NaN/inf và giá âm.
- Chạy end-to-end trên dữ liệu thật; CPU được kiểm tra, CUDA được kiểm tra khi tương thích.

---

# 12. Việc cần làm ngay

Bắt đầu từ **Milestone 1**:

```text
1. Bổ sung project structure còn thiếu, tạo môi trường và dependencies
2. Kiểm tra lại train.csv và test.csv đã có trong data/, không ghi đè dữ liệu
3. Tạo notebooks/01_eda.ipynb
4. Load dữ liệu
5. In shape + info + missing values
6. Phân tích SalePrice
7. So sánh SalePrice và log1p(SalePrice)
8. Phân loại numeric/categorical features
```

Sau khi hoàn tất 8 bước trên mới chuyển sang preprocessing và PyTorch.

---

## Nguyên tắc làm bài

> **Làm pipeline đúng trước, tối ưu score sau.**

Ưu tiên theo thứ tự:

```text
Correctness
    ↓
No Data Leakage
    ↓
Reproducibility
    ↓
Validation
    ↓
MLP Tuning
    ↓
Kaggle Score
```

---

# 13. Nhật ký triển khai

## 2026-09-26 — Khảo sát và cập nhật kế hoạch

- File sửa: `plan.md`; chưa tạo/sửa code hoặc dữ liệu.
- Lý do: phản ánh repository thực tế, ghi quyết định holdout/final/Kaggle, thống nhất E0–E4 và đường dẫn, bổ sung tiêu chí kiểm chứng.
- Đã kiểm tra: danh sách toàn bộ file, nội dung kế hoạch và data description; đọc CSV bằng Python standard library để kiểm tra shape, duplicate, Id, schema, thống kê target; kiểm tra Python/package availability và `nvidia-smi`.
- Kết quả chính: dữ liệu thật đủ để bắt đầu Milestone 1; scaffold và dependency còn thiếu như mục 0.
- Validation RMSE: **chưa có — chưa training**.
- Tồn tại: chưa có EDA notebook, pipeline, model, tests hay artifact; chưa xác minh CUDA. Nộp Kaggle và score do người dùng cung cấp sau khi submission được kiểm định.
- Trạng thái: chưa nghiệm thu milestone nào; các checkbox giữ nguyên `[ ]`.

Mỗi milestone tiếp theo thêm một mục nhật ký gồm: ngày, file tạo/sửa, mục đích, lệnh kiểm chứng/kết quả, artifact, best epoch/validation RMSE nếu có, vấn đề tồn tại và checkbox đã được nghiệm thu.

## 2026-09-26 — Milestone 1 nghiệm thu

- File tạo/sửa: `src/{data,eda,utils}.py`, `src/__init__.py`, `notebooks/01_eda.ipynb`, `docs/01_business_understanding.md`, `docs/02_data_understanding.md`, `requirements.txt`, `.gitignore`, `plan.md`.
- Lý do: đọc dữ liệu thật, mô tả target/features/missing/outliers, tạo biểu đồ và tài liệu business/data understanding.
- Kiểm chứng: chạy `run_eda()` trên dữ liệu thật và `jupyter nbconvert --execute notebooks/01_eda.ipynb` từ kernel sạch; đều thành công. Năm biểu đồ đã lưu trong `outputs/figures/`.
- Kết quả: train 1460×81, test 1459×80; 36 numeric + 43 categorical; không duplicate. Skewness target 1.883, log target 0.121. Giữ Id 524 và 1299 để baseline kiểm định.
- Validation RMSE: chưa có, vì chưa huấn luyện.
- Tồn tại: EDA chỉ mô tả tương quan, không diễn giải nhân quả. Baseline imputation sẽ mất tín hiệu “không có tiện ích” ở một số cột.

## 2026-09-26 — Milestone 2 nghiệm thu

- File tạo/sửa: `src/preprocess.py`, `src/dataset.py`, `notebooks/02_preprocessing.ipynb`, `docs/03_data_preparation.md`, `plan.md`.
- Lý do: chia holdout trước khi fit, chuyển 79 feature thành tensor float32 hữu hạn với fitted pipeline tái sử dụng.
- Kiểm chứng: thực thi notebook 02 từ kernel sạch trên dữ liệu thật, thành công. Shapes train/val/test `(1168, 286)`, `(292, 286)`, `(1459, 286)`; target `(1168, 1)`, `(292, 1)`; mọi array hữu hạn. Dense memory khoảng 1.27/0.32/1.59 MiB.
- Validation RMSE: chưa có, vì chưa huấn luyện.
- Tồn tại: kiểm tra hồi quy tự động cho leakage và reload artifact được thêm ở giai đoạn tests; semantics `NA` là hạn chế baseline.

## 2026-09-26 — Milestone 3 nghiệm thu

- File tạo/sửa: `src/{config,model,train,evaluate}.py`, `scripts/run_train.py`, `configs/baseline.yaml`, `docs/04_modeling.md`, `tests/test_pipeline.py`, `plan.md`; artifact `models/best_mlp.pt`, `models/experiments/E1/`, `outputs/metrics/e1_history.csv`, bốn biểu đồ E1.
- Lý do: MLP baseline PyTorch, AdamW/MSE, lịch sử đầy đủ, early stopping và checkpoint tốt nhất. Khởi tạo output bias bằng mean log-price của training fold vì lần chạy zero-intercept đầu tiên kém hơn constant baseline; không thay đổi kiến trúc hoặc dùng validation để khởi tạo.
- Kiểm chứng: `python scripts/run_train.py` trên dữ liệu thật, `pytest -q` (3 passed), tự reload best checkpoint và đối chiếu RMSE với dự đoán độc lập trong training code.
- Kết quả: E1 best epoch **10**, validation RMSE log-price **0.141518**; dừng ở epoch 40. `device=auto` chọn CPU vì installed PyTorch là CPU-only; CUDA forward/backward không khả dụng trong wheel này.
- Tồn tại: chỉ một holdout; E0/E2/E3/E4 chưa chạy. Chưa đo Kaggle score.

## 2026-09-26 — Milestone 4 nghiệm thu

- File tạo/sửa: `notebooks/03_mlp_experiments.ipynb`, `docs/05_evaluation.md`, `plan.md`; artifact `models/experiments/E0`–`E4`, `outputs/metrics/experiments.csv`, `e0_history.csv`–`e4_history.csv`, bốn biểu đồ cho mỗi cấu hình.
- Lý do: so sánh đủ 5 cấu hình trên cùng holdout và preprocessing, chọn theo validation RMSE.
- Kiểm chứng: `python scripts/run_train.py --all-experiments` thành công, notebook 03 chạy từ kernel sạch; mỗi checkpoint reloaded và đối chiếu RMSE trong training code.
- Kết quả: E0 **0.133316** (epoch 3), E1 0.141518 (10), E2 0.136446 (4), E3 0.142573 (11), E4 0.138196 (24). E0 thắng theo validation RMSE.
- Tồn tại: chọn bằng một holdout có thể lạc quan; Kaggle score chưa có. Final model và submission chuyển sang Milestone 5.

## 2026-09-26 — Milestone 5: artifact cục bộ đã kiểm chứng, chờ Kaggle

- File tạo/sửa: `src/predict.py`, `scripts/run_predict.py`, `docs/06_deployment.md`, `README.md`, `PROJECT_STRUCTURE.md`, `tests/test_pipeline.py`, `plan.md`; artifact `models/final/best_mlp.pt`, `models/final/preprocessor.joblib`, `outputs/submission.csv`.
- Lý do: train final E0 trên cùng split, dự đoán test và tạo CSV Kaggle được kiểm định. Ghi hướng dẫn setup/chạy lại và kiến trúc file.
- Kiểm chứng: `python scripts/run_train.py --final`, `python scripts/run_predict.py`, mở lại submission CSV đối chiếu `test.csv`, `python -m pytest -q` (5 passed); cả ba notebook đã chạy từ kernel sạch.
- Kết quả: final E0 best epoch **3**, validation RMSE log-price **0.133316**. Submission 1459 dòng, đúng hai cột và thứ tự Id, toàn bộ giá hữu hạn/không âm, không cần clamp; khoảng giá 49,901.78–767,002.19.
- Thiết bị: CPU; wheel PyTorch 2.14.0+cpu không hỗ trợ CUDA. Phiên bản dependency đã kiểm chứng trong README.
- Trạng thái tại thời điểm tạo artifact: chờ người dùng upload lên Kaggle và cung cấp score; kết quả được ghi ở mục nhật ký tiếp theo.

## 2026-09-26 — Kiểm toán sau triển khai

- File sửa: `notebooks/01_eda.ipynb`, `notebooks/03_mlp_experiments.ipynb`, `src/data.py`, `src/train.py`, `src/dataset.py`, `tests/test_pipeline.py`, `docs/06_deployment.md`, `README.md`, `.gitignore`, `plan.md`.
- Lý do: notebook experiments có thể gọi `src.train` khi chưa có bảng kết quả; EDA hiện dtype/missing theo từng cột; `test_size` trong config được dùng thật; tests kiểm tra thay đổi cả validation/test và checkpoint roundtrip; final run giữ lịch sử/biểu đồ riêng, không ghi đè E0.
- Kiểm chứng: 6 tests đạt; notebook 01 và 03 chạy lại từ kernel sạch; đọc độc lập cả sáu checkpoint (E0–E4, final) và đối chiếu dự đoán holdout, best epoch, RMSE, preprocessing và hình vẽ; chạy lại `--final`/prediction, xác nhận SHA-256 của `experiments.csv` không đổi.
- Kết quả không đổi: E0/final validation RMSE **0.133316**, submission 1459 dòng hợp lệ. Final history và bốn hình final có artifact riêng.
- Tồn tại tại thời điểm kiểm toán: Kaggle upload/score chờ người dùng; các hướng cải thiện ở mục 10 vẫn là tùy chọn sau baseline. Automatic command review chặn lệnh xóa log/notebook copy tạm, nên các file tạm còn trên đĩa; `.gitignore` loại chúng khỏi Git nếu repository được khởi tạo.

## 2026-09-26 — Milestone 5 nghiệm thu sau phản hồi Kaggle

- Người dùng báo đã nộp submission và cung cấp Kaggle score **0.13543**. Đây là kết quả do người dùng cung cấp; không có submission ID hoặc ảnh leaderboard để xác minh độc lập.
- File sửa: `plan.md`, `README.md`, `docs/06_deployment.md`; không thay đổi model, preprocessing hay submission đã kiểm định.
- Đối chiếu: validation RMSE log-price cục bộ của final E0 là **0.133316**; Kaggle score được báo là **0.13543**. Đây là hai tập đánh giá khác nhau, không dùng Kaggle score để chọn lại cấu hình.
- Trạng thái: tất cả checkbox bắt buộc của Milestone 1–5 và Definition of Done đã hoàn thành; mục 10 là cải thiện tùy chọn sau baseline.
