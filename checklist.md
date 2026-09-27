# Checklist tiểu luận: Dự đoán giá nhà bằng MLP theo CRISP-DM

> Đánh dấu `[x]` khi đã **viết và kiểm tra** mục tương ứng trong bản báo cáo. Checklist này bám theo yêu cầu ở [instruct.md](instruct.md) và kết quả hiện có của dự án; các đề xuất cải thiện ở cuối chưa phải kết quả đã thực hiện.

## 1. Chuẩn bị tư liệu và bố cục

- [x] Đọc [README.md](README.md), [plan.md](plan.md), [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) và sáu tệp trong `docs/` để thống nhất phạm vi bài.
- [x] Đối chiếu mọi số liệu sẽ dùng với `data/`, notebook đã chạy, [experiments.csv](outputs/metrics/experiments.csv) và các biểu đồ trong `outputs/figures/`; không tự tạo kết quả.
- [x] Lập dàn ý gồm: trang bìa, mục lục, mở đầu, sáu giai đoạn CRISP-DM, kết luận, tài liệu tham khảo và phụ lục nếu cần.
- [x] Đánh số chương và tiểu mục trong bản báo cáo theo dạng **1, 1.1, 1.2, 1.3; 2, 2.1, 2.2...**; dùng cùng hệ thống tiêu đề để tạo mục lục tự động.
- [x] Dự kiến 4.000–5.000 từ cho phần nội dung chính theo `instruct.md`. Gợi ý phân bổ: mở đầu 250–350; Business Understanding 350–450; Data Understanding 650–750; Data Preparation 650–750; Modeling 650–800; Evaluation 850–1.000; Deployment 350–450; kết luận 250–350 từ.
- [x] Làm trang bìa; để trống tên sinh viên, lớp và giảng viên nếu chưa có thông tin.
- [x] Viết mở đầu: lý do chọn đề tài, mục tiêu nghiên cứu, phạm vi dữ liệu Ames/Kaggle, phương pháp MLP và bố cục bài.

### Gợi ý hệ thống đề mục cho bản báo cáo

| Chương | Các tiểu mục đề xuất |
|---|---|
| **1. Mở đầu** | 1.1 Lý do chọn đề tài; 1.2 Mục tiêu; 1.3 Phạm vi và phương pháp |
| **2. Business Understanding** | 2.1 Bài toán; 2.2 Mục tiêu và tiêu chí thành công; 2.3 Phạm vi so sánh mô hình |
| **3. Data Understanding** | 3.1 Nguồn và cấu trúc dữ liệu; 3.2 Chất lượng dữ liệu; 3.3 Phân tích khám phá; 3.4 Ngoại lệ |
| **4. Data Preparation** | 4.1 Chia dữ liệu và biến đổi đích; 4.2 Xử lý biến số; 4.3 Xử lý biến phân loại; 4.4 Kiểm soát rò rỉ dữ liệu |
| **5. Modeling** | 5.1 Kiến trúc MLP; 5.2 Thiết lập E0–E4; 5.3 Quy trình huấn luyện |
| **6. Evaluation** | 6.1 Thước đo; 6.2 So sánh thí nghiệm; 6.3 Phân tích đường học và sai số; 6.4 Giới hạn đánh giá |
| **7. Deployment** | 7.1 Lưu mô hình và tiền xử lý; 7.2 Tạo submission; 7.3 Kết quả Kaggle được báo lại |
| **8. Kết luận** | 8.1 Kết quả chính; 8.2 Hạn chế; 8.3 Hướng phát triển |

Đặt **Tài liệu tham khảo** và **Phụ lục** sau chương 8 theo quy định trình bày của môn học.

## 2. Business Understanding — Hiểu bài toán

- [x] Phát biểu bài toán hồi quy: dự đoán `SalePrice` từ các thuộc tính của nhà; `Id` chỉ dùng để nhận diện bản ghi.
- [x] Nêu mục tiêu thực nghiệm: so sánh các cấu hình MLP trên cùng tập validation và tạo `Id,SalePrice` cho 1.459 nhà trong test.
- [x] Xác định tiêu chí chọn mô hình là RMSE trên `log1p(SalePrice)` của tập validation; giải thích đây là nghiên cứu học thuật và bài nộp Kaggle, chưa phải dịch vụ định giá nhà thực tế.
- [x] Nêu rõ phạm vi thực nghiệm **chỉ so sánh năm cấu hình MLP**, chưa có baseline tuyến tính hoặc cây. Nếu yêu cầu học phần là so sánh nhiều họ mô hình, ghi đây là khoảng trống của nghiên cứu và cân nhắc bổ sung thí nghiệm trước khi nộp.
- [x] Nêu sản phẩm đầu ra: mô hình, bộ tiền xử lý, bảng kết quả và `outputs/submission.csv`.
- [x] Dẫn nguồn [docs/01_business_understanding.md](docs/01_business_understanding.md), [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) và [IBM CRISP-DM](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview).

## 3. Data Understanding — Hiểu dữ liệu

- [x] Mô tả dữ liệu gốc: `train.csv` 1.460 dòng × 81 cột; `test.csv` 1.459 dòng × 80 cột; 79 đặc trưng đầu vào gồm 36 biến số và 43 biến phân loại.
- [x] Trình bày `SalePrice`: trung bình 180.921,2; trung vị 163.000; phân phối lệch phải. Phân tích việc `log1p` làm độ lệch giảm từ khoảng 1,883 xuống 0,121.
- [x] Chèn [target_histograms.png](outputs/figures/target_histograms.png); viết ít nhất một đoạn so sánh phân phối giá gốc và sau biến đổi log.
- [x] Chèn [missing_values.png](outputs/figures/missing_values.png); phân tích các cột thiếu nhiều như `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu` và `LotFrontage`.
- [x] Đọc [data_description.txt](data/data_description.txt); giải thích rằng `NA` ở một số cột có nghĩa là **không có tiện ích**, không đơn thuần là quên ghi dữ liệu.
- [x] Chèn [numeric_scatter.png](outputs/figures/numeric_scatter.png); nhận xét quan hệ của diện tích sinh hoạt, diện tích tầng hầm, garage, năm xây với giá. Phân biệt tương quan với quan hệ nhân quả.
- [x] Nêu các điểm ngoại lệ tiềm năng `Id` 524 và 1299; giải thích quyết định **giữ lại** trong baseline để giữ dữ liệu gốc và quy trình nhất quán. Không khẳng định loại bỏ chúng sẽ cải thiện kết quả khi chưa có thí nghiệm đối chứng.
- [x] Dẫn nguồn [docs/02_data_understanding.md](docs/02_data_understanding.md) và [01_eda-executed.ipynb](notebooks/01_eda-executed.ipynb).

## 4. Data Preparation — Chuẩn bị dữ liệu

- [x] Vẽ hoặc mô tả đúng thứ tự pipeline: bỏ `Id`/tách `SalePrice` → `log1p` biến đích → chia 80/20 với seed 42 → fit tiền xử lý trên train → transform validation/test.
- [x] Ghi đúng kích thước sau chia và biến đổi: train `(1168, 286)`, validation `(292, 286)`, test `(1459, 286)`.
- [x] Giải thích biến số: điền thiếu bằng trung vị rồi `StandardScaler`.
- [x] Giải thích biến phân loại: điền giá trị phổ biến rồi `OneHotEncoder(handle_unknown="ignore")`.
- [x] Nêu rõ imputer, scaler và encoder chỉ được **fit trên 1.168 dòng train**, sau đó dùng lại cho validation/test để tránh rò rỉ ở bước tiền xử lý.
- [x] Giải thích vì sao học trên `log1p(SalePrice)` và dùng `expm1` để trả dự đoán về đơn vị giá bán.
- [x] Nêu hạn chế của cách điền giá trị phổ biến khi `NA` biểu thị “không có tiện ích”; chưa khẳng định đã xử lý theo cách khác.
- [x] Dẫn nguồn [data.py](src/data.py), [preprocess.py](src/preprocess.py), [docs/03_data_preparation.md](docs/03_data_preparation.md) và [02_preprocessing-executed.ipynb](notebooks/02_preprocessing-executed.ipynb).

## 5. Modeling — Xây dựng mô hình

- [x] Giải thích đầu vào 286 chiều và đầu ra một giá trị `log1p(SalePrice)` của MLP.
- [x] Mô tả E0: kiến trúc `286 → 64 → 32 → 1`, BatchNorm/ReLU, dropout `0`, weight decay `0`. E0 vẫn là **MLP**, không phải mô hình dự đoán hằng số.
- [x] Trình bày vai trò của E1–E4 và các thay đổi trong [baseline.yaml](configs/baseline.yaml): số tầng/nút, dropout, learning rate và weight decay.
- [x] Giải thích MSE loss, AdamW, batch size 32, seed 42, tối đa 500 epoch và early stopping sau 30 epoch không cải thiện validation RMSE.
- [x] Nêu cách lưu checkpoint tốt nhất và nạp lại trước khi đánh giá; không chép toàn bộ mã huấn luyện vào thân bài.
- [x] Dẫn nguồn [model.py](src/model.py), [train.py](src/train.py) và [docs/04_modeling.md](docs/04_modeling.md).

## 6. Evaluation — Đánh giá

- [x] Viết công thức RMSE giữa giá trị thực và dự đoán **trên thang `log1p(SalePrice)`**; giải thích điểm này không có đơn vị USD và không phải “sai 13,3%”.
- [x] Lập bảng đủ E0–E4 từ [experiments.csv](outputs/metrics/experiments.csv): kiến trúc, dropout, learning rate, weight decay, epoch tốt nhất và validation RMSE.
- [x] Ghi đúng kết quả: E0 `0,133316` (epoch 3); E1 `0,141518`; E2 `0,136446`; E3 `0,142573`; E4 `0,138196`.
- [x] Chèn [e0_rmse.png](outputs/figures/e0_rmse.png); giải thích E0 tốt nhất ở epoch 3 nhưng tiếp tục chạy đến epoch 33 trước khi early stopping. Train RMSE giảm từ khoảng `0,09707` xuống `0,04075`, còn validation RMSE tăng từ `0,13332` lên `0,14273`, gợi ý overfitting.
- [x] Chèn [e0_actual_predicted.png](outputs/figures/e0_actual_predicted.png) và [e0_residuals.png](outputs/figures/e0_residuals.png); nhận xét độ bám quanh đường lý tưởng, các sai số lớn và vùng giá cực trị mà không suy diễn nguyên nhân chưa kiểm chứng.
- [x] Kết luận **E0 tốt nhất trong năm cấu hình trên holdout này**; không kết luận MLP tốt hơn những thuật toán chưa được thử hoặc một siêu tham số riêng lẻ chắc chắn gây ra chênh lệch.
- [x] Rà soát đường đi dữ liệu khi viết về leakage: imputer/scaler/encoder và bước cập nhật trọng số chỉ fit trên train; test không tham gia huấn luyện; validation được dùng cho early stopping và chọn cấu hình, nên điểm validation sau lựa chọn không còn là đánh giá độc lập.
- [x] Nêu giới hạn: một lần chia 80/20, cùng 292 dòng validation dùng cho early stopping và chọn cấu hình, chưa cross-validation, chưa tập kiểm tra nội bộ độc lập.
- [x] Phân biệt metric cục bộ dùng `log1p` với mô tả metric chính thức của [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation) dùng log của giá bán; không gọi hai điểm là cùng một phép đo trên cùng dữ liệu.

## 7. Deployment — Tạo kết quả sử dụng

- [x] Mô tả bước chọn E0, lưu `models/final/best_mlp.pt` và `models/final/preprocessor.joblib`.
- [x] Ghi chính xác rằng lần chạy `final` dùng lại **cùng split 80/20 và seed 42**, không fit trên toàn bộ 1.460 dòng có nhãn và không phải một lần đánh giá mới.
- [x] Mô tả quá trình đọc `test.csv`, biến đổi bằng preprocessor đã lưu, dự đoán, áp dụng `expm1` và tạo `outputs/submission.csv`.
- [x] Xác nhận CSV có 1.459 dòng, hai cột `Id,SalePrice`, đúng thứ tự `Id`, giá hữu hạn và không âm.
- [x] Nếu nhắc điểm Kaggle `0,13543`, ghi rõ **người dùng báo lại; workspace chưa có bằng chứng để xác minh độc lập**. Không tự gán điểm đó là Public hoặc Private leaderboard.
- [x] Dẫn nguồn [docs/06_deployment.md](docs/06_deployment.md) và [predict.py](src/predict.py).

## 8. Kết luận, tham khảo và kiểm tra bản nộp

- [x] Kết luận trả lời mục tiêu ban đầu: đã tạo pipeline và submission; E0 là cấu hình tốt nhất theo validation hiện có.
- [x] Viết hạn chế và hướng phát triển dưới dạng **đề xuất chưa thực hiện**: xử lý `NA` theo nghĩa “không có tiện ích”, cross-validation, baseline Ridge/tree, feature engineering và phân tích sai số theo phân khúc giá.
- [x] Lập tài liệu tham khảo cho CRISP-DM, Kaggle và những nguồn thật sự đã dùng; không tạo nguồn giả. Phân biệt nguồn bên ngoài với số liệu/biểu đồ do dự án tạo.
- [x] Đánh số, đặt chú thích và ghi nguồn **“Kết quả thực nghiệm của dự án”** cho từng hình; mỗi hình phải có đoạn phân tích tương ứng trong nội dung.
- [x] Kiểm tra bảng E0–E4, mọi số liệu, tên file và phát biểu về validation/Kaggle khớp với artifact gốc.
- [x] Xuất `reports/bao_cao_crisp_dm.docx` với ảnh PNG **nhúng trực tiếp**, rồi xuất `reports/bao_cao_crisp_dm.pdf` từ DOCX.
- [x] Tạo mục lục theo tiêu đề và số trang của bản đã dàn trang; kiểm tra lại sau lần xuất PDF cuối.
- [x] Mở và kiểm tra cả DOCX/PDF: đủ ảnh, bảng, chú thích, công thức, số trang và ký tự tiếng Việt; không có ảnh bị cắt hoặc đường dẫn ảnh thay cho ảnh thật.
