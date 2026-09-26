# BÁO CÁO DỰ ĐOÁN GIÁ NHÀ BẰNG MLP PYTORCH THEO CRISP-DM

Sinh viên: ............................................................

Lớp: ....................................................................

Giảng viên hướng dẫn: ...............................................

## MỤC LỤC

[Mục lục được cập nhật theo số trang của bản DOCX đã dàn trang.]

## 1. MỞ ĐẦU

Giá bán của một căn nhà chịu ảnh hưởng đồng thời của diện tích, chất lượng xây dựng, vị trí, tuổi công trình và nhiều tiện ích khác. Quan hệ giữa các yếu tố này thường không đơn giản: cùng một mức tăng diện tích có thể gắn với mức tăng giá khác nhau tùy chất lượng hoặc khu vực. Vì vậy, dự đoán giá từ thông tin mô tả nhà ở là một bài toán phù hợp để khảo sát quy trình học máy có giám sát. Bộ dữ liệu House Prices của Kaggle đặt ra nhiệm vụ dự đoán `SalePrice` cho từng `Id` trong tập kiểm tra; kết quả nộp gồm đúng hai cột `Id` và `SalePrice` [1]. Báo cáo này trình bày một dự án sử dụng mạng nơ-ron truyền thẳng nhiều lớp, hay Multilayer Perceptron (MLP), được cài đặt bằng PyTorch để giải quyết nhiệm vụ đó.

Đề tài được tổ chức theo sáu giai đoạn của CRISP-DM: hiểu bài toán, hiểu dữ liệu, chuẩn bị dữ liệu, xây dựng mô hình, đánh giá và triển khai [2]. Cách trình bày này giúp lần theo từng quyết định từ dữ liệu gốc đến tệp dự đoán, đồng thời làm rõ kết quả nào đã được đo trực tiếp và kết quả nào chỉ được báo lại. Mục tiêu không phải chứng minh MLP là phương pháp tốt nhất cho dữ liệu dạng bảng, mà là xây dựng và kiểm chứng một pipeline có thể chạy lại, xử lý dữ liệu thiếu, ngăn rò rỉ thông tin từ validation và so sánh nhiều cấu hình trên cùng một tiêu chí.

Phạm vi thực nghiệm gồm 1.460 căn nhà có giá bán trong `train.csv`, 1.459 căn nhà chưa có nhãn trong `test.csv` và năm cấu hình E0–E4. Dữ liệu được chia một lần thành 80% để huấn luyện và 20% để validation với seed 42. Mô hình học `log1p(SalePrice)` và được chọn bằng RMSE trên thang log của tập validation. Dự án đã tạo tệp submission và người dùng báo điểm Kaggle 0.13543; báo cáo không xem con số đó là kết quả được xác minh độc lập. Các kết luận thực nghiệm chủ yếu dựa trên tệp kết quả, lịch sử huấn luyện, notebook đã chạy và hình có sẵn trong thư mục dự án [3]–[6].

## 2. BUSINESS UNDERSTANDING — HIỂU BÀI TOÁN

Về bản chất, đây là bài toán hồi quy có giám sát. Mỗi quan sát là một căn nhà với các thuộc tính đã ghi nhận, còn biến cần dự đoán là giá bán liên tục `SalePrice`. Đầu ra hữu ích của dự án là một ước lượng giá cho từng nhà trong tập kiểm tra theo đúng thứ tự `Id`. `Id` chỉ dùng để ghép và kiểm tra kết quả nộp; đưa nó vào mạng như một đặc trưng sẽ biến mã định danh thành tín hiệu giả. Ngoài điểm số, dự án cần duy trì khả năng giải thích đường đi của dữ liệu: cột nào bị loại, biến nào được điền thiếu, bộ biến đổi nào được lưu và checkpoint nào tạo ra dự đoán cuối.

Giá bán có phân phối lệch phải, nên tối ưu trực tiếp sai số bình phương trên đơn vị tiền có thể khiến một số giao dịch giá rất cao chi phối hàm mất mát. Dự án biến đổi mục tiêu bằng `y_log = log1p(SalePrice)`, tức `log(1 + SalePrice)`. Biến đổi này nén khoảng cách giữa các mức giá cao và giúp mô hình học trên một thang đo ổn định hơn. Khi xuất submission, dự đoán được đưa về đơn vị giá bằng `expm1`. Đây là phép biến đổi của biến mục tiêu, không phải một thao tác điền thiếu hoặc chuẩn hóa đặc trưng. Giá trị RMSE dùng để chọn mô hình vì thế cũng có đơn vị là log-price, không phải đô la hay một sai số tiền tệ trực tiếp.

Tiêu chí thành công cục bộ là validation RMSE thấp nhất sau khi khôi phục checkpoint tốt nhất của mỗi cấu hình. Năm cấu hình cùng dùng một phép chia dữ liệu, cách tiền xử lý và seed, giúp phép so sánh có ý nghĩa hơn so với năm lần chạy trên những tập khác nhau. Kết quả train loss chỉ hỗ trợ chẩn đoán quá khớp; nó không quyết định cấu hình thắng. Tiêu chí đầu ra là `submission.csv` có đúng hai cột, đủ 1.459 dòng, `Id` nguyên thứ tự và mọi giá dự đoán hữu hạn, không âm. Các tiêu chí này phản ánh cả chất lượng ước lượng lẫn khả năng triển khai một quy trình không làm hỏng định dạng nộp bài [4], [5].

Trang cuộc thi Kaggle mô tả việc đánh giá trên log của giá dự đoán và giá quan sát [1]. Mã nguồn của dự án lại tính RMSE giữa các giá trị `log1p` trên holdout cục bộ [5]. Hai phép đo gần nhau về ý tưởng nhưng không đồng nhất hoàn toàn về công thức, và quan trọng hơn là dùng hai tập dữ liệu khác nhau. Do đó, điểm validation 0.133316 và điểm Kaggle được báo 0.13543 phải được nêu riêng. Không thể suy ra chênh lệch giữa chúng chỉ do chất lượng mô hình thay đổi; tập mẫu và cách tính điểm cũng là những yếu tố khác nhau.

## 3. DATA UNDERSTANDING — HIỂU DỮ LIỆU

Dữ liệu gốc gồm 1.460 dòng và 81 cột ở tập train, 1.459 dòng và 80 cột ở tập test. Tập test không chứa `SalePrice`; sau khi bỏ `Id` và biến mục tiêu, cả hai tập có 79 thuộc tính mô tả nhà. Theo kiểu dữ liệu mà mã đọc CSV nhận được, 36 thuộc tính là số và 43 thuộc tính là phân loại. Không có dòng trùng hoàn toàn, và `Id` không trùng trong từng tệp. Việc kiểm tra hình dạng, tên cột và định danh trước khi phân tích giúp phát hiện sớm sai tệp hoặc sai thứ tự cột, nhất là vì dữ liệu kiểm tra sẽ được dự đoán sau khi mô hình đã huấn luyện [3], [4].

Biến mục tiêu có giá trung bình 180.921,20, trung vị 163.000 và độ lệch chuẩn 79.442,50. Trung bình cao hơn trung vị, phù hợp với đuôi phải trên histogram. Độ lệch skewness của `SalePrice` là 1,883, giảm còn 0,121 sau `log1p`. Các con số này xuất hiện trong notebook EDA đã chạy, đồng thời phù hợp với hai phân phối trong Hình 1 [3]. Hình không chứng minh dữ liệu sau biến đổi là chuẩn tuyệt đối; vẫn có quan sát ở hai đuôi. Tuy vậy, dạng phân phối cân đối hơn là lý do thực nghiệm rõ ràng để dùng log-price làm đích học.

![target_histograms](../outputs/figures/target_histograms.png)

*Hình 1. Phân phối SalePrice trước và sau biến đổi log1p. Nguồn: Kết quả thực nghiệm của dự án.*

Hình 1 cho thấy nhiều căn nhà tập trung ở vùng giá thấp và trung bình, trong khi một số căn giá rất cao kéo dài trục ngang ở biểu đồ gốc. Sau biến đổi, khối quan sát trải đều hơn quanh vùng trung tâm; sự khác biệt giữa các mức giá cực cao không còn chiếm phần lớn thang đo. Khi đọc kết quả mô hình, cần nhớ rằng khoảng cách trên trục log không chuyển thẳng thành một khoảng tiền cố định. Cùng một sai số log có thể tương ứng với mức chênh lệch tiền khác nhau tùy giá trị ban đầu.

Dữ liệu thiếu là vấn đề nổi bật. Trong tập train, `PoolQC` thiếu 1.453 dòng, `MiscFeature` thiếu 1.406, `Alley` thiếu 1.369, `Fence` thiếu 1.179, `FireplaceQu` thiếu 690 và biến số `LotFrontage` thiếu 259. Hình 2 thể hiện chênh lệch lớn giữa vài cột gần như trống và phần còn lại. Từ điển dữ liệu giải thích `NA` của nhiều cột phân loại là “không có” tiện ích, chẳng hạn không có hồ bơi, ngõ hoặc gara [3]. Bộ đọc dữ liệu hiện tại coi chuỗi `NA` và ô rỗng là giá trị thiếu, song giữ chuỗi `None` hợp lệ của `MasVnrType`. Vì thế biểu đồ thiếu phản ánh quy tắc đọc này, không có nghĩa mọi dòng thiếu đều là lỗi thu thập dữ liệu.

![missing_values](../outputs/figures/missing_values.png)

*Hình 2. Hai mươi cột có số dòng thiếu lớn nhất trong tập train. Nguồn: Kết quả thực nghiệm của dự án.*

Hình 2 gợi ý rằng lựa chọn điền giá trị phổ biến nhất cho biến phân loại có một đánh đổi cụ thể: khi `NA` mang nghĩa “không có”, thao tác điền có thể biến thông tin về sự vắng mặt thành một loại tiện ích phổ biến. Dự án vẫn giữ cách xử lý nhất quán này để có baseline rõ ràng và tránh sửa quy tắc giữa các thí nghiệm. Ở hướng phát triển, phân biệt “không có” với “không biết” có thể là một thay đổi đáng kiểm chứng bằng cross-validation, nhưng không nên trình bày nó như cải thiện đã được đo.

Đối với quan hệ giữa thuộc tính số và giá, các hệ số tương quan lớn nhất ghi trong EDA là `OverallQual` 0,791, `GrLivArea` 0,709, `GarageArea` 0,623, `TotalBsmtSF` 0,614 và `YearBuilt` 0,523 [3]. Đây là tương quan tuyến tính trên dữ liệu quan sát, không chứng minh quan hệ nhân quả. Chất lượng tổng thể và diện tích có vẻ chứa tín hiệu quan trọng; đồng thời những biến này có thể liên hệ với nhau. MLP có khả năng kết hợp nhiều đầu vào sau mã hóa, nhưng nó chỉ học từ các mẫu hiện có và không tự bảo đảm khả năng ngoại suy tốt cho căn nhà rất khác tập train.

![numeric_scatter](../outputs/figures/numeric_scatter.png)

*Hình 3. Quan hệ giữa bốn thuộc tính số và SalePrice. Nguồn: Kết quả thực nghiệm của dự án.*

Hình 3 cho thấy xu hướng giá tăng theo diện tích sử dụng trên mặt đất, diện tích tầng hầm và diện tích gara, nhưng đám mây điểm phân tán rộng. Với `YearBuilt`, nhà mới hơn thường nằm ở vùng giá cao hơn, song năm xây dựng một mình không giải thích được toàn bộ biến thiên. Trên biểu đồ `GrLivArea`, hai căn có diện tích trên 4.000 foot vuông mà giá dưới 300.000, mang `Id` 524 và 1299, nổi bật như ứng viên ngoại lệ. Dự án giữ nguyên chúng: loại điểm bất thường chỉ vì chúng làm sai số lớn sẽ khiến đánh giá đẹp hơn theo cách khó biện minh. Cần khảo sát nguyên nhân và đo tác động trên một quy trình đánh giá phù hợp trước khi quyết định loại bỏ.

## 4. DATA PREPARATION — CHUẨN BỊ DỮ LIỆU

Giai đoạn chuẩn bị bắt đầu bằng việc tách `SalePrice` khỏi ma trận đặc trưng, bỏ `Id` khỏi đầu vào mạng và giữ bản sao `Id` của tập test để tạo submission. Sau `log1p`, dữ liệu train được chia ngẫu nhiên thành 1.168 dòng huấn luyện và 292 dòng validation với `test_size=0.2`, `random_state=42`. Thao tác chia diễn ra trước khi fit bộ tiền xử lý. Đây là ranh giới quan trọng: nếu dùng thống kê của toàn bộ train.csv để điền thiếu hoặc chuẩn hóa, tập validation sẽ gián tiếp tác động đến mô hình dù không tham gia bước cập nhật trọng số, làm cho ước lượng holdout kém tin cậy [4], [5].

`ColumnTransformer` của dự án tạo hai nhánh. Nhánh biến số điền giá trị thiếu bằng trung vị tính trên phần huấn luyện, rồi dùng `StandardScaler` để đưa mỗi cột về thang đo thích hợp cho tối ưu bằng gradient. Trung vị ít bị kéo bởi giá trị cực đoan hơn trung bình, phù hợp với dữ liệu nhà ở có các cột diện tích và kích thước lệch. Việc scale đặc trưng không xóa thông tin về thứ tự lớn nhỏ của từng biến; nó thay đổi đơn vị biểu diễn để các trọng số của mạng không phải xử lý những cột có độ lớn rất khác nhau. Các giá trị trung vị, trung bình và độ lệch chuẩn đều được học trên 1.168 dòng train fold.

Nhánh phân loại điền thiếu bằng giá trị xuất hiện nhiều nhất rồi dùng `OneHotEncoder(handle_unknown="ignore")`. One-hot chuyển mỗi mức phân loại thành một chiều nhị phân, tránh gán thứ tự số học tùy tiện cho những nhãn như khu vực hoặc vật liệu. Tùy chọn bỏ qua mức chưa gặp khi biến đổi validation và test giúp pipeline không thất bại nếu tập mới có một nhãn ngoài tập fit. Đầu ra của cả hai nhánh được ghép thành một ma trận đặc trưng; bộ biến đổi đã fit và thứ tự cột đầu vào được lưu chung để phép suy luận về sau dùng đúng cấu trúc đã học. Nhược điểm của baseline là điền mode cho `NA` có ý nghĩa “không có”; báo cáo giữ rõ giới hạn đó thay vì coi đây là cách xử lý duy nhất hợp lý [3]–[5].

Notebook tiền xử lý đã chạy cho thấy ba ma trận train, validation và test lần lượt có kích thước `(1168, 286)`, `(292, 286)` và `(1459, 286)`. Số chiều tăng từ 79 lên 286 chủ yếu do mã hóa one-hot. Pipeline tạo đầu ra dense vì bộ dữ liệu tương đối nhỏ: các mảng `float32` chiếm khoảng 1,27 MiB, 0,32 MiB và 1,59 MiB. Giá trị sau biến đổi được kiểm tra hữu hạn, còn nhãn log có dạng `(N, 1)` để khớp đầu ra vô hướng của MLP. Kiểm tra hình dạng và dtype ở đây không chỉ mang tính kỹ thuật; chúng ngăn lỗi phát sinh muộn trong phép nhân ma trận hoặc khi tính loss [6].

Các `DataLoader` dùng batch size 32. Train loader trộn thứ tự bằng bộ sinh có seed cố định, còn validation và test không trộn; dự đoán test vì vậy vẫn khớp thứ tự `Id` gốc. BatchNorm cần nhiều hơn một ví dụ trong batch huấn luyện, nên nếu batch cuối chỉ có một dòng, batch đó bị bỏ riêng trong vòng lặp cập nhật. Khi tính loss và RMSE để báo cáo, mã vẫn duyệt toàn bộ train fold và validation fold ở chế độ `eval()`, kể cả dòng không nằm trong batch cập nhật cuối. Điều này phân biệt dữ liệu dùng để cập nhật trọng số ở một epoch với dữ liệu dùng để đo chỉ số sau epoch.

Tiền xử lý không bổ sung feature engineering, không xóa hai ứng viên ngoại lệ và không fit lại trên toàn bộ dữ liệu có nhãn. Nhờ đó, năm thí nghiệm sử dụng cùng một biểu diễn đầu vào và sự khác biệt trong bảng kết quả chủ yếu đến từ cấu hình mạng cùng tối ưu hóa. Quy trình này cũng giúp truy vết: mô hình cuối phải đi cùng đúng preprocessor mà nó đã dùng, không thể tùy tiện thay bằng một bộ mã hóa mới dù cùng tên cột. Khả năng tái lập vẫn có giới hạn thông thường của môi trường tính toán, nhưng seed và các artifact được lưu giúp giảm những nguồn khác biệt dễ tránh.

## 5. MODELING — XÂY DỰNG MÔ HÌNH

MLP của dự án nhận 286 chiều đầu vào sau tiền xử lý và trả về một giá trị `log1p(SalePrice)`. Lớp cuối là tuyến tính, không áp dụng ReLU hay sigmoid, vì mô hình cần biểu diễn giá trị log liên tục mà không bị giới hạn bởi một hàm kích hoạt đầu ra. Các lớp ẩn dùng phép biến đổi tuyến tính và ReLU. Với mỗi lớp ẩn trước lớp ẩn cuối, mã đặt BatchNorm rồi ReLU, tiếp theo là dropout nếu cấu hình có tỷ lệ lớn hơn 0; lớp ẩn cuối dùng Linear và ReLU. Một cài đặt MLP có tham số kích thước lớp và dropout được dùng chung cho E0–E4, tránh việc mỗi thí nghiệm có một phiên bản kiến trúc khó đối chiếu [5].

E1 là cấu hình baseline ban đầu với các lớp ẩn 128–64–32, dropout 0,15 và 0,10, learning rate 0,001 và weight decay 0,0001. Cấu hình E0 nhỏ hơn, 64–32, dropout 0, learning rate 0,001 và weight decay 0. Dù tên E0 là “naive baseline” trong kế hoạch, nó vẫn là một MLP có BatchNorm ở lớp ẩn đầu theo cài đặt chung. E2 tăng độ rộng mạng lên 256–128–64; E3 giảm learning rate của mạng kiểu E1; E4 tăng dropout và weight decay. Tập cấu hình này tạo một phép so sánh nhỏ về độ rộng, tốc độ học và mức điều chuẩn, nhưng chưa phải tìm kiếm siêu tham số toàn diện.

Mô hình tối thiểu hóa MSE của dự đoán trên thang log bằng AdamW. Sau từng epoch, mã tính MSE đầy đủ của train và validation ở chế độ `eval()` và `no_grad()`, sau đó lấy căn để được RMSE. MSE được tổng hợp theo số phần tử, tránh sai lệch do batch cuối có kích thước khác. Batch size là 32, trần huấn luyện là 500 epoch và early stopping có patience 30 epoch tính theo validation RMSE. Khi một epoch cải thiện điểm validation, checkpoint mới được lưu; sau khi dừng, mã nạp lại checkpoint tốt nhất và kiểm tra RMSE của nó khớp chỉ số đã ghi. Kết quả vì thế không dựa vào trọng số của epoch cuối nếu trước đó đã tốt hơn [5].

Đầu ra của mạng lúc khởi tạo thông thường có thể nằm gần 0, trong khi nhãn log của bộ dữ liệu này nằm xấp xỉ 12. Dự án khởi tạo bias của lớp cuối bằng trung bình nhãn log của train fold, chỉ sử dụng dữ liệu được phép học. Quyết định này giúp điểm xuất phát phù hợp hơn và đã được ghi trong mã huấn luyện. Seed 42 được đặt cho Python, NumPy, PyTorch và bộ sinh của DataLoader. Tùy chọn `device=auto` chỉ chọn CUDA sau phép thử forward/backward; môi trường đã kiểm chứng dùng bản PyTorch chỉ có CPU, nên các lần chạy được ghi nhận trên CPU. Các chi tiết này có thể ảnh hưởng đến diễn biến tối ưu và cần được nêu khi giải thích kết quả.

![e0_rmse](../outputs/figures/e0_rmse.png)

*Hình 4. RMSE train và validation theo epoch của E0. Nguồn: Kết quả thực nghiệm của dự án.*

Hình 4 cho thấy RMSE train giảm đều sau những epoch đầu, trong khi RMSE validation đạt mức thấp nhất rất sớm rồi dao động quanh một mức cao hơn. Theo tệp thí nghiệm, E0 đạt validation RMSE 0,133316 tại epoch 3. Khoảng cách giữa hai đường về sau là dấu hiệu mô hình ngày càng khớp tập huấn luyện hơn mà không cải thiện trên holdout. Early stopping và việc nạp lại checkpoint tốt nhất làm cho mô hình được chọn vẫn là trạng thái ở epoch 3, thay vì trạng thái cuối đường cong. Biểu đồ là bằng chứng về diễn biến trong lần chia này; nó không tự chứng minh mức quá khớp sẽ giống hệt trên những phép chia khác.

## 6. EVALUATION — ĐÁNH GIÁ

Chỉ số chính của dự án là căn bậc hai trung bình bình phương sai số giữa nhãn thật và dự đoán trên `log1p(SalePrice)` của 292 căn nhà validation. Ký hiệu ngắn gọn là `RMSE = sqrt(mean((y_log - ŷ_log)^2))`. Cùng một split, preprocessor và seed được áp dụng cho cả năm cấu hình. Tệp `experiments.csv` lưu siêu tham số, epoch tốt nhất và validation RMSE; bảng dưới đây làm tròn RMSE tới sáu chữ số thập phân mà không thay đổi thứ hạng [4], [6].

*Bảng 1. So sánh đầy đủ năm cấu hình trên cùng holdout 80/20. Nguồn: Kết quả thực nghiệm của dự án.*

| Thí nghiệm | Lớp ẩn | Dropout | Learning rate | Weight decay | Epoch tốt nhất | Validation RMSE |
|---|---|---|---:|---:|---:|---:|
| E0 | 64–32 | 0 | 0,001 | 0 | 3 | 0,133316 |
| E1 | 128–64–32 | 0,15; 0,10 | 0,001 | 0,0001 | 10 | 0,141518 |
| E2 | 256–128–64 | 0,20; 0,15 | 0,001 | 0,0001 | 4 | 0,136446 |
| E3 | 128–64–32 | 0,15; 0,10 | 0,0005 | 0,0001 | 11 | 0,142573 |
| E4 | 128–64–32 | 0,25; 0,15 | 0,001 | 0,001 | 24 | 0,138196 |

E0 là cấu hình thắng theo quy tắc đã đặt vì có validation RMSE nhỏ nhất, 0,133316. E2 đứng sau E0 với 0,136446; E4 đạt 0,138196; E1 đạt 0,141518 và E3 đạt 0,142573. Không nên suy ra mạng nhỏ luôn tốt hơn mạng lớn từ năm con số này. Các cấu hình khác nhau đồng thời ở độ rộng và mức điều chuẩn, còn độ dao động giữa các cách chia dữ liệu chưa được đo. Bảng cho thấy trong thiết lập cụ thể của dự án, mạng đơn giản E0 tổng quát hóa tốt nhất trên holdout đã chọn, mặc dù loss train về sau vẫn có thể tiếp tục giảm.

![e0_actual_predicted](../outputs/figures/e0_actual_predicted.png)

*Hình 5. Log-price thực tế và dự đoán của E0 trên validation. Nguồn: Kết quả thực nghiệm của dự án.*

Hình 5 đặt giá trị thật trên trục ngang và giá trị dự đoán trên trục dọc; đường chéo đỏ biểu diễn dự đoán khớp hoàn toàn. Nhiều điểm nằm gần đường này, nhất là vùng giữa, cho thấy mô hình nắm được quan hệ chung của dữ liệu. Tuy nhiên có các điểm lệch rõ ở vùng log-price thấp và cao. Ở phần giá cao, một số điểm nằm dưới đường chéo, tức dự đoán thấp hơn thực tế; ở phần giá thấp có những điểm dự đoán cao hơn thực tế. Đây là quan sát từ biểu đồ, không phải một ước lượng định lượng về thiên lệch theo từng khoảng giá. Muốn khẳng định thiên lệch có hệ thống cần chia nhóm và tính chỉ số trên nhiều fold hoặc một tập kiểm tra có nhãn độc lập.

![e0_residuals](../outputs/figures/e0_residuals.png)

*Hình 6. Sai số thực tế trừ dự đoán theo log-price dự đoán của E0. Nguồn: Kết quả thực nghiệm của dự án.*

Ở Hình 6, sai số được định nghĩa là giá trị log thật trừ giá trị log dự đoán. Đường ngang 0 là mốc không sai số; điểm ở phía trên thể hiện dự đoán thấp, điểm ở phía dưới thể hiện dự đoán cao. Đám mây điểm phần lớn quanh 0 nhưng vẫn có vài sai số âm và dương khá lớn. Sự tồn tại của các điểm này giải thích vì sao một RMSE duy nhất chưa mô tả hết trải nghiệm dự đoán đối với từng căn nhà. Biểu đồ cũng không cho thấy một mẫu hình tuyến tính đơn giản theo giá dự đoán, nhưng số mẫu validation chỉ 292 nên nhận xét này cần được xem là mô tả định tính.

Một phép chia holdout cố định giúp quy trình gọn, dễ đối chiếu và tiết kiệm thời gian huấn luyện, nhưng chỉ cung cấp một ước lượng có điều kiện theo seed 42. Nếu đổi cách chia, tỷ lệ nhà giá cao, khu vực hoặc ngoại lệ trong validation có thể thay đổi và thứ hạng E0–E4 cũng có thể đổi. Hơn nữa, chính holdout đó được dùng để chọn trong năm cấu hình và để early stopping, nên điểm thấp nhất có thể lạc quan đối với khả năng tổng quát hóa thật. Không có cross-validation, khoảng tin cậy hoặc một test nội bộ thứ hai để đo mức biến thiên này. Do đó, kết luận “E0 tốt nhất” chỉ áp dụng cho giao thức đánh giá đã thực hiện, không phải một khẳng định chắc chắn cho mọi lần lấy mẫu.

Dự án cũng chưa thử feature engineering, ví dụ biểu diễn tường minh tuổi nhà, tổng diện tích liên quan hoặc trạng thái “không có” của tiện ích. Điều đó giới hạn khả năng khai thác ngữ nghĩa trong từ điển dữ liệu. Hai ứng viên ngoại lệ vẫn nằm trong tập gốc; chưa có thí nghiệm loại bỏ hoặc xử lý riêng. Các hướng cải thiện này là giả thuyết nghiên cứu sau baseline, không phải bằng chứng rằng điểm sẽ giảm. Nếu mở rộng thực nghiệm, cần giữ nguyên quy tắc fit tiền xử lý trong từng fold và tách rõ quy trình chọn cấu hình khỏi ước lượng cuối để tránh lạc quan do lựa chọn mô hình.

## 7. DEPLOYMENT — TRIỂN KHAI

Sau khi có bảng E0–E4, lệnh final của dự án đọc tệp kết quả và chọn cấu hình có validation RMSE thấp nhất, với tên thí nghiệm làm quy tắc phụ khi bằng điểm. Final run dùng lại seed 42 và cùng cách chia 80/20, train lại E0 trên phần train fold, giữ validation để early stopping. Preprocessor final chỉ fit trên train fold và được lưu cùng checkpoint. Cách làm này giữ lại một điểm holdout có thể kiểm tra, nhưng không tận dụng toàn bộ 1.460 nhãn để cập nhật mô hình; đây là một đánh đổi đã được ghi trong tài liệu triển khai, không phải bước train lại trên tất cả dữ liệu [4], [5].

Khi dự đoán, chương trình nạp checkpoint cùng fitted preprocessor, kiểm tra số chiều đầu vào, biến đổi `test.csv` theo thứ tự cột huấn luyện, rồi chạy mạng ở chế độ đánh giá. Đầu ra log được kiểm tra hữu hạn trước khi áp dụng `expm1`. Sau phép nghịch đảo, chương trình kiểm tra tiếp giá trị hữu hạn, chặn giá âm về 0 nếu xuất hiện và đếm số trường hợp bị chặn. Trong lần chạy đã ghi nhận, không có giá nào cần chặn. Tệp CSV chỉ được ghi sau khi chương trình xác nhận đúng số dòng, đúng hai cột `Id,SalePrice`, thứ tự `Id` trùng tập test và mọi giá không âm. Các kiểm tra này bảo vệ định dạng nộp bài ngay cả khi quá trình huấn luyện hoàn tất bình thường.

Artifact cuối đã chọn E0, ghi best epoch 3 và validation RMSE 0,133316. Submission có 1.459 dòng, với giá dự đoán trong khoảng 49.901,78 đến 767.002,19 theo lần kiểm tra được ghi lại. Đây là khoảng giá của dự đoán trong tệp, không phải khoảng giá thật của tập test vì nhãn test không công khai trong dự án. Final run có checkpoint, preprocessor, history và hình riêng; nó không ghi đè các artifact thí nghiệm E0. Vì vậy việc đối chiếu giữa kết quả lúc chọn cấu hình và kết quả tạo tệp cuối vẫn thực hiện được [4].

Người dùng báo đã tải tệp submission lên Kaggle và nhận điểm 0.13543 vào ngày 26-09-2026. Workspace không có submission ID hay ảnh leaderboard để xác minh độc lập, nên báo cáo ghi rõ nguồn của số này là phản hồi người dùng. Validation RMSE 0,133316 là phép đo cục bộ trên 292 mẫu đã tách; điểm 0.13543 phản ánh báo cáo về hệ thống đánh giá ngoài dự án. Không dùng điểm Kaggle để chọn lại cấu hình hoặc sửa mô hình. Phân biệt nguồn và tập đánh giá của hai chỉ số giúp tránh trình bày một kết quả chưa kiểm chứng như phép đo đã được tái tạo trong môi trường dự án.

Ở mức triển khai thực tế, dự án hiện tạo một tệp dự đoán hàng loạt phục vụ Kaggle, chưa có dịch vụ API, giao diện người dùng hoặc quy trình giám sát sai số khi có dữ liệu mới. Nếu mở rộng thành ứng dụng định giá, cần kiểm soát phiên bản schema, theo dõi dữ liệu thiếu và độ lệch phân phối của đầu vào, lưu lịch sử model, và đánh giá lại khi có giá bán thực tế. Những bước đó vượt phạm vi báo cáo hiện tại; không nên gán cho artifact `submission.csv` vai trò của một hệ thống định giá đang vận hành liên tục.

## 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Dự án đã hoàn thành một pipeline dự đoán giá nhà bằng MLP PyTorch theo sáu giai đoạn CRISP-DM: hiểu yêu cầu nộp bài, khảo sát dữ liệu, tiền xử lý không dùng thông tin validation để fit, huấn luyện năm cấu hình, đánh giá trên holdout và tạo submission hợp lệ. Các notebook đã chạy và artifact đi kèm cho phép kiểm tra hình dạng dữ liệu, bảng kết quả và diễn biến học. Trong giao thức hiện tại, E0 với các lớp ẩn 64–32 đạt validation RMSE log-price 0,133316 tại epoch 3 và được dùng cho mô hình cuối. Điểm Kaggle 0.13543 chỉ là kết quả do người dùng cung cấp, chưa có bằng chứng độc lập trong workspace.

Kết quả này minh họa rằng mạng lớn hơn hoặc điều chuẩn mạnh hơn không mặc nhiên cho điểm validation thấp hơn trên một bộ dữ liệu nhỏ. Hình RMSE cho thấy tầm quan trọng của checkpoint tốt nhất và early stopping; hình dự đoán và sai số cho thấy mô hình vẫn có các trường hợp lệch đáng kể. Giá trị của báo cáo nằm ở chuỗi quyết định và kiểm tra có thể truy vết, không chỉ ở một con số xếp hạng. Một holdout duy nhất không đo được độ ổn định của thứ hạng mô hình, nên mọi nhận xét về khả năng tổng quát hóa cần giữ đúng giới hạn đó.

Hướng phát triển ưu tiên là thử cross-validation với tiền xử lý fit độc lập trong từng fold, đồng thời giữ một tập đánh giá cuối không tham gia chọn cấu hình nếu dữ liệu cho phép. Sau đó có thể kiểm chứng cách mã hóa riêng cho trạng thái “không có” tiện ích, tạo đặc trưng dựa trên hiểu biết về nhà ở và đánh giá tác động của hai ứng viên ngoại lệ bằng thí nghiệm được kiểm soát. Việc so sánh thêm với các mô hình tabular khác cũng hữu ích, nhưng chỉ nên kết luận sau khi dùng cùng giao thức đánh giá. Những đề xuất này chưa được thực hiện trong dự án và không được tính vào điểm số đã báo cáo.

## 9. TÀI LIỆU THAM KHẢO

[1] Kaggle, “House Prices - Advanced Regression Techniques,” trang mô tả cuộc thi và quy cách đánh giá, https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/description (truy cập 26-09-2026).

[2] IBM, “CRISP-DM Help Overview,” https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=dm-crisp-help-overview (truy cập 26-09-2026).

[3] Dự án House Prices MLP, `data/data_description.txt`, `data/train.csv`, `data/test.csv`, `docs/02_data_understanding.md` và notebook EDA đã chạy; dữ liệu, từ điển và thống kê của dự án.

[4] Dự án House Prices MLP, `plan.md`, `README.md`, `PROJECT_STRUCTURE.md` và sáu tệp trong `docs/`; quyết định triển khai, kết quả nghiệm thu và giới hạn.

[5] Dự án House Prices MLP, `configs/baseline.yaml`, mã trong `src/` và `scripts/`; cấu hình và cài đặt pipeline.

[6] Dự án House Prices MLP, các notebook đã chạy, `outputs/metrics/experiments.csv`, các history CSV và sáu hình được trích trong báo cáo; kết quả thực nghiệm.

## 10. PHỤ LỤC — LỆNH TÁI LẬP

Từ thư mục gốc của dự án, lần lượt chạy các lệnh sau trong môi trường Python đáp ứng `requirements.txt`:

`python scripts/run_train.py --all-experiments`

`python scripts/run_train.py --final`

`python scripts/run_predict.py`

`python -m pytest -q`

Lệnh đầu tạo bảng E0–E4; lệnh final chọn cấu hình theo bảng đó; lệnh predict tạo `outputs/submission.csv`. Những lệnh này mô tả cách tái lập pipeline đã có, không được thực thi lại để tạo số liệu mới cho báo cáo.
