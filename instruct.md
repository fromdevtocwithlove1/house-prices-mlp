Hãy viết một bài tiểu luận tiếng Việt dài 4.000–5.000 từ về dự án dự đoán giá nhà bằng MLP PyTorch theo 6 giai đoạn CRISP-DM.

Thư mục dự án: C:\Users\PC\Desktop\house-prices-mlp

Trước khi viết, hãy đọc:
- plan.md, README.md và toàn bộ docs/ , PROJECT_STRUCTURE.md
- data/data_description.txt và thông tin tổng quan của data/train.csv, data/test.csv
- notebooks/, ưu tiên các notebook đã chạy để kiểm tra kết quả thực tế
- configs/baseline.yaml, src/ và scripts/ để mô tả đúng phương pháp
- outputs/metrics/experiments.csv và các biểu đồ trong outputs/figures/

Cấu trúc báo cáo:
1. Trang bìa (để trống tên sinh viên, lớp, giảng viên nếu chưa có thông tin)
2. Mục lục
3. Mở đầu: lý do chọn đề tài, mục tiêu và phạm vi
4. Business Understanding
5. Data Understanding
6. Data Preparation
7. Modeling
8. Evaluation
9. Deployment
10. Kết luận và hướng phát triển
11. Tài liệu tham khảo
12. Phụ lục nếu cần

Trong bài, hãy giải thích các quyết định xử lý dữ liệu, phép biến đổi log1p(SalePrice), kiến trúc và cách huấn luyện MLP. Lập bảng so sánh đầy đủ E0–E4. Ghi rõ E0 đạt validation RMSE 0.133316; điểm Kaggle 0.13543 là kết quả người dùng báo lại và chưa được xác minh độc lập. Phân biệt hai loại điểm này. Nêu giới hạn của việc đánh giá bằng một lần chia holdout 80/20 và việc chưa dùng cross-validation hoặc feature engineering.

Chèn trực tiếp ảnh từ outputs/figures/ vào đúng phần nội dung liên quan, tối thiểu:
- target_histograms.png: phân phối SalePrice trước và sau biến đổi
- missing_values.png: dữ liệu thiếu
- numeric_scatter.png: quan hệ giữa đặc trưng số và giá bán
- e0_rmse.png: diễn biến RMSE khi huấn luyện E0
- e0_actual_predicted.png: giá trị thực tế so với dự đoán
- e0_residuals.png: phân tích sai số

Mỗi ảnh cần có số thứ tự, chú thích, nguồn “Kết quả thực nghiệm của dự án” và ít nhất một đoạn phân tích trong bài. Chỉ dùng ảnh thực sự tồn tại; có thể chọn thêm ảnh phù hợp trong outputs/figures/.

Đầu ra bắt buộc:
- Tạo thư mục reports/ trong dự án.
- Xuất reports/bao_cao_crisp_dm.docx với ảnh PNG được NHÚNG trực tiếp vào tài liệu, không chỉ đặt đường dẫn hoặc liên kết.
- Xuất reports/bao_cao_crisp_dm.pdf từ bản DOCX.
- Tạo mục lục khớp tiêu đề và số trang của bản đã dàn trang.
- Kiểm tra lại rằng DOCX và PDF đều hiển thị đầy đủ ảnh, bảng, chú thích và ký tự tiếng Việt.

Không tự tạo số liệu, kết quả thí nghiệm, hình ảnh hoặc tài liệu tham khảo. Nếu không thể đọc tệp hay tạo DOCX/PDF, hãy nói rõ giới hạn đó và cung cấp bản Markdown có cú pháp ảnh trỏ đến đúng tệp trong outputs/figures/.
