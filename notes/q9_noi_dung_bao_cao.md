# Câu 9 — So sánh với scikit-learn

## Thiết lập

So sánh Gradient Descent tự viết, nghiệm phương trình chuẩn và scikit-learn LinearRegression trên cùng 7.622 quan sát train và 1.905 quan sát test. Cả ba dự đoán PE từ AT chuẩn hóa bằng trung bình và độ lệch chuẩn của train. Mã NumPy dùng \(X=[\mathbf{1},z]\). LinearRegression nhận riêng cột \(z\) và khớp hệ số chặn bằng fit_intercept=True. Các hệ số được quy về đơn vị gốc trước khi trình bày.

Kết quả GD lấy từ baseline câu 4. Khi đo thời gian, chương trình chạy lại chính thuật toán GD của nhóm, với \(\gamma=0{,}10\), khởi tạo \((0,0)\), dung sai gradient \(10^{-6}\) và tối đa 10.000 bước. Lần chạy lại hội tụ sau 190 cập nhật và khớp hệ số đã lưu. Nghiệm phương trình chuẩn cũng được đối chiếu với câu 5.

## Kết quả

| Phương pháp | Hệ số chặn (MW) | Hệ số góc (MW/°C) | \(J_{\mathrm{train}}\) (MW²) | MSE train (MW²) | MSE test (MW²) | \(R^2_{\mathrm{test}}\) |
|---|---:|---:|---:|---:|---:|---:|
| GD tự viết | 496,977983872 | −2,169684131 | 14,5922002593 | 29,184400519 | 30,547660083 | 0,895158680 |
| Nghiệm giải tích | 496,977984877 | −2,169684135 | 14,5922002593 | 29,184400519 | 30,547659991 | 0,895158681 |
| scikit-learn | 496,977984877 | −2,169684135 | 14,5922002593 | 29,184400519 | 30,547659991 | 0,895158681 |

So với nghiệm giải tích, chênh lệch dự đoán lớn nhất của GD khoảng \(9{,}98\times10^{-7}\) MW trên train và \(9{,}90\times10^{-7}\) MW trên test. Sai khác nhỏ này phù hợp với tiêu chí dừng theo gradient. Chênh lệch lớn nhất của scikit-learn trên cả hai tập khoảng \(1{,}71\times10^{-13}\) MW, ở mức sai số dấu phẩy động. Cả ba đạt chất lượng dự đoán tương đương trên bộ dữ liệu này. \(J_{\mathrm{train}}\) là nửa MSE train.

## Thời gian khớp mô hình

Dữ liệu và ma trận thiết kế được chuẩn bị trong bộ nhớ trước khi đo. Mỗi phương pháp khởi động một lần rồi đo 30 lần bằng perf_counter, với giới hạn một luồng tính toán cho các thư viện số. Phép đo chỉ gồm bước khớp mô hình. Các giá trị trung vị và khoảng min–max nằm trong results/q9_comparison.csv.

Trong lần đo đóng gói, trung vị xấp xỉ 7,27 ms với GD, 0,020 ms với phương trình chuẩn và 0,241 ms với scikit-learn. Phương trình chuẩn chỉ giải hệ 2×2 sau khi lập hệ. Bản LinearRegression 1.8.0 đã chạy dùng bộ giải least squares trực tiếp cho đầu vào dense. GD thực hiện 190 cập nhật trên toàn bộ train và lưu lịch sử. Các số đo phản ánh cấu hình, dữ liệu và môi trường của lần chạy này.

Chạy python run_q9.py để tạo lại bảng. results/q9_summary.json ghi phiên bản môi trường và số lần đo.
