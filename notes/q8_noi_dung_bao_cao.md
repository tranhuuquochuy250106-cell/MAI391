# Câu 8 — Dự đoán, đánh giá mô hình và phân tích phần dư

## Mô hình và dự đoán

Dùng nghiệm Gradient Descent của câu 4, với \(\theta_0=454{,}380870243\), \(\theta_1=-16{,}150470138\) trên biến chuẩn hóa \(z=(AT-\mu_{\mathrm{train}})/\sigma_{\mathrm{train}}\). Trên đơn vị gốc, phương trình là \(\widehat{PE}\approx496{,}977983872-2{,}169684131\,AT\). Mô hình dự đoán công suất giảm khoảng 2,17 MW khi nhiệt độ tăng 1°C. Hệ số này mô tả quan hệ trong mô hình hồi quy.

| AT (°C) | PE dự đoán (MW) |
|---:|---:|
| 10 | 475,281 |
| 20 | 453,584 |
| 30 | 431,887 |

Các nhiệt độ trên là đầu vào minh họa nằm trong phạm vi AT quan sát trên train. Bảng chỉ trình bày giá trị mô hình dự đoán. Hình q8_fit.png cho thấy đường hồi quy đi theo xu hướng giảm chung của đám mây dữ liệu train và test, đồng thời vẫn còn độ phân tán quanh đường.

## Đánh giá định lượng

Đặt \(e_i=y_i-\hat y_i\). Chỉ số MSE là \(\sum_i e_i^2/n\), RMSE là căn bậc hai của MSE và \(R^2=1-\sum_i e_i^2/\sum_i(y_i-\bar y)^2\), với \(\bar y\) tính riêng trên từng tập. Khi tổng bình phương quanh trung bình bằng 0, \(R^2\) không xác định và trường tương ứng sẽ để trống.

| Tập | Quan sát | MSE (MW²) | RMSE (MW) | \(R^2\) |
|---|---:|---:|---:|---:|
| Train | 7.622 | 29,184401 | 5,402259 | 0,899372 |
| Test | 1.905 | 30,547660 | 5,526994 | 0,895159 |

MSE train bằng \(2J_{\mathrm{train}}\) trong sai số số học, đúng với cách định nghĩa hàm chi phí. Trên test, mô hình có RMSE khoảng 5,53 MW và \(R^2\) khoảng 0,895. RMSE test cao hơn train khoảng 0,125 MW. Câu 5 kiểm chứng mức độ GD tiến gần nghiệm tối ưu, còn các chỉ số ở đây đo sai số dự đoán so với PE quan sát.

## Phân tích phần dư

Hai hình q8_residual_train.png và q8_residual_test.png vẽ \(e=PE_{\mathrm{thực}}-\widehat{PE}\) theo PE dự đoán, dùng cùng giới hạn trục. Phần dư dương biểu thị dự đoán thấp. Đường đậm nối trung bình phần dư của 10 nhóm theo phân vị PE dự đoán trên train. Các ngưỡng nhóm được cố định từ train rồi áp dụng lại trên test.

- **Train.** Trung bình phần dư khoảng +3,93 MW ở nhóm nóng nhất (AT trung bình 31,10°C), −2,23 MW ở nhóm AT trung bình 23,77°C và +1,49 MW ở nhóm lạnh nhất (7,16°C). Xu hướng dương ở hai đầu và âm ở vùng giữa cho thấy mô hình tuyến tính còn bỏ sót độ cong. Độ lệch chuẩn phần dư theo nhóm dao động khoảng 4,43–5,66 MW, chưa cho thấy dạng phễu tăng đều. Điểm có phần dư âm lớn nhất lệch khoảng −39,75 MW.
- **Test.** Đường trung bình phần dư có xu hướng tương tự. Nhóm có PE dự đoán cao nhất có độ lệch chuẩn phần dư khoảng 6,11 MW và chứa một điểm lệch khoảng −45,91 MW. Các điểm lệch lớn được giữ trong đánh giá.
- **Đầu vào cho câu 10.** Dựa trên độ cong quan sát được trên train, có cơ sở thử thêm \(AT^2\). Việc chọn mô hình cần dựa vào train hoặc validation nội bộ. Kết quả test ở câu 8 là mốc đánh giá baseline.

Chạy python run_q8.py trong thư mục MAI391_Final để tái tạo hình và bảng. q8_predictions.csv lưu toàn bộ dự đoán và phần dư theo source_row. q8_residual_bins.csv lưu thống kê từng nhóm.
