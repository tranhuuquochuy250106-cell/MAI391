# Câu 10 — Cải thiện mô hình bằng hồi quy bậc hai

## Lý do và cách chọn mô hình

Ở câu 8, trung bình phần dư train dương ở hai đầu dải nhiệt độ và âm ở vùng giữa. Chẳng hạn, ba nhóm có AT trung bình 7,16°C, 23,77°C và 31,10°C có phần dư trung bình lần lượt khoảng +1,49, −2,23 và +3,93 MW. Để biểu diễn độ cong này, nhóm thử thêm đặc trưng bình phương của AT chuẩn hóa.

Chia nội bộ 7.622 dòng train thành 6.098 dòng khớp và 1.524 dòng validation, bằng NumPy default_rng với seed 42. Với cả hai ứng viên, mọi thông số chuẩn hóa ở vòng này được tính từ 6.098 dòng khớp. Mô hình bậc hai dùng \(z=(AT-\mu_{AT})/\sigma_{AT}\) và \(w=(z^2-\mu_{z^2})/\sigma_{z^2}\). Hàm dự đoán mở rộng là \(\hat y=\theta_0+\theta_1z+\theta_2w\). Cả hai ứng viên dùng Batch Gradient Descent tự viết từ câu 4, khởi tạo bằng 0, learning rate 0,10, dung sai gradient \(10^{-6}\) và tối đa 10.000 cập nhật.

| Ứng viên | MSE validation (MW²) | RMSE validation (MW) | \(R^2\) validation |
|---|---:|---:|---:|
| Bậc một | 30,9462 | 5,5629 | 0,8907 |
| Bậc hai | 28,5160 | 5,3400 | 0,8993 |

Mô hình bậc hai được chọn theo MSE validation thấp hơn. Sau khi chốt bậc, tính lại thông số chuẩn hóa và khớp mô hình trên toàn bộ train. Kết quả thu được \(\theta\approx(454{,}380870;\,-15{,}936280;\,1{,}535251)\) trong hệ \([1,z,w]\), sau 190 lần cập nhật. Baseline trong bảng dưới lấy đúng nghiệm GD đã chốt ở câu 4. Hai mô hình được đánh giá trên cùng 1.905 dòng test của nhóm.

## Kết quả và diễn giải

| Mô hình | Tập | MSE (MW²) | RMSE (MW) | \(R^2\) |
|---|---|---:|---:|---:|
| Baseline bậc một | Train | 29,1844 | 5,4023 | 0,8994 |
| Bậc hai | Train | 26,8733 | 5,1839 | 0,9073 |
| Baseline bậc một | Test | 30,5477 | 5,5270 | 0,8952 |
| Bậc hai | Test | 27,9210 | 5,2840 | 0,9042 |

MSE test giảm khoảng 8,60%, RMSE giảm khoảng 0,243 MW và \(R^2\) tăng khoảng 0,0090. Hình q10_comparison.png cho thấy mô hình bậc hai giảm độ lệch của trung bình phần dư ở hai đầu dải AT trên train. Ở ô phải, điểm mờ là phần dư test của hai mô hình và đường nối là trung bình phần dư train theo nhóm nhiệt độ. Phần dư test vẫn có các quan sát lệch lớn, nên kết luận cải thiện dựa trên các chỉ số tổng hợp của tập test này.

## Tái lập

Chạy python run_q10.py từ thư mục MAI391_Final sau khi có kết quả câu 4 và câu 8. q10_validation.csv lưu kết quả chọn bậc, q10_comparison.csv lưu chỉ số theo tập, q10_test_predictions.csv lưu dự đoán và phần dư theo source_row. q10_summary.json lưu thông số của mô hình đã chọn.
