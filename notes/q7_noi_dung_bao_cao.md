# Câu 7 — Phân tích hội tụ của Gradient Descent

## Thiết lập và cách đọc hình

Năm cấu hình dùng cùng 7.622 quan sát train, cùng chuẩn hóa \(z=(AT-\mu_{AT})/\sigma_{AT}\), cùng điểm khởi tạo \(\theta^{(0)}=(0,0)\), tối đa 10.000 lần cập nhật. Tiêu chí hội tụ là \(\|\nabla J(\theta^{(k)})\|_2\le 10^{-6}\). Thuật toán cũng dừng nếu chi phí vượt \(10^{12}\max(J^{(0)},1)\) hoặc gặp giá trị không hữu hạn. Bước 0 biểu thị trạng thái khởi tạo.

Hình 7a (figures/q7_convergence.png) hiển thị \(J_{\mathrm{train}}\) và khoảng cách \(\Delta J\) tới nghiệm giải tích câu 5 theo số cập nhật. Hình 7b (figures/q7_learning_rate.png) so sánh \(\Delta J\) của năm cấu hình. Ô bên phải hình 7b theo dõi sai số có dấu của hệ số chặn, chia cho sai số tại bước 0. Dấu đổi khi tham số đi qua phía bên kia của nghiệm. Ô này tập trung vào bốn chuỗi nhiều bước để quan sát dao động trong 12 lần cập nhật đầu.

Để tránh triệt tiêu số khi lấy hiệu hai chi phí gần bằng nhau, dùng dạng bậc hai \(\Delta J=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)\) với \(H=X^\top X/n\). Công thức đúng tại nghiệm tối ưu của bài toán bình phương tối thiểu. Giá trị bằng 0 được bỏ qua trên trục log, còn bảng giữ giá trị đã tính. Với \(\gamma=1\), giá trị cỡ \(10^{-27}\) phản ánh độ chính xác dấu phẩy động.

## Kết quả trên train

| Learning rate | Lần cập nhật | Trạng thái | \(J\) cuối (MW²) | \(\Delta J\) cuối (MW²) |
|---:|---:|---|---:|---:|
| 0,01 | 1.984 | Hội tụ chậm | 14,5922002593 | \(4{,}95\times10^{-13}\) |
| 0,10 | 190 | Hội tụ, cấu hình cơ sở | 14,5922002593 | \(4{,}23\times10^{-13}\) |
| 1,00 | 1 | Hội tụ sau một cập nhật | 14,5922002593 | \(1{,}82\times10^{-27}\) |
| 1,85 | 123 | Hội tụ, tham số dao động tắt dần | 14,5922002593 | \(4{,}48\times10^{-13}\) |
| 2,05 | 284 | Phân kỳ, dừng do vượt ngưỡng chi phí | \(1{,}12\times10^{17}\) | \(1{,}12\times10^{17}\) |

Baseline giảm chi phí từ 103.375,999083 MW² xuống 14,5922002593 MW². Tại bước 190, chuẩn gradient còn \(9{,}20\times10^{-7}\), đạt tiêu chí hội tụ. Trục log cho thấy khoảng cách tới nghiệm vẫn giảm trong các bước cuối dù đường \(J\) nhìn gần như phẳng trên trục tuyến tính.

## Giải thích ảnh hưởng của learning rate

Đặt \(e_k=\theta^{(k)}-\theta^*\). Vì gradient của hàm bậc hai là \(H e_k\), ta có \(e_{k+1}=(I-\gamma H)e_k\). Điều kiện hội tụ từ mọi khởi tạo là \(0<\gamma<2/\lambda_{\max}(H)\) khi \(H\) xác định dương. Ở đây \(H\approx I\) nhờ chuẩn hóa AT trên train với ddof = 0 và dùng một cột hệ số chặn, nên ngưỡng trên xấp xỉ 2.

- Với 0,01 và 0,10, hệ số nhân sai số xấp xỉ 0,99 và 0,90. Sai số co lại cùng phía, trong đó 0,01 cần nhiều bước hơn.
- Với 1,85, hệ số nhân xấp xỉ −0,85. Tham số vượt qua nghiệm ở mỗi lần cập nhật trong 12 bước đầu và biên độ giảm. Vì chi phí phụ thuộc bậc hai vào sai số, chi phí vẫn giảm dù tham số đổi phía.
- Với 2,05, hệ số nhân xấp xỉ −1,05. Tham số đổi phía với biên độ tăng. Lần chạy này dừng vì chi phí vượt ngưỡng khi các giá trị lưu vẫn hữu hạn.
- Với 1,00, hệ số nhân gần 0 nên một lần cập nhật đạt dung sai trong thiết lập cụ thể này. Baseline 0,10 được giữ để nối tiếp quỹ đạo nhiều bước đã dùng ở câu 6.

Chạy python run_q7.py trong thư mục gốc để tạo lại hình và bảng. Kết quả chi tiết ở results/q7_runs.csv và results/q7_summary.json.
