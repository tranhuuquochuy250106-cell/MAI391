# Câu 5 — Nghiệm bình phương tối thiểu và đối chiếu Gradient Descent

## Thiết lập

Dùng 7.622 quan sát train của bộ dữ liệu TV2_shared_v1. Đặt \(z_i=(AT_i-\mu_{AT})/\sigma_{AT}\), với \(\mu_{AT}=19{,}63286408\ ^\circ\mathrm C\), \(\sigma_{AT}=7{,}44369649\ ^\circ\mathrm C\) tính trên train theo quy ước ddof = 0. Ma trận \(X=[\mathbf 1,z]\) có kích thước \(7622\times 2\), vector \(y\) chứa PE tính bằng MW. Tập test gồm 1.905 quan sát. Câu 5 chỉ dùng AT của test để đối chiếu dự đoán giữa hai nghiệm.

## Suy ra nghiệm

Từ hàm chi phí ở câu 3,

\[
J(\theta)=\frac{1}{2n}(X\theta-y)^\top(X\theta-y),
\qquad
\nabla J(\theta)=\frac{1}{n}X^\top(X\theta-y).
\]

Cho gradient bằng 0 thu được hệ phương trình chuẩn

\[
(X^\top X)\theta^*=X^\top y.
\]

Khai triển theo hai tham số,

\[
\begin{bmatrix}
n & \sum_i z_i\\
\sum_i z_i & \sum_i z_i^2
\end{bmatrix}
\begin{bmatrix}\theta_0^*\\\theta_1^*\end{bmatrix}
=
\begin{bmatrix}\sum_i y_i\\\sum_i z_i y_i\end{bmatrix}.
\]

Định thức của \(X^\top X\) bằng \(n\sum_i(z_i-\bar z)^2\). Vì AT có phương sai dương trên train, hai cột của \(X\) độc lập tuyến tính và Hessian \(X^\top X/n\) xác định dương. Do đó nghiệm của hệ là nghiệm cực tiểu toàn cục duy nhất. Chương trình giải hệ bằng \(\texttt{numpy.linalg.solve}\).

Vì \(z\) được chuẩn hóa trên train với ddof = 0, \(\bar z=0\) và \(\sum_i z_i^2=n\) trong sai số dấu phẩy động. Chia hai vế hệ cho \(n\) cho ma trận hệ số xấp xỉ \(I_2\) và vế phải xấp xỉ \([454{,}3808711624,\ -16{,}1504701706]^\top\).

## Kiểm tra độc lập và đổi về đơn vị gốc

Tính trực tiếp trên AT gốc bằng công thức hiệp phương sai chia phương sai,

\[
\beta_1=
\frac{\sum_i(AT_i-\overline{AT})(y_i-\bar y)}
     {\sum_i(AT_i-\overline{AT})^2},
\qquad
\beta_0=\bar y-\beta_1\overline{AT}.
\]

Đổi nghiệm chuẩn hóa về cùng thang bằng \(\beta_1=\theta_1^*/\sigma_{AT}\) và \(\beta_0=\theta_0^*-\beta_1\mu_{AT}\). Hai phép tính khớp nhau trong sai số tuyệt đối \(10^{-9}\) theo từng hệ số. Phương trình thu được là

\[
\widehat{PE}\approx496{,}977984877-2{,}169684135\,AT.
\]

Trong mô hình đã khớp, AT tăng 1°C tương ứng với PE dự đoán giảm khoảng 2,17 MW. Hệ số góc mô tả quan hệ trong mô hình hồi quy.

## Đối chiếu với baseline câu 4

| Đại lượng | Nghiệm giải tích | Gradient Descent |
|---|---:|---:|
| \(\theta_0\) theo \(z\) (MW) | 454,380871162 | 454,380870243 |
| \(\theta_1\) theo \(z\) (MW trên một đơn vị \(z\)) | −16,150470171 | −16,150470138 |
| Hệ số chặn theo AT gốc (MW) | 496,977984877 | 496,977983872 |
| Hệ số góc theo AT gốc (MW/°C) | −2,169684135 | −2,169684131 |
| \(J_{\mathrm{train}}\) (MW²) | 14,5922002593049 | 14,5922002593053 |
| Chuẩn L2 gradient trên train | ≈ 1,04 × 10⁻¹³ | ≈ 9,20 × 10⁻⁷ |
| Chênh lệch dự đoán lớn nhất trên train (MW) | 0 (mốc) | ≈ 9,98 × 10⁻⁷ |
| Chênh lệch dự đoán lớn nhất trên test (MW) | 0 (mốc) | ≈ 9,90 × 10⁻⁷ |

Gradient Descent dừng sau 190 lần cập nhật khi chuẩn L2 của gradient nhỏ hơn \(10^{-6}\). Chênh lệch tuyệt đối của \(J_{\mathrm{train}}\) vào khoảng \(4{,}23\times10^{-13}\) MW² trong lần chạy này, sát giới hạn chính xác dấu phẩy động. Trên cả train và test, độ lệch dự đoán tối đa đều dưới ngưỡng nhóm đặt ra là 0,001 MW. Kết quả xác nhận GD tiến đến nghiệm tối ưu của bài toán đã thiết lập. Chất lượng dự đoán so với PE quan sát sẽ được đánh giá ở câu 8.

## Tái lập

Chạy \(\texttt{python run\_q5.py}\) trong thư mục gốc. Lấy số liệu đầy đủ từ \(\texttt{results/q5\_solution.csv}\) và \(\texttt{results/q5\_summary.json}\). Chương trình kiểm tra dữ liệu, cấu hình, thứ tự quan sát, lịch sử và dự đoán đã lưu của baseline trước khi xuất bảng. TV3 có thể dùng hai file này để kiểm tra chéo.
