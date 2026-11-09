# Câu 3 Hàm chi phí và gradient của hồi quy tuyến tính đơn

## 3.1 Mục tiêu

Nhóm cài đặt hàm dự đoán, hàm chi phí và gradient bằng các phép toán NumPy để làm nền cho Gradient Descent ở câu 4. Câu 3 đánh giá các hàm tại những tham số xác định trước, chưa chạy thuật toán tối ưu, chưa xác định tham số cuối và chưa đánh giá hiệu quả dự đoán trên test. Các phép tính dữ liệu thật dùng 7.622 quan sát train của bộ TV2_shared_v1.

## 3.2 Mô hình và dữ liệu đầu vào

Đặt z là AT đã chuẩn hóa bằng trung bình và độ lệch chuẩn train của câu 1:

\[
z_i=\frac{AT_i-\mu_{AT,train}}{s_{AT,train}}.
\]

Trong đó, trung bình train là 19,632864077670°C, độ lệch chuẩn dùng ddof=0 là 7,443696485468°C. Mã luôn đọc giá trị đầy đủ từ scaling.json thay vì dùng số đã làm tròn. PE giữ đơn vị MW.

Mô hình hồi quy tuyến tính đơn được viết:

\[
\hat y_i=f_\theta(z_i)=\theta_0+\theta_1z_i.
\]

Với dạng ma trận:

\[
X=\begin{bmatrix}1&z_1\\1&z_2\\\vdots&\vdots\\1&z_n\end{bmatrix},\qquad
\theta=\begin{bmatrix}\theta_0\\\theta_1\end{bmatrix},\qquad
\hat y=X\theta.
\]

Trong ký hiệu toán học, y và theta được xem là vector cột. Trong NumPy, dự án thống nhất y.shape=(7622,), theta.shape=(2,) và X.shape=(7622,2); X @ theta cho vector dự đoán shape (7622,). Cột đầu tiên toàn số 1 biểu diễn hệ số chặn; cột thứ hai chỉ chứa z. source_row và các biến V, AP, RH không được đưa vào ma trận cơ sở.

## 3.3 Xây dựng hàm chi phí

Đặt r_i = y_hat_i - y_i là sai số dùng bên trong hàm chi phí. Tổng bình phương sai số là:

\[
SSE(\theta)=\sum_{i=1}^{n}(\hat y_i-y_i)^2.
\]

Đề sử dụng hàm mục tiêu:

\[
J(\theta)=\frac{1}{2n}\sum_{i=1}^{n}(\theta_0+\theta_1z_i-y_i)^2
=\frac{1}{2n}\lVert X\theta-y\rVert_2^2.
\]

Bình phương giúp các sai số trái dấu không triệt tiêu nhau. Chia cho n đưa tổng sai số về thang trung bình; hệ số 1/2 giúp triệt tiêu hệ số 2 khi lấy đạo hàm. Hệ số 1/2 không làm đổi nghiệm cực tiểu so với MSE thông thường, nhưng có ảnh hưởng tới độ lớn gradient nên phải giữ định nghĩa nhất quán khi chọn learning rate.

MSE thông thường được định nghĩa:

\[
MSE(\theta)=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2=2J(\theta).
\]

Quan hệ này áp dụng khi dùng cùng tập dữ liệu và cùng dự đoán. Vì PE giữ MW, J và MSE có đơn vị MW². Khi tính trên một tập khác, n phải là số quan sát của tập đang truyền vào hàm, không lấy cố định số dòng train.

## 3.4 Suy ra gradient

Áp dụng quy tắc dây chuyền cho từng số hạng bình phương. Với theta0, đạo hàm của biểu thức bên trong bằng 1:

\[
\frac{\partial J}{\partial\theta_0}
=\frac{1}{2n}\sum_{i=1}^{n}2(\theta_0+\theta_1z_i-y_i)
=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i).
\]

Với theta1, đạo hàm bên trong bằng z_i:

\[
\frac{\partial J}{\partial\theta_1}
=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)z_i.
\]

Ghép hai đạo hàm riêng thành vector:

\[
\nabla J(\theta)=\frac{1}{n}X^\top(X\theta-y).
\]

Mã dùng r = y_hat - y để tính gradient. Phần dư dùng cho câu 8 theo đề là e = y - y_hat, tức e = -r. Hai định nghĩa cho cùng bình phương nhưng khi lấy đạo hàm phải giữ dấu nhất quán.

## 3.5 Triển khai bằng NumPy

Các hàm được tích hợp nguyên vẹn từ src/linear_model.py của TV3:

```python
predict(X, theta)
compute_cost(X, y, theta)
compute_gradient(X, y, theta)
```

Phép tính cốt lõi:

```python
y_hat = X @ theta
error = y_hat - y
J = float(error @ error / (2 * len(y)))
gradient = X.T @ error / len(y)
```

data_io.py đọc đúng dữ liệu và tham số chuẩn hóa đã chốt ở câu 1. linear_model.py kiểm tra shape, dữ liệu hữu hạn và các trường hợp tràn số. Không dùng scikit-learn hoặc bộ tối ưu có sẵn trong câu 3.

## 3.6 Ví dụ tính tay

Chọn hai quan sát z=[-1,1], y=[1,3]. Tại theta=[0,0], dự đoán là [0,0], sai số r=[-1,-3]. Khi đó:

\[
SSE=1+9=10,\qquad J=\frac{10}{2\times2}=2.5,\qquad MSE=5.
\]

Hai thành phần gradient:

\[
\frac{\partial J}{\partial\theta_0}=\frac{-1-3}{2}=-2,\qquad
\frac{\partial J}{\partial\theta_1}=\frac{(-1)(-1)+(-3)(1)}{2}=-1.
\]

Tại theta=[2,1], hai dự đoán là [1,3], trùng hoàn toàn y; J=0 và gradient=[0,0]. Tham số trong ví dụ được chọn để kiểm chứng công thức, không phải tham số học được trên dữ liệu CCPP.

Nguồn kết quả: results/q3_cost_examples.csv.

## 3.7 Kiểm tra gradient độc lập

Nhóm dùng sai phân trung tâm để đối chiếu đạo hàm đã cài đặt:

\[
g_j^{FD}(\theta)\approx
\frac{J(\theta+\varepsilon e_j)-J(\theta-\varepsilon e_j)}{2\varepsilon},
\]

trong đó e_j là vector cơ sở ở tọa độ j. Phép kiểm tra được thực hiện tại nhiều điểm, bao gồm điểm không phải nghiệm, để tránh việc một công thức sai chỉ tình cờ cho gradient bằng 0 tại nghiệm.

Bốn điểm trên ví dụ hai quan sát dùng epsilon=1e-5, ngưỡng sai khác tuyệt đối 1e-8. Ba điểm trên train dùng epsilon=1e-4, ngưỡng 1e-6 để xét sai số số chấm động trong phép trừ các giá trị chi phí lớn. Kết quả chạy trong môi trường kiểm chứng: sai khác lớn nhất trên ví dụ khoảng 1,5143e-10; trên train khoảng 1,2183e-8. Các điểm đều đạt ngưỡng đã đặt. Kết quả và ngưỡng đầy đủ nằm trong q3_gradient_checks.csv.

Sai phân hữu hạn chỉ là phép kiểm tra; Gradient Descent ở câu 4 sẽ sử dụng công thức gradient giải tích, không dùng sai phân để thay đạo hàm.

## 3.8 Tính thử trên dữ liệu train

Tại theta=[0,0], trên 7.622 quan sát train:

| Đại lượng | Giá trị gần đúng |
|---|---:|
| J ban đầu | 103375,999083 MW² |
| MSE ban đầu | 206751,998166 MW² |
| Đạo hàm theo theta0 | -454,380871 |
| Đạo hàm theo theta1 | 16,150470 |

Chi phí ban đầu lớn vì mô hình đang dự đoán PE=0 MW cho mọi quan sát, trong khi PE thực tập trung quanh 454 MW. Đây là giá trị ở điểm khởi tạo, chưa phải kết quả tối ưu hay chỉ số chất lượng cuối của mô hình. Không dùng bảng này để kết luận mô hình tuyến tính dự đoán kém.

## 3.9 Kết luận và bàn giao

Các phép tính tay, đối chiếu gradient bằng sai phân hữu hạn, kiểm tra shape và kiểm tra dữ liệu thật đều đạt. Câu 3 đã xác nhận tính đúng của hàm dự đoán, half-MSE và gradient trên các trường hợp đã kiểm tra; không phải chứng minh số học cho mọi đầu vào có thể có. Các hàm sẵn sàng để dùng trong vòng lặp cập nhật tham số ở câu 4.

Chạy python run_q3.py trong thư mục MAI391_Final. Kết quả gồm q3_cost_examples.csv, q3_gradient_checks.csv, q3_verification.csv và q3_summary.json. Chương trình hiện có 24 kiểm tra, gồm 7 điểm sai phân hữu hạn. Các file dữ liệu chung của câu 1 giữ nguyên.
