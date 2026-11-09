# Câu 6 — Bề mặt chi phí và quỹ đạo tối ưu

## Cách tạo hình

Dùng 7.622 quan sát train của TV2_shared_v1 với ma trận thiết kế \(X=[\mathbf{1},z]\), trong đó \(z=(AT-\mu_{AT})/\sigma_{AT}\) được chuẩn hóa bằng thống kê train. Cả hai hình dùng \(\theta_0,\theta_1\) trong hệ tọa độ này. Chi phí là \(J(\theta)=\|X\theta-y\|^2/(2n)\), đơn vị MW².

Đặt \(\delta=\theta-\theta^*\), với \(\theta^*\) là nghiệm giải tích câu 5. Chương trình tính chi phí trên lưới \(161\times161\) bằng khai triển bậc hai chính xác:

\[
J(\theta)=J(\theta^*)+\nabla J(\theta^*)^\top\delta+
\frac{1}{2}\delta^\top\left(\frac{X^\top X}{n}\right)\delta.
\]

Hình 3D vẽ quỹ đạo \((\theta_0^{(k)},\theta_1^{(k)},J^{(k)})\) từ lịch sử câu 4. Hình 2D chiếu cùng quỹ đạo lên mặt phẳng tham số. Cả hai hình dùng đủ 191 trạng thái gốc, gồm khởi tạo và 190 lần cập nhật. Ô phóng to dùng một lưới riêng và giới hạn vùng nhìn quanh nghiệm giải tích. Tam giác là điểm khởi tạo, ngôi sao là điểm cuối GD.

## Kết quả và nhận xét

- **Hình 6a — Bề mặt 3D** (`figures/q6_surface.png`). Hàm chi phí có dạng paraboloid lồi. GD bắt đầu tại \((0,0)\) với \(J\approx103\,375{,}999083\ \mathrm{MW}^2\), đi xuống vùng đáy và kết thúc tại \((454{,}380870243;\,-16{,}150470138)\) với \(J\approx14{,}5922002593\ \mathrm{MW}^2\).
- **Hình 6b — Đường đồng mức** (`figures/q6_contour.png`). Các đường bao gần tròn vì chuẩn hóa train với ddof = 0 làm \(X^\top X/n\) xấp xỉ ma trận đơn vị. Với learning rate 0,1, hướng dịch chuyển gần như thẳng về tâm. Khoảng cách giữa các trạng thái liên tiếp giảm khi tới vùng đáy. Ô phóng to cho thấy điểm cuối GD gần như trùng với nghiệm giải tích ở tỷ lệ hiển thị.
- **Kiểm tra quỹ đạo.** Tính lại \(J\) trực tiếp tại mọi trạng thái đã lưu và đối chiếu với giá trị tính bằng công thức bề mặt. Sai khác tuyệt đối tối đa giữa công thức bề mặt và phép tính trực tiếp dọc quỹ đạo là khoảng \(2{,}91\times10^{-11}\ \mathrm{MW}^2\). Số liệu đầy đủ nằm trong `results/q6_summary.json`.

Chạy `python run_q6.py` ở thư mục gốc để tạo lại hai hình. Câu 7 sẽ dùng chính lịch sử đã lưu để phân tích \(J\) theo số vòng lặp và so sánh learning rate.
