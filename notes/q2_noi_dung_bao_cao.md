# Câu 2 Khám phá và trực quan hóa dữ liệu

## 2.1 Mục tiêu và phạm vi

Phần khám phá dữ liệu nhằm mô tả phân phối của nhiệt độ môi trường AT và công suất điện PE, nhận diện chiều và dạng quan hệ giữa hai biến, đồng thời kiểm tra các điểm cần xem xét trước khi xây dựng hồi quy tuyến tính đơn. Các thống kê, biểu đồ và nhận xét ở câu 2 dùng tập train gồm 7.622 quan sát của bộ TV2_shared_v1. Nhóm không chia lại dữ liệu, không thay thông số chuẩn hóa và không sử dụng test để chọn dạng mô hình. Câu 2 không huấn luyện mô hình hay báo cáo kết quả dự đoán trên test.

## 2.2 Thống kê mô tả

| Đại lượng | AT tính bằng °C | PE tính bằng MW |
|---|---:|---:|
| Số quan sát | 7.622 | 7.622 |
| Trung bình | 19,6329 | 454,3809 |
| Độ lệch chuẩn mẫu | 7,4442 | 17,0312 |
| Nhỏ nhất | 1,81 | 420,26 |
| Phân vị 25 phần trăm | 13,53 | 439,74 |
| Trung vị | 20,285 | 451,785 |
| Phân vị 75 phần trăm | 25,71 | 468,22 |
| Lớn nhất | 37,11 | 495,35 |

Nguồn: results/q2_descriptive_stats.csv. Độ lệch chuẩn mô tả trong bảng là độ lệch chuẩn mẫu, ddof=1. Độ lệch chuẩn dùng để chuẩn hóa ở câu 1 là ddof=0 và được giữ nguyên trong scaling.json. Hai giá trị chênh nhau rất nhỏ nhưng có mục đích khác nhau; không thay thông số chuẩn hóa bằng số đã làm tròn trong báo cáo.

AT và PE có miền giá trị đủ rộng để khảo sát mối quan hệ. Khoảng giữa hai tứ phân vị của AT là 13,53–25,71°C, trong khi PE có khoảng tương ứng 439,74–468,22 MW. Trung bình và trung vị của hai biến không trùng nhau. Việc mô tả toàn bộ phân phối cần kết hợp histogram, không chỉ dựa vào một giá trị trung bình.

## 2.3 Biểu đồ phân tán và quan hệ giữa hai biến

Chèn figures/q2_scatter.png.

Chú thích gợi ý: Hình 2.1 Mối quan hệ giữa nhiệt độ môi trường AT và công suất điện PE trên tập train. Đường màu cam nối các trung bình theo khoảng AT rộng 4°C có ít nhất 30 quan sát; đây là đường mô tả theo nhóm, không phải đường hồi quy đã huấn luyện.

Đám điểm cho thấy quan hệ nghịch rõ rệt: tại vùng AT thấp, PE thường cao hơn, còn tại vùng AT cao, PE thường thấp hơn. Hệ số tương quan Pearson trên train là r = -0,948352, thể hiện mối liên hệ tuyến tính nghịch mạnh. Kết quả này tạo cơ sở để sử dụng hồi quy tuyến tính đơn làm mô hình cơ sở, nhưng không khẳng định đường thẳng mô tả toàn bộ cấu trúc dữ liệu.

Đường trung bình theo khoảng gợi ý độ dốc có thể thay đổi giữa vùng nhiệt độ trung bình và hai đầu miền quan sát. Một số quan sát nằm xa xu hướng chung cũng xuất hiện trên scatter. Đây là các dấu hiệu cần kiểm tra thêm bằng phần dư ở câu 8 trước khi quyết định cách cải thiện tại câu 10. Không loại chúng chỉ vì mô hình tuyến tính có thể dự đoán kém.

Pearson là thước đo liên hệ, không phải bằng chứng nhân quả. Cũng không diễn giải |r| = 0,9484 thành “độ chính xác dự đoán 94,84%”. Giá trị pearson_r_squared trong file là bình phương tương quan train; nó không thay cho R² test ở câu 8.

## 2.4 Phân phối của AT và PE

Chèn figures/q2_distributions.png.

Chú thích gợi ý: Hình 2.2 Histogram của AT và PE trên tập train; đường đứt nét biểu diễn trung vị.

Các histogram cho thấy quan sát không phân bố đồng đều trên toàn miền. AT có nhiều quan sát ở các vùng nhiệt độ trung gian và có nhiều đỉnh cục bộ; PE cũng thể hiện nhiều vùng tập trung. Vì vậy, không mô tả hai biến như các phân phối chuẩn đã được xác nhận. Độ lệch thống kê của AT khoảng -0,138, của PE khoảng 0,304; độ lệch nhỏ không đồng nghĩa phân phối chuẩn hoặc đơn đỉnh.

Hồi quy tuyến tính không yêu cầu bản thân AT và PE phải có phân phối chuẩn. Tính phù hợp của đường thẳng và cấu trúc sai số sẽ được đánh giá tiếp bằng phần dư; histogram hai biến chưa đủ để kết luận về những giả định đó.

## 2.5 Boxplot và các điểm cần kiểm tra

Chèn figures/q2_boxplots.png.

Chú thích gợi ý: Hình 2.3 Boxplot của AT và PE trên tập train, dùng quy tắc râu 1,5 lần khoảng tứ phân vị.

Theo quy tắc 1,5 × IQR, hàng rào của AT là -4,74 đến 43,98°C; hàng rào của PE là 397,02 đến 510,94 MW. Không có quan sát train vượt hàng rào của từng biến. Nhóm không loại thêm dòng nào ở câu 2.

Kết quả này chỉ cho thấy không có ngoại lệ đơn biến theo quy tắc đang xét. Nó không chứng minh dữ liệu không có điểm bất thường trong quan hệ AT–PE. Một điểm có giá trị AT và PE riêng lẻ nằm trong hàng rào vẫn có thể lệch nhiều khỏi xu hướng chung, nên cần kiểm tra phần dư và dữ liệu gốc trước khi quyết định xử lý.

## 2.6 Trung bình theo khoảng nhiệt độ và giới hạn diễn giải

Nguồn: results/q2_binned_means.csv. Các khoảng rộng 4°C bao phủ toàn bộ train, từ 0 đến 40°C; tổng số quan sát trong các khoảng bằng 7.622. Khoảng 0–4°C có 33 quan sát, khoảng 32–36°C có 180 quan sát, trong khi khoảng 36–40°C chỉ có một quan sát. Vì vậy, các nhận xét về hai đầu miền dữ liệu cần thận trọng.

Tất cả các khoảng vẫn có trong bảng. Chỉ những khoảng có ít nhất 30 quan sát mới được nối thành đường trung bình trên scatter; ngưỡng này là lựa chọn trình bày, không phải kiểm định thống kê và không làm mất điểm khỏi scatter. Độ lệch chuẩn mẫu của PE tại khoảng chỉ có một quan sát không xác định, nên ô tương ứng để trống. Đây không phải dữ liệu PE bị thiếu.

## 2.7 Kết luận phục vụ các câu sau

Dữ liệu train cho thấy liên hệ nghịch mạnh giữa AT và PE, hỗ trợ chọn hồi quy tuyến tính đơn làm mô hình cơ sở. Tuy nhiên, dấu hiệu thay đổi độ dốc và các điểm lệch khỏi xu hướng cần được kiểm chứng bằng phần dư sau khi fit. Câu 2 không tự xác nhận hồi quy bậc hai tốt hơn, không xóa thêm dữ liệu và không sử dụng test để lựa chọn mô hình.

## Hồ sơ và cách chạy

Chạy python run_q2.py trong thư mục MAI391_Final. Chương trình kiểm tra dữ liệu câu 1, tạo ba biểu đồ và bốn bảng EDA, sau đó xác nhận bằng results/q2_verification.csv. results/q2_provenance.json ghi tập sử dụng, quy ước thống kê, cách chia khoảng và dấu vân tay dữ liệu đầu vào. Các tệp đầu vào của câu 1 không thay đổi.
