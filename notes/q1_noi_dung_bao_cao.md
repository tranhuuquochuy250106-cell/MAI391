# Câu 1 Thu thập và chuẩn bị dữ liệu

## Nguồn và phạm vi

Nhóm sử dụng bộ Combined Cycle Power Plant do Tüfekci và Kaya công bố trên UCI Machine Learning Repository năm 2014, DOI 10.24432/C5002N. Dữ liệu gồm 9.568 quan sát thực tế thu thập trong giai đoạn 2006–2011, khi nhà máy vận hành đầy tải. Các biến được biểu diễn bằng giá trị trung bình theo giờ. Nhóm chỉ sử dụng Sheet1 của Folds5x2_pp.xlsx; năm sheet là các bản xáo trộn của cùng dữ liệu, không ghép thành các quan sát mới.

Nguồn: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

Ngày truy cập: ghi ngày nhóm thực tế truy cập nguồn. Không suy ra ngày truy cập từ ngày tạo gói hoặc ngày chạy mã.

## Các biến

| Biến | Ý nghĩa | Đơn vị |
|---|---|---|
| AT | Nhiệt độ môi trường | °C |
| V | Độ chân không khí xả | cm Hg |
| AP | Áp suất môi trường | mbar |
| RH | Độ ẩm tương đối | % |
| PE | Công suất điện đầu ra ròng | MW |

Mô hình cơ sở dùng duy nhất AT làm biến đầu vào và PE làm biến mục tiêu. Các cột khác được giữ để kiểm tra và truy vết dữ liệu, không đưa vào mô hình tuyến tính đơn. source_row là số dòng Excel bổ sung để nối đúng quan sát giữa các phần.

## Chất lượng và chính sách làm sạch

Không phát hiện giá trị thiếu hoặc không hữu hạn trong năm cột của Sheet1. Nhóm phát hiện 41 bản sao dư giống hoàn toàn trên năm cột và giữ lần xuất hiện đầu tiên, còn 9.527 dòng. Vì thiếu ID đo lường và thời gian gốc, không khẳng định chắc chắn nguyên nhân trùng; loại bản sao là chính sách xử lý được công bố của nhóm. Không loại dòng chỉ vì AT hoặc cặp AT–PE lặp lại, không loại điểm cực trị để cải thiện chỉ số.

Sheet1 có 55 dòng RH vượt 100%, với giá trị lớn nhất 100,16%. Nhóm ghi nhận bất thường này và giữ nguyên dữ liệu; chưa xác định nguyên nhân đo lường. RH không phải biến đầu vào của mô hình cơ sở. Giữ dữ liệu không đồng nghĩa xác nhận các giá trị này hợp lệ về vật lý.

## Chia tập và chuẩn hóa

Từ 9.527 dòng sạch, nhóm chia một lần bằng numpy.random.default_rng(42).permutation: lấy round(0,2 × 9.527) = 1.905 chỉ số đầu làm test; 7.622 chỉ số còn lại làm train. Mỗi tập được lưu theo thứ tự source_row. Hai tập không có source_row giao nhau và hợp của chúng khớp dữ liệu sạch.

AT được chuẩn hóa bằng trung bình train 19,632864077670 °C và độ lệch chuẩn train 7,443696485468 °C, dùng ddof=0. Test và đầu vào dự đoán mới sử dụng đúng hai tham số này. PE giữ đơn vị MW. Đây là lựa chọn triển khai của nhóm; đề không ấn định seed hoặc tỷ lệ chia tập.

## Giới hạn

Bản phát hành đã được xáo trộn và không có dấu thời gian để khôi phục thứ tự quan sát. Vì vậy, đánh giá trên tập test giữ lại không được mô tả là dự báo tương lai theo thời gian. Việc dùng dữ liệu tại một nhà máy trong điều kiện đầy tải cũng giới hạn khả năng khái quát sang nhà máy và chế độ vận hành khác.

## Bằng chứng tái lập

Chạy python run_q1.py để tái tạo dữ liệu và kiểm tra độc lập. results/data_audit.csv ghi xử lý dữ liệu; results/q1_verification.csv lưu kết quả kiểm tra; results/data_contract.json ghi quy ước và dấu vân tay SHA256 của các file chung. Các số liệu mô hình không thuộc phạm vi câu 1.
