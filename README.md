# MAI391 Final

## Phạm vi

Gói này gồm đủ câu 1–10 trên bộ dữ liệu TV2_shared_v1. Câu 1–8 là phần bắt buộc, câu 9–10 là phần cộng điểm.

## Chạy toàn bộ từ dữ liệu gốc

Cài phụ thuộc bằng lệnh python -m pip install -r requirements_q9.txt. Mở notebooks/project_master.ipynb trong Jupyter hoặc VS Code với cùng môi trường Python rồi chạy các ô từ trên xuống. Có thể chạy lần lượt python run_q1.py đến python run_q10.py tại thư mục MAI391_Final. Câu 1 tái tạo các tập dữ liệu từ Excel gốc. Các câu sau ghi kết quả vào results và figures.

## Chạy câu 10

Sau khi có baseline câu 4 và kết quả câu 8, chạy từ thư mục MAI391_Final.

~~~bash
python run_q10.py
~~~

Kết quả kiểm tra là Q10 CHECK: PASS (degree 2 selected on validation). MSE test giảm từ 30,547660 xuống 27,921013 MW² trên cùng tập test.

- src/improvement.py: chia validation nội bộ, chuẩn hóa và huấn luyện bằng Gradient Descent.
- results/q10_validation.csv và q10_comparison.csv: kết quả chọn bậc và so sánh hai mô hình.
- results/q10_test_predictions.csv và q10_summary.json: dự đoán, phần dư, cấu hình.
- figures/q10_comparison.png và notes/q10_noi_dung_bao_cao.md: hình, phương pháp và diễn giải.

## Chạy câu 9

Chạy từ thư mục MAI391_Final.

~~~bash
python -m pip install -r requirements_q9.txt
python run_q9.py
~~~

Câu 9 so sánh GD, phương trình chuẩn và scikit-learn trên cùng train/test đã chốt. Cần baseline câu 4, nghiệm câu 5 và bảng chỉ số câu 8. Kết quả kiểm tra là Q9 CHECK: PASS (3 methods, 30 timed fits per method).

- results/q9_comparison.csv: hệ số, chi phí, chỉ số test, sai khác dự đoán và thời gian khớp.
- results/q9_summary.json: thiết lập đo, phiên bản môi trường và kết quả kiểm tra.
- notes/q9_noi_dung_bao_cao.md: bảng và nhận xét để đưa vào báo cáo.

## Chạy câu 8

Trong cùng môi trường với câu 4, chạy từ thư mục MAI391_Final.

~~~bash
python run_q8.py
~~~

Câu 8 đánh giá baseline GD câu 4 trên train và test đã chốt. Kết quả kiểm tra là Q8 CHECK: PASS (9527 predictions verified).

- figures/q8_fit.png: đường hồi quy trên dữ liệu quan sát.
- figures/q8_residual_train.png và q8_residual_test.png: phần dư theo PE dự đoán.
- results/q8_metrics.csv, q8_predictions.csv và q8_examples.csv: chỉ số, dự đoán theo source_row và ví dụ suy luận.
- results/q8_residual_bins.csv và q8_summary.json: thống kê theo nhóm, tham số và kết quả kiểm tra.
- notes/q8_noi_dung_bao_cao.md: kết quả và nhận xét phần dư cho báo cáo.

## Chạy câu 7

Trong cùng môi trường với câu 4, chạy từ thư mục MAI391_Final.

~~~bash
python run_q7.py
~~~

Câu 7 đọc năm lịch sử trên train ở câu 4 và nghiệm giải tích ở câu 5. Nếu thiếu các đầu vào này, chạy câu 4 rồi câu 5 trước. Kết quả kiểm tra là Q7 CHECK: PASS (5 runs, 2587 states verified).

- figures/q7_convergence.png: chi phí theo bước và khoảng cách đến tối ưu của baseline.
- figures/q7_learning_rate.png: so sánh tốc độ hội tụ, đổi phía của tham số và phân kỳ.
- results/q7_runs.csv và q7_summary.json: bảng kết quả, cấu hình và kiểm tra.
- notes/q7_noi_dung_bao_cao.md: thuyết minh và cách đọc hình.

## Chạy câu 6

Dùng cùng môi trường với câu 4. Chạy lệnh sau trong thư mục MAI391_Final.

~~~bash
python run_q6.py
~~~

Kết quả cuối là Q6 CHECK: PASS (191 states verified). Nếu thiếu đầu vào, chạy câu 4 rồi câu 5 trước.

- figures/q6_surface.png: bề mặt chi phí 3D và quỹ đạo GD.
- figures/q6_contour.png: đường đồng mức toàn quỹ đạo và vùng phóng to quanh nghiệm.
- results/q6_summary.json: tọa độ, chi phí và kết quả xác minh lịch sử.
- notes/q6_noi_dung_bao_cao.md: phương pháp, chú thích và nhận xét cho báo cáo.

Cả hai hình dùng baseline câu 4 trên train thật, gồm đủ 191 trạng thái từ bước 0 đến 190.

## Chạy câu 5

Mở terminal trong thư mục MAI391_Final. Nếu đã có môi trường của câu 4, chạy trực tiếp:

~~~bash
python run_q5.py
~~~

Kết quả cuối phải có `Q5 CHECK: PASS`. Chương trình đọc dữ liệu đã chốt và baseline lưu từ câu 4. Nếu thiếu baseline, chạy `python run_q4.py` trước. Câu 5 khớp nghiệm trên train và đối chiếu dự đoán với GD trên cả train và test.

- src/analytical.py: nghiệm phương trình chuẩn, kiểm tra bằng công thức hiệp phương sai trên AT gốc và đổi thang hệ số.
- results/q5_solution.csv: hệ số hai thang, J_train, chuẩn gradient và chênh lệch dự đoán của hai phương pháp.
- results/q5_summary.json: nghiệm tham chiếu, hệ phương trình trên mỗi mẫu và sai khác với GD.
- notes/q5_noi_dung_bao_cao.md: suy ra công thức, bảng số liệu và diễn giải để đưa vào báo cáo.

Kết quả dự đoán của GD lệch tối đa khoảng 9,98 × 10⁻⁷ MW trên train và 9,90 × 10⁻⁷ MW trên test so với nghiệm giải tích. Các câu 6–8 tiếp tục dùng quỹ đạo và nghiệm GD từ baseline câu 4. Câu 9 có thể dùng dòng analytical trong q5_solution.csv làm mốc đối chiếu.

## Chạy câu 4

Mở terminal trong thư mục MAI391_Final, tại vị trí có run_q4.py.

```bash
python -m pip install -r requirements_q4.txt
python run_q4.py
```

Chương trình dùng dữ liệu và chuẩn hóa đã chốt, không chia tập lại. Hàm Batch Gradient Descent của TV3 được giữ nguyên, chỉ tích hợp thêm chương trình điều phối và kiểm chứng. Các thử nghiệm learning rate chỉ dùng train. Nghiệm giải tích là mốc kiểm tra, không thay thế cập nhật GD; phần suy ra đầy đủ thuộc câu 5.

Kết quả baseline: learning rate 0,1, theta ban đầu [0,0], tolerance=1e-6 theo chuẩn L2 gradient, max_iter=10000. Chương trình hội tụ sau 190 cập nhật và lưu 191 trạng thái gồm bước 0. J_train cuối khoảng 14,592200259; theta theo z khoảng [454,380870243; -16,150470138]. Mô hình theo AT gốc là PE_hat ≈ 496,977983872 - 2,169684131 × AT. Kết quả cuối phải có `Q4 CHECK: PASS (27 checks)`.

- src/gradient_descent.py: thuật toán GD của TV3.
- src/q4_training.py: chạy train, kiểm tra và xuất kết quả.
- src/reference.py: nghiệm độc lập để kiểm chứng; không được dùng cập nhật GD.
- results/baseline_run.npz: nghiệm GD, lịch sử và dự đoán train/test.
- results/baseline_config.json: cấu hình, stop_rule/stop_reason, phiên bản và dấu vân tay dữ liệu.
- results/q4_baseline_trace.csv: trạng thái ở từng bước.
- results/q4_learning_rate_runs.csv và q4_*_history.npz: lịch sử các learning rate.
- results/q4_summary.json và q4_verification.csv: kết quả và kiểm tra.
- figures/q4_baseline_cost.png: minh họa đường chi phí baseline.
- notes/q4_noi_dung_bao_cao.md: quy tắc cập nhật, ví dụ, tiêu chí dừng và diễn giải.

baseline_run.npz giữ 5 trường cơ bản và thêm lịch sử các run small, appropriate, large, divergent để các câu sau dùng chung. baseline_config.json ghi rõ tiêu chí chuẩn gradient, không dùng chênh lệch hai cost. Dự đoán test được xuất sau khi baseline cố định, chưa tính chỉ số chất lượng test trong câu 4. Gói bước 4 chứa đủ câu 1–4; giữ ghi chú riêng trước khi chuyển bản mới.

## Chạy câu 3

Mở terminal trong thư mục MAI391_Final, tại vị trí có run_q3.py.

```bash
python -m pip install -r requirements_q3.txt
python run_q3.py
```

Kết quả cuối phải có `Q3 CHECK: PASS (24 checks; 7 finite-difference points)`. Chương trình xác nhận dữ liệu câu 1, kiểm tra hàm dự đoán, hàm half-MSE và gradient của TV3 bằng ví dụ tính tay và sai phân hữu hạn. Đây chưa phải lần chạy Gradient Descent; không xuất nghiệm tối ưu hoặc chỉ số dự đoán test.

- src/linear_model.py: giữ nguyên các hàm của TV3, dùng NumPy.
- src/data_io.py: đọc dữ liệu và thông số chuẩn hóa chung, tạo X=[1,z].
- src/q3_checks.py: phép kiểm tra và xuất bằng chứng.
- results/q3_cost_examples.csv: ví dụ tính tay và giá trị ở điểm khởi tạo trên train.
- results/q3_gradient_checks.csv: đối chiếu 7 điểm bằng sai phân hữu hạn.
- results/q3_verification.csv: kết quả các kiểm tra.
- results/q3_summary.json: kích thước dữ liệu, kết quả và phạm vi kiểm chứng.
- notes/q3_noi_dung_bao_cao.md: mô hình, suy ra công thức, giải thích và thuyết minh.

Tại theta=[0,0], J_train xấp xỉ 103375,999083 và MSE_train bằng 2J trong sai số số học. Đây là chi phí trước tối ưu. Không nhầm với chi phí cuối ở câu 4. Các file dữ liệu và chuẩn hóa của câu 1 không thay đổi. Gói bước 3 có đủ câu 1–3; giữ ghi chú cá nhân riêng trước khi chuyển sang thư mục bản mới.

## Chạy câu 2

Mở terminal trong thư mục MAI391_Final, tại vị trí có run_q2.py.

```bash
python -m pip install -r requirements_q2.txt
python run_q2.py
```

run_q2.py kiểm tra bộ dữ liệu câu 1 rồi chạy EDA trên train và kiểm tra các kết quả. Nếu chưa có dữ liệu xử lý, chạy python run_q1.py trước. Câu 2 không chia lại train/test và không thay thông số chuẩn hóa.

Kết quả cuối phải có `Q2 CHECK: PASS (17 checks)`. Cả 7.622 dòng train đều được bao phủ trong bảng theo khoảng nhiệt độ. Câu 2 không loại thêm dòng nào và không huấn luyện mô hình.

- figures/q2_scatter.png: quan hệ AT–PE, tương quan và đường trung bình theo khoảng có ít nhất 30 quan sát.
- figures/q2_distributions.png: histogram hai biến.
- figures/q2_boxplots.png: boxplot theo quy tắc 1,5 × IQR.
- results/q2_descriptive_stats.csv, q2_correlation.csv, q2_iqr_check.csv, q2_binned_means.csv: bảng EDA.
- results/q2_provenance.json và q2_verification.csv: quy ước, nguồn đầu vào và kiểm tra.
- notes/q2_noi_dung_bao_cao.md: hướng dẫn chèn hình và thuyết minh.

Độ lệch chuẩn mô tả trong câu 2 dùng ddof=1; chuẩn hóa câu 1 vẫn dùng ddof=0. Không đổi scaling.json theo số đã làm tròn trong bảng mô tả. Bình phương tương quan train không phải R² test.

Mã EDA được tích hợp từ TV2 và sửa biên khoảng nhiệt độ để không bỏ sót quan sát 37,11°C. Khoảng 36–40°C có một quan sát; bảng vẫn lưu khoảng này nhưng không đưa vào đường nối trung bình. Gói bước 2 đã chứa toàn bộ bước 1 nên có thể giải nén ra một thư mục mới và dùng làm bản làm việc tiếp theo. Nếu đã sửa ghi chú riêng trong gói bước 1, sao lưu hoặc chuyển ghi chú trước khi thay thư mục.

## Chạy câu 1

Giải nén ZIP, mở terminal trong thư mục MAI391_Final, tại vị trí có run_q1.py.

```bash
python -m pip install -r requirements_q1.txt
python run_q1.py
```

Nếu máy dùng lệnh `py` thay cho `python`, thay nhất quán ở cả hai lệnh. Nếu đã có môi trường chứa numpy, pandas và openpyxl, có thể dùng môi trường đó. requirements_q1.txt liệt kê phụ thuộc; chưa kiểm thử cài mới trên máy người dùng.

Kết quả cuối phải có `Q1 DATA CHECK: PASS`: raw 9568, clean 9527, train 7622 và test 1905. Các file kết quả đã được tạo và kiểm tra sẵn. Khi chạy lại, script tái tạo và ghi đè file sinh ra; không sửa Excel gốc.

## Quy ước thống nhất

- Nguồn: dữ liệu Excel gốc trong gói TV2, chỉ đọc Sheet1.
- Giữ bản đầu trong mỗi nhóm trùng toàn bộ 5 cột, loại 41 bản sao dư. Đây là chính sách xử lý của nhóm, không khẳng định các dòng trùng là lỗi đo.
- Dùng AT dự đoán PE. AT có đơn vị độ C, PE có đơn vị MW.
- Cách chia của TV2: default_rng(42), round(0.2*n) quan sát làm test, giữ thứ tự source_row trong mỗi tập.
- Chuẩn hóa AT bằng mean và std của train, ddof=0; PE giữ nguyên.
- source_row là số dòng Excel, không dùng làm đặc trưng.
- Giữ nguyên RH > 100% và ghi nhận bất thường. Không diễn giải ngưỡng kiểm tra trong mã TV2 là xác nhận hợp lệ về mặt vật lý.
- Không thay train.csv, test.csv hoặc scaling.json riêng lẻ. Nếu thay quy ước, phải chạy lại toàn bộ mô hình, hình và chỉ số phụ thuộc.

## Nội dung chính

- src/data_prep.py: mã chuẩn bị dữ liệu gốc của TV2; bổ sung encoding UTF-8 khi ghi JSON để hỗ trợ Windows.
- src/verify_data.py: kiểm tra độc lập cách làm sạch, chia tập, thứ tự dòng, dữ liệu và chuẩn hóa.
- results/data_contract.json: mô tả bộ dữ liệu chung và SHA256 của đầu vào.
- results/q1_verification.csv: các kiểm tra PASS.
- notes/q1_noi_dung_bao_cao.md: nội dung gợi ý để hoàn thiện phần câu 1.

## Bàn giao cho các câu sau

TV3 phải chạy trên train/test và scaling.json trong gói này, không tự chia lại. Trong gói TV3 đã nộp, dùng chế độ --use-prepared sau khi nhận các file tương ứng. TV4 và câu 9 chỉ nhận baseline được tạo trên cùng dữ liệu này.
