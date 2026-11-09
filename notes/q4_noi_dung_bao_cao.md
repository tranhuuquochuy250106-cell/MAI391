# Câu 4 Tự cài đặt và thực thi Gradient Descent

## 4.1 Mục tiêu và phạm vi

Nhóm sử dụng Batch Gradient Descent tự cài đặt để tìm hai tham số của mô hình PE theo AT đã chuẩn hóa. Các hàm dự đoán, half-MSE và gradient ở câu 3 được giữ nguyên. Hàm gradient_descent của TV3 được tích hợp nguyên vẹn vào project chung; chương trình điều phối đọc đúng bộ dữ liệu TV2_shared_v1 gồm 7.622 dòng train và 1.905 dòng test, không tự chia lại dữ liệu.

Việc huấn luyện và thử learning rate chỉ dùng train. Sau khi cố định baseline, chương trình xuất dự đoán train và test để bàn giao; chưa tính MSE, RMSE hay R² trên test trong câu 4. Nghiệm giải tích được dùng như mốc kiểm tra độc lập, không được dùng thay bước cập nhật của GD. Phần suy ra và so sánh nghiệm giải tích đầy đủ thuộc câu 5.

## 4.2 Quy tắc cập nhật

Với hàm mục tiêu đã xác nhận ở câu 3:

\[
J(\theta)=\frac{1}{2n}\lVert X\theta-y\rVert_2^2,\qquad
\nabla J(\theta)=\frac1n X^\top(X\theta-y),
\]

Gradient Descent thực hiện:

\[
\theta^{(k+1)}=\theta^{(k)}-\alpha\nabla J(\theta^{(k)}).
\]

Alpha là learning rate dương. Gradient chỉ hướng tăng cục bộ của J; cập nhật theo dấu trừ nhằm giảm hàm mục tiêu khi bước phù hợp. Batch nghĩa là mỗi gradient sử dụng toàn bộ train, không lấy mini-batch hay một quan sát riêng lẻ.

Hai thành phần của theta được cập nhật đồng thời từ cùng trạng thái cũ. Mã tạo một vector updated mới từ gradient đã tính; không cập nhật theta0 trước rồi dùng theta0 mới để tính đạo hàm theo theta1.

## 4.3 Cấu hình baseline

| Thành phần | Giá trị hoặc quy ước |
|---|---|
| Đặc trưng | AT chuẩn hóa bằng mean/std train của câu 1 |
| Mục tiêu | PE giữ đơn vị MW |
| Ma trận X | 7.622 dòng, cột 1 và cột AT chuẩn hóa |
| Tham số ban đầu | theta0=0, theta1=0 |
| Learning rate baseline | 0,1 |
| Số cập nhật tối đa | 10.000 |
| Tolerance | 1e-6 theo chuẩn L2 gradient |
| Ngưỡng kiểm tra dự đoán với nghiệm giải tích | 0,001 MW trên train |

Baseline alpha=0,1 được cố định để có quỹ đạo nhiều bước dễ quan sát và giải thích trong các câu sau. Đây không phải khẳng định alpha=0,1 là learning rate nhanh nhất. Seed và cách chia tập không đổi trong quá trình huấn luyện.

## 4.4 Các bước của thuật toán

1. Đọc dữ liệu và thông số chuẩn hóa đã chốt; tạo X và y đúng shape.
2. Khởi tạo theta và lưu theta, J ở bước 0.
3. Tính gradient trên toàn bộ train tại theta hiện tại.
4. Nếu chuẩn L2 gradient không vượt tolerance, báo converged và dừng.
5. Nếu chưa đạt, cập nhật đồng thời cả hai tham số; tính và lưu chi phí mới.
6. Tiếp tục đến khi đạt tiêu chí, hết giới hạn cập nhật hoặc gặp tình trạng phân kỳ, không hữu hạn.

Giả mã cốt lõi:

```python
theta = theta_initial.copy()
save(theta, compute_cost(X, y, theta))
for k in range(max_iter):
    gradient = compute_gradient(X, y, theta)
    if norm(gradient) <= tolerance:
        status = 'converged'
        break
    theta = theta - learning_rate * gradient
    save(theta, compute_cost(X, y, theta))
```

Đây là mô tả để giải thích thuật toán. Mã thực còn kiểm tra gradient ở trạng thái cuối, xử lý số không hữu hạn và ngưỡng phân kỳ, xem src/gradient_descent.py. Kết quả returned mô tả trạng thái cuối đã lưu.

## 4.5 Ví dụ một bước cập nhật

Tiếp tục ví dụ câu 3 với z=[-1,1], y=[1,3], theta=[0,0]. Gradient tại điểm này là [-2,-1]. Với alpha=0,1:

\[
\theta^{(1)}=[0,0]-0.1[-2,-1]=[0.2,0.1].
\]

Dự đoán sau bước đầu là 0,1 và 0,3, tương ứng mảng Python [0.1, 0.3]. Chi phí giảm từ 2,5 xuống 2,025. Kiểm tra này xác nhận cả hai tham số được cập nhật từ cùng gradient ban đầu.

## 4.6 Điều kiện dừng và trạng thái

\[
\lVert\nabla J(\theta)\rVert_2\le 10^{-6}.
\]

Tolerance được áp dụng cho chuẩn gradient, không phải chênh lệch hai chi phí liên tiếp. Việc đường chi phí trông phẳng không tự chứng minh đã hội tụ. Chương trình phân biệt:

| Trạng thái | Ý nghĩa |
|---|---|
| converged | Đã đạt chuẩn gradient theo tolerance |
| max_iter | Hết số cập nhật cho phép, chưa đạt tiêu chí |
| diverged | Chi phí vượt ngưỡng bảo vệ thực nghiệm |
| nonfinite | Gradient hoặc bước mới không hữu hạn, giữ trạng thái hợp lệ cuối |

Ngưỡng phân kỳ trong mã là J lớn hơn 1e12 lần max(J ban đầu,1). Đây là cơ chế dừng thí nghiệm khi chi phí bùng nổ, không phải phát biểu rằng máy đã tính đến vô cực. Lịch sử baseline và ví dụ phân kỳ được kiểm tra hữu hạn, có các cặp theta và cost cùng trạng thái.

## 4.7 Kết quả baseline trên dữ liệu chung

| Đại lượng | Giá trị gần đúng |
|---|---:|
| Trạng thái | converged |
| Số lần cập nhật | 190 |
| Số trạng thái lịch sử gồm bước 0 | 191 |
| theta0 theo AT chuẩn hóa | 454,380870243 |
| theta1 theo AT chuẩn hóa | -16,150470138 |
| J train ban đầu | 103375,999082787 MW² |
| J train cuối | 14,592200259 MW² |
| MSE train cuối | 29,184400519 MW² |
| Chuẩn gradient cuối | 9,19964e-7 |
| Sai khác dự đoán lớn nhất với nghiệm giải tích trên train | 9,97628e-7 MW |

Chi phí giảm mạnh từ điểm khởi tạo và chuẩn gradient cuối nhỏ hơn 1e-6, phù hợp điều kiện dừng. Sai khác dự đoán với nghiệm tham chiếu nhỏ hơn ngưỡng 0,001 MW đã đặt. Khớp nghiệm tham chiếu xác nhận việc tối ưu trên train trong các kiểm tra này, không tự chứng minh khả năng dự đoán ngoài mẫu.

Nguồn: results/q4_summary.json và results/q4_baseline_trace.csv. Có thể chèn figures/q4_baseline_cost.png làm hình minh họa baseline; phân tích hội tụ đầy đủ ở câu 7 sẽ dùng lại cùng lịch sử này.

## 4.8 Chuyển về đơn vị AT gốc

Vì z=(AT-mu)/s, mô hình theo AT gốc có hệ số:

\[
b=\frac{\theta_1}{s},\qquad a=\theta_0-\frac{\theta_1\mu}{s}.
\]

Do đó:

\[
\widehat{PE}\approx496.977983872-2.169684131\,AT.
\]

Đây là cùng mô hình, được biểu diễn trong đơn vị gốc. Không so trực tiếp slope -16,15047 theo z với slope -2,169684 theo AT như hai kết quả mâu thuẫn. Theo đường hồi quy đã học, AT cao hơn 1°C gắn với PE dự đoán thấp hơn khoảng 2,17 MW trong phạm vi dữ liệu. Không diễn giải đây là tác động nhân quả đã được kiểm chứng. AT=0 nằm ngoài khoảng quan sát, nên không xem hệ số chặn là một chế độ nhà máy đã đo.

## 4.9 Các thử nghiệm learning rate phục vụ bàn giao

Các lần chạy dùng cùng X_train, theta ban đầu, tolerance và max_iter. Kết quả:

| Run | Alpha | Số cập nhật | Trạng thái |
|---|---:|---:|---|
| small | 0,01 | 1.984 | converged |
| baseline | 0,1 | 190 | converged |
| large | 1,85 | 123 | converged |
| divergent | 2,05 | 284 | diverged |
| one_step | 1 | 1 | converged |

Trong hồi quy một biến đã chuẩn hóa kèm intercept, H=X.T@X/n xấp xỉ ma trận đơn vị. Vì vậy alpha=1 đạt gần nghiệm sau một bước trong bài toán đặc biệt này. Không áp dụng kết luận này máy móc cho bài toán nhiều biến hoặc dữ liệu chưa chuẩn hóa. Với alpha=1,85, tham số có thể đổi phía quanh nghiệm trong khi chi phí vẫn giảm; đổi phía tham số không có nghĩa chi phí bắt buộc dao động tăng giảm.

Baseline vẫn giữ alpha=0,1 để minh họa quỹ đạo nhiều bước, không dùng điểm số test để chọn learning rate. Các lần chạy được lưu để câu 7 tiếp tục phân tích, không tạo một baseline riêng cho từng câu.

## 4.10 Kiểm chứng và bàn giao

run_q4.py thực hiện 27 kiểm tra, gồm phép tính tay một bước, không làm đổi theta của người gọi, điều kiện dừng, trạng thái phân kỳ, toàn bộ lịch sử, quy tắc cập nhật từng bước, gradient cuối, đối chiếu nghiệm tham chiếu, chuyển hệ số và tính nhất quán của file xuất. Các kiểm tra đều đạt trong môi trường đã chạy.

Các file chính:

- baseline_run.npz: theta_final, theta_history, cost_history, pred_train, pred_test; có thêm source_row và lịch sử thử learning rate để bàn giao.
- baseline_config.json: cấu hình, ý nghĩa tiêu chí dừng, số bước, data_version, dấu vân tay train/test/scaling và xác nhận is_mock_data=false.
- q4_baseline_trace.csv: tham số, chi phí và chuẩn gradient ở từng trạng thái từ bước 0.
- q4_learning_rate_runs.csv và q4_*_history.npz: các thí nghiệm trên train.
- q4_verification.csv và q4_summary.json: bằng chứng kiểm tra và kết quả tóm tắt.

Lịch sử thứ k luôn chứa theta sau k cập nhật và J tương ứng. Không thay tham số trong baseline_run.npz bằng nghiệm giải tích; nghiệm giải tích chỉ nằm trong trường tham chiếu riêng. Câu 6–8 và câu 9 sẽ đọc cùng baseline này. Khi thay dữ liệu hoặc thuật toán phải chạy lại các phần phụ thuộc và cập nhật báo cáo.

Chạy python run_q4.py trong thư mục MAI391_Final. Chương trình dùng dữ liệu chung sẵn có, không gọi bước chuẩn bị để chia lại train/test. Kết quả xác nhận trong câu 4 là độ đúng của thuật toán và phép tối ưu; đánh giá dự đoán test thuộc câu 8.
