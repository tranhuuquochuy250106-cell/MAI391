# MAI391 — Notebook nộp bài

## Cách chạy trên một máy Windows khác

Giữ nguyên cấu trúc thư mục khi giải nén. Cần cài Python 3.10 trở lên và kết nối Internet để cài thư viện.

1. Giải nén toàn bộ thư mục `MAI391_Code`.
2. Nhấp đúp `run_all.bat`.
3. Lần đầu, chương trình tạo môi trường Python riêng `.venv`, cài các thư viện từ `requirements.txt`, rồi chạy lần lượt câu 1–10. Chờ tới khi hiện `All questions completed successfully` và nhấn phím bất kỳ để đóng cửa sổ.

Không cần cài hoặc chọn Jupyter kernel để chạy các file Python. Nếu chỉ muốn chạy một câu sau khi đã cài đặt xong, mở terminal tại thư mục giải nén và dùng `.\.venv\Scripts\python.exe run_qN.py` (thay N bằng số câu). Các câu 2–10 cần kết quả của các câu trước, nên nên chạy đủ thứ tự từ câu 1.

Nếu muốn chạy notebook trong VS Code/Jupyter: chọn Python interpreter `.venv\Scripts\python.exe` bên trong thư mục đã giải nén rồi chọn **Run All**. Không chọn kernel Remote/SSH/WSL/Container nếu dữ liệu đang ở máy Windows cục bộ.

## Chạy toàn bộ câu 1–10

Có thể chạy bằng `run_all.bat` hoặc từng file Python. Nếu tự chạy thủ công, mở terminal tại thư mục `MAI391_Code`, cài dependencies một lần:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_q1.py
.\.venv\Scripts\python.exe run_q2.py
.\.venv\Scripts\python.exe run_q3.py
.\.venv\Scripts\python.exe run_q4.py
.\.venv\Scripts\python.exe run_q5.py
.\.venv\Scripts\python.exe run_q6.py
.\.venv\Scripts\python.exe run_q7.py
.\.venv\Scripts\python.exe run_q8.py
.\.venv\Scripts\python.exe run_q9.py
.\.venv\Scripts\python.exe run_q10.py
```

Mỗi script `run_qN.py` chạy riêng câu N; các câu 2–10 cần kết quả đầu vào của những câu trước, vì vậy hãy chạy theo thứ tự. Giữ nguyên các thư mục `data/`, `src/`, `results/` và `figures/`. Không đưa `.venv/` vào file ZIP; máy nhận sẽ tự tạo môi trường riêng lần đầu chạy `run_all.bat`.

Notebook đã lưu sẵn kết quả chạy: câu 5–10 hiển thị bảng số liệu, chỉ số đánh giá và biểu đồ trực tiếp dưới từng ô code. Chạy **Run All** sẽ cập nhật lại các kết quả này. Các bước xử lý cũng tạo dữ liệu dẫn xuất trong `data/processed/`, bảng chi tiết và log kiểm chứng trong `results/`, hình trong `figures/`.

## Nội dung

- `MAI391_Code.ipynb`: mã nguồn đầy đủ, hướng dẫn và kết quả kiểm tra theo từng câu 1–10.
- `run_all.bat`: thiết lập môi trường Python riêng và chạy tuần tự câu 1–10 trên Windows.
- `run_q1.py`–`run_q10.py`: script chạy theo từng câu; `src/` chứa các module được những script này sử dụng.
- `requirements.txt`: danh sách duy nhất các thư viện Python cần thiết.
- Nguồn dữ liệu: Combined Cycle Power Plant, UCI Machine Learning Repository, DOI [10.24432/C5002N](https://doi.org/10.24432/C5002N).

Khi lưu/nộp, giữ nguyên notebook, mã nguồn, workbook gốc và các kết quả được tạo sau khi chạy **Run All**.
