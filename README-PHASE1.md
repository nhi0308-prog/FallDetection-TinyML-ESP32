# Phase 1 — Fall Detection TinyML ESP32

## 1. Mục tiêu

Thiết lập Python 3.12 x64, kiểm tra TensorFlow trên CPU và rà soát hai ZIP KFall. Hai script chạy độc lập. Script KFall không cần import TensorFlow.

Repo gốc ở commit `8a96bc6` đã có hai notebook và bản xuất INT8 nhị phân 50 Hz. Bộ Phase 1 có thể dùng như thư mục riêng hoặc chép các file bổ sung vào repo đã clone. Nếu repo trên máy bạn đã có `.gitignore`, hãy gộp nội dung. Mô hình nhị phân hiện tại dự đoán hai lớp. Mục tiêu bốn lớp Idle, Walking, Active, Fall sẽ được chuẩn bị ở các giai đoạn tiếp theo.

## 2. Cấu trúc và mã nguồn đầy đủ

| Đường dẫn | Chức năng |
|---|---|
| `requirements-phase1.txt` | Phiên bản thư viện cho Python 3.12 x64 |
| `scripts/check_environment.py` | Import các thư viện, chạy một lượt Conv1D trên CPU và ghi phiên bản đã cài |
| `scripts/check_kfall.py` | Kiểm tra từng CSV, workbook, mã trial, chất lượng mẫu và liên kết nhãn té ngã |
| `data/raw/kfall/` | Nơi đặt hai ZIP gốc do bạn đã tải |
| `.gitignore` | Loại môi trường ảo, dữ liệu cục bộ và kết quả chạy khỏi Git |
| `results/phase1/` | Các báo cáo được ghi khi chạy script |

## 3. Mở và chạy trong VS Code trên Windows

Giải nén bộ Phase 1. Mở thư mục chứa `requirements-phase1.txt` bằng **File → Open Folder**. Nếu đã clone repo, có thể chép bộ Phase 1 vào thư mục repo và mở repo.

Chép `sensor_data_new_2.zip` và `label_data_new_2.zip` vào `data/raw/kfall/`. Giữ nguyên hai file ZIP. Theo điều kiện của KFall, quyền tải được cấp qua nhóm tác giả và không chuyển bộ dữ liệu cho bên thứ ba.

Mở **Terminal → New Terminal**, chọn PowerShell và chạy:

```powershell
py -0p
py -3.12 -c "import platform, struct; print(platform.python_version(), struct.calcsize('P') * 8)"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe scripts/check_kfall.py
```

Trong VS Code, cài extension **Python** của Microsoft nếu chưa có. Nhấn **Ctrl+Shift+P → Python: Select Interpreter** và chọn `.venv\Scripts\python.exe`. Các lệnh trên gọi thẳng Python trong môi trường ảo nên không cần chạy script kích hoạt.

Nếu đặt ZIP ở vị trí khác, truyền đường dẫn trực tiếp:

```powershell
.\.venv\Scripts\python.exe scripts/check_kfall.py --sensor "D:\KFall\sensor_data_new_2.zip" --labels "D:\KFall\label_data_new_2.zip"
```

## 4. Cách đọc kết quả

- `results/phase1/environment.json` ghi phiên bản thực tế và kết quả Conv1D `PASS`, kích thước đầu ra `[1, 98, 1]`.
- `results/phase1/kfall_summary.json` ghi số trial, số subject, thống kê theo task, sampling interval và tổng lỗi.
- `results/phase1/kfall_issues.csv` ghi vị trí cần xem lại. Mã thoát 1 báo có lỗi audit, mã 2 báo không mở được đầu vào hoặc tham số sai.

KFall thu ở 100 Hz. Gia tốc dùng g, tốc độ góc dùng độ/giây. Cột Euler chỉ được kiểm tra khi đọc CSV. Đầu vào IMU dự kiến gồm sáu cột `AccX, AccY, AccZ, GyrX, GyrY, GyrZ`. Hai mốc onset/impact được tra theo giá trị `FrameCounter` trong đúng trial. Task ID 20–34 là Fall theo bảng nhãn XLSX. Task ID 35 và 36 thuộc nhóm ADL trong ZIP đã kiểm tra.

## 5. Xử lý lỗi

| Hiện tượng | Cách kiểm tra |
|---|---|
| `py -3.12` không tìm thấy Python | Cài [Python 3.12 x64](https://www.python.org/downloads/windows/), mở lại VS Code và chạy `py -0p`. |
| VS Code báo thiếu module | Chọn đúng interpreter `.venv\Scripts\python.exe`, rồi chạy lại lệnh cài dependencies bằng chính interpreter đó. |
| TensorFlow báo thiếu DLL khi import | Kiểm tra Python x64 và Microsoft Visual C++ Redistributable theo [hướng dẫn TensorFlow cho Windows](https://www.tensorflow.org/install/pip#windows-native). |
| Không mở được ZIP | Kiểm tra tên và vị trí hai ZIP hoặc dùng `--sensor` và `--labels`. |
| Audit báo `unresolved_fall_frames` | Kiểm tra trial ghi trong CSV báo cáo và đối chiếu onset/impact với `FrameCounter` của trial đó. |

## 6. Tiêu chí hoàn thành

TensorFlow in `PASS`. Audit đọc được 5.075 CSV, 32 workbook, liên kết đủ 2.346 nhãn Fall và ghi `Errors: 0`. Xem chi tiết các cảnh báo trước khi bắt đầu Phase 2. Phase 1 kết thúc ở bước kiểm tra, trước khi tạo cửa sổ và huấn luyện.

Tài liệu tham khảo gồm [Trang chính thức KFall](https://sites.google.com/view/kfalldataset/home), [Yu, Jang và Xiong (2021)](https://doi.org/10.3389/fnagi.2021.692865), [repository](https://github.com/nhi0308-prog/FallDetection-TinyML-ESP32).

## 7. Kết quả chạy với hai ZIP bạn gửi

## Experiment results

Large experiment outputs and model checkpoints are not stored in this repository.

Verified summary metrics are available in `kfall-summary-verified.json`.

Run the training and evaluation scripts to reproduce the results.

- Python 3.12.14 và TensorFlow 2.20.0 trên Linux x64 chạy Conv1D trên CPU thành công. Các gói trực tiếp trong requirements có wheel cho Windows x64 và Python 3.12. Việc chạy thực tế trên máy Windows của bạn được kiểm tra bằng lệnh ở mục 3.
- Audit đọc đủ 5.075/5.075 CSV từ 32 subject, gồm 2.729 ADL và 2.346 Fall. Cả 2.346 nhãn Fall khớp được với hai mốc frame. Trung vị bước lấy mẫu là 10 ms.
- Audit có 0 lỗi và 2 cảnh báo `trial_id_above_six`. Hai trial là `S11T27R07` và `S14T22R07`, đều có workbook và frame hợp lệ. Giữ cả hai trong dữ liệu.
- Tệp `kfall-summary-verified.json` là bản tổng hợp từ lần kiểm tra này. Hai ZIP dữ liệu gốc được đặt riêng theo mục 3.
