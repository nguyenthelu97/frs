Dự án: Face recognition local (employee)

Cài đặt:
- Tạo virtualenv, cài requirements.txt:
  pip install -r requirements.txt

Các script:
- python register.py     # Thêm nhân viên mới (nhập id, tên), chụp webcam và lưu encoding
- python recognize.py    # Chạy nhận dạng realtime, lưu history vào data/history.json
- python export_history.py  # Xuất file excel từ history

Dữ liệu:
- data/employees.json
- data/history.json

Gợi ý: cần camera và thư viện face_recognition (dlib). Nếu cài gặp lỗi, xem tài liệu face_recognition/dlib.
