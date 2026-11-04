# Quy trình thay đổi

1. Làm trên nhánh riêng hoặc giahuy theo phạm vi đã thống nhất; không sửa nhánh main ngoài yêu cầu.
2. Tách thay đổi cấu trúc mã khỏi thay đổi prompt/model/QA/policy. Luật mới cần tests và ghi rõ thay đổi hành vi.
3. Chạy `python -m pytest -q` và `python scripts/maintenance/check_repository.py` trước commit. Test không cần khóa hoặc gọi model.
4. Không dùng `git add .` trên workspace chứa dữ liệu/API keys. Xem diff và danh sách staged, quét thông tin xác thực trước push.
5. Giữ nguyên bản nguồn và lý do trước/sau khi sửa dữ liệu. Signal kỹ thuật không được chuyển thành nhãn lỗi xác nhận nếu chưa đủ căn cứ.
6. Publish HF, chạy API trả phí hoặc khởi động GPU là tác vụ riêng; không xảy ra trong CI hoặc chỉ vì import module.

Source layout hiện còn các module tên cũ và compatibility bootstrap để tránh thay đổi tính toán. Khi refactor thành package cần giữ regression tests và khả năng đọc đúng checkpoint tương ứng.
