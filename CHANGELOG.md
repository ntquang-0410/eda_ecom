# Changelog

## 2026-11-04 — Tổ chức mã xử lý nhánh giahuy

- Hợp nhất mã title 1688/Tiki, COMETKiwi, QA repair, judge, ghép và hậu kiểm từ các workspace đã dùng.
- Chia mã core, scripts theo chức năng, notebook template, tests, docs và archive.
- Thêm manage.py, môi trường test khai báo rõ, CI kiểm tra offline và source inventory/checksum.
- Sửa đường dẫn source/template sau khi di chuyển; sinh notebook/bundle vào outputs/generated.
- Không thêm API key, checkpoint, private generated notebook hoặc dataset mới vào commit.
- Bổ sung local read-only audit hai bộ judged; không tuyên bố đã rà ngữ nghĩa toàn bộ.
- Giữ nguyên các chính sách strict/no-floor lịch sử dưới tên/phạm vi riêng, không thay dữ liệu HF.
