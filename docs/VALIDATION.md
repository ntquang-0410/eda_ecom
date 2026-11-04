# Kiểm tra bản tổ chức mã — 04/11/2026

Mốc mã được kiểm tra: `c4864fc` trên nhánh `giahuy`, base trước khi tổ chức là `862c1a4b3483c47722c940a895ef2dd82e4c0d7a`.

## Kết quả

- **226 tests passed, 66 subtests passed** trong môi trường xử lý cũ.
- Lặp lại cùng bộ test trong virtual environment mới, Python **3.12.14**, chỉ cài `requirements.txt`: **226 passed, 66 subtests passed**.
- Kiểm tra offline 183 file văn bản: compile Python, mẫu credential phổ biến, notebook outputs và published checksum của **164 file nguồn được chuyển**: không có lỗi.
- `git diff --cached --check`: không có lỗi khoảng trắng trước commit mã.
- Dựng thành công tại local: sampled Qwen v6 notebook/bundle, Kaggle Qwen notebook, hai notebook QA repair và notebook scorer COMET1688 sử dụng Kaggle Secrets. Chỉ build source, không chạy cell/model/API.
- Điểm vào `manage.py list` và chuyển tiếp `--help` được kiểm thử; các chương trình dataset một lần không thực hiện I/O khi import.

## Phạm vi của kết quả

Đây là kiểm tra code và hồi quy bằng dữ liệu tự dựng, fake judge/client, temporary SQLite/Parquet. Không xác nhận lại chất lượng dịch của dataset thật, không chạy lại job Kaggle và không thử tích hợp API trả phí. CI Ubuntu/Python 3.11 được cấu hình nhưng chưa có kết quả remote tại thời điểm viết.

Quét theo mẫu credential không phải chứng minh tuyệt đối không có mọi dạng thông tin nhạy cảm. Các file khóa, private output notebooks, checkpoints và input riêng đã được loại khỏi danh sách chuyển nguồn; chỉ source-only templates được giữ.

## Push

Commit đã tạo local. Lần push thử chưa thành công: thông tin đăng nhập dùng trước đây bị GitHub từ chối xác thực; tài khoản cached khác không có quyền ghi repository. Cần đăng nhập tài khoản được cấp quyền `ntquang-0410/eda_ecom` trước khi push; không thay đổi remote bằng URL nhúng token và không force-push.
