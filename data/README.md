# Dữ liệu local

Dữ liệu phát hành nằm trên Hugging Face; GitHub chứa mã xử lý. Các Parquet/CSV đã được theo dõi trong commit giahuy cũ giữ nguyên, không đại diện mặc nhiên cho release mới nhất.

Quy ước mới:

- `data/inputs/`: input local do người chạy chuẩn bị, ghi revision/checksum.
- `data/snapshot/`: snapshot bất biến cho judge/ghép, theo tài liệu từng stage.
- `data/processed/`: intermediate local, có thể chứa nội dung private.
- `outputs/`: báo cáo, notebook sinh, cache và ứng viên phát hành ngoài data.

Không commit file dữ liệu mới, cache SQLite, token hoặc đầu ra API. Không suy đoán file thiếu hoặc copy dữ liệu từ checkpoint cũ ghi đè bản mới. Các bảng judged dùng khóa `(source, product_id)` để phân biệt hai nguồn.
