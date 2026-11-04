# Mã lịch sử và công cụ chuyên biệt

- `eda/`: ba script từ base giahuy; có đường dẫn máy cá nhân và tác vụ khi chạy file.
- `infrastructure/`: các controller và công cụ checkpoint cũ còn được module hoặc test sử dụng. Không phải quy trình batch được khuyến nghị hiện tại.
- `release_history/`: script lựa chọn no-floor để truy vết chính sách từng phát hành. Script thực hiện I/O ngay khi import; **không import hoặc chạy thử trên HF thật**.
- `judged_release_history/`: audit/provenance/schema/patch judged theo từng snapshot. Các script sửa tay lịch sử giữ lý do trong code, không được áp lại vào bản khác mà không kiểm chứng.

Các probe và script publish/chỉnh metadata dùng một lần đã được bỏ khỏi cây hiện tại; có thể tra lại tại commit `c35396e` trong lịch sử Git.

Archive là nguồn tham khảo có truy vết. Nó không được tự discovery/chạy trong CI và không có cam kết portable. Dùng entry point ở manage.py cho tác vụ duy trì; xem docs/source_inventory.json để tìm file cũ.
