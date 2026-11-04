# Mã lịch sử và công cụ chuyên biệt

- `eda/`: ba script từ base giahuy; có đường dẫn máy cá nhân và tác vụ khi chạy file.
- `infrastructure/`: các controller interactive, boot probes, phiên bản môi trường hoặc công cụ chuyển checkpoint đã dùng. Không phải quy trình batch được khuyến nghị hiện tại.
- `release_history/`: các lần tổ chức/publish HF, chỉnh metadata hoặc chọn no-floor từng chạy. Nhiều file thực hiện I/O ngay khi import và chứa đường dẫn dữ liệu riêng; **không import để khám phá và không chạy thử trên HF thật**.
- `judged_release_history/`: audit/provenance/schema/patch judged theo từng snapshot. Các script sửa tay lịch sử giữ lý do trong code, không được áp lại vào bản khác mà không kiểm chứng.

Archive là nguồn tham khảo có truy vết. Nó không được tự discovery/chạy trong CI và không có cam kết portable. Dùng entry point ở manage.py cho tác vụ duy trì; xem docs/source_inventory.json để tìm file cũ.
