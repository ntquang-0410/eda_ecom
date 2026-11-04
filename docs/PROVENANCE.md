# Nguồn gốc mã và phạm vi thay đổi

## Nguồn đã hợp nhất

1. Base `giahuy`: commit `862c1a4b3483c47722c940a895ef2dd82e4c0d7a`. Ba script EDA và notebook được chuyển vào archive/eda và notebooks/eda; README/walkthrough cũ giữ ở docs/history/eda.
2. Workspace title: mã dịch, repair, COMETKiwi, giám sát, dataset assembly, kiểm thử và các lượt upload trước đây của người dùng.
3. Gói judge bàn giao của nhóm: scripts judge/ft_clean từ snapshot `diagnostics3` đã thực chạy, có tích hợp provider; tài liệu/PowerShell companion từ gói handoff. Đây không phải toàn bộ mã đều do một thành viên viết; giữ chú thích và credit có sẵn.
4. Workspace hậu kiểm judged: các script audit nhiều vòng, provenance, schema chín cột và xuất bản lịch sử. Được lưu để truy vết, không coi là chương trình portable sẵn chạy.
5. Gói ghép Tiki–1688: script, quy tắc JSON và tests bàn giao. Giữ nguyên luật gói; không thay thế chúng bằng các luật mới chỉ vì tổ chức repository.

`source_inventory.json` ghi source group, tên file gốc, đường dẫn mới, SHA-256 gốc và SHA-256 sau điều chỉnh. Nó là bản đồ chuyển nguồn, không phải manifest dữ liệu hoặc xác nhận các đầu vào private đã có trên Git.

## Thay đổi kỹ thuật trong lần tổ chức này

- Phân nhóm nguồn và lịch sử; thêm điểm vào `manage.py`, import bootstrap tương thích và source/template lookup cho notebook builders.
- Sửa đường dẫn đọc source/checksum sau khi chuyển file; ghi notebook/ZIP sinh ra vào outputs/generated.
- Giữ checkpoint logic, prompt, model và luật QA cũ; SHA code mới không tương thích tự động với worker lock lịch sử.
- Bỏ outputs/execution counts của notebook template; không copy notebook scorer private có nhúng khóa.
- Lưu template judge batch/resume đã thay khóa/endpoint tài khoản bằng Kaggle Secrets; không phát hành input riêng của lượt chạy.
- Từ chối nhúng khóa trong builder COMET được công bố; thay bằng Kaggle Secrets.
- Chỉnh các test từng phụ thuộc dữ liệu thật trên máy để dùng fixture nhỏ tự dựng.
- Thêm local release audit read-only có kiểm tra checksum, schema và alignment, cùng tests.

## Không thực hiện

Không chạy API/model, không huấn luyện lại, không đánh giá lại dataset thật, không sửa dữ liệu trên HF, không viết lại lịch sử Git, không sửa nhánh main, không xóa workspace đang chứa checkpoint. Không chuyển private outputs/ZIP/khóa vào GitHub.

Các lượt publish có đường dẫn local và thao tác remote nằm trong archive, không được chạy lại chỉ để thử import. Tài liệu lịch sử giữ số liệu tại thời điểm viết; không suy ra trạng thái dataset hiện tại từ các số đó.
