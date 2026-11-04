# E-commerce Chinese–Vietnamese Data Pipeline

Mã xử lý của nhánh **`giahuy`** cho dữ liệu thương mại điện tử 1688 và Tiki: dịch title bằng Gemma/Qwen, QA kỹ thuật, so sánh COMETKiwi, ghép bộ song ngữ, judge so sánh và kiểm tra dữ liệu trước fine-tuning.

Đây là hướng xử lý riêng của Gia Huy. Mã trên nhánh `main` về CADO, dữ liệu tổng hợp và thí nghiệm LoRA là một hướng khác; không được nhập đè vào nhánh này.

## Bắt đầu

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python manage.py list
.\.venv\Scripts\python -m pytest -q
```

Lệnh `list` chỉ hiển thị các điểm vào; không tải model, không gọi API và không chạy pipeline. Môi trường model GPU/COMETKiwi được tách khỏi môi trường kiểm tra dữ liệu. Xem [hướng dẫn vận hành](docs/OPERATIONS.md) trước khi chạy một tác vụ.

## Cấu trúc dự án

```text
src/                     QA, cache, dịch title và theo dõi batch dùng chung
scripts/
  translation/           Runner dịch và resume
  notebooks/             Builder notebook và worker Qwen/Kaggle
  comparison/            Chuẩn bị, chấm và lựa chọn COMETKiwi
  audit/                 Audit title và kiểm tra release chỉ đọc
  datasets/              Tạo full dataset và đối chiếu category
  merge/                 Ghép 1688/Tiki theo cấu hình
  judge/                 Judge so cặp, theo dõi chi phí, ghép quyết định
  ft_clean/              Bộ luật và chuẩn bị dữ liệu fine-tune
  monitoring/            Đọc trạng thái và tiến độ an toàn
notebooks/
  eda/                   Notebook EDA cũ đã bỏ outputs
  templates/             Template nguồn để dựng notebook, không có khóa
tests/                   Kiểm thử tự dựng cho title, ghép và audit
docs/                    Kiến trúc, vận hành, nguồn gốc, lịch sử
archive/                 Script lịch sử, probe và xuất bản một lần
data/                    Dữ liệu local; xem data/README.md
outputs/                 Kết quả sinh và checkpoint, không commit
```

Các module phẳng cũ được giữ tên để bảo toàn imports và thuật toán. `project_bootstrap.py` giải quyết đường dẫn sau khi phân nhóm; `manage.py` cung cấp một điểm vào chung. Danh sách chuyển file và SHA-256 nằm trong [source inventory](docs/source_inventory.json).

## Các giai đoạn

1. **Dịch title:** 1688 Trung→Việt; Tiki Việt→Trung. Dùng cache/checkpoint riêng cho từng model và nguồn.
2. **QA và repair:** phát hiện chữ còn sót, số liệu, đầu ra trống/cắt cụt/lặp; repair có giới hạn và giữ dấu vết bản gốc.
3. **COMETKiwi:** chấm hai ứng viên trên cùng nguồn; giữ cả điểm, ứng viên và lý do chọn.
4. **Ghép dữ liệu:** ghép title/description theo ID, áp luật và ghi sản phẩm loại riêng. Loại do trần nhóm hoặc trùng description không đồng nghĩa dịch sai.
5. **Judge:** so sánh title và description theo pipeline cụm/pair, checkpoint quyết định và ngân sách; ghép lựa chọn theo luật đã định.
6. **Audit release:** kiểm tra cấu trúc, ID, chữ/số và các dấu hiệu bất thường; giữ hàng cần xem thêm thay vì tuyên bố đã kiểm chứng ngữ nghĩa.

Xem [kiến trúc và ranh giới từng giai đoạn](docs/ARCHITECTURE.md).

## Chính sách lựa chọn và giới hạn

- Script COMET `compare_1688_cometkiwi.py` giữ chính sách **strict lịch sử**: QA trước, floor 0.72, có thể HOLD. Đây không phải chính sách cuối đã áp dụng cho tất cả title.
- Chính sách xuất title được người dùng chọn sau đó là **điểm cao hơn, không floor, QA mang tính cảnh báo**. Script xuất lịch sử tương ứng ở `archive/release_history/export_comet_best_without_floor.py`; cần đúng đầu vào và trạng thái generation của lượt cũ, không chạy tùy tiện.
- Bộ judged đã qua luật và lựa chọn; `judge_chon_title` biểu diễn model của bản nháp title cuối đang giữ, không luôn là lựa chọn trực tiếp của judge. Sửa tay phải tra ở dấu vết riêng.
- `judge_chon_description` tổng hợp nguồn các mảnh; pair đếm riêng key và value. Từ điển/luật/nguồn chưa rõ không được gán thành Gemma/Qwen.
- COMETKiwi, QA pass hoặc judge không chứng minh mọi câu đúng nghĩa. Các lượt rà ngữ nghĩa lịch sử có giới hạn phạm vi; không công bố “zero defects” như kết luận tổng quát.

## Dữ liệu và bảo mật

Dataset được phát hành riêng tại [Hugging Face](https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/tree/main/data/LLM). Lần tổ chức code này **không thay đổi dữ liệu trên HF**, không chạy dịch/judge mới và không thay đổi nhánh `main`.

Không commit token, API key, private notebook đã nhúng khóa, SQLite, ZIP, weights hoặc output chứa dữ liệu. Khóa đi qua biến môi trường hoặc Kaggle Secrets. Các file Parquet/CSV EDA đã được theo dõi từ lịch sử Git vẫn được giữ; dữ liệu mới không thêm vào Git. Xem [quy ước dữ liệu](data/README.md) và [quy trình đóng góp](CONTRIBUTING.md).

## Tài liệu

- [Vận hành và các lệnh](docs/OPERATIONS.md)
- [Kiến trúc hệ thống](docs/ARCHITECTURE.md)
- [Nguồn gốc mã và phạm vi chuyển đổi](docs/PROVENANCE.md)
- [Đọc các tài liệu lịch sử đúng ngữ cảnh](docs/history/README.md)
- [Thay đổi của bản tổ chức này](CHANGELOG.md)

Các notebook/script lịch sử cần private inputs/checkpoints gắn với từng lượt. Chỉ có code không đủ tái tạo mọi kết quả đã phát hành; không giả lập hoặc thay thế dữ liệu còn thiếu.
