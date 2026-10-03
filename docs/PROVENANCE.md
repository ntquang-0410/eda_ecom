# PROVENANCE: file nào đến từ đâu (chuyển folder ngày 01/10/2026)

Folder này tách ra từ `D:\nam4_ky1\e-commerce - crawl\ecom_crawler` khi dự án
sang bước EDA + Xử lý dữ liệu. Danh sách đầy đủ từng file, nguồn, SHA256 nguồn,
và trạng thái: [`copy_manifest.csv`](copy_manifest.csv) (41 file: 33 giống hệt
byte, 7 script chỉ sửa đường dẫn, và `WALKTHROUGH.md` đã thêm mục 0 + nhật ký
01/10 sau lúc copy). `CLAUDE.md` không nằm trong sổ vì là file viết lại: mục 1-4
giống hệt byte `ecom_crawler/CLAUDE.md`, mục 5 viết mới.

## Gốc của từng nhóm

| Nhóm | Nguồn | Ghi chú |
|---|---|---|
| `notebooks/eda-bilingual_zh_vi.ipynb` | notebook của Quang, Nhật Anh thêm vào `ecom_crawler` rồi chỉnh sửa; ngày 28/09 Claude viết lại mục 5 và 7 → 28 cell | Các cell còn lại giữ nguyên byte so với bản 19 cell ngay trước khi Claude sửa (bản đó đã gồm chỉnh sửa của Nhật Anh nếu có) |
| `notebooks/archive/…BACKUP.ipynb` | bản 19 cell ngay trước khi Claude sửa 28/09 | Không có trong git nào |
| `scripts/eda/`, `scripts/notebook/` | thư mục tạm của phiên Claude 28/09 | 7 file đã sửa đường dẫn (xem `copy_manifest.csv`); cùng logic, đọc `data/snapshot/` thay vì tải HF |
| `scripts/archive/` | như trên | `patch_nb.py` và `new_cells.py` chỉ chạy được trên bản 19 cell (có `assert`), `title_latin_explore.py` sinh `bi.parquet` cũ. **Giữ làm lịch sử, chưa sửa đường dẫn** |
| `src/textnorm.py` | `ecom_crawler/core/textnorm.py` | Chuẩn hoá đã áp lên mọi dòng trước khi ghi (NFC, đổi chữ/số toàn chiều, bỏ HTML và ký tự vô hình). Bước xử lý phải biết để không làm lại hoặc làm khác. Chỉ dùng stdlib |
| `data/snapshot/` | `ecom_crawler/data/snapshot_tmp/` | Xem `data/snapshot/SNAPSHOT.md` |
| `data/eda/` | `ecom_crawler/data/eda/` | Giữ nguyên đường dẫn con vì WALKTHROUGH dẫn theo. 8 file trong `title_audit/` và `data_issues/` đã chạy lại ở đây: giống hệt bản gốc |
| `docs/ecom_crawler_README.md` | `ecom_crawler/README.md` | Nguyên văn: schema một dòng, mục 9 (snapshot), mục 10 (việc còn lại của bước này) |
| `docs/PROJECT_CHARTER_team_CLAUDE.md` | `CLAUDE.md` gốc của nhóm, Nhật Anh đính kèm (dạng tài liệu) vào phiên Claude 15/09/2026 08:32 UTC; lấy lại từ transcript của phiên (dòng 158) ngày 01/10 | **Nguyên văn** (10.577 ký tự, SHA256 của văn bản gốc `2E1E227976CAFC90CEE2CDE1A6968EACE3995C46238AD21BDA3C539A214F944A`), chỉ thêm khung chú thích ở đầu: mục nào còn hiệu lực, mục nào lỗi thời. Đây là nơi duy nhất có mục tiêu đề tài, mô hình nền, 8 giai đoạn, quy tắc tập test, các bẫy dữ liệu |
| `docs/design/*.html` | thư mục tạm của phiên Claude 26 và 28/09 | Thiết kế pipeline làm sạch + EDA 1688-Tiki. Số liệu trong đó chưa kiểm lại, đối chiếu WALKTHROUGH mục 0 |

## Tra cứu "tại sao" khi cần

Transcript gốc của phiên Claude làm việc từ 15/09 đến 01/10/2026 (JSONL, 49 MB):
`C:\Users\ADMIN\.claude\projects\d--nam4-ky1-AI-4-IOT-D4\dee1c3c9-eb48-4d7f-8a4a-9dcbb11cc25f.jsonl`.
Dùng để tra lý do của một quyết định cũ (tìm theo từ khoá), không nạp cả file.
**Không copy vào folder này** vì có thể chứa nội dung nhạy cảm (token, tài khoản), và
Claude Code có thể dọn file này về sau; nếu cần giữ lâu thì tự sao lưu ở nơi riêng tư.

## Cố ý KHÔNG chuyển

- `data/raw/` (155 MB): nguồn thật của crawl nhưng README cấm dùng để phân tích;
  có bản mirror trên HF. Chỉ có dữ liệu của worker_nhat_anh.
- Mã crawler (`core/`, `parsers/`, `scripts/`, `main.py`, `config.py`), `firebase/`,
  `.env*`, `*.bat`, `.pw_profile*` (cookie đăng nhập 1688), `.venv` cũ.
- Cache sinh lại được: `bi.parquet`, `attr_pairs.parquet` (nay tự sinh vào
  `data/interim/`), `review_*.tsv`, log chạy thử.
- Các `eda_*.csv` trong thư mục tạm: là kết quả chạy thử 150 dòng, **trùng tên**
  với bản đầy đủ trong `data/eda/full_run/`, chép sang sẽ nhầm.
