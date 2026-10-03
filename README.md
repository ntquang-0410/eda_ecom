# e-commerce - EDA

Bước EDA và Xử lý dữ liệu của tiểu luận "Tối ưu hoá mô hình dịch máy Việt–Trung
cho TMĐT xuyên biên giới". Dữ liệu crawl từ 1688.com đã xong (xem
`data/snapshot/SNAPSHOT.md`); folder này chỉ phân tích và làm sạch, không crawl.

**Đọc trước:** `CLAUDE.md` (quy tắc làm việc) và `WALKTHROUGH.md` (nhật ký,
các vấn đề của data kèm bằng chứng, giải pháp, việc còn nợ). Mục 0 của
WALKTHROUGH nói rõ đang ở đâu.

## Dựng môi trường (Windows, PowerShell)

```
python -m venv .venv                       # Python 3.13.x
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không cần `.env` hay token. Snapshot có sẵn trong `data/snapshot/`.

## Chạy lại các bước đã có (từ gốc folder)

```
.venv\Scripts\python.exe scripts\eda\attr_split_check.py     # -> data\interim\attr_pairs.parquet
.venv\Scripts\python.exe scripts\eda\title_audit.py          # -> data\eda\title_audit\
.venv\Scripts\python.exe scripts\eda\build_tokens_edited.py  # -> title_latin_tokens_editted.csv
.venv\Scripts\python.exe scripts\eda\data_issues.py          # -> data\eda\data_issues\
```

Bốn bước này chạy lại ra file **giống hệt từng byte** bản gốc (kiểm 01/10/2026).
`scripts\eda\sim_full.py` (similarity toàn bộ, khoảng 20 phút) và
`scripts\notebook\run_nb_harness.py` cần Ollama chạy `bge-m3` ở
`127.0.0.1:11434`; chưa chạy lại sau khi chuyển folder (xem WALKTHROUGH).

## Cấu trúc

```
notebooks/   eda-bilingual_zh_vi.ipynb (+ archive/)
scripts/     eda/ · notebook/ · archive/
src/         textnorm.py (chuẩn hoá đã áp lên dữ liệu crawl)
docs/        PROJECT_CHARTER_team_CLAUDE.md, PROVENANCE.md, copy_manifest.csv, design/, ecom_crawler_README.md
data/        snapshot/ (bất biến) · eda/ (kết quả) · interim/ (cache, sinh lại được) · processed/ (đầu ra bước xử lý, đang rỗng)
```
