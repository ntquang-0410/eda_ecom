# eda_ecom — nhánh `nhatanh_updating`

Phần việc EDA của Nhật Anh cho tiểu luận "Tối ưu hoá mô hình dịch máy
Việt–Trung cho TMĐT xuyên biên giới" (data crawl từ 1688.com, repo crawler:
[`ecom_crawler`](https://github.com/ntquang-0410/ecom_crawler)).

**Đọc [WALKTHROUGH.md](WALKTHROUGH.md) trước** — đó là tài liệu chính: cấu
trúc, luồng chạy, các vấn đề tìm được trong data (mục 5) kèm bằng chứng, đề
xuất giải pháp (mục 6), và nhật ký từng buổi làm việc (mục 7).

## Nội dung nhánh này

- `eda-bilingual_zh_vi.ipynb` — notebook EDA + similarity BAAI/bge-m3
  (mục 5 của notebook: negative baseline, ngưỡng theo category, cấp thuộc tính).
- `data/eda/title_similarity_full.csv`, `title_similarity_summary.json` —
  similarity đầy đủ 16.348 dòng.
- `data/eda/title_audit/` — audit tiếng Anh/pinyin trong `title_vi` + các lỗi
  khác (dịch sát chữ, lặp từ, mất số...), xem WALKTHROUGH mục 4.3–4.4.
- `data/eda/data_issues/` — tổng hợp mọi vấn đề của data (D00–D17), xếp mức
  độ, kèm bằng chứng theo `product_id`. Xem WALKTHROUGH mục 5–6.

**Nhánh này không chứa `bilingual_zh_vi.parquet`** (đã có ở nhánh `main`,
~25MB) — muốn chạy lại notebook thì lấy file đó từ nhánh `main` hoặc từ HF
Hub (`data/snapshot/bilingual_zh_vi.parquet`, xem README của `ecom_crawler`).

Script tạo ra các file CSV audit hiện **chưa được commit** (đang hard-code
đường dẫn máy Nhật Anh) — nhắn Nhật Anh nếu cần chạy lại hoặc mở rộng.
