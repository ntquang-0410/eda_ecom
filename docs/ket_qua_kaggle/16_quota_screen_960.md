# Quota screen 960 (bản cũ, có lỗi bỏ mẫu)

5 bảng trọng số × 2 bộ, rank 16, max_len 960, 1.200 bước.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 47 (pair), 48 (cụm), 46 (smoke) |
| Thời gian | 09/11/2026 |
| Dùng ở đâu | bản thử đầu, bị lỗi bỏ mẫu — chỉ để đối chiếu |
| Ghi chú | Có lỗi: trainer bỏ mẫu dài >960 trong khi ngạch coi như cắt. Không dùng để chốt. |

## 1. Kết quả từng cấu hình

### `kaggle_quota_screen\quota_report.json`

| Bộ | Cấu hình | Seed | Macro | Chính 10/90 | Title zh→vi | Mô tả zh→vi | Bước | Lệch ngạch |
|---|---|---|---|---|---|---|---|---|
| pair | natural | 42 | 69.35 | — | — | — | 1200 | — |
| pair | balanced | 42 | 70.25 | — | — | — | 1200 | — |
| pair | title_heavy | 42 | 71.51 | — | — | — | 1200 | — |
| pair | attr_heavy | 42 | 67.65 | — | — | — | 1200 | — |
| pair | tiki_heavy | 42 | 70.21 | — | — | — | 1200 | — |

- 5 job lỗi (bỏ khỏi bảng): cum/natural: ValueError: id_tham_chieu_trung; cum/balanced: ValueError: id_tham_chieu_trung; cum/title_heavy: ValueError: id_tham_chieu_trung; cum/desc_heavy: ValueError: id_tham_chieu_trung; cum/tiki_heavy: ValueError: id_tham_chieu_trung.

- Bản này **chưa lưu điểm tách title/mô tả**, nên không tính được cột Chính 10/90; chỉ có macro.

### `kaggle_quota_screen\version48\quota_report.json`

| Bộ | Cấu hình | Seed | Macro | Chính 10/90 | Title zh→vi | Mô tả zh→vi | Bước | Lệch ngạch |
|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 74.06 | — | — | — | 1200 | — |
| cum | balanced | 42 | 72.55 | — | — | — | 1200 | — |
| cum | title_heavy | 42 | 70.07 | — | — | — | 1200 | — |
| cum | desc_heavy | 42 | 73.33 | — | — | — | 1200 | — |
| cum | tiki_heavy | 42 | 71.49 | — | — | — | 1200 | — |

- Bản này **chưa lưu điểm tách title/mô tả**, nên không tính được cột Chính 10/90; chỉ có macro.

### `kaggle_quota_smoke\quota_report.json`

| Bộ | Cấu hình | Seed | Macro | Chính 10/90 | Title zh→vi | Mô tả zh→vi | Bước | Lệch ngạch |
|---|---|---|---|---|---|---|---|---|
| cum | balanced | 42 | 64.41 | — | — | — | 100 | — |
| pair | natural | 42 | 62.01 | — | — | — | 100 | — |

- Bản này **chưa lưu điểm tách title/mô tả**, nên không tính được cột Chính 10/90; chỉ có macro.

## 2. Đọc kết quả

- `natural` là bảng dữ liệu tự nhiên; các mức khác là can thiệp có chủ đích.
- Với bản 960: có lỗi bỏ mẫu dài, **không dùng để chốt**.
- Kết luận cuối ở `docs/QUOTA_KET_LUAN.md`.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_quota_screen\quota_report.json | 26,959 | 3175932e09aa6803 |
| kaggle_quota_screen\version47\quota_report.json | 26,959 | 3175932e09aa6803 |
| kaggle_quota_screen\version48\quota_report.json | 21,714 | 3adef3453f7b2d2e |
| kaggle_quota_smoke\quota_report.json | 5,638 | a70d680880b96ea6 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
