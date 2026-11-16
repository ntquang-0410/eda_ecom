# Quota bước 1 — 3 cấu hình × 2 bộ × 2 seed

Kiểm lại sau khi sửa metric: title 10/90, guardrail, 2 seed.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 54 (dry), 55 (chạy thật) |
| Thời gian | 10/11/2026 |
| Dùng ở đâu | chốt bảng natural (2 seed, metric 10/90) |
| Ghi chú | Kết quả dùng để chốt natural; xem docs/QUOTA_KET_LUAN.md. |

## 1. Kết quả từng cấu hình

### `kaggle_quota_step1\quota_report.json`

- cum: bỏ 0 mẫu, dài nhất 2047, natural title 10.79%.
- pair: bỏ 0 mẫu, dài nhất 1907, natural title 10.34%.
| Bộ | Cấu hình | Seed | Macro | Chính 10/90 | Title zh→vi | Mô tả zh→vi | Bước | Lệch ngạch |
|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 72.05 | 86.91 | 54.60 | 90.50 | 241 | 0.0000 |
| cum | natural | 123 | 71.70 | 86.72 | 53.77 | 90.38 | 240 | 0.0001 |
| cum | balanced | 42 | 73.19 | 85.12 | 58.27 | 88.10 | 234 | 0.0000 |
| cum | balanced | 123 | 73.03 | 84.96 | 57.69 | 87.99 | 234 | 0.0000 |
| cum | title_heavy | 42 | 72.36 | 81.64 | 59.22 | 84.13 | 232 | 0.0003 |
| cum | title_heavy | 123 | 72.50 | 82.03 | 59.02 | 84.59 | 233 | 0.0001 |
| pair | natural | 42 | 68.55 | 80.79 | 49.80 | 84.24 | 107 | 0.0001 |
| pair | natural | 123 | 68.49 | 79.16 | 50.16 | 82.38 | 107 | 0.0000 |
| pair | balanced | 42 | 68.75 | 74.74 | 53.46 | 77.10 | 107 | 0.0000 |
| pair | balanced | 123 | 68.45 | 76.14 | 54.53 | 78.54 | 107 | 0.0000 |
| pair | attr_heavy | 42 | 67.66 | 78.18 | 51.58 | 81.14 | 107 | 0.0001 |
| pair | attr_heavy | 123 | 68.71 | 79.54 | 51.19 | 82.69 | 107 | 0.0000 |

## 2. Đọc kết quả

- `natural` là bảng dữ liệu tự nhiên; các mức khác là can thiệp có chủ đích.
- Với bản 960: có lỗi bỏ mẫu dài, **không dùng để chốt**.
- Kết luận cuối ở `docs/QUOTA_KET_LUAN.md`.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_quota_step1\quota_report.json | 90,297 | fd2bbc880ba3051c |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
