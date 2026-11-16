# Quota screen 2.048 (đã sửa lỗi độ dài)

5 cấu hình × 2 bộ, cap 2.048, loss chuẩn hoá theo token đích, 1 seed.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 53 |
| Thời gian | 10/11/2026 |
| Dùng ở đâu | bản sửa lỗi độ dài, 1 seed |
| Ghi chú | Bản sửa đầu tiên; phát hiện metric 'Chính' sai trọng số ở pair, đã sửa ở bước sau. |

## 1. Kết quả từng cấu hình

### `kaggle_quota_screen_v2\quota_report.json`

- cum: bỏ 0 mẫu, dài nhất 2047, natural title 10.79%.
- pair: bỏ 0 mẫu, dài nhất 1907, natural title 10.34%.
| Bộ | Cấu hình | Seed | Macro | Chính 10/90 | Title zh→vi | Mô tả zh→vi | Bước | Lệch ngạch |
|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 72.12 | — | — | — | 241 | 0.0000 |
| cum | balanced | 42 | 72.99 | — | — | — | 234 | 0.0000 |
| cum | title_heavy | 42 | 72.35 | — | — | — | 232 | 0.0003 |
| cum | desc_heavy | 42 | 71.36 | — | — | — | 238 | 0.0001 |
| cum | tiki_heavy | 42 | 72.55 | — | — | — | 233 | 0.0001 |
| pair | natural | 42 | 68.48 | — | — | — | 107 | 0.0001 |
| pair | balanced | 42 | 68.82 | — | — | — | 107 | 0.0000 |
| pair | title_heavy | 42 | 67.84 | — | — | — | 108 | 0.0000 |
| pair | attr_heavy | 42 | 68.84 | — | — | — | 107 | 0.0001 |
| pair | tiki_heavy | 42 | 66.84 | — | — | — | 107 | 0.0000 |

- Bản này **chưa lưu điểm tách title/mô tả**, nên không tính được cột Chính 10/90; chỉ có macro.

## 2. Đọc kết quả

- `natural` là bảng dữ liệu tự nhiên; các mức khác là can thiệp có chủ đích.
- Với bản 960: có lỗi bỏ mẫu dài, **không dùng để chốt**.
- Kết luận cuối ở `docs/QUOTA_KET_LUAN.md`.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_quota_screen_v2\quota_report.json | 27,742 | 6be5661410912e2a |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
