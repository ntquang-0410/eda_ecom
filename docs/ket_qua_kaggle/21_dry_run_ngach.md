# Dry run ngạch và chọn mẫu

Dựng ngạch, kiểm tỉ lệ và độ dài trước khi train.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 52, 54 (và các lần dry 8B/Ecom) |
| Thời gian | 09–10/11/2026 |
| Dùng ở đâu | chặn lỗi độ dài/ngạch trước khi chạy GPU |
| Ghi chú | 0 GPU cho phần lớn; dùng để chặn lỗi trước khi chạy thật. |

## 1. Các lần dry

| Bộ | Mẫu | Bỏ | Dài nhất | Ngân sách token đích | Số job | Job lỗi |
|---|---|---|---|---|---|---|
| cum | 37123 | 0 | 2047 | 1725636 | 10 | 0 |
| pair | 331296 | 0 | 1907 | 817840 | 10 | 0 |

- Dry run dùng để chặn lỗi độ dài/ngạch trước khi tốn GPU; các bộ 4B/8B/Ecom dùng chung một bảng ngạch nên số giống nhau.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_quota_dry\quota_dry.json | 12,648 | 719fea593aea3905 |
| kaggle_quota_dry2\quota_dry.json | 15,031 | d6dbb58e7138aaf4 |
| kaggle_dry_chon\quota_dry.json | 12,743 | f1a635f165260112 |
| kaggle_dry_mh\nlp-dry-8b\quota_dry.json | 6,809 | c4cd8b3f4c05e0dd |
| kaggle_dry_mh\nlp-dry-ecom\quota_dry.json | 7,392 | dd7a676d54d9b28a |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
