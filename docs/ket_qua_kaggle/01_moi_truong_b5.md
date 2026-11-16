# Kiểm môi trường và đo tốc độ (B5)

Chạy khói để kiểm GPU, tokenizer, loss giảm, lưu/nạp adapter và đo token/giây.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | không lưu (bản 1–4) |
| Thời gian | 04–05/11/2026 |
| Dùng ở đâu | chốt cấu hình chạy (micro_bs, ngân sách, tốc độ) |
| Ghi chú | Không dùng làm kết quả; chỉ để chốt cấu hình chạy. |

## 1. Các lần chạy

| Lần | Giây | GPU | VRAM (GB) | Token/s | Mẫu pool | Đạt |
|---|---|---|---|---|---|---|
| b5 | 50 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 95.0 | — | — | False |
| b5_lan2 | 184 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 95.0 | — | 46120 | False |
| b5_lan3 | 126 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 95.0 | — | 46120 | False |
| b5_lan4 | 600 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 95.0 | 3883.0 | 46120 | True |

## 2. Đo tốc độ riêng

| Cấu hình | Giây | Token/s |
|---|---|---|
| — | 0.0 | 3823.5 |
| — | 0.0 | 3753.0 |
| — | 0.0 | 3772.0 |
| — | 0.0 | 5873.0 |
| — | 0.0 | 5767.2 |

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_b5/bao_cao_b5.json | 1,318 | ae5e6e8eccfb3034 |
| kaggle_b5_lan2/bao_cao_b5.json | 4,292 | 2a1cf37603fa080e |
| kaggle_b5_lan3/bao_cao_b5.json | 2,331 | c702669ded87961b |
| kaggle_b5_lan4/bao_cao_b5.json | 14,254 | 5fc702e50006c6e1 |
| kaggle_do_toc_do/bao_cao_b5.json | 1,723 | cb2d49caa125a8a1 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
