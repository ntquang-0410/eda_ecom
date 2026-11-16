# B8 fold 2 — cụm ~11%

Kiểm độc lập thứ hai cho tỉ trọng ~11% ở cụm.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 34 |
| Thời gian | 06/11/2026 16:51 → 18:20 |
| Dùng ở đâu | kiểm độc lập fold 2, đủ luật 3 fold cho cụm |
| Ghi chú | Hoàn tất luật 3 fold cho bản cụm. |

## 1. Việc đã chạy

### `kaggle_b8f2\phien_1_20261106_165058\bao_cao_b8f2.json`

- Thời lượng: **5365 giây**; GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`; mô hình: `Qwen/Qwen3-4B, 1cfa9a7208912126459214e8b04321603b3df60c`.
- Cấu hình: lr 0.0001, rank 16, max_len 2048, micro_bs 32, seed 42.

| Việc | Bộ | Fold | Cấu hình | Epoch | Bước | chrF macro | vi→zh | zh→vi | Giây/mẫu | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| cum_f2_p11 | cum | 2 | lr2e-4_r64 | 2 | 1839 | 75.31 | 73.76 | 76.87 | 0.347 |  |

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_b8f2\phien_1_20261106_165058\bao_cao_b8f2.json | 13,366 | 15dd537e3bfacc97 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
