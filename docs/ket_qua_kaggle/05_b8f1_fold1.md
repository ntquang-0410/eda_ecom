# B8 fold 1 — mốc 1:1 và cụm ~11%

Kiểm tỉ trọng ~11% trên fold 1 (kiểm độc lập đầu tiên).

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 32 |
| Thời gian | 06/11/2026 10:28 → 14:05 |
| Dùng ở đâu | kiểm độc lập fold 1 (luật 3 fold) |
| Ghi chú | Nối cum_f1 từ bước 1.102 của bản 25. |

## 1. Việc đã chạy

### `kaggle_b8f1\phien_1_20261106_102741\bao_cao_b78.json`

- Thời lượng: **13079 giây**; GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`; mô hình: `Qwen/Qwen3-4B, 1cfa9a7208912126459214e8b04321603b3df60c`.
- Cấu hình: lr 0.0001, rank 16, max_len 2048, micro_bs 32, seed 42.

| Việc | Bộ | Fold | Cấu hình | Epoch | Bước | chrF macro | vi→zh | zh→vi | Giây/mẫu | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0 | cum | 0 | lr2e-4_r64 | 2 | 0 | 75.76 | 75.62 | 75.90 | 0.327 | dùng lại adapter b6 (khớp B6) |
| pair_f0 | pair | 0 | lr2e-4_r64 | 2 | 1843 | 75.26 | 74.35 | 76.17 | 0.314 |  |
| cum_f1 | cum | 1 | lr2e-4_r64 | 2 | 1800 | 74.74 | 72.12 | 77.37 | 0.309 |  |
| cum_f1_p11 | cum | 1 | lr2e-4_r64 | 2 | 1800 | 75.14 | 72.05 | 78.24 | 0.309 |  |
| cum_f2 | cum | 2 | lr2e-4_r64 | 2 | 1839 | 74.16 | 71.88 | 76.44 | 0.344 |  |

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_b8f1\phien_1_20261106_102741\bao_cao_b78.json | 93,432 | 4a65cc65113e1e99 |
| kaggle_b8f1\phien_1_20261106_102741\tong_hop\bao_cao_b78.json | 4,637 | 67cdeac7bc3350b8 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
