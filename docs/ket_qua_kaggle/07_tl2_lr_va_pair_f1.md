# Kiểm lr 1e-4 (cụm) và pair fold 1

Kiểm lr 1e-4 ở mức ~11% và mốc pair fold 1.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 36 |
| Thời gian | 06/11/2026 21:00 → 07/11/2026 01:10 |
| Dùng ở đâu | loại lr 1e-4; mốc pair fold 1 |
| Ghi chú | Lr 1e-4 không đạt; giữ 2e-4. |

## 1. Việc đã chạy

### `kaggle_tl2\phien_1_20261106_210034\bao_cao_tl2.json`

- Thời lượng: **14962 giây**; GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`; mô hình: `Qwen/Qwen3-4B, 1cfa9a7208912126459214e8b04321603b3df60c`.
- Cấu hình: lr 0.0001, rank 16, max_len 2048, micro_bs 32, seed 42.

| Việc | Bộ | Fold | Cấu hình | Epoch | Bước | chrF macro | vi→zh | zh→vi | Giây/mẫu | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0_p11_lr1e4 | cum | 0 | lr2e-4_r64 (lr 0.0001) | 2 | 1837 | 75.97 | 75.97 | 75.97 | 0.329 |  |
| pair_f1 | pair | 1 | lr2e-4_r64 | 2 | 1778 | 74.80 | 71.96 | 77.64 | 0.296 |  |
| pair_f1_p11 | pair | 1 | lr2e-4_r64 | 2 | 1778 | 74.93 | 71.73 | 78.13 | 0.297 |  |

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_tl2\phien_1_20261106_210034\bao_cao_tl2.json | 37,102 | 3aeb0ceceed84efe |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
