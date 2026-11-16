# Pair fold 2 — mốc 1:1 và ~11%

Hoàn tất luật 3 fold riêng cho pair.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 38 |
| Thời gian | 07/11/2026 02:31 → 05:17 |
| Dùng ở đâu | đủ luật 3 fold cho pair |
| Ghi chú | Bản 37 là khói. |

## 1. Việc đã chạy

### `kaggle_tl3\phien_1_20261107_073538\bao_cao_tl3.json`

- Thời lượng: **9964 giây**; GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`; mô hình: `Qwen/Qwen3-4B, 1cfa9a7208912126459214e8b04321603b3df60c`.
- Cấu hình: lr 0.0001, rank 16, max_len 2048, micro_bs 32, seed 42.

| Việc | Bộ | Fold | Cấu hình | Epoch | Bước | chrF macro | vi→zh | zh→vi | Giây/mẫu | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| pair_f2 | pair | 2 | lr2e-4_r64 | 2 | 1859 | 73.70 | 71.16 | 76.24 | 0.374 |  |
| pair_f2_p11 | pair | 2 | lr2e-4_r64 | 2 | 1859 | 74.66 | 72.08 | 77.25 | 0.356 |  |

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_tl3\phien_1_20261107_073538\bao_cao_tl3.json | 25,098 | d2064b08c30f13f2 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
