# B7 + B8 — pair fold 0, cụm fold 1

Mốc 1:1 và mức ~11% title cho pair fold 0; mốc cụm fold 1 nối từ bước 1.102.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 25 |
| Thời gian | 05/11/2026 18:13 → 20:43 |
| Dùng ở đâu | mốc pair fold 0 và cụm fold 1 |
| Ghi chú | Dừng ở trần 2,5 giờ; bản 26 nối tiếp bị hủy, không có báo cáo. |

## 1. Việc đã chạy

### `kaggle_b78\phien_1_20261105_181337\bao_cao_b78.json`

- Thời lượng: **9002 giây**; GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`; mô hình: `Qwen/Qwen3-4B, 1cfa9a7208912126459214e8b04321603b3df60c`.
- Cấu hình: lr 0.0001, rank 16, max_len 2048, micro_bs 32, seed 42.

| Việc | Bộ | Fold | Cấu hình | Epoch | Bước | chrF macro | vi→zh | zh→vi | Giây/mẫu | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0 | cum | 0 | lr2e-4_r64 | 2 | 0 | 75.76 | 75.62 | 75.90 | 0.327 | dùng lại adapter b6 (khớp B6) |
| pair_f0 | pair | 0 | lr2e-4_r64 | 2 | 1843 | 75.26 | 74.35 | 76.17 | 0.314 |  |
| cum_f1 | cum | 1 | lr2e-4_r64 | 1 | 1102 | — | — | — | — | chưa chấm |

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_b78\phien_1_20261105_181337\bao_cao_b78.json | 50,284 | 5db0da1ba64ce5fb |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
