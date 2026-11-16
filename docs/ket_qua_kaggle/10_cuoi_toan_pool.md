# Bản cuối trên toàn pool (cum_cuoi, pair_cuoi)

Huấn luyện 2 epoch trên 23.017 sản phẩm, loại 6.386 title bị cờ.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 43 |
| Thời gian | 07/11/2026 10:55 → 14:07 |
| Dùng ở đâu | adapter giao cho demo, không có điểm |
| Ghi chú | Không còn fold giữ lại nên không có điểm chất lượng; không đo test. |

## 1. Việc đã chạy

### `kaggle_cuoi\phien_1_20261107_105449\bao_cao_cuoi.json`

- Thời lượng: **11519 giây**; GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`; mô hình: `Qwen/Qwen3-4B, 1cfa9a7208912126459214e8b04321603b3df60c`.
- Cấu hình: lr 0.0001, rank 16, max_len 2048, micro_bs 32, seed 42.

| Việc | Bộ | Fold | Cấu hình | Epoch | Bước | chrF macro | vi→zh | zh→vi | Giây/mẫu | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| cum_cuoi | cum | — | lr2e-4_r64 | 2 | 1990 | — | — | — | — | chưa chấm |
| pair_cuoi | pair | — | lr2e-4_r64 | 2 | 2008 | — | — | — | — | chưa chấm |

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_cuoi\phien_1_20261107_105449\bao_cao_cuoi.json | 150,253 | 935608d6631f6ff8 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
