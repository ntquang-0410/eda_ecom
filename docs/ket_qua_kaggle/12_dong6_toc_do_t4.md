# Đo tốc độ trên T4 fp16 (đóng gói máy chủ)

Đo thời gian nạp và sinh của adapter trên Tesla T4 để tính công suất máy chủ demo.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | không lưu trong file kết quả |
| Thời gian | 12/11/2026 |
| Dùng ở đâu | tính công suất máy chủ demo trên T4 |
| Ghi chú | Chỉ đo tốc độ, không đo chất lượng. |

## 1. Tốc độ trên T4 fp16

### `kaggle_dong6\bao_cao_dong6.json`

- GPU: `Tesla T4`; VRAM 14.6 GB; dtype `fp16`; tổng 5758 giây.

| Bộ | Adapter | Mục eval | Lượt gọi | Giây nạp | Giây sinh | Giây/lượt |
|---|---|---|---|---|---|---|
| cum | cum_f0_p11 | 1200 | 1200 | 1756 | 2039 | 1.70 |
| pair | pair_f0_p11 | 1198 | 1165 | 9 | 1938 | 1.66 |

### `kaggle_dong6_khoi\bao_cao_dong6.json`

- GPU: `Tesla T4`; VRAM 14.6 GB; dtype `bf16`; tổng 971 giây.

| Bộ | Adapter | Mục eval | Lượt gọi | Giây nạp | Giây sinh | Giây/lượt |
|---|---|---|---|---|---|---|
| cum | cum_f0_p11 | 40 | 40 | 120 | 461 | 11.51 |
| pair | pair_f0_p11 | 40 | 40 | 9 | 375 | 9.37 |

### `kaggle_dong6_khoi_fp16\bao_cao_dong6.json`

- GPU: `Tesla T4`; VRAM 14.6 GB; dtype `fp16`; tổng 2467 giây.

| Bộ | Adapter | Mục eval | Lượt gọi | Giây nạp | Giây sinh | Giây/lượt |
|---|---|---|---|---|---|---|
| cum | cum_f0_p11 | 40 | 40 | 2278 | 186 | 4.65 |
| pair | — | — | — | — | — | — |

## 2. Dùng để làm gì

- Tính công suất máy chủ demo trên T4 (1,66–1,70 giây/lượt).
- Không đo chất lượng.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_dong6\bao_cao_dong6.json | 7,040 | 1a8ca526c9816e6a |
| kaggle_dong6_khoi\bao_cao_dong6.json | 6,648 | 6713338807fc5d5e |
| kaggle_dong6_khoi_fp16\bao_cao_dong6.json | 3,631 | d4012b409450293c |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
