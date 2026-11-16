# B6 — dò siêu tham số (fold 0, bản cụm)

Quét lr × rank, chọn cấu hình cho mọi bước sau.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 20, 21, 22 (3 phiên nối tiếp) |
| Thời gian | 05/11/2026 09:58 → 16:10 |
| Dùng ở đâu | chọn lr 2e-4, rank 64 cho mọi bước sau |
| Ghi chú | Phiên 1–2 dừng ở trần 2,5 giờ; phiên 3 chạy nốt. |

## 1. Cấu hình từng phần

| Phần | Giây | Bước | lr | rank | Đạt | Số cấu hình |
|---|---|---|---|---|---|---|
| phiên 1 (09:58) | 9003 | 100 | 0.0001 | 16 | False | 7 |
| phiên 2 (12:33) | 9002 | 100 | 0.0001 | 16 | False | 7 |
| phiên 3 (15:07) | 3778 | 100 | 0.0001 | 16 | True | 7 |
| nối phần 1 | 164 | 100 | 0.0001 | 16 | False | 4 |
| nối phần 2 (xong) | 273 | 100 | 0.0001 | 16 | True | 4 |

## 2. Loss kiểm theo cấu hình (ghép 3 phiên)

| Cấu hình | Loss kiểm ep1 | Loss kiểm ep2 |
|---|---|---|
| lr1e-4_r16 | 0.3900 | — |
| lr1e-4_r16_attn | 0.4435 | — |
| lr1e-4_r64 | 0.3680 | 0.3587 |
| lr2e-4_r16 | 0.3770 | — |
| lr2e-4_r64 | 0.3703 | 0.3512 |
| lr5e-5_r16 | 0.4090 | — |
| lr5e-5_r64 | 0.3808 | — |

## 3. Điểm hai cấu hình được chấm

| Cấu hình | chrF macro | Title (TB 2 hướng) | Mô tả (TB 2 hướng) | Giây/mẫu |
|---|---|---|---|---|
| lr1e-4_r64 | 75.51 | 60.21 | 90.81 | 0.372 |
| lr2e-4_r64 | 75.76 | 60.22 | 91.30 | 0.327 |

- Chọn **lr 2e-4, rank 64**: loss thấp nhất; chênh chrF so 1e-4 là +0,33 [−0,12; +0,81], hòa.
- LoRA chỉ attention kém hơn all-linear: −2,20 [−2,75; −1,65].

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_b6/phien_3_20261105_150627/bao_cao_b6.json | 10,872 | c3157cfe47db05fe |
| kaggle_b6/phien_1_20261105_095809/bao_cao_b6.json | 3,481 | 285ce79a430465b8 |
| kaggle_b6/phien_2_20261105_123231/bao_cao_b6.json | 4,843 | 605015eeb00e5a3d |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
