# Thử tỉ trọng title fold 0 (cụm và pair)

So mốc 1:1 với ~11% và 75% title, cả hai bộ.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 29 |
| Thời gian | 05/11/2026 23:20 → 06/11/2026 05:00 |
| Dùng ở đâu | chọn tỉ trọng title ~11%, bỏ 75% |
| Ghi chú | Phiên 5,67 giờ; kết quả là cơ sở chọn ~11%. |

## 1. Mốc và các mức

### cum

| Mức | % title đặt | % title thực | w_title | Loss kiểm ep2 | chrF | Title | Mô tả | Hỏng |
|---|---|---|---|---|---|---|---|---|
| p11 (~11%) | 10.74 | 10.74 | 0.12029 | 0.349 | 76.14 | 60.36 | 91.92 | False |
| p75 (75%) | 75.00 | 75.00 | 3.03912 | 0.357 | 75.66 | 60.67 | 90.65 | False |

Mốc 1:1: chrF 75.76, title 60.22, mô tả 91.30.

### pair

| Mức | % title đặt | % title thực | w_title | Loss kiểm ep2 | chrF | Title | Mô tả | Hỏng |
|---|---|---|---|---|---|---|---|---|
| p11 (~11%) | 10.86 | 10.86 | 0.11818 | 0.351 | 75.16 | 59.39 | 90.94 | False |
| p75 (75%) | 75.00 | 75.00 | 2.95906 | 0.364 | 74.60 | 59.75 | 89.46 | False |

Mốc 1:1: chrF 75.26, title 60.35, mô tả 90.17.


## 2. Kết luận đã khóa

- Cụm: mức ~11% đạt 3 fold (xem mục 3 của `TONG_HOP_KET_QUA_NHAT_ANH.md`).
- Pair: mức 75% thua; ~11% đạt luật 3 fold riêng.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_tl/phien_1_20261105_231941/tong_hop/bao_cao_ti_le.json | 11,669 | 2d7a74114b6f3df1 |
| kaggle_tl/phien_1_20261105_231941/tong_hop/kiem_dong_moi.json | 43,764 | 62d7c2c742b91e7c |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
