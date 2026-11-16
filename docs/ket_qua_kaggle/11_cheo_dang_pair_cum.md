# Thử chéo dạng pair ↔ cụm

Đo 600 sản phẩm fold 0, mô tả zh→vi, đổi dạng đầu vào và adapter.

|  |  |
|---|---|
| Notebook | nlp-group3 (các bản cheo) |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | không lưu trong file kết quả |
| Thời gian | 08–09/11/2026 |
| Dùng ở đâu | chọn dạng cho extension (pair hay cụm) |
| Ghi chú | Dùng adapter fold 0 cum_f0_p11 / pair_f0_p11. |

## 1. Bảng đo (600 sản phẩm fold 0, mô tả zh→vi)

| Adapter | Dạng vào | n | chrF câu TB | Đúng số mảnh | Giữ số | Rò lời dẫn | Rỗng |
|---|---|---|---|---|---|---|---|
| cum_f0_p11 | cum | 600 | 93.69 | 0.9983 | 0.9839 | 0.0 | 0.0 |
| cum_f0_p11 | pair | 600 | 92.72 | 0.9767 | 0.9805 | 0.0 | 0.0 |
| pair_f0_p11 | cum | 600 | 93.21 | 0.9967 | 0.9842 | 0.0 | 0.0 |
| pair_f0_p11 | pair | 600 | 93.73 | 1.0000 | 0.9842 | 0.0 | 0.0 |
| zero_shot | cum | 600 | 56.01 | 0.9433 | 0.9721 | 0.0 | 0.0 |
| zero_shot | pair | 600 | 55.88 | 0.1017 | 0.9636 | 0.0 | 0.0 |

## 2. Kết luận

- Dùng đúng adapter cho đúng dạng thì hai dạng tương đương (+0,04 [−0,16; +0,23]).
- Đưa sai dạng làm giảm 0,47–1,01 điểm.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| cheo_ket_qua/ket_qua.json | 5,051 | b347edea4843012d |
| cheo_ket_qua/bang.md | 2,713 | d6304eb3e426dd9d |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
