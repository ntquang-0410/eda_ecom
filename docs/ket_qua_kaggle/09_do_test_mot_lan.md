# Đo test khóa MỘT lần

Zero-shot và adapter fold 0 trên toàn bộ test 2.558 sản phẩm (5.116 mục mỗi nhánh).

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 41 |
| Thời gian | 07/11/2026 08:06 → 10:39 |
| Dùng ở đâu | điểm test duy nhất của bản fold 0 |
| Ghi chú | Không huấn luyện. Bản 39–40 là khói. Không đo lại sau lần này. |

## 1. Kết quả trên test khóa (5.116 mục mỗi nhánh)

| Bộ | Hướng | Mục | Zero-shot | Fine-tune | Chênh [95%] |
|---|---|---|---|---|---|
| cum | vi->zh | 2146 | 41.49 | 75.63 | +34.14 [33.30; 34.95] |
| cum | zh->vi | 2970 | 47.22 | 77.36 | +30.14 [29.66; 30.63] |
| pair | vi->zh | 2146 | 43.92 | 75.41 | +31.50 [30.73; 32.24] |
| pair | zh->vi | 2970 | 46.34 | 77.65 | +31.31 [30.81; 31.83] |

## 2. Ghi chú

- Đo **một lần** theo luật khóa trước: thắng khi cận dưới 95% > 0.
- Sau lần này không đo lại test; mọi so sánh sau đó chỉ trên dev.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_do_test/phien_1_20261107_080604/tong_hop/tong_hop_do_test.json | 6,295 | e409c30b7a96719f |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
