# C0 — chấm adapter official trên dev

Chấm cum_f0_p11 / pair_f0_p11 trên đúng bộ eval của screen.

|  |  |
|---|---|
| Notebook | nlp-group3 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | 49 (lỗi adapter), 50 (cụm), 51 (pair) |
| Thời gian | 09/11/2026 |
| Dùng ở đâu | so candidate với adapter đang dùng |
| Ghi chú | Dùng để so candidate với bản đang dùng. |

## 1. Điểm C0 trên dev

| File | Bộ | Adapter | n | chrF | vi→zh | zh→vi |
|---|---|---|---|---|---|---|
| c0_report.json | pair | pair_f0_p11 | 1198 | 74.55 | 75.65 | 73.45 |
| c0_report_v50_cum.json | cum | cum_f0_p11 | 1200 | 76.49 | 75.46 | 77.52 |

- C0 cụm 76,49 (1.200 mẫu) và pair 74,55 (1.198 mẫu), chấm trên đúng eval của screen.
- Bản 49 lỗi đường dẫn adapter (`khong_thay_adapter`), đã sửa ở bản 50–51.
- Dùng để so với các bảng quota; không phải điểm test.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_c0\c0_report.json | 3,220 | cc997fe1a308cbee |
| kaggle_c0\c0_report_v50_cum.json | 4,180 | 727b425f2c333928 |
| kaggle_c0\c0_report_v51_pair.json | 3,220 | cc997fe1a308cbee |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
