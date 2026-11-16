# Demo — máy chủ dịch và đo lượng tử hoá

Đo Q4_K_M / Q5_K_M + LoRA trên RTX 3050, thử 13 trang 1688 thật.

|  |  |
|---|---|
| Notebook | demo cục bộ + Kaggle T4 |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | không lưu |
| Thời gian | 14–15/11/2026 |
| Dùng ở đâu | chọn Q5_K_M cho demo, kiểm 13 trang thật |
| Ghi chú | Q5_K_M đạt cổng; Q4_K_M bị loại. |

## 1. Lượng tử hoá trên RTX 3050

| Mô hình | Sản phẩm | Dòng eval | chrF câu TB zh→vi fp16 | 4-bit | chrF corpus fp16 | 4-bit | Khối lệch dòng | Tỉ lệ rỗng |
|---|---|---|---|---|---|---|---|---|
| q4km-lora-pair-f0 | 150 | 164 | 79.76 | 76.95 | 71.58 | 68.61 | 1 | 0.0061 |
| q5km-lora-pair-f0 | 150 | 164 | 79.76 | 79.62 | 71.58 | 71.15 | 0 | 0.0000 |

## 2. Thử trang 1688 thật

- Số trang: **13**; đạt kiểm cấu trúc: **13/13**.
- Q5_K_M + LoRA đạt cổng (tụt 0,14 điểm); Q4_K_M bị loại (tụt 2,81).

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| demo_do_luong_tu/ket_qua_q5km-lora-pair-f0.json | 4,232 | 759313adf95ee7aa |
| demo_do_luong_tu/ket_qua_q4km-lora-pair-f0.json | 4,249 | 824aaba5338052e0 |
| demo_thu_trang_bo_dem.json | 46,825 | e14c9ed99977f54c |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
