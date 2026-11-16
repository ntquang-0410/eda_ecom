# Moses — phrase-based, hai hướng

Baseline cổ điển: Moses với Hán tách từng chữ, hai biến thể (học đủ / khớp 4M token).

|  |  |
|---|---|
| Notebook | các phiên moses |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | không lưu trong file kết quả |
| Thời gian | 10–12/11/2026 |
| Dùng ở đâu | dòng Moses trong bảng năm mô hình |
| Ghi chú | Moses pair mạnh vì ~87% dòng thuộc tính đã có nguyên văn trong train. |

## 1. Hai hướng

| Hướng | Tách | Biến thể | Dòng train | chrF dev | n eval | Giây |
|---|---|---|---|---|---|---|
| zh-vi | char | train | 269895 | 75.14 | 598 | 3598 |
| zh-vi | char | khop4m_s42 | 37528 | 71.82 | 598 | 2974 |
| vi-zh | char | train | 61401 | 71.46 | 600 | 2270 |
| vi-zh | char | khop4m_s42 | 9006 | 66.11 | 600 | 1544 |

## 2. Ghi chú

- Moses pair mạnh vì ~87% dòng thuộc tính eval đã có nguyên văn trong train; lợi thế mất khi dịch cả khối.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_moses_v2_zhvi/moses_report.json | 731 | 0726a08c005577e9 |
| kaggle_moses_v2_vizh/moses_report.json | 727 | ca87c047fa867350 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
