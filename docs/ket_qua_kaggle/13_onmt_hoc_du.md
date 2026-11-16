# OpenNMT — học từ đầu, đủ dữ liệu

Baseline cổ điển: Transformer nhỏ của OpenNMT, huấn luyện trên dòng của bản pair.

|  |  |
|---|---|
| Notebook | các phiên onmt |
| Tài khoản | trinhnhatanh |
| Bản Kaggle | không lưu trong file kết quả |
| Thời gian | 10–11/11/2026 |
| Dùng ở đâu | dòng OpenNMT trong bảng năm mô hình |
| Ghi chú | Dùng để so trong bảng năm mô hình. |

## 1. Điểm trên eval fold 0

| Nhóm | chrF |
|---|---|
| zh->vi/title | 48.27 |
| zh->vi/mo_ta | 77.28 |
| chinh_w0.10 | 74.38 |
| vi->zh/title | 29.64 |
| vi->zh/mo_ta | 64.38 |
| macro | 54.89 |

## 2. Huấn luyện

| Hướng | Dòng train | Token | Bước | Giây train | Giây tổng |
|---|---|---|---|---|---|
| zh-vi | 269895 | 3778854 | 22740 | 2263 | 2280 |
| vi-zh | 61401 | 934205 | 5620 | 1617 | 1629 |

- OpenNMT yếu ở vi→zh vì chỉ có khoảng 61 nghìn dòng hướng này (token ít hơn zh→vi khoảng 4 lần).

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| kaggle_onmt_that/cham_onmt.json | 6,946 | ed018d51fee4cc87 |
| kaggle_onmt_that/onmt_report.json | 6,205 | ad760c20bec8cf84 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
