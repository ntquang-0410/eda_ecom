# Dịch dữ liệu song ngữ (song_ngu 1 → 4)

Sinh bản dịch Trung–Việt cho thuộc tính/title bằng Gemma/Qwen + từ điển, chọn bằng luật và Judge.

|  |  |
|---|---|
| Notebook | các phiên Kaggle tài khoản 1 và 2 (Gemma/Qwen) |
| Tài khoản | trinhnhatanh + honghuynanh |
| Bản Kaggle | không lưu trong file kết quả |
| Thời gian | 16–26/10/2026 |
| Dùng ở đâu | đầu vào của mọi bước fine-tune và Judge |
| Ghi chú | Đây là đầu vào của toàn bộ fine-tune; không phải phiên huấn luyện. |

## 1. Quy mô

| Chỉ số | Giá trị |
|---|---|
| Sản phẩm | 16,342 |
| Cụm (block) cần gọi | 16,346 |
| Cặp (cặp tên–giá trị) đưa vào | 402,498 |
| Trường cần xuất | 457,413 |
| Ký tự Trung cần xem | 2,403,687 |

## 2. Kết quả chọn bản dịch (song_ngu_4)

| Chỉ số | Giá trị |
|---|---|
| Cặp giữ | 401,692 |
| Bỏ vì không có bản | 806 |
| Cách chọn | AB |
| Gemma: ok / lỗi cấu trúc / chưa dịch | 16,314 / 31 / 1 |
| Qwen: ok / lỗi cấu trúc / chưa dịch | 1,718 / 15 / 14,613 |

## 3. Nguồn bản dịch cuối

| Nguồn | Số mục |
|---|---|
| giu_ban_dich_don_vi | 206,673 |
| tu_dien_quy_tac | 172,338 |
| gemma_sua | 15,664 |
| dich_moi | 6,999 |
| qwen_sua | 18 |

- Độ tin cậy cấp: cao 188,362, vua 188,086, thap 25,244.
- Bản cuối dùng cho fine-tune là bộ đã qua Judge; xem mục 2 của `TONG_HOP_KET_QUA_NHAT_ANH.md`.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| song_ngu_4/tom_tat_cum_20261026_2123.json | 2,132 | 4b19b3f80f38f2e7 |
| song_ngu_4/the_chuan_bi.json | 1,869 | 0930858199ac9130 |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
