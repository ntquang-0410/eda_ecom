# A5. Lọc luật cho fine-tune (trên máy)

[← Mục lục](INDEX.md) · [← A4](A4_bo_song_ngu_4_ra_soat_theo_cum_co_ngu_canh_1688.md) · [A6 →](A6_judge_deepseek_chon_ban_dich_gemma_hay_qwen_cum.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | A5 |
| Giai đoạn | A. Dữ liệu: dịch, lọc, Judge |
| Notebook Kaggle | – (`scripts/ft_clean/xay_ft_clean.py`) |
| Tài khoản | – (chạy trên máy) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 30/10/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_clean` |

## Mục đích

Loại khỏi bộ official v2 (bản cụm, 25.956 sản phẩm, HF commit `4a23574`) các dòng chắc chắn hỏng theo luật cứng trước khi đưa vào Judge và fine-tune; cờ mềm chỉ ghi lại, không loại.

## Phễu lọc (`ft_clean/pheu_buoc0_4.json`)

Vào: **25.956** sản phẩm (`bilingual_official_context.parquet`, sha256 `8fd10827c9e21ab2…`). Mỗi dòng chỉ bị loại theo lý do đầu tiên nó dính; không sửa nội dung.

| Thứ tự | Lý do loại | Số loại | Theo nguồn |
|---|---|---|---|
| 1 | ro_loi_nhac_title | 46 | `{"1688": 46}` |
| 2 | so_doan_lech | 49 | `{"1688": 38, "tiki": 11}` |
| 3 | han_sot_ban_viet | 7 | `{"1688": 7}` |
| 4 | vuot_tran_khuon | 279 | `{"tiki": 279}` |

| Mục | Giá trị |
|---|---|
| Còn lại | 25.575 |
| Theo nguồn | `{"1688": 14837, "tiki": 10738}` |
| Theo hướng | `{"zh->vi": 14837, "vi->zh": 10738}` |
| Theo nhóm hàng | `{"home": 3416, "mother_baby": 3414, "beauty": 3303, "electronics": 3133, "food": 3084, "bags": 2857, "auto": 2750, "shoes": 2299, "fashion": 1319}` |
| Căn hàng chắc (số đoạn và số khớp) | 22.706 (88.8%) |
| Cờ mềm (chỉ ghi, không loại) | `{"mem_latin_la_ban_trung": 62, "mem_viet_sot_title_zh": 16, "mem_v2_review": 24}` |
| Nhóm mô tả trùng | `{"so_nhom_co_tu_2_dong": 643, "so_dong_thuoc_nhom_trung": 1629}` |
| Trần mỗi khuôn thuộc tính | 200 |

## Kết luận của phiên

- Còn **25.575** sản phẩm (1688 14.837, Tiki 10.738); đây là tập sản phẩm của cả hai bộ cụm và pair sau Judge.

## Ghi chú, giới hạn, sự cố

- Luật `so_doan_lech` có thể loại oan khi giá trị chứa chuỗi giống "nhãn:" bên trong.
- Mã có 6 bài thử trên dữ liệu tự dựng (`scripts/ft_clean/test_ft_clean.py`).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_clean/pheu_buoc0_4.json` | 1.524 | `02f73fbd05df5b1d` |
