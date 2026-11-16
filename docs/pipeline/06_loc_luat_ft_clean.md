# 06. Lọc luật trước fine-tune (`ft_clean`)

[← 05](05_gop_tiki_vao_1688_official.md) · [Tổng quan](00_TONG_QUAN.md) · [07 →](07_judge_deepseek.md)

Thời gian: 30/10/2026, chạy trên máy. Nguồn:
- `scripts/ft_clean/` (README, `luat_ft.py`, `xay_ft_clean.py`, `xuat_ban_pair.py`);
- `data/processed/ft_clean/pheu_buoc0_4.json`;
- [`ket_qua_kaggle/A5`](../ket_qua_kaggle/A5_loc_luat_cho_fine_tune_tren_may.md);
- WALKTHROUGH Trả lời 603.

## 1. Đầu vào

Bản official v2: `bilingual_official_context.parquet`, 25.956 sản phẩm, HF `4a23574`, SHA256 kiểm theo manifest `8fd10827...`. Bản pair v2 đi kèm, cùng tập và cùng thứ tự `product_id`.

## 2. Phễu (loại theo thứ tự, mỗi sản phẩm chỉ tính lý do đầu tiên) [ĐO]

| Bước | Luật | Loại |
|---|---|---|
| B1a | `title_vi` của 1688 rò lời nhắc hoặc ảo giác. Đã đọc cả 46 title, đều là rò thật, ví dụ "mã sản phẩm: 12345, giá: 250.000 VND" | 46 |
| B1b | Số đoạn thuộc tính hai bản khác nhau (chỉ tách ở "; " khi phần sau có dạng "nhãn:") | 49 (1688 38, Tiki 11) |
| B1c | Còn chữ Hán trong bản Việt 1688 | 7 |
| B2 | Trần 200 dòng cho mỗi khuôn thuộc tính lặp. Chỉ Tiki bị cắt; khuôn lớn nhất 445 dòng | 279 |
| | **Còn lại** | **25.575** (1688 14.837, Tiki 10.738) |

- Căn hàng chắc (số đoạn khớp và số trong từng đoạn khớp): 22.706/25.575 = 88,8%.
- Danh sách 25.575 sản phẩm giữ lại được áp **giống hệt** cho bản pair: `xuat_ban_pair.py` xuất `ft_buoc4_giu_pair.parquet`. Script kiểm cùng tập, cùng thứ tự, title và nguồn dịch title giống hệt.
- Bản cụm và bản pair khác mô tả ở 25.922/25.956 sản phẩm, gồm cả khác dấu nối ("\n" và "; ") lẫn nội dung dịch.

## 3. Lưu ý

- Ở B1b còn có thể có ca bị loại oan khi chuỗi "nhãn:" nằm bên trong một giá trị. Chưa kiểm từng ca.
- Từ bước này mọi số liệu dùng **25.575 sản phẩm**. Bộ sau Judge (file 07) có cùng danh sách này.
