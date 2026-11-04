# Làm sạch bộ chính thức v2 cho fine-tune (bước 0-4, 30/10/2026)

Nguồn: `data/snapshot/bilingual_official_v2/bilingual_official_context.parquet` (bản cụm, 25.956 sản phẩm, commit HF `4a23574`,
SHA256 kiểm trong script). Kết quả: `data/processed/ft_clean/` (không commit). Không sửa nội dung, không đụng HF.

```powershell
python -I -X utf8 scripts/ft_clean/xay_ft_clean.py        # chạy thật (vài giây)
python -I -X utf8 scripts/ft_clean/test_ft_clean.py       # 6 bài thử trên dữ liệu tự dựng
```

## Đầu ra
- `ft_candidate_v2.parquet`: cả 25.956 dòng + mọi cờ (`co_*` cứng, `mem_*` mềm), `ly_do_loai_ft`, `giu_ft`, `huong_goc`, `khuon_id`, `nhom_mo_ta_id`.
- `ft_buoc4_giu.parquet`: dòng giữ lại (9 cột gốc + `huong_goc`, `khuon_id`, `nhom_mo_ta_id`, `can_hang_chac`, cờ mềm).
- `ft_buoc4_loai.parquet`: dòng loại + mã lý do (khôi phục được). `pheu_buoc0_4.json`: phễu + tiêu chí.

## Phễu (loại theo thứ tự, lý do đầu tiên)
| Bước | Luật | Loại |
|---|---|---|
| B1a | `ro_loi_nhac_title`: title_vi 1688 rò lời nhắc / ảo giác ("mã sản phẩm: không có, số lượng...", "mã sản phẩm: 12345, giá: 250.000 VND") | 46 (1688) |
| B1b | `so_doan_lech`: số đoạn thuộc tính hai bản khác nhau (tách ở "; " chỉ khi phần sau có dạng "nhãn:"). Có thể oan nếu giá trị chứa "nhãn:" bên trong | 49 (1688 38, Tiki 11) |
| B1c | `han_sot_ban_viet`: chữ Hán còn trong bản Việt 1688, trừ số đăng ký mỹ phẩm | 7 |
| B2 | `vuot_tran_khuon`: trần 200 dòng mỗi khuôn (chuỗi nhãn thuộc tính của bản gốc), thứ tự theo crc32(product_id) | 279 (Tiki) |
| | **Còn lại** | **25.575** (1688 14.837, Tiki 10.738) |

## Cờ mềm (chỉ ghi, không loại; dùng ở bước judge/chọn tập sạch)
`mem_latin_la_ban_trung` 62 (Tiki: từ Latin trong bản Trung không có trong bản Việt, vd. "Stabilizer"), `mem_viet_sot_title_zh` 16,
`mem_v2_review` 24 (danh sách cần xem của Huy).

## B3, B4
- B3 `huong_goc`: 1688 = `zh->vi` (gốc Trung, bản Việt do máy dịch); Tiki = `vi->zh` (gốc Việt, bản Trung do máy dịch). Dùng để huấn luyện theo hướng hoặc gắn thẻ hướng.
- B4 căn hàng: số đoạn khớp ở các dòng giữ lại; `can_hang_chac` (số đoạn khớp VÀ số trong từng đoạn khớp) = 22.706/25.575 = 88,8%. Phần còn lại phần lớn là quy đổi đơn vị hoặc số ở dạng khác; chưa kiểm thứ tự đoạn.
- `nhom_mo_ta_id`: các dòng cùng mô tả (cả hai bản) cùng nhóm; khi chia train/dev/test sau này phải giữ cùng một phía (chưa chia, chờ lệnh).

## Chưa làm
CometKiwi (bước 5, tùy chọn), judge (bước 6), người chấm mù (bước 7), đóng gói `ft_clean_v1` (bước 8).
