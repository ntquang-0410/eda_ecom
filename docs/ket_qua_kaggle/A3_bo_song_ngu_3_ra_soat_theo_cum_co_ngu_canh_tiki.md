# A3. Bộ song ngữ 3: rà soát theo cụm có ngữ cảnh (Tiki)

[← Mục lục](INDEX.md) · [← A2](A2_bo_song_ngu_2_thuoc_tinh_tiki_vi_zh_dich_theo_do.md) · [A4 →](A4_bo_song_ngu_4_ra_soat_theo_cum_co_ngu_canh_1688.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | A3 |
| Giai đoạn | A. Dữ liệu: dịch, lọc, Judge |
| Notebook Kaggle | `trinhnhatanh/song-ngu-3-cum`, `trinhnhatanh/song-ngu-3-cum-qwen` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 23–24/10/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/song_ngu_3` |

## Mục đích

Đưa cả cụm thuộc tính của một sản phẩm (đã dịch theo đơn vị ở A2) cho hai mô hình rà soát có ngữ cảnh: giữ, sửa hoặc dịch mới từng cặp; cặp đã khóa bằng luật không cho sửa.

## Thẻ chuẩn bị `song_ngu_3/the_chuan_bi_v2.json`

| Trường | Giá trị |
|---|---|
| buoc | song_ngu_3.chuan_bi_v2 |
| ngay_chay | 2026-10-23 20:34:22 |
| sha256_bang_cap | 78b5a1249c05584d7a6045b07436a25bd4463d2915a8a61633065d3ca6006be5 |
| bo_2 | bo_song_ngu_2_day_du_20261023_1656.parquet |
| cap | 87.139 |
| cum | 13.208 |
| san_pham | 13.187 |
| ten_khoa | 85.803 |
| ten_nhap | 1.335 |
| ten_dich_moi | 1 |
| gia_tri_khoa | 72.850 |
| gia_tri_nhap | 14.206 |
| gia_tri_dich_moi | 83 |
| gia_tri_khoa_doi_gr | 284 |
| san_pham_khong_co_tieu_de | 0 |
| mau_thu | `{"san_pham": 200, "cum": 201, "cap": 1365, "truong_can_xuat": 256}` |

## Tóm tắt `song_ngu_3/tom_tat_cum_20261024_1616.json`

| Trường | Giá trị |
|---|---|
| cap_tong | 87.139 |
| cap_co_ban_cuoi | 87.112 |
| cap_khong_co_ban_cuoi | 27 |
| nguon_gia_tri | `{"khoa": 72850, "giu_nhap": 12096, "sua_A": 2073, "dich_moi_A": 49, "khong_co": 27, "sua_B": 25, "lui_nhap": 12, "dich_moi_B": 7}` |
| nguon_ten | `{"khoa": 85803, "giu_nhap": 1116, "sua_A": 219, "dich_moi_A": 1}` |
| dong_thuan | `{"khoa": 72850, "ca_hai_giu_nhap": 11848, "chi_A_sua": 1679, "chi_B_sua": 291, "ca_hai_sua_giong": 229, "ca_hai_sua_khac": 214, "chi_mot_mo_hinh_giu_nhap": 28}` |
| sua_khong_ly_do | 361 |
| cum_sua_nhieu | 16 |
| gia_tri_khac_don_vi | 2.465 |
| ten_khac_don_vi | 298 |
| cum_A | `{"ok": 13206, "loi_cau_truc": 2}` |
| cum_B | `{"ok": 13204, "loi_cau_truc": 4}` |
| san_pham | 13.187 |
| san_pham_mat_cap | 25 |
| san_pham_khong_con_cap_nao | 0 |
| do_tin_cay_cap | `{"cao": 76820, "vua": 8967, "thap": 1325}` |
| do_tin_cay_san_pham | `{"cao": 6979, "vua": 4982, "thap": 1226}` |

## Ghi chú, giới hạn, sự cố

- `A` và `B` trong bảng là hai mô hình rà soát theo thứ tự trong mã (A = Gemma, B = Qwen).
- Thư mục còn các lần chấm mù mẫu (`cham_mau_*`, `cham_that_*`); file `khoa_KHONG_XEM.json` là khóa giải mã, không mở khi chấm.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `song_ngu_3/the_chuan_bi_v2.json` | 574 | `26f398a25ca6f533` |
| `song_ngu_3/tom_tat_cum_20261024_1616.json` | 1.014 | `b25bce41986ad75b` |
