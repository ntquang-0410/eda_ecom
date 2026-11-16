# A4. Bộ song ngữ 4: rà soát theo cụm có ngữ cảnh (1688)

[← Mục lục](INDEX.md) · [← A3](A3_bo_song_ngu_3_ra_soat_theo_cum_co_ngu_canh_tiki.md) · [A5 →](A5_loc_luat_cho_fine_tune_tren_may.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | A4 |
| Giai đoạn | A. Dữ liệu: dịch, lọc, Judge |
| Notebook Kaggle | `trinhnhatanh/song-ngu-4-cum-gemma` (+ `-p1`, `-p2`, `-p3`), `trinhnhatanh/song-ngu-4-cum-qwen`, `trinhnhatanh/dl-group2` (Qwen dự phòng trên RTX) |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 24–26/10/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/song_ngu_4` |

## Mục đích

Như A3 cho 1688: mỗi cụm thuộc tính được Gemma rà soát (Qwen chỉ phủ một phần), luật trượt chặn bản sửa xấu.

## Thẻ chuẩn bị `song_ngu_4/the_chuan_bi.json`

| Trường | Giá trị |
|---|---|
| buoc | song_ngu_4.chuan_bi |
| ngay_chay | 2026-10-24 22:47:28 |
| nguon | `{"bang_cap": {"file": "bang_cap.parquet", "sha256": "ac4c79d8b829011c43ce470089a25625643db6cd41680799a95d879119becb5e"}, "bo_1": {"file": "bo_song_ngu_1_day_du_20261022_2126.parquet", "sha256": "8c9f0458424790ff0b4345ebe5fa6d0271bc75711950d7be70d74f619bd53494"}, "don_vi": {"file": "don_vi_da_chon_20261022_2126.parquet", "sha256": "5a79bb8c8f77f65a3018165542bea736e628031f71ae1cad4547beaf43b298a4"},` |
| max_zh_xem | 80 |
| max_trong_so_phan | 2.000 |
| cap_dung_bo_1 | 403.830 |
| cap_ten_hang_vo_nghia | 6 |
| cap_bo_1_cong_bo | 394.662 |
| cap_bo_1_cat_rieng_dua_vao | 9.162 |
| cap_bo_dai_khong_co_ban | 1.326 |
| cap | 402.498 |
| san_pham | 16.342 |
| cum | 16.346 |
| cum_can_goi | 16.346 |
| san_pham_nhieu_phan | 4 |
| nguon_ten_vao | `{"nhap": 233195, "tu_dien_p6": 167411, "tu_dien_p6b": 1306, "chep_nguyen": 524, "moi": 62}` |
| nguon_gt_vao | `{"nhap": 216354, "quy_tac_co_khong": 55154, "quy_tac_chep_nguyen": 49128, "tu_dien_p7": 48881, "quy_tac_ten_hang": 13872, "moi": 7802, "dai_giu_bo_1": 6037, "quy_tac_khoang": 2566, "quy_tac_ma_that": 2479, "quy_tac_diem_anh": 153, "quy_tac_ngay": 72}` |
| truong_can_xuat | 457.413 |
| ky_tu_trung_can_xem | 2.403.687 |
| san_pham_khong_co_tieu_de | 0 |
| mau_thu | `{"san_pham": 200, "cum": 200, "cum_can_goi": 200, "cap": 4802, "truong_can_xuat": 5400, "theo_danh_muc": {"auto": 23, "bags": 23, "beauty": 22, "electronics": 22, "fashion": 22, "food": 22, "home": 22, "mother_baby": 22, "shoes": 22}, "hat_giong": 20261024}` |

## Tóm tắt `song_ngu_4/tom_tat_cum_20261026_2123.json`

| Trường | Giá trị |
|---|---|
| cap_vao | 402.498 |
| cap_giu | 401.692 |
| cap_bo_khong_co_ban | 806 |
| cach_chon | AB |
| nhat_quan | `{"gia_tri": {"独立站:Trạm độc lập": 7, "镜头:thấu kính": 1, "无印风:Phong cách không nhãn hiệu": 2, "拉链:Ngăn kéo": 1, "厚底:Đế độn mũi": 3, "套头:Xỏ chân": 1, "日式:Phong cách tối giản Nhật": 2, "中筒:Thân trung": 10, "短筒:Thân ngắn": 7}, "gia_tri_bi_luat_chan": 0, "ten": 75}` |
| cum_gemma | `{"ok": 16314, "loi_cau_truc": 31, "chua_dich": 1}` |
| nguon_ten | `{"giu_nhap": 232189, "khoa": 169241, "sua_A": 942, "lui_nhap": 63, "dich_moi_A": 49, "khong_co": 13, "sua_B": 1}` |
| nguon_gia_tri | `{"giu_nhap": 199841, "khoa": 178373, "sua_A": 15665, "dich_moi_A": 7004, "lui_nhap": 801, "khong_co": 797, "sua_B": 17}` |
| nguon_ban_dich | `{"giu_ban_dich_don_vi": 206673, "tu_dien_quy_tac": 172338, "gemma_sua": 15664, "dich_moi": 6999, "qwen_sua": 18}` |
| do_tin_cay_cap | `{"cao": 188362, "vua": 188086, "thap": 25244}` |
| luat_truot_gia_tri_gemma | `{"latin_moi": 702, "mat_viet": 592, "do_dai": 218, "han_sot": 59, "lech_so_muc": 51, "mat_ma": 45, "lech_so": 28, "jin_sai": 25, "jin_can": 5, "loi_thua": 2, "lo_ngu_canh": 2}` |
| luat_truot_ten_gemma | `{"mat_viet": 52, "han_sot": 13, "latin_moi": 13, "do_dai": 11, "mat_ma": 5, "lech_so": 4}` |
| sua_khong_ly_do | 2.907 |
| cum_sua_nhieu | 0 |
| gia_tri_khac_bo_1 | 22.710 |
| ten_khac_bo_1 | 988 |
| cum_qwen | `{"chua_dich": 14613, "ok": 1718, "loi_cau_truc": 15}` |
| qwen_thay_gemma_gia_tri | 17 |
| qwen_thay_gemma_ten | 1 |
| qwen_dich_moi_khong_dung | 12 |
| san_pham | 16.342 |
| san_pham_mat_cap | 736 |
| san_pham_khong_con_cap_nao | 0 |
| do_tin_cay_san_pham | `{"thap": 12020, "vua": 4322}` |
| nhan | 20261026_2123 |
| thu_muc_gemma | data\processed\song_ngu_4\tu_kaggle_that_gemma_gop |
| thu_muc_qwen | data\processed\song_ngu_4\that_qwen_du_phong |
| giay | 146 |

## Ghi chú, giới hạn, sự cố

- `cum_qwen.chua_dich` lớn: Qwen chỉ dịch được một phần cụm trong thời gian có GPU; phần còn lại theo Gemma.
- Sau bước này các bộ được Huy gộp thành bộ official v2 (`bilingual_official_context.parquet`), rồi qua lọc luật (A5) và Judge (A6).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `song_ngu_4/the_chuan_bi.json` | 1.869 | `0930858199ac9130` |
| `song_ngu_4/tom_tat_cum_20261026_2123.json` | 2.132 | `4b19b3f80f38f2e7` |
