# A2. Bộ song ngữ 2: thuộc tính Tiki (vi→zh) dịch theo đơn vị

[← Mục lục](INDEX.md) · [← A1](A1_bo_song_ngu_1_thuoc_tinh_1688_dich_theo_don_vi_g.md) · [A3 →](A3_bo_song_ngu_3_ra_soat_theo_cum_co_ngu_canh_tiki.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | A2 |
| Giai đoạn | A. Dữ liệu: dịch, lọc, Judge |
| Notebook Kaggle | các notebook dịch và chấm của tài khoản 1 |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 23/10/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/song_ngu_2` |

## Mục đích

Như A1 nhưng cho sản phẩm Tiki: gốc tiếng Việt, dịch sang tiếng Trung.

## Thẻ nguồn gốc `song_ngu_2/the_nguon_goc_20261023_1656.json`

Bước: `song_ngu_2.hoan_tat (5, 6b, 6c, 7)`, ngày chạy 2026-10-23 16:56:06. Mô hình theo thẻ: `{"A": "api_tuy_chon", "B": "qwen3_14b"}`; ngưỡng chọn: `{"tau": 60.0, "kappa": 0.4, "kappa_ngan": 0.3, "kappa_cat": 0.3, "kappa_cat_ngan": 0.22, "delta": 0.25, "uu_tien": "A", "chi_doi_khi_truot": true, "do_dai_ngan": 3, "do_dai_nguon_ngan": 12, "q_do_dai": [0.0, 0.999], "ghi_chu": "chọn từ phân bố điểm thật của bộ Tiki, không có nhãn đúng sai; Kiwi chỉ quyết định độ tin cậy và việc cất riêng khi cả hai bản dưới sàn; bản B chỉ dùng khi bản A trượt luật; so hai bản bằng chuan_hoa_zh"}`.

Chuẩn bị (bước 1–3):

| Trường | Giá trị |
|---|---|
| buoc | song_ngu_2.chuan_bi (1-3) |
| ngay_chay | 2026-10-23 09:15:06 |
| file_nguon | tiki_eda.parquet |
| sha256_file_nguon | 7af2a2620b8ca2def3b80762cb8f99196bfe7c64baeaeaa8b99d974df570d3bd |
| so_san_pham | 13.200 |
| so_cap | 89.300 |
| so_cap_goc_truoc_lam_sach | 89.325 |
| dau_ngan_gia_tri | ,  |
| split_nguon_san_pham | `{"train": 10566, "test": 1328, "dev": 1306}` |
| trang_thai_vao | `{"dung": 87267, "van_loai": 2033}` |
| ly_do_van_loai | `{"to_chuc": 1952, "lien_he": 81}` |
| dinh_tuyen_cap_dung | `{"mo_hinh": 66152, "ten_hang": 13126, "do_luong": 4418, "chep_nguyen": 2125, "ma_model": 1307, "co_chu": 139}` |
| don_vi | `{"ten": 830, "gia_tri": 12688, "ky_tu_viet_gia_tri": 1040898}` |

Các lượt dịch / chấm (thẻ `kaggle` trong file; `api_tuy_chon` = Gemma gọi qua API, `qwen3_14b` = Qwen3-14B-AWQ chạy vLLM trên Kaggle, `kiwi` = CometKiwi chấm trên Kaggle):

| Thẻ | Mô hình | Repo / bản dịch của | Backend / GPU | Bắt đầu | Kết thúc | Số đơn vị | Thời lượng |
|---|---|---|---|---|---|---|---|
| the_kaggle | – | – | 2 GPU | 2026-10-23 04:26:56 | – | – | – |
| the_kiwi_api_tuy_chon | Unbabel/wmt22-cometkiwi-da | api_tuy_chon | – | 2026-10-23 04:20:57 | 2026-10-23 04:25:30 | 13.518 | 274 giây (0.08 giờ) |
| the_kiwi_qwen3_14b | Unbabel/wmt22-cometkiwi-da | qwen3_14b | – | 2026-10-23 04:20:56 | 2026-10-23 04:24:54 | 13.518 | 238 giây (0.07 giờ) |
| the_qwen3_14b | qwen3_14b | Qwen/Qwen3-14B-AWQ | vllm | 2026-10-23 03:52:32 | 2026-10-23 04:19:41 | 13.518 | 1.629 giây (0.45 giờ) |
| the_api_tuy_chon | api_tuy_chon | gemma-4-26b-a4b-it | api | 2026-10-23 02:35:55 | 2026-10-23 03:17:00 | 13.518 | 2.465 giây (0.68 giờ) |

Kết quả chọn theo `don_vi`:

| Trường | Giá trị |
|---|---|
| tong | 13.518 |
| giu | 13.439 |
| cach_chon | `{"khac_chu_cung_nghia_A": 5660, "giong_het": 3879, "gan_giong_A": 2214, "uu_tien_duoi_san_A": 448, "tu_dien_p7": 384, "quy_tac_thoi_han": 219, "chi_B": 212, "tu_dien_p6": 175, "chi_A": 108, "quy_tac_khoi_luong": 72, "quy_tac_co_khong": 68}` |
| ly_do_cat | `{"khac_nhieu_diem_khong_du": 37, "ca_hai_truot": 37, "gan_giong_diem_thap": 4, "mot_ban_qua_luat_diem_thap": 1}` |

Kết quả chọn theo `cap`:

| Trường | Giá trị |
|---|---|
| giu | 87.056 |
| cat_rieng | 2.244 |
| do_tin_cay | `{"cao": 77018, "vua": 9181, "thap": 857}` |
| cach_chon | `{"tu_dien_p7": 31928, "quy_tac_co_khong": 13729, "quy_tac_ten_hang": 13099, "khac_chu_cung_nghia_A": 6386, "quy_tac_thoi_han": 5993, "giong_het": 4516, "quy_tac_do_luong": 4417, "gan_giong_A": 2508, "quy_tac_chep_nguyen": 2125, "quy_tac_ma_model": 1307, "uu_tien_duoi_san_A": 465, "chi_B": 218, "quy_tac_co_chu": 139, "quy_tac_khoi_luong": 113, "chi_A": 113}` |

File ra: `{"day_du": "bo_song_ngu_2_day_du_20261023_1656.parquet", "cong_bo": "bo_song_ngu_2_cong_bo_20261023_1656.parquet", "cat_rieng": "cat_rieng_20261023_1656.parquet"}`

## Kết luận của phiên

- Bản cuối của bộ này là nguồn cho bộ cụm Tiki (A3).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `song_ngu_2/the_chuan_bi.json` | 888 | `aa24d642c17e066d` |
| `song_ngu_2/the_nguon_goc_20261023_1656.json` | 10.452 | `cd378b421810d636` |
