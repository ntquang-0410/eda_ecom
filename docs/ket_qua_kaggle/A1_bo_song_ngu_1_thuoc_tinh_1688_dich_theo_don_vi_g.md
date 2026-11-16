# A1. Bộ song ngữ 1: thuộc tính 1688 dịch theo đơn vị (Gemma, Qwen3-14B, CometKiwi)

[← Mục lục](INDEX.md) · [A2 →](A2_bo_song_ngu_2_thuoc_tinh_tiki_vi_zh_dich_theo_do.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | A1 |
| Giai đoạn | A. Dữ liệu: dịch, lọc, Judge |
| Notebook Kaggle | các notebook dịch và chấm của tài khoản 1 (thẻ `kaggle` bên dưới) |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 15–22/10/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/song_ngu_1` |

## Mục đích

Dịch tên và giá trị thuộc tính của sản phẩm 1688 sang tiếng Việt theo từng đơn vị bằng hai mô hình, chấm từng bản bằng CometKiwi (`wmt22-cometkiwi-da`, không cần tham chiếu) rồi chọn bản theo luật ngưỡng. Phần cố định (có/không, mã thật, tên hàng, từ điển đã duyệt) đi theo luật, không qua mô hình.

## Thẻ nguồn gốc `song_ngu_1/the_nguon_goc_20261022_2126.json`

Bước: `hoan_tat (5, 6b, 7)`, ngày chạy 2026-10-22 21:26:31. Mô hình theo thẻ: `{"A": "api_tuy_chon", "B": "qwen3_14b"}`; ngưỡng chọn: `{"tau": 60.0, "kappa": 0.5, "kappa_ngan": 0.35, "delta": 0.25, "uu_tien": "A", "chi_doi_khi_truot": true, "do_dai_ngan": 6, "do_dai_zh_ngan": 8, "ghi_chu": "tạm; chọn từ phân bố điểm, không có nhãn đúng sai. delta là chênh điểm CometKiwi để bỏ mô hình ưu tiên (từ 15/10/2026). Luật lap_tu từ 21/10/2026: phép tỉ lệ từ không trùng bỏ qua danh sách có số mục khớp nguồn"}`.

Chuẩn bị (bước 1–3):

| Trường | Giá trị |
|---|---|
| buoc | chuan_bi (1-3) |
| ngay_chay | 2026-10-15 16:58:07 |
| file_cap | bilingual_zh_vi_desc_pairs(nhatanh).parquet |
| sha256_file_cap | 48F11FF3C7661BF17791AE286001C82C8C7B8420518D6FD5CCA841CF49A0D447 |
| so_cap | 405.275 |
| ma_loai_ban_dau | `{"T2_CHUA_KIEM": 36184, "DS07": 2762, "DS09_dai": 74, "RONG": 1369, "DS10": 554, "DS17": 110, "DS03": 62, "DS25": 2, "DS18": 3}` |
| trang_thai_vao | `{"dung": 403830, "van_loai": 1371, "cho_xem": 74}` |
| dinh_tuyen_cap_dung | `{"mo_hinh": 338273, "chep_nguyen": 49128, "ten_hang": 13878, "ma_that": 2479, "ngay": 72}` |
| don_vi | `{"ten": 1980, "gia_tri": 41707, "ky_tu_trung_gia_tri": 2387310}` |
| khop_voi_ban_v1_phan_quy_tac | `{"ma_that": 1.0, "ten_hang": 1.0}` |

Các lượt dịch / chấm (thẻ `kaggle` trong file; `api_tuy_chon` = Gemma gọi qua API, `qwen3_14b` = Qwen3-14B-AWQ chạy vLLM trên Kaggle, `kiwi` = CometKiwi chấm trên Kaggle):

| Thẻ | Mô hình | Repo / bản dịch của | Backend / GPU | Bắt đầu | Kết thúc | Số đơn vị | Thời lượng |
|---|---|---|---|---|---|---|---|
| the_api_tuy_chon_phien_1 | api_tuy_chon | gemma-4-26b-a4b-it | api | 2026-10-16 00:34:49 | 2026-10-16 01:35:17 | 43.687 | 3.629 giây (1.01 giờ) |
| the_api_tuy_chon_phien_2 | api_tuy_chon | gemma-4-26b-a4b-it | api | 2026-10-16 02:58:26 | 2026-10-16 04:22:52 | 43.687 | 5.067 giây (1.41 giờ) |
| the_kaggle_cham_kiwi_a | – | – | 2 GPU | 2026-10-16 06:49:06 | – | – | – |
| the_kaggle_qwen3_14b | – | – | 2 GPU | 2026-10-21 13:53:48 | – | – | – |
| the_kiwi_api_tuy_chon | Unbabel/wmt22-cometkiwi-da | api_tuy_chon | – | 2026-10-16 06:31:07 | 2026-10-16 06:45:34 | 43.662 | 868 giây (0.24 giờ) |
| the_kiwi_qwen3_14b | Unbabel/wmt22-cometkiwi-da | qwen3_14b | – | 2026-10-21 13:37:50 | 2026-10-21 13:53:13 | 43.687 | 925 giây (0.26 giờ) |
| the_qwen3_14b | qwen3_14b | Qwen/Qwen3-14B-AWQ | vllm | 2026-10-21 09:25:27 | 2026-10-21 13:35:24 | 43.687 | 14.998 giây (4.17 giờ) |

Kết quả chọn theo `don_vi`:

| Trường | Giá trị |
|---|---|
| tong | 43.687 |
| giu | 37.907 |
| cach_chon | `{"khac_chu_cung_nghia_A": 14937, "gan_giong_A": 10866, "giong_het": 6917, "chi_A": 2024, "uu_tien_duoi_san_A": 1258, "chi_B": 672, "tu_dien_p7": 628, "quy_tac_co_khong": 385, "tu_dien_p6": 119, "quy_tac_khoang": 80, "quy_tac_diem_anh": 14, "tu_dien_p6b": 7}` |
| ly_do_cat | `{"khac_nhieu_diem_khong_du": 3277, "gan_giong_diem_thap": 1053, "mot_ban_qua_luat_diem_thap": 951, "ca_hai_truot": 499}` |

Kết quả chọn theo `cap`:

| Trường | Giá trị |
|---|---|
| giu | 394.662 |
| cat_rieng | 10.613 |
| do_tin_cay | `{"cao": 188708, "vua": 187755, "thap": 18199}` |
| cach_chon | `{"giong_het": 90157, "khac_chu_cung_nghia_A": 86082, "quy_tac_co_khong": 55154, "quy_tac_chep_nguyen": 49125, "tu_dien_p7": 48881, "gan_giong_A": 31541, "quy_tac_ten_hang": 13872, "chi_B": 5451, "uu_tien_duoi_san_A": 4937, "chi_A": 4192, "quy_tac_khoang": 2566, "quy_tac_ma_that": 2479, "quy_tac_diem_anh": 153, "quy_tac_ngay": 72}` |

File ra: `{"day_du": "bo_song_ngu_1_day_du_20261022_2126.parquet", "cong_bo": "bo_song_ngu_1_cong_bo_20261022_2126.parquet", "cat_rieng": "cat_rieng_20261022_2126.parquet"}`

## Kết luận của phiên

- Bản cuối của bộ này là nguồn cho bộ cụm 1688 (A4).

## Ghi chú, giới hạn, sự cố

- Ngưỡng chọn (`tau`, `kappa`, `delta`) ghi trong thẻ là tạm, đặt từ phân bố điểm, không có nhãn đúng/sai.
- Thư mục còn các lần thử mô hình dịch khác: `hy_mt_thu` (HY-MT1.5-7B), `qwen14b_thu`, `qwen30b_thu`.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `song_ngu_1/the_chuan_bi.json` | 921 | `7072b34bd864ae7b` |
| `song_ngu_1/the_nguon_goc_20261022_2126.json` | 14.617 | `696adad602759843` |
