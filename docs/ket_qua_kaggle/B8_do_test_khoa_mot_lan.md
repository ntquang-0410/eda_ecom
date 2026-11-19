# B8. Đo test khóa MỘT lần

[← Mục lục](INDEX.md) · [← B7](B7_pair_fold_2_luat_3_fold_ban_pair.md) · [B9 →](B9_ban_cuoi_tren_toan_pool_cum_cuoi_pair_cuoi.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B8 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` + Dataset riêng `ft-qwen3-4b-do-khoa` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 41 (bản 39, 40 là khói) |
| Thời gian | 07/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_do_test` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_do_test/phien_1_20261107_080604/bao_cao_do_test.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | 0.18.1 | 9.188 giây (2.55 giờ) | có | chi_co_do_khoa_dung_ten |

## Mục đích

Đo một lần duy nhất adapter fold 0 (`cum_f0_p11`, `pair_f0_p11` của tài khoản 1, kho `ft-qwen3-4b-tl-luu`) so với zero-shot trên toàn bộ tập test khóa; luật thắng đặt trước khi đo.

## Kết quả: fine-tune (adapter fold 0, ~11% title) so với zero-shot trên TEST KHÓA

chrF theo mục, bootstrap theo sản phẩm (seed 42). Luật: thắng khi cận dưới khoảng 95% của chênh > 0.

| Nhánh | Hướng | Sản phẩm | Mục | chrF zero-shot | chrF fine-tune | Chênh [95%] | Thắng (luật khóa) |
|---|---|---|---|---|---|---|---|
| cum | vi->zh | 1.073 | 2.146 | 41.49 | 75.63 | +34.14 [+33.30; +34.95] | có |
| cum | zh->vi | 1.485 | 2.970 | 47.22 | 77.36 | +30.14 [+29.66; +30.63] | có |
| pair | vi->zh | 1.073 | 2.146 | 43.92 | 75.41 | +31.50 [+30.73; +32.24] | có |
| pair | zh->vi | 1.485 | 2.970 | 46.34 | 77.65 | +31.31 [+30.81; +31.83] | có |

Theo loại:

| Nhánh | Hướng | Loại | Zero-shot | Fine-tune | Chênh [95%] |
|---|---|---|---|---|---|
| cum | vi->zh | title | 45.40 | 57.53 | +12.13 [+10.85; +13.33] |
| cum | vi->zh | mo_ta | 37.58 | 93.73 | +56.15 [+55.02; +57.29] |
| cum | zh->vi | title | 37.66 | 60.94 | +23.28 [+22.48; +24.15] |
| cum | zh->vi | mo_ta | 56.78 | 93.78 | +37.00 [+36.50; +37.45] |
| pair | vi->zh | title | 45.51 | 57.53 | +12.02 [+10.76; +13.27] |
| pair | vi->zh | mo_ta | 42.33 | 93.30 | +50.97 [+50.11; +51.88] |
| pair | zh->vi | title | 37.66 | 61.23 | +23.57 [+22.72; +24.46] |
| pair | zh->vi | mo_ta | 55.01 | 94.06 | +39.05 [+38.54; +39.60] |

Kiểm riêng (tỉ lệ trên số mục; `da_cat_loi_dan` = bản sinh có lời dẫn bị cắt bỏ; cột Tham chiếu = cùng phép kiểm áp lên chính bản dịch tham chiếu):

| Nhánh | Hướng | Kiểm | Zero-shot | Fine-tune | Tham chiếu |
|---|---|---|---|---|---|
| cum | vi->zh | sinh_lap | 0.000 | 0.000 | – |
| cum | vi->zh | giu_so_dat | 0.938 | 0.954 | 0.956 |
| cum | vi->zh | giu_latin_dat | 0.591 | 0.669 | 0.658 |
| cum | vi->zh | sai_chu_viet | 0.050 | 0.052 | 0.048 |
| cum | vi->zh | da_cat_loi_dan | 0.414 | 0.000 | – |
| cum | zh->vi | sinh_lap | 0.015 | 0.001 | – |
| cum | zh->vi | giu_so_dat | 0.894 | 0.890 | 0.885 |
| cum | zh->vi | giu_latin_dat | 0.954 | 0.986 | 0.988 |
| cum | zh->vi | sai_chu_viet | 0.308 | 0.058 | 0.049 |
| cum | zh->vi | da_cat_loi_dan | 0.326 | 0.000 | – |
| pair | vi->zh | sinh_lap | 0.000 | 0.000 | – |
| pair | vi->zh | giu_so_dat | 0.938 | 0.961 | 0.957 |
| pair | vi->zh | giu_latin_dat | 0.593 | 0.678 | 0.658 |
| pair | vi->zh | sai_chu_viet | 0.050 | 0.055 | 0.048 |
| pair | vi->zh | da_cat_loi_dan | 0.452 | 0.000 | – |
| pair | zh->vi | sinh_lap | 0.016 | 0.001 | – |
| pair | zh->vi | giu_so_dat | 0.878 | 0.894 | 0.891 |
| pair | zh->vi | giu_latin_dat | 0.949 | 0.985 | 0.988 |
| pair | zh->vi | sai_chu_viet | 0.304 | 0.058 | 0.049 |
| pair | zh->vi | da_cat_loi_dan | 0.353 | 0.000 | – |

Cảnh báo theo luật: không có.

## Kết luận của phiên

- Fine-tune thắng zero-shot ở **mọi nhánh, mọi hướng** (cận dưới xa 0).

## Ghi chú, giới hạn, sự cố

- Không đổi tỉ lệ, lr hay adapter sau khi thấy số test. Bản cuối (B9) **không** đo lại trên test.
- Chênh lớn một phần vì tham chiếu là bản dịch máy cùng kiểu với dữ liệu huấn luyện.
- Bổ sung 18/11 (Bước 0, chỉ đọc tệp tổng hợp của chính lần đo này, không chấm lại): 86 đến 87% dòng mô tả zh→vi (73% vi→zh) của test đã có trong train; trên dòng mô tả **mới** fine-tune vẫn hơn zero-shot khoảng 30 điểm chrF corpus (cụm zh→vi +30,1; pair zh→vi +30,6; pair vi→zh +29,7), chất lượng dòng mới khoảng 76 (không phải 92–94); chưa có khoảng tin cậy. Chi tiết `docs/phan_bien_vong2/KET_QUA_B8_DONG_MOI.md`.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_do_test/phien_1_20261107_080604/bao_cao_do_test.json` | 890 | `d6630c34c52c2255` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/do_test_trang_thai.json` | 18.212 | `c93d6e1a3d4bdeed` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_cum_ft.jsonl` | 5.415.370 | `b07848660d0ee51a` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_cum_zero_shot.jsonl` | 6.023.822 | `16c01bc90afc2041` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_diem_cum_ft.jsonl` | 6.255.416 | `98a73d23621392ec` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_diem_cum_zero_shot.jsonl` | 6.253.539 | `9ee6a28ef9774250` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_diem_pair_ft.jsonl` | 6.209.285 | `b85762c13ed0c6d1` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_diem_pair_zero_shot.jsonl` | 6.243.863 | `b1c5f0c9d9ba67dc` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_pair_ft.jsonl` | 5.318.098 | `ffcb1f0132bf6263` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/sinh_pair_zero_shot.jsonl` | 6.092.578 | `70a75897c8285c67` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/nlp-group3.log` | 4.354 | `ed85f413488b6cb1` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/tong_hop/tong_hop_do_test.json` | 6.295 | `e409c30b7a96719f` |
| `ft_run/kaggle_do_test/phien_1_20261107_080604/tong_hop/tong_hop_do_test.md` | 2.630 | `ffb0e4c7faa78ee6` |
| `ft_run/kaggle_do_test/tu_dong.log` | 866 | `edb84e2407f2fee5` |
