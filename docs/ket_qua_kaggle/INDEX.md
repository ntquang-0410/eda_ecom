# Mục lục kết quả Kaggle hướng Nhật Anh

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py` lúc 18/11/2026 00:05 (ICT). Trang tổng hợp: [`docs/TONG_HOP_KET_QUA_NHAT_ANH.md`](../TONG_HOP_KET_QUA_NHAT_ANH.md). Nhật ký đầy đủ các phiên 4B (bản 20–43): [`docs/PHU_LUC_NHAT_KY_CHAY.md`](../PHU_LUC_NHAT_KY_CHAY.md).

Mỗi phiên chạy thật một file. Output gốc nằm trên máy Nhật Anh ở `data/processed/` (không có trong git); mỗi file ghi sha256 của file gốc để đối chiếu. Mỗi file có: thông tin phiên (notebook, tài khoản, bản, thời gian, GPU), mục đích, bảng số, kết luận, ghi chú/giới hạn, danh sách file gốc.

Tài khoản Kaggle: 1 = `trinhnhatanh` (tới 10/11/2026, và notebook demo 15/11), 2 = `honghuynanh` (từ 11/11/2026). GPU RTX Pro 6000 có được qua notebook gắn cuộc thi ARC Prize (nhóm báo đã được ban tổ chức đồng ý); từ 13/11/2026 notebook mới chỉ được cấp T4.

Hai cách huấn luyện cần phân biệt khi đọc: **công thức chính** (B, E: rank 64, 2 epoch, `w_title`) và **vòng sàng lọc** (C, D: rank 16, một lượt, ~4M token, lấy mẫu quota). Điểm hai cách không so thẳng với nhau, trừ ở E3 (cùng eval).

## A. Dữ liệu: dịch, lọc, Judge

| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |
|---|---|---|---|---|
| [A1](A1_bo_song_ngu_1_thuoc_tinh_1688_dich_theo_don_vi_g.md) | [Bộ song ngữ 1: thuộc tính 1688 dịch theo đơn vị (Gemma, Qwen3-14B, CometKiwi)](A1_bo_song_ngu_1_thuoc_tinh_1688_dich_theo_don_vi_g.md) | 1 (`trinhnhatanh`) | – | 15–22/10/2026 |
| [A2](A2_bo_song_ngu_2_thuoc_tinh_tiki_vi_zh_dich_theo_do.md) | [Bộ song ngữ 2: thuộc tính Tiki (vi→zh) dịch theo đơn vị](A2_bo_song_ngu_2_thuoc_tinh_tiki_vi_zh_dich_theo_do.md) | 1 (`trinhnhatanh`) | – | 23/10/2026 |
| [A3](A3_bo_song_ngu_3_ra_soat_theo_cum_co_ngu_canh_tiki.md) | [Bộ song ngữ 3: rà soát theo cụm có ngữ cảnh (Tiki)](A3_bo_song_ngu_3_ra_soat_theo_cum_co_ngu_canh_tiki.md) | 1 (`trinhnhatanh`) | – | 23–24/10/2026 |
| [A4](A4_bo_song_ngu_4_ra_soat_theo_cum_co_ngu_canh_1688.md) | [Bộ song ngữ 4: rà soát theo cụm có ngữ cảnh (1688)](A4_bo_song_ngu_4_ra_soat_theo_cum_co_ngu_canh_1688.md) | 1 (`trinhnhatanh`) | – | 24–26/10/2026 |
| [A5](A5_loc_luat_cho_fine_tune_tren_may.md) | [Lọc luật cho fine-tune (trên máy)](A5_loc_luat_cho_fine_tune_tren_may.md) | – (chạy trên máy) | – | 30/10/2026 |
| [A6](A6_judge_deepseek_chon_ban_dich_gemma_hay_qwen_cum.md) | [Judge DeepSeek chọn bản dịch Gemma hay Qwen (cụm và pair)](A6_judge_deepseek_chon_ban_dich_gemma_hay_qwen_cum.md) | – (gọi API DeepSeek, không chạy trên Kaggle; thí điểm: Nhật Anh trên máy; lần chạy đầy đủ: Huy) | – | thí điểm 29–31/10/2026; bản phát hành tải về 03/11/2026 |
| [A7](A7_chia_tap_theo_san_pham_kiem_ro_ri_gan_co_title_t.md) | [Chia tập theo sản phẩm, kiểm rò rỉ, gắn cờ title (trên máy)](A7_chia_tap_theo_san_pham_kiem_ro_ri_gan_co_title_t.md) | – (chạy trên máy) | – | 04–05/11/2026 |

## B. Fine-tune Qwen3-4B (công thức chính)

| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |
|---|---|---|---|---|
| [B0](B0_dung_mau_huan_luyen_va_dong_goi_dataset_kaggle_t.md) | [Dựng mẫu huấn luyện và đóng gói Dataset Kaggle (trên máy)](B0_dung_mau_huan_luyen_va_dong_goi_dataset_kaggle_t.md) | – (dựng trên máy; Dataset ở tài khoản 1 và 2) | – | 04–12/11/2026 |
| [B1](B1_b6_do_sieu_tham_so_qwen3_4b_lr_rank.md) | [B6: dò siêu tham số Qwen3-4B (lr × rank)](B1_b6_do_sieu_tham_so_qwen3_4b_lr_rank.md) | 1 (`trinhnhatanh`) | 20, 21, 22 (nối nhau) | 05/11/2026 |
| [B2](B2_b7_b8_phan_dau_lora_attention_pair_fold_0_cum_fo.md) | [B7 + B8 phần đầu: LoRA attention, pair fold 0, cụm fold 1 (dở)](B2_b7_b8_phan_dau_lora_attention_pair_fold_0_cum_fo.md) | 1 (`trinhnhatanh`) | 25 (bản 26 nối tiếp bị hủy) | 05/11/2026 |
| [B3](B3_thu_ti_trong_title_fold_0_11_va_75_cum_va_pair.md) | [Thử tỉ trọng title fold 0 (~11% và 75%, cụm và pair)](B3_thu_ti_trong_title_fold_0_11_va_75_cum_va_pair.md) | 1 (`trinhnhatanh`) | 29 | 05–06/11/2026 |
| [B4](B4_b8_fold_1_2_moc_1_1_cum_11_fold_1.md) | [B8 fold 1, 2 (mốc 1 : 1) + cụm ~11% fold 1](B4_b8_fold_1_2_moc_1_1_cum_11_fold_1.md) | 1 (`trinhnhatanh`) | 32 | 06/11/2026 |
| [B5](B5_cum_11_fold_2_luat_3_fold_ban_cum.md) | [Cụm ~11% fold 2 + luật 3 fold bản cụm](B5_cum_11_fold_2_luat_3_fold_ban_cum.md) | 1 (`trinhnhatanh`) | 34 | 06/11/2026 |
| [B6](B6_kiem_lr_1e_4_cum_pair_fold_1.md) | [Kiểm lr 1e-4 (cụm) + pair fold 1](B6_kiem_lr_1e_4_cum_pair_fold_1.md) | 1 (`trinhnhatanh`) | 36 | 06–07/11/2026 |
| [B7](B7_pair_fold_2_luat_3_fold_ban_pair.md) | [Pair fold 2 + luật 3 fold bản pair](B7_pair_fold_2_luat_3_fold_ban_pair.md) | 1 (`trinhnhatanh`) | 38 (bản 37 là khói) | 07/11/2026 |
| [B8](B8_do_test_khoa_mot_lan.md) | [Đo test khóa MỘT lần](B8_do_test_khoa_mot_lan.md) | 1 (`trinhnhatanh`) | 41 (bản 39, 40 là khói) | 07/11/2026 |
| [B9](B9_ban_cuoi_tren_toan_pool_cum_cuoi_pair_cuoi.md) | [Bản cuối trên toàn pool (cum_cuoi, pair_cuoi)](B9_ban_cuoi_tren_toan_pool_cum_cuoi_pair_cuoi.md) | 1 (`trinhnhatanh`) | 43 (bản 42 là khói) | 07/11/2026 |

## C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc)

| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |
|---|---|---|---|---|
| [C1](C1_khao_sat_quota_vong_1_rank_16_max_len_960_1_200.md) | [Khảo sát quota vòng 1 (rank 16, max_len 960, 1.200 bước)](C1_khao_sat_quota_vong_1_rank_16_max_len_960_1_200.md) | 1 (`trinhnhatanh`) | 47 (pair), 48 (cụm) | 09/11/2026 |
| [C2](C2_c0_adapter_chinh_thuc_cham_tren_eval_cua_vong_sa.md) | [C0: adapter chính thức chấm trên eval của vòng sàng lọc](C2_c0_adapter_chinh_thuc_cham_tren_eval_cua_vong_sa.md) | 1 (`trinhnhatanh`) | 50 (cụm), 51 (pair) | 09/11/2026 |
| [C3](C3_khao_sat_quota_vong_2_ngan_sach_token_co_dinh_ma.md) | [Khảo sát quota vòng 2 (ngân sách token cố định, max_len 2.048)](C3_khao_sat_quota_vong_2_ngan_sach_token_co_dinh_ma.md) | 1 (`trinhnhatanh`) | 53 | 10/11/2026 |
| [C4](C4_khao_sat_quota_vong_3_3_cau_hinh_2_bo_2_seed_ban.md) | [Khảo sát quota vòng 3: 3 cấu hình × 2 bộ × 2 seed (bảng đã chốt)](C4_khao_sat_quota_vong_3_3_cau_hinh_2_bo_2_seed_ban.md) | 1 (`trinhnhatanh`) | 55 | 10/11/2026 |
| [C5](C5_tai_lap_o_tai_khoan_2_check1.md) | [Tái lập ở tài khoản 2 (check1)](C5_tai_lap_o_tai_khoan_2_check1.md) | 2 (`honghuynanh`) | – | 11/11/2026 |
| [C6](C6_qwen3_4b_natural_voi_lr_1e_4.md) | [Qwen3-4B natural với lr 1e-4](C6_qwen3_4b_natural_voi_lr_1e_4.md) | 2 (`honghuynanh`) | – | 12/11/2026 |

## D. So sánh năm mô hình (vòng sàng lọc)

| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |
|---|---|---|---|---|
| [D0](D0_danh_sach_mau_natural_dung_chung_cho_cac_llm_cha.md) | [Danh sách mẫu natural dùng chung cho các LLM (chạy khô, CPU)](D0_danh_sach_mau_natural_dung_chung_cho_cac_llm_cha.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [D1](D1_qwen3_8b_vong_sang_loc_cum_pair_2_seed.md) | [Qwen3-8B, vòng sàng lọc (cụm + pair, 2 seed)](D1_qwen3_8b_vong_sang_loc_cum_pair_2_seed.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [D2](D2_ecomgpt_7b_vong_sang_loc_cum_pair_2_seed.md) | [EcomGPT-7B, vòng sàng lọc (cụm + pair, 2 seed)](D2_ecomgpt_7b_vong_sang_loc_cum_pair_2_seed.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [D3](D3_opennmt_transformer_nho_hoc_tu_dau_pair_dich_ban.md) | [OpenNMT Transformer nhỏ, học từ đầu (pair + dịch bản cụm)](D3_opennmt_transformer_nho_hoc_tu_dau_pair_dich_ban.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [D4](D4_moses_vong_1_chon_cach_tach_chu_han_tung_chu_hay.md) | [Moses vòng 1: chọn cách tách chữ Hán (từng chữ hay jieba)](D4_moses_vong_1_chon_cach_tach_chu_han_tung_chu_hay.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [D5](D5_moses_vong_2_hoc_du_du_lieu_va_khop_4m_token_kem.md) | [Moses vòng 2: học đủ dữ liệu và khớp 4M token, kèm eval bản cụm](D5_moses_vong_2_hoc_du_du_lieu_va_khop_4m_token_kem.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [D6](D6_tong_hop_nam_mo_hinh_tren_may_khong_gpu.md) | [Tổng hợp năm mô hình (trên máy, không GPU)](D6_tong_hop_nam_mo_hinh_tren_may_khong_gpu.md) | – (Qwen3-4B lấy từ C4, tài khoản 1; các mô hình khác tài khoản 2) | – | 12–14/11/2026 |

## E. Thử chéo và công thức chính trên eval năm mô hình

| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |
|---|---|---|---|---|
| [E1](E1_huan_luyen_lai_adapter_fold_0_o_tai_khoan_2_cum.md) | [Huấn luyện lại adapter fold 0 ở tài khoản 2 (cum_f0_p11, pair_f0_p11)](E1_huan_luyen_lai_adapter_fold_0_o_tai_khoan_2_cum.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [E2](E2_thu_cheo_dang_pair_va_dang_cum.md) | [Thử chéo dạng pair và dạng cụm](E2_thu_cheo_dang_pair_va_dang_cum.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| [E3](E3_cong_thuc_chinh_tren_eval_nam_mo_hinh_dong_thu_s.md) | [Công thức chính trên eval năm mô hình ("dòng thứ sáu", T4 fp16)](E3_cong_thuc_chinh_tren_eval_nam_mo_hinh_dong_thu_s.md) | 2 (`honghuynanh`) | – | 14/11/2026 |

## F. Demo và việc chưa chạy được

| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |
|---|---|---|---|---|
| [F1](F1_cong_nen_de_chay_demo_tren_laptop_khong_phai_kag.md) | [Cổng nén để chạy demo trên laptop (không phải Kaggle)](F1_cong_nen_de_chay_demo_tren_laptop_khong_phai_kag.md) | – | – | 14/11/2026 |
| [F2](F2_may_chu_dich_demo_tren_kaggle_thu_13_trang_1688.md) | [Máy chủ dịch demo trên Kaggle + thử 13 trang 1688 thật](F2_may_chu_dich_demo_tren_kaggle_thu_13_trang_1688.md) | 1 (`trinhnhatanh`) | – | 15/11/2026 |
| [F3](F3_qwen3_8b_theo_cong_thuc_chinh_khong_chay_duoc_bi.md) | [Qwen3-8B theo công thức chính: không chạy được (bị cấp T4)](F3_qwen3_8b_theo_cong_thuc_chinh_khong_chay_duoc_bi.md) | 2 (`honghuynanh`) | – | 13/11/2026 |

## Phiên chạy thử ngắn (khói, chạy khô, biên dịch): không có file riêng

Các phiên này chỉ kiểm mã, môi trường, bộ nhớ hoặc tốc độ trước phiên thật; không dùng làm kết quả. Cột Đạt và Thời lượng đọc từ file báo cáo nếu có (`–` là phiên không ghi báo cáo, chỉ có log).

| Thư mục | Kiểm cái gì | Notebook | Đạt | Thời lượng | GPU | Tải về lúc |
|---|---|---|---|---|---|---|
| `ft_run/kaggle_b5` | B5: chạy thử môi trường và mã huấn luyện, lần 1 | `trinhnhatanh/nlp-group3` | không | 50 giây (0.01 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 06:57 |
| `ft_run/kaggle_b5_lan2` | B5 lần 2 | `trinhnhatanh/nlp-group3` | không | 184 giây (0.05 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 07:01 |
| `ft_run/kaggle_b5_lan3` | B5 lần 3 | `trinhnhatanh/nlp-group3` | không | 126 giây (0.03 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 07:08 |
| `ft_run/kaggle_b5_lan4` | B5 lần 4 | `trinhnhatanh/nlp-group3` | có | 600 giây (0.17 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 07:20 |
| `ft_run/kaggle_do_toc_do` | đo tốc độ token/giây | `trinhnhatanh/nlp-group3` | có | 503 giây (0.14 giờ) | – | 05/11/2026 07:32 |
| `ft_run/kaggle_nguyen_nhan` | tìm nguyên nhân chậm | `trinhnhatanh/nlp-group3` | có | 256 giây (0.07 giờ) | – | 05/11/2026 07:48 |
| `ft_run/kaggle_b6_thu` | khói B6 | `trinhnhatanh/nlp-group3` | có | 406 giây (0.11 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 08:07 |
| `ft_run/kaggle_b6_thu2` | khói B6 lần 2 | `trinhnhatanh/nlp-group3` | có | 377 giây (0.10 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 08:16 |
| `ft_run/kaggle_b6_noi` | khói nối phiên B6 (phan1 → phan2) | `trinhnhatanh/nlp-group3` | không, có | 437 giây (0.12 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 09:57 |
| `ft_run/kaggle_b78_thu` | khói B7 + B8 (hai phần + tổng hợp) | `trinhnhatanh/nlp-group3` | không, có | 748 giây (0.21 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 18:09 |
| `ft_run/kaggle_tl_thu` | khói thử tỉ lệ (hai phần) | `trinhnhatanh/nlp-group3` | không, có | 581 giây (0.16 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 05/11/2026 23:19 |
| `ft_run/kaggle_tl3_thu` | khói pair fold 2 (bản 37) | `trinhnhatanh/nlp-group3` | có | 364 giây (0.10 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 07/11/2026 02:30 |
| `ft_run/kaggle_do_test_thu` | khói đo test (bản 39, 40) | `trinhnhatanh/nlp-group3` | có | 232 giây (0.06 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 07/11/2026 08:05 |
| `ft_run/kaggle_cuoi_thu` | khói bản cuối (bản 42) | `trinhnhatanh/nlp-group3` | có | 340 giây (0.09 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 07/11/2026 10:54 |
| `ft_run/kaggle_quota_smoke` | khói khảo sát quota (bản 46) | `trinhnhatanh/nlp-group3` | – | – | NVIDIA RTX PRO 6000 Blackwell Server Edition | 09/11/2026 10:54 |
| `ft_run/kaggle_quota_dry` | chạy khô quota trước C3 (bản 52) | `trinhnhatanh/nlp-group3` | – | – | NVIDIA RTX PRO 6000 Blackwell Server Edition | 10/11/2026 00:19 |
| `ft_run/kaggle_quota_dry2` | chạy khô quota trước C4 (bản 54) | `trinhnhatanh/nlp-group3` | – | – | NVIDIA RTX PRO 6000 Blackwell Server Edition | 10/11/2026 12:23 |
| `ft_run/kaggle_dry_mh` | chạy khô danh sách mẫu cho 8B và EcomGPT (CPU) | `honghuynanh/nlp-dry-8b, nlp-dry-ecom` | – | – | cpu | 12/11/2026 06:49 |
| `ft_run/kaggle_smoke_mh` | khói 8B và EcomGPT (bộ nhớ, tốc độ) | `honghuynanh/nlp-smoke-8b, nlp-smoke-ecom` | – | – | NVIDIA RTX PRO 6000 Blackwell Server Edition | 12/11/2026 06:57 |
| `ft_run/kaggle_onmt_khoi` | khói OpenNMT | `honghuynanh/nlp-onmt-khoi` | – | 317 giây (0.09 giờ) | – | 12/11/2026 11:32 |
| `ft_run/kaggle_moses_build` | biên dịch Moses | `honghuynanh/nlp-moses-build` | – | – | – | 12/11/2026 12:04 |
| `ft_run/kaggle_moses_diag` | chẩn đoán lỗi biên dịch Moses | `honghuynanh/nlp-moses-diag` | – | – | – | 12/11/2026 12:11 |
| `ft_run/kaggle_moses_do` | đo thời gian Moses | `honghuynanh/nlp-moses-do` | – | – | – | 12/11/2026 10:57 |
| `ft_run/kaggle_moses_khoi` | khói Moses (từng chữ, khớp 4M) | `honghuynanh/nlp-moses-khoi` | – | 520 giây (0.14 giờ) | – | 12/11/2026 15:01 |
| `ft_run/kaggle_cheo_khoi` | khói huấn luyện cho thử chéo | `honghuynanh/nlp-group3` | có | 340 giây (0.09 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 12/11/2026 20:27 |
| `ft_run/kaggle_cheo_khoi2` | khói sinh thử chéo | `honghuynanh/nlp-cheo` | có | 763 giây (0.21 giờ) | NVIDIA RTX PRO 6000 Blackwell Server Edition | 12/11/2026 22:12 |
| `ft_run/kaggle_dong6_khoi` | khói dòng thứ sáu (bf16 trên T4) | `honghuynanh/nlp-dong6` | có | 971 giây (0.27 giờ) | Tesla T4 | 14/11/2026 00:25 |
| `ft_run/kaggle_dong6_khoi_fp16` | khói dòng thứ sáu (fp16 trên T4) | `honghuynanh/nlp-dong6` | không | 2.467 giây (0.69 giờ) | Tesla T4 | 14/11/2026 08:18 |

## Ngoài Kaggle

- A5 (lọc luật), A6 (Judge DeepSeek, gọi API), D6 (tổng hợp năm mô hình), phần chấm của E2 và F1 (cổng nén) không chạy trên Kaggle nhưng có file riêng ở trên vì là bước chính.
- Chia tập theo sản phẩm + test khóa và lọc title bị gắn cờ (`loc_title.py`): chạy trên máy; số liệu và hash ở trang tổng hợp mục 2.5, 2.6.
