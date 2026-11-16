> **Phụ lục — bản chụp ngày 07/11/2026.** File này là `NHAT_KY_CHAY.md` gốc nằm trong `data/processed/ft_run/` (không vào git). Nó chỉ phủ các phiên tới 07/11; các phiên sau (quota 09–10/11, demo 14–15/11, so sánh 5 mô hình 10–13/11) nằm trong [`ket_qua_kaggle/INDEX.md`](ket_qua_kaggle/INDEX.md). Số trong bảng lấy từ file kết quả, không gõ tay.

# NHẬT KÝ CÁC LẦN CHẠY FINE-TUNE QWEN3-4B + LORA

Sinh tự động bởi `scripts/ft_train/nhat_ky_chay.py` lúc 07/11/2026 14:19 ICT. Mọi số lấy từ file kết quả trong `data/processed/ft_run/kaggle_*`, không gõ tay. Nhãn: [ĐO] = từ file; điểm chrF là độ giống bản dịch máy dùng làm tham chiếu, chưa phải người chấm. Không dùng tập test ở bất kỳ lần chạy nào trong bảng này.

## 1. Các phiên Kaggle (kết quả chính)

| Phiên | Bản Kaggle | Bắt đầu (ICT) | Kết thúc (ICT) | Giây | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| B6 phiên 1 | 20 | 05/11/2026 09:58:22 | 05/11/2026 12:28:25 | 9003 | 2.50 | 2.5 | False | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | dò siêu tham số, dừng có trật tự ở giờ tối đa |
| B6 phiên 2 | 21 | 05/11/2026 12:33:15 | 05/11/2026 15:03:17 | 9002 | 2.50 | 2.5 | False | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | nối từ phiên 1 |
| B6 phiên 3 | 22 | 05/11/2026 15:07:27 | 05/11/2026 16:10:25 | 3778 | 1.05 | 2.5 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | nối từ phiên 2, xong B6 |
| B7 + B8 phần đầu | 25 | 05/11/2026 18:13:55 | 05/11/2026 20:43:57 | 9002 | 2.50 | 2.5 | False | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | attn, pair_f0, cum_f1 dừng ở bước 1.102/1.800 do hết giờ tối đa; bản 26 (nối tiếp) bị Nhật Anh hủy, không có báo cáo |
| Thử tỉ lệ fold 0 | 29 | 05/11/2026 23:20:00 | 06/11/2026 05:00:04 | 20404 | 5.67 | 7.5 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | cum/pair × ~11% và 75% title |
| B8 fold 1, 2 mốc 1 : 1 + cụm ~11% fold 1 | 32 | 06/11/2026 10:28:00 | 06/11/2026 14:05:59 | 13079 | 3.63 | 7.5 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | nối cum_f1 từ bước 1.102 của bản 25, rồi cum_f1_p11, cum_f2 |
| Cụm ~11% fold 2 | 34 | 06/11/2026 16:51:18 | 06/11/2026 18:20:43 | 5365 | 1.49 | 3.0 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | cum_f2_p11 |
| Kiểm lr 1e-4 (cụm) + pair fold 1 | 36 | 06/11/2026 21:00:52 | 07/11/2026 01:10:13 | 14962 | 4.16 | 6.0 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | cum_f0_p11_lr1e4, pair_f1, pair_f1_p11 |
| Pair fold 2 | 38 | 07/11/2026 02:31:18 | 07/11/2026 05:17:23 | 9964 | 2.77 | 3.5 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | pair_f2, pair_f2_p11; bản 37 là khói, bản 38 là phiên thật |
| Đo test MỘT lần | 41 | 07/11/2026 08:06:18 | 07/11/2026 10:39:26 | 9188 | 2.55 | 4.5 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | không huấn luyện: zero-shot và adapter fold 0 (cum_f0_p11, pair_f0_p11) trên toàn bộ 5.116 mẫu test mỗi nhánh; bản 39, 40 là khói |
| Bản cuối trên toàn pool | 43 | 07/11/2026 10:55:02 | 07/11/2026 14:07:01 | 11519 | 3.20 | 5.5 | True | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 | cum_cuoi, pair_cuoi: toàn pool, loại 6.386 title bị cờ; bản 42 là khói |

`Xong hết việc = False` nghĩa là phiên dừng có trật tự vì hết giờ tối đa cài đặt, việc dở được nối ở phiên sau (không phải lỗi).

Tổng giờ GPU của các phiên chính: **32.02 giờ** (chưa tính phiên khói bên dưới). `khoa_test` ở mọi phiên: chi_co_do_khoa_dung_ten, khong_co_file_test_trong_input.

Phiên khói, thử môi trường và đo tốc độ (không dùng làm kết quả): kaggle_b5 50 giây; kaggle_b5_lan2 184 giây; kaggle_b5_lan3 126 giây; kaggle_b5_lan4 600 giây; kaggle_do_toc_do 503 giây; kaggle_nguyen_nhan 256 giây; kaggle_b6_thu 406 giây; kaggle_b6_thu2 377 giây; kaggle_b6_noi/phan1 164 giây; kaggle_b6_noi/phan2 273 giây; kaggle_b78_thu/phan1 301 giây; kaggle_b78_thu/phan2 447 giây; kaggle_tl_thu/phan1 188 giây; kaggle_tl_thu/phan2 393 giây.

## 2. Thông số cố định cho mọi lần chạy huấn luyện [đọc từ mã `kaggle_train.py`]

- Mô hình nền `Qwen/Qwen3-4B` (revision 1cfa9a72), nạp bf16, `attn_implementation=sdpa`, tắt suy luận (`enable_thinking=False`).
- LoRA: `lora_alpha = 2 × rank`, `lora_dropout = 0,05`, `target_modules = all-linear` (riêng lần thử attention: q_proj, k_proj, v_proj, o_proj).
- Tối ưu: AdamW, `weight_decay = 0`, cắt gradient `clip_grad_norm_ = 1,0`; lịch lr: warmup 3% rồi giảm cosine về 0, tính cho cả 2 epoch.
- Dữ liệu: `max_len` 2.048 token, `micro_bs` 32, ngân sách 4.096 token mỗi lô nhỏ, không bật gradient checkpointing; seed 42 (xáo lô theo seed + chỉ số epoch).
- Số epoch: B6 vòng 1 là 1 epoch; mọi lần chạy từ vòng 2 về sau là 2 epoch.
- Kiểm loss: 3.000 mẫu kiểm cố định (750 mỗi hướng × loại), nằm trong fold kiểm; loss không trọng số, điểm = trung bình theo hướng, title và mô tả trọng số bằng nhau.
- Sinh bản dịch: greedy (`do_sample=False`), trần `max_new_tokens` theo độ dài đích tham chiếu cho mọi lần chạy huấn luyện và chấm theo fold, 1.000 mẫu (250 mỗi hướng × loại) giống nhau giữa các cấu hình để so theo cặp. Riêng lần đo test (mục 9) dùng trần theo độ dài NGUỒN `min(2048, 3,5 × token nguồn + 64)`, chọn từ tỉ lệ token đích/nguồn lớn nhất đo trên pool (2,94), và bộ chấm có thêm mốc tham chiếu `ref_*` và quy tắc cắt lời dẫn `dong_ngu_canh`.
- Loss có trọng số (từ đợt thử tỉ lệ): mẫu title nhân `w_title`, mẫu mô tả nhân 1; `he_chuan = he_chuan_goc × tổng_trọng_số / số_mẫu` giữ độ lớn bước tương đương; đánh giá vẫn không trọng số. Tỉ lệ title là siêu tham số cố định trong một lần train.
- Chia dữ liệu: `chia_tap_v2`, seed 20261104, test 10% khóa cứng (2.558 sản phẩm), pool 23.017 chia 5 fold theo nhóm gần-trùng; Dataset Kaggle chỉ có mẫu pool.

## 3. Thông số huấn luyện của từng việc

| Việc | Từ trạng thái | Bản Kaggle | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title mục tiêu | % title thực | % title nếu w=1 | w_title | he_chuan | Bước clip (title / trộn / mô tả) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr5e-5_r16 | B6 | 20–22 | cum | 0 | 5e-05 | 16 | 33030144 | 1 | 929 | 37123 | 13061056 | – | – | – | – | – | – |
| lr1e-4_r16 | B6 | 20–22 | cum | 0 | 0.0001 | 16 | 33030144 | 1 | 929 | 37123 | 13061056 | – | – | – | – | – | – |
| lr2e-4_r16 | B6 | 20–22 | cum | 0 | 0.0002 | 16 | 33030144 | 1 | 929 | 37123 | 13061056 | – | – | – | – | – | – |
| lr5e-5_r64 | B6 | 20–22 | cum | 0 | 5e-05 | 64 | 132120576 | 1 | 929 | 37123 | 13061056 | – | – | – | – | – | – |
| lr1e-4_r64 | B6 | 20–22 | cum | 0 | 0.0001 | 64 | 132120576 | 2 | 1837 | 37123 | 13061056 | – | – | – | – | – | – |
| lr2e-4_r64 | B6 | 20–22 | cum | 0 | 0.0002 | 64 | 132120576 | 2 | 1837 | 37123 | 13061056 | – | – | – | – | – | – |
| lr1e-4_r16_attn | B6 | 20–22 | cum | 0 | 0.0001 | 16 | 11796480 | 1 | 929 | 37123 | 13061056 | – | – | – | – | – | – |
| pair_f0 | B7+B8 phần đầu | 25 | pair | 0 | 0.0002 | 64 | 132120576 | 2 | 1843 | 37114 | 12395741 | – | – | – | – | – | – |
| cum_f1 | B8 fold 1, 2 + cụm ~11% fold 1 | 25→32 | cum | 1 | 0.0002 | 64 | 132120576 | 2 | 1800 | 35938 | 12853300 | – | – | – | – | – | 12/93 / 2/391 / 0/214 |
| cum_f0_p11 | Thử tỉ lệ fold 0 | 29 | cum | 0 | 0.0002 | 64 | 132120576 | 2 | 1837 | 37123 | 13061056 | 10.74 | 10.74 | 49.65 | 0.12029 | 1.7831 | 1/263 / 5/1011 / 6/563 |
| cum_f0_p75 | Thử tỉ lệ fold 0 | 29 | cum | 0 | 0.0002 | 64 | 132120576 | 2 | 1837 | 37123 | 13061056 | 75.00 | 75.00 | 49.65 | 3.03912 | 0.4955 | 262/263 / 456/1011 / 3/563 |
| pair_f0_p11 | Thử tỉ lệ fold 0 | 29 | pair | 0 | 0.0002 | 64 | 132120576 | 2 | 1843 | 37114 | 12395741 | 10.86 | 10.86 | 50.30 | 0.11818 | 1.7868 | 1/263 / 5/1054 / 6/526 |
| pair_f0_p75 | Thử tỉ lệ fold 0 | 29 | pair | 0 | 0.0002 | 64 | 132120576 | 2 | 1843 | 37114 | 12395741 | 75.00 | 75.00 | 50.30 | 2.95906 | 0.5055 | 263/263 / 494/1054 / 5/526 |
| cum_f1_p11 | B8 fold 1, 2 + cụm ~11% fold 1 | 25→32 | cum | 1 | 0.0002 | 64 | 132120576 | 2 | 1800 | 35938 | 12853300 | 10.74 | 10.74 | 49.88 | 0.11859 | 1.7852 | 1/266 / 3/964 / 6/570 |
| cum_f2 | B8 fold 1, 2 + cụm ~11% fold 1 | 25→32 | cum | 2 | 0.0002 | 64 | 132120576 | 2 | 1839 | 37134 | 12975641 | – | – | – | – | – | 112/256 / 110/1016 / 3/567 |
| cum_f2_p11 | Cụm ~11% fold 2 | 34 | cum | 2 | 0.0002 | 64 | 132120576 | 2 | 1839 | 37134 | 12975641 | 10.74 | 10.74 | 49.72 | 0.11966 | 1.7838 | 3/256 / 5/1016 / 4/567 |
| cum_f0_p11_lr1e4 | Kiểm lr 1e-4 (cụm) + pair fold 1 | 36 | cum | 0 | 0.0001 | 64 | 132120576 | 2 | 1837 | 37123 | 13061056 | 10.74 | 10.74 | 49.65 | 0.12029 | 1.7831 | 2/263 / 10/1011 / 8/563 |
| pair_f1 | Kiểm lr 1e-4 (cụm) + pair fold 1 | 36 | pair | 1 | 0.0002 | 64 | 132120576 | 2 | 1778 | 35931 | 12203089 | – | – | – | – | – | 94/234 / 101/1019 / 5/525 |
| pair_f1_p11 | Kiểm lr 1e-4 (cụm) + pair fold 1 | 36 | pair | 1 | 0.0002 | 64 | 132120576 | 2 | 1778 | 35931 | 12203089 | 10.86 | 10.86 | 49.84 | 0.11964 | 1.7838 | 0/234 / 9/1019 / 6/525 |
| pair_f2 | Pair fold 2 | 38 | pair | 2 | 0.0002 | 64 | 132120576 | 2 | 1859 | 37126 | 12314160 | – | – | – | – | – | 132/288 / 131/1027 / 3/544 |
| pair_f2_p11 | Pair fold 2 | 38 | pair | 2 | 0.0002 | 64 | 132120576 | 2 | 1859 | 37126 | 12314160 | 10.86 | 10.86 | 50.35 | 0.11740 | 1.7877 | 2/288 / 7/1027 / 3/544 |
| cum_cuoi | Bản cuối trên toàn pool | 43 | cum | – | 0.0002 | 64 | 132120576 | 2 | 1990 | 39734 | 15652689 | 10.74 | 10.74 | 40.77 | 0.17247 | 1.5299 | 1/207 / 2/923 / 7/860 |
| pair_cuoi | Bản cuối trên toàn pool | 43 | pair | – | 0.0002 | 64 | 132120576 | 2 | 2008 | 39724 | 14829898 | 10.86 | 10.86 | 41.42 | 0.16915 | 1.5334 | 3/237 / 8/931 / 6/840 |

`fold kiểm` là fold dùng để kiểm; huấn luyện bằng bốn fold còn lại của pool. Ô `Bước clip` là số bước bị cắt gradient / số bước, tách theo loại lô (lô toàn title / lô trộn / lô toàn mô tả); `–` là việc chạy trước khi ghi thống kê này. Dòng B6 ở vòng 1 chỉ có 1 epoch. Hai lần thử `lr1e-4_r16` và `lr1e-4_r16_attn` ở epoch 1 được sinh và chấm lại ở B7 từ adapter B6 (không train lại), nằm ở bảng kết quả bên dưới.

## 4. Kết quả từng việc (fold kiểm, 1.000 mẫu sinh)

| Việc | Điểm loss ep1 | Điểm loss ep2 | chrF chung (corpus) | chrF chung (theo câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 | Giây mỗi mẫu | Token/giây ep2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr5e-5_r16 | 0.4090 | – | – | – | – | – | – | – | – / – | – / – | – | – | 5836 |
| lr1e-4_r16 | 0.3900 | – | – | – | – | – | – | – | – / – | – / – | – | – | 5838 |
| lr2e-4_r16 | 0.3770 | – | – | – | – | – | – | – | – / – | – / – | – | – | 5839 |
| lr5e-5_r64 | 0.3808 | – | – | – | – | – | – | – | – / – | – / – | – | – | 5546 |
| lr1e-4_r64 | 0.3680 | 0.3587 | 75.51 | 76.29 | 60.21 | 90.81 | – | – | – / – | – / – | 3 | 0.372 | 5564 |
| lr2e-4_r64 | 0.3703 | 0.3512 | 75.76 | 76.62 | 60.22 | 91.30 | 75.90 | 75.62 | 895 / 77.17 | 456 / 77.79 | 1 | 0.327 | 5574 |
| lr1e-4_r16_attn | 0.4435 | – | – | – | – | – | – | – | – / – | – / – | – | – | 7781 |
| pair_f0 | 0.3798 | 0.3587 | 75.26 | 76.47 | 60.35 | 90.17 | 76.17 | 74.35 | 849 / 76.36 | 454 / 75.12 | 0 | 0.314 | 5619 |
| cum_f1 | 0.3881 | 0.3661 | 74.74 | 75.41 | 58.56 | 90.92 | 77.37 | 72.12 | 902 / 78.32 | 416 / 76.27 | 1 | 0.309 | 5563 |
| lr1e-4_r16_attn (epoch 1, chấm lại ở B7) | – | – | 71.54 | 72.77 | 55.77 | 87.31 | 71.12 | 71.95 | – / – | – / – | 4 | 0.297 | – |
| lr1e-4_r16 (epoch 1, chấm lại ở B7) | – | – | 73.89 | 74.98 | 58.34 | 89.43 | 73.50 | 74.28 | – / – | – / – | 2 | 0.327 | – |
| cum_f0_p11 | 0.3668 | 0.3487 | 76.14 | 76.73 | 60.36 | 91.92 | 76.43 | 75.86 | 895 / 79.05 | 456 / 78.20 | 1 | 0.331 | 5532 |
| cum_f0_p75 | 0.3806 | 0.3573 | 75.66 | 76.56 | 60.67 | 90.65 | 75.65 | 75.67 | 895 / 74.54 | 456 / 77.06 | 3 | 0.335 | 5532 |
| pair_f0_p11 | 0.3723 | 0.3514 | 75.16 | 76.39 | 59.39 | 90.94 | 76.00 | 74.33 | 849 / 78.68 | 454 / 75.91 | 1 | 0.316 | 5572 |
| pair_f0_p75 | 0.3921 | 0.3643 | 74.60 | 75.89 | 59.75 | 89.46 | 75.49 | 73.72 | 844 / 74.07 | 454 / 75.13 | 1 | 0.349 | 5574 |
| cum_f1_p11 | 0.3851 | 0.3592 | 75.14 | 75.81 | 58.89 | 91.40 | 78.24 | 72.05 | 907 / 80.45 | 416 / 76.34 | 0 | 0.309 | 5531 |
| cum_f2 | 0.3775 | 0.3538 | 74.16 | 75.74 | 60.33 | 87.99 | 76.44 | 71.88 | 911 / 73.40 | 473 / 68.51 | 1 | 0.344 | 5592 |
| cum_f2_p11 | 0.3687 | 0.3478 | 75.31 | 76.33 | 60.67 | 89.96 | 76.87 | 73.76 | 911 / 76.79 | 473 / 73.04 | 1 | 0.347 | 5534 |
| cum_f0_p11_lr1e4 | 0.3692 | 0.3571 | 75.97 | 76.76 | 60.29 | 91.65 | 75.97 | 75.97 | 895 / 78.27 | 456 / 77.97 | 1 | 0.329 | 5538 |
| pair_f1 | 0.3888 | 0.3697 | 74.80 | 75.63 | 59.40 | 90.21 | 77.64 | 71.96 | 846 / 76.98 | 422 / 74.91 | 1 | 0.296 | 5615 |
| pair_f1_p11 | 0.3846 | 0.3636 | 74.93 | 75.91 | 59.41 | 90.46 | 78.13 | 71.73 | 846 / 79.90 | 422 / 73.78 | 1 | 0.297 | 5571 |
| pair_f2 | 0.3856 | 0.3593 | 73.70 | 75.72 | 60.41 | 86.99 | 76.24 | 71.16 | 856 / 74.57 | 479 / 66.77 | 3 | 0.374 | 5633 |
| pair_f2_p11 | 0.3771 | 0.3536 | 74.66 | 76.19 | 60.74 | 88.58 | 77.25 | 72.08 | 858 / 76.76 | 479 / 69.82 | 1 | 0.356 | 5587 |
| cum_cuoi | 0.2504 | 0.1599 | – | – | – | – | – | – | – / – | – / – | – | – | 5525 |
| pair_cuoi | 0.2515 | 0.1655 | – | – | – | – | – | – | – / – | – / – | – | – | 5567 |

Việc không có chrF là việc chỉ chạy tới đó (vòng 1 B6 chỉ so loss, `cum_f1` bản dở). `Dòng mới` là dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train; `chrF chung (corpus)` là điểm chính dùng để so.

## 5. Loss không trọng số theo hướng × loại, ở epoch cuối

| Việc | Epoch | zh->vi title | zh->vi mo_ta | vi->zh title | vi->zh mo_ta |
|---|---|---|---|---|---|
| lr5e-5_r16 | 1 | 0.7832 | 0.0916 | 0.6412 | 0.1199 |
| lr1e-4_r16 | 1 | 0.7437 | 0.0813 | 0.6208 | 0.1142 |
| lr2e-4_r16 | 1 | 0.7170 | 0.0728 | 0.6085 | 0.1097 |
| lr5e-5_r64 | 1 | 0.7286 | 0.0756 | 0.6092 | 0.1100 |
| lr1e-4_r64 | 2 | 0.6669 | 0.0603 | 0.6042 | 0.1036 |
| lr2e-4_r64 | 2 | 0.6494 | 0.0543 | 0.5999 | 0.1013 |
| lr1e-4_r16_attn | 1 | 0.8565 | 0.1151 | 0.6721 | 0.1305 |
| pair_f0 | 2 | 0.6539 | 0.0544 | 0.6098 | 0.1169 |
| cum_f1 | 2 | 0.6323 | 0.0559 | 0.6749 | 0.1014 |
| cum_f0_p11 | 2 | 0.6545 | 0.0442 | 0.6005 | 0.0956 |
| cum_f0_p75 | 2 | 0.6553 | 0.0651 | 0.6018 | 0.1069 |
| pair_f0_p11 | 2 | 0.6558 | 0.0434 | 0.5952 | 0.1111 |
| pair_f0_p75 | 2 | 0.6595 | 0.0647 | 0.6082 | 0.1248 |
| cum_f1_p11 | 2 | 0.6332 | 0.0462 | 0.6627 | 0.0946 |
| cum_f2 | 2 | 0.6397 | 0.0574 | 0.5886 | 0.1296 |
| cum_f2_p11 | 2 | 0.6406 | 0.0477 | 0.5824 | 0.1207 |
| cum_f0_p11_lr1e4 | 2 | 0.6802 | 0.0475 | 0.6051 | 0.0957 |
| pair_f1 | 2 | 0.6383 | 0.0526 | 0.6722 | 0.1159 |
| pair_f1_p11 | 2 | 0.6411 | 0.0422 | 0.6628 | 0.1083 |
| pair_f2 | 2 | 0.6444 | 0.0492 | 0.5922 | 0.1515 |
| pair_f2_p11 | 2 | 0.6460 | 0.0407 | 0.5845 | 0.1434 |
| cum_cuoi | 2 | 0.3491 | 0.0230 | 0.2314 | 0.0362 |
| pair_cuoi | 2 | 0.3579 | 0.0230 | 0.2386 | 0.0426 |

## 6. So sánh theo cặp (bootstrap theo mục, khoảng tin cậy 95%)

`chênh` = mức đầu trừ mức sau như ghi ở cột 2. Khoảng chỉ tính dao động do chọn mẫu kiểm, KHÔNG tính dao động do huấn luyện lại (chưa chạy seed khác). Khoảng chứa 0 là hòa.

| Nhóm so sánh | Cặp / chỉ số | Chênh | Thấp | Cao |
|---|---|---|---|---|
| B6 chrF chung (theo câu), epoch 2 | lr2e-4_r64 - lr1e-4_r64 | 0.33 | -0.12 | 0.81 |
| B7 chrF chung, epoch 1, fold 0 | LoRA attention (q,k,v,o) − LoRA mọi lớp, cùng lr1e-4_r16 | -2.20 | -2.75 | -1.65 |
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản cum | ~11% − mốc ~50% | 0.11 | -0.39 | 0.61 |
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản cum | 75% − mốc ~50% | -0.06 | -0.51 | 0.42 |
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản pair | ~11% − mốc ~50% | -0.08 | -0.54 | 0.36 |
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản pair | 75% − mốc ~50% | -0.58 | -0.99 | -0.19 |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 1.43 | 0.54 | 2.32 |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | 0.79 | -0.27 | 1.93 |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | 0.14 | -0.99 | 1.34 |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | -0.31 | -1.24 | 0.64 |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | 0.24 | -0.86 | 1.35 |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 0.69 | 0.39 | 1.00 |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 1.36 | 0.57 | 2.19 |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | 0.84 | -0.54 | 2.15 |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | -0.96 | -1.81 | -0.15 |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | -0.72 | -1.57 | 0.13 |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | -1.19 | -2.20 | -0.25 |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 0.82 | 0.52 | 1.16 |
| Fold 1: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 1.95 | 1.13 | 2.82 |
| Fold 1: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | 0.90 | -0.16 | 2.00 |
| Fold 1: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | 0.33 | -0.64 | 1.30 |
| Fold 1: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | 0.28 | -0.54 | 1.14 |
| Fold 1: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | 0.93 | -0.07 | 1.95 |
| Fold 1: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 0.84 | 0.54 | 1.14 |
| Fold 2: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 2.69 | 1.76 | 3.69 |
| Fold 2: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | 1.23 | 0.23 | 2.35 |
| Fold 2: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | 0.33 | -0.46 | 1.16 |
| Fold 2: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | 0.20 | -0.61 | 1.11 |
| Fold 2: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | -0.38 | -1.35 | 0.50 |
| Fold 2: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 1.07 | 0.75 | 1.41 |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | dòng mới mô tả zh→vi (theo sản phẩm) | -0.29 | -0.97 | 0.38 |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | dòng mới vi→zh | 0.02 | -0.79 | 0.81 |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | title hai hướng (chrF corpus) | -0.08 | -0.86 | 0.67 |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | title hai hướng (theo mục) | 0.24 | -0.58 | 1.11 |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | title zh→vi (theo mục) | -0.58 | -1.47 | 0.24 |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | mô tả zh→vi toàn bộ (theo mục) | -0.27 | -0.55 | 0.02 |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | chrF chung (theo câu) | 0.03 | -0.43 | 0.47 |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 2.10 | 1.23 | 2.94 |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | -0.00 | -1.27 | 1.12 |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | 0.01 | -0.96 | 1.00 |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | 0.20 | -0.77 | 1.22 |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | 0.12 | -0.90 | 1.13 |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 0.88 | 0.62 | 1.14 |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | 0.28 | -0.26 | 0.80 |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 1.36 | 0.57 | 2.19 |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | 0.84 | -0.54 | 2.15 |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | -0.96 | -1.81 | -0.15 |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | -0.72 | -1.57 | 0.13 |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | -1.19 | -2.20 | -0.25 |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 0.82 | 0.52 | 1.16 |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | -0.08 | -0.54 | 0.36 |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 2.10 | 1.23 | 2.94 |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | -0.00 | -1.27 | 1.12 |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | 0.01 | -0.96 | 1.00 |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | 0.20 | -0.77 | 1.22 |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | 0.12 | -0.90 | 1.13 |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 0.88 | 0.62 | 1.14 |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | 0.28 | -0.26 | 0.80 |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | 1.58 | 0.72 | 2.44 |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | 0.17 | -1.14 | 1.55 |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | 0.34 | -0.65 | 1.25 |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | 0.14 | -0.74 | 1.03 |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | 0.68 | -0.29 | 1.66 |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | 0.94 | 0.58 | 1.36 |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | 0.47 | 0.03 | 0.94 |
| Bản cụm, fold 0: mức ~11% − mốc 1 : 1 | chrF chung theo câu | 0.11 | -0.39 | 0.61 |
| Bản cụm, fold 1: mức ~11% − mốc 1 : 1 | chrF chung theo câu | 0.40 | -0.05 | 0.85 |
| Bản cụm, fold 2: mức ~11% − mốc 1 : 1 | chrF chung theo câu | 0.59 | 0.16 | 1.05 |

**Tiêu chí 3 fold (bản cụm, ~11% − 1 : 1)** [ĐO, `kiem_3fold.json`]: chênh chrF chung corpus theo fold = 0.381, 0.402, 1.150; trung bình 0.644; độ lệch chuẩn 0.438; kết luận đổi tỉ lệ: **True** (cả 3 fold dương, trung bình 0.64 lớn hơn độ lệch chuẩn mẫu 0.44, không học lệch); điều kiện thêm fold 2: dat.

## 7. Ghi chú và sự cố cần nhớ khi bị hỏi

- cum_f0: các trạng thái B7/B8 chỉ ghi tham chiếu tới kết quả B6 cùng cấu hình (lr2e-4_r64, fold 0), không phải lần chạy riêng; ở bảng dưới việc này nằm ở dòng `lr2e-4_r64`.
- cum_f1: bản dở ở 'B7+B8 phần đầu' (epoch 1, đã chạy 1102 bước) được thay bằng bản xong ở 'B8 fold 1, 2 + cụm ~11% fold 1' (tiếp tục từ checkpoint). Thống kê clip gradient của cum_f1 chỉ tính phần chạy ở bản sau (698 bước cuối).
- lr2e-4_r64 (mốc cum_f0): zh→vi, vi→zh và dòng mới lấy từ lần chấm lại bản dịch B6 ở đợt thử tỉ lệ (`bao_cao_ti_le.json`, mốc); chrF chung chấm lại trùng điểm B6 75,7613 (độ lệch 0).
- Bản Kaggle 26 (nối B7+B8 phần hai) bị Nhật Anh hủy; `cum_f1` dừng ở bước 1.102/1.800 rồi được nối ở bản 32 từ checkpoint lưu trong notebook `ft-qwen3-4b-b78-luu` (có cả `opt.pt`, trạng thái tối ưu, và vị trí lô). Chạy khói đã kiểm nối đúng số bước, nhưng cum_f1 vẫn là một lần huấn luyện đi qua hai phiên và hai bản mã Kaggle (25 rồi 32).
- Mã `kaggle_train.py` bản cũ KHÔNG được lưu riêng: B6 (bản 20–22) và B7+B8 phần đầu (bản 25) chạy bằng bản mã trước lần sửa lúc 23:06 ngày 05/11/2026; từ bản 29 trở đi chạy bằng bản hiện tại (mtime 23:06 05/11, không đổi sau đó; bản Kaggle 34 còn nguyên trong `notebooks/kaggle_api/nlp-group3/nlp-group3.ipynb`). Lần sửa đó thêm (theo plan v0.6.1, bước 1): loss có trọng số theo tỉ lệ title, danh sách việc lấy từ cấu hình, chọn trạng thái tiến xa hơn. Em chưa chạy lại cùng một cấu hình bằng hai bản mã để so kết quả; bằng chứng duy nhất là phép thử trong phiên: với w = 1 loss có trọng số trùng `out.loss` của mô hình (3,4583 = 3,4583, ghi ở `thu_loss_1_1`), và bản dịch B6 chấm lại trên máy ra đúng điểm B6 (chỉ chứng minh bản dịch không bị đổi, không chứng minh đường mã).
- Gradient bị cắt (clip 1,0) khác nhau rõ giữa các việc: mốc 1 : 1 `cum_f2` bị cắt 112/256 bước lô toàn title, còn `cum_f2_p11` chỉ 3/256; mức 75% bị cắt gần như mọi bước title (262/263, 263/263). Tức ở mức 1 : 1 các lô title cũng đã bị cắt thường xuyên; thống kê này chưa có cho `cum_f0` và `pair_f0` (chạy trước khi ghi).
- Kết luận hiện tại: bản cụm đạt luật khóa 3 fold và chốt 10,74% title, lr 2e-4 (kiểm lr 1e-4 ở mức này KHÔNG đạt: chrF chung theo câu +0,03 [-0,43; +0,47]); bản pair đạt luật 3 fold riêng của pair (khóa trước ở WALKTHROUGH Prompt 804, `kiem_3fold_pair.py`) và Nhật Anh duyệt chốt 10,86%; tập test đã đo MỘT lần (mục 9): fine-tune thắng zero-shot ở mọi nhánh, mọi hướng; bản cuối đã huấn luyện (mục 10). CÁC ĐIỂM CHƯA được người hay LLM chấm so cặp (B9); chrF chỉ đo độ giống bản dịch máy dùng làm tham chiếu.
- Các mức tỉ lệ đã thử: ~11% (3 fold cho cả cụm và pair), ~50% (mốc, 3 fold cho cả hai bản), 75% (fold 0 cả hai bản). Mức 75% bị clip gradient gần như mọi bước title nên tác dụng thực thấp hơn danh nghĩa. Fold 0 cũng là nơi phát hiện tín hiệu dòng mới zh→vi nên chỉ fold 1, 2 là kiểm độc lập; luật pair 3 fold tính trung bình title nên một fold title tụt (fold 0: title hai hướng -0,96, zh→vi -1,19) có thể được fold khác bù.
- Sự cố khi vận hành, ghi để phản biện: tiến trình tự động trên máy (`tl3_tu_dong`) từng chết giữa chừng khi chờ khói; cổng khói được chạy tay bằng đúng hàm `kiem_khoi_tl3` rồi mới đẩy phiên thật. Phiên Kaggle chạy độc lập với máy nên không mất kết quả. Khói đo test lần đầu (bản 39) lỗi do đường dẫn manifest, đã sửa (bản 40). Script kiểm khói bản cuối em viết có ngưỡng chất lượng áp nhầm cho khói 15 bước nên báo không đạt dù cơ chế đều đạt (bản 42); ngưỡng đó được áp lại và đạt ở bản thật.

## 8. Dấu vân tay để chứng minh số không bị sửa

### Dữ liệu và phép chia

- HF `ntquang0410/zh-vie_ecom`, thư mục `data/LLM/bilingual_judged`, commit `82aa9c9`
  - `bo_cum_chon_deepseek.parquet`: `e0a625b2c1159133af24a8ae43c3ec4af4911ada68e10cf392920b8e49a20def`
  - `bo_pair_chon_deepseek.parquet`: `d4535c62ed1e3692f1563143d10c5c612aaec30d5e574eb9969ef003c627230d`
  - `bo_cum_danh_dau.parquet`: `945c441f213c0442c788110b836d8aca51f0b36fa7b1e70b87fc2957c38840c6`
  - `bo_pair_danh_dau.parquet`: `12b66f24e4bc3718e6c784a927c50c4f0a02c84c1079791bb96734a984bddc24`
- Phép chia `chia_tap_v2`: sha256 file `d66a39c7dbe76bd9f51524a4cb35f4192220641db4981ffcd0b4d3f6d0ee4fa3`, sha256 danh sách test `0d7c7c1535d115481657e6faa7aa206c1acad76219dcf31aa5964dbfc0e09101`, test 2558, pool 23017, seed 20261104, ngưỡng Jaccard {'title_zh': 0.5, 'title_vi': 0.5, 'mo_ta_zh': 0.7, 'mo_ta_vi': 0.7, 'mo_ta_zh_cum': 0.7, 'mo_ta_vi_cum': 0.7, 'chua_cung_hieu': 0.7}

### Mã (sha256 hiện tại)

| Tệp | Sửa lần cuối (ICT) | sha256 |
|---|---|---|
| kaggle_train.py | 07/11/2026 08:23 | `dc7b030fa10ede8f391a2376a0cc4ffd71adf28a6bc1db654fef7b951e344b88` |
| cham_diem.py | 07/11/2026 07:55 | `58c92797c4c43583fd7f2519509c02a4a356b324e37d1d983f26368f64d4ab12` |
| dong_goi_kaggle.py | 07/11/2026 08:23 | `2cd1c4b8f2ca5d302be1714f8890ecbe198f72a386388818ce1f521427d0f337` |
| chung.py | 04/11/2026 23:41 | `f6853cb9cef187c7af40cefe72c59adc87934dcb20842c0b3a26236c455d9645` |
| dung_mau.py | 04/11/2026 23:45 | `021ea59d55eca2e17f26ec8d602eefa3961477e2211e998e297f96b95e84675a` |
| tong_hop_ti_le.py | 06/11/2026 10:31 | `e63cd36e4e7737fa1ef58e83c58a99cde8fddd1f1ca5846ad9831b1eb50975de` |
| kiem_fold_ti_le.py | 06/11/2026 20:47 | `efeefd45bc2d96c148f59c36b0c3eb4f8a09928daf406d8ea533099bf86bc7d9` |
| kiem_3fold_ti_le.py | 06/11/2026 18:30 | `49afa81636bba9618b553992df044a50358bcbb722b76c074c9b772c787344b8` |
| kiem_dong_moi.py | 06/11/2026 09:28 | `1f0905c7445942b07111f4c127289cc5b8dad2b6d5fa62df72a6b7c53bfc3343` |
| tong_hop_b78.py | 05/11/2026 17:51 | `0cf1df0136efa6a1bb3e8f55bbf8f500920d3dc7d6b9a18a9f9b8c063272cd9d` |

### Bản dịch sinh ra (sinh_ep2.jsonl)

| Tệp | Byte | sha256 |
|---|---|---|
| kaggle_b6/ket_qua_cuoi/b6/lr1e-4_r64/sinh_ep2.jsonl | 946476 | `85302a373c9319a5c5cd8ccef3e1af1b02b2ea7ca6a3b2a51208527ee14deef6` |
| kaggle_b6/ket_qua_cuoi/b6/lr2e-4_r64/sinh_ep2.jsonl | 932776 | `935df8a51848de096eebfcac46f2b2607c473b1c3e66601efff97fa92c0da78e` |
| kaggle_b78/phien_1_20261105_181337/b78/pair_f0/sinh_ep2.jsonl | 917530 | `7ef70d0ce4c9d004aead679e47188f7ab727166b19e57462d3360a2d82a94357` |
| kaggle_b8f1/phien_1_20261106_102741/b78/cum_f1/sinh_ep2.jsonl | 926385 | `0c5335d801574dd0f9d945416daba1c5149be98a45ea4e10df530e7a797942b2` |
| kaggle_b8f1/phien_1_20261106_102741/b78/cum_f1_p11/sinh_ep2.jsonl | 928281 | `8f7bd4179fb7cd97e44cb5cdb261c8adc6f12b2b71abab19a58213cb51a0d7f7` |
| kaggle_b8f1/phien_1_20261106_102741/b78/cum_f2/sinh_ep2.jsonl | 996461 | `cc6c743175fe384941435df0705224ee3c0a5492d9182394bbd5fa493498b00a` |
| kaggle_b8f1/phien_1_20261106_102741/b78/pair_f0/sinh_ep2.jsonl | 917530 | `7ef70d0ce4c9d004aead679e47188f7ab727166b19e57462d3360a2d82a94357` |
| kaggle_b8f2/phien_1_20261106_165058/b8f2/cum_f2_p11/sinh_ep2.jsonl | 999139 | `78f0f06b6d64964be5ed4b79a698f0d0d086fc3815ec4c6eedff4edc5232da83` |
| kaggle_tl/phien_1_20261105_231941/tl/cum_f0_p11/sinh_ep2.jsonl | 934312 | `37136af7a0042805a0ffe3c117a7d496e2630e1163dee88365f22ef1633f490d` |
| kaggle_tl/phien_1_20261105_231941/tl/cum_f0_p75/sinh_ep2.jsonl | 933754 | `7523021c40aa4845505544036e6291809662c19983bfa43ecc59e6a163983082` |
| kaggle_tl/phien_1_20261105_231941/tl/pair_f0_p11/sinh_ep2.jsonl | 918798 | `905aba16a13af802070699351d7e42ead9481e1967ec28ab7683d5ce908da8c9` |
| kaggle_tl/phien_1_20261105_231941/tl/pair_f0_p75/sinh_ep2.jsonl | 921440 | `e3f173187be2187d0468ef0423f9767c7cbb2e0a96d9ad6dd69f7a3e05b7ee54` |
| kaggle_tl2/phien_1_20261106_210034/tl2/cum_f0_p11_lr1e4/sinh_ep2.jsonl | 933464 | `228b5c15b6dba2096e8f91fb7606c83735a003296dacd62f2faa99e5d68898f6` |
| kaggle_tl2/phien_1_20261106_210034/tl2/pair_f1/sinh_ep2.jsonl | 905243 | `2b6c49eb5175c8082d4b89846ffb3a344939fe8cb02bedb6458cc6ffe2344417` |
| kaggle_tl2/phien_1_20261106_210034/tl2/pair_f1_p11/sinh_ep2.jsonl | 905277 | `dbb30d1d0a63ec3d10a3814ee4a21a92ae4808a1c8580cd0463c92f49334af86` |
| kaggle_tl3/phien_1_20261107_073538/tl3/pair_f2/sinh_ep2.jsonl | 985257 | `c9fc11280c091655eca5acf7a096e1e146a6746cb875300de3695f4b3290796f` |
| kaggle_tl3/phien_1_20261107_073538/tl3/pair_f2_p11/sinh_ep2.jsonl | 983311 | `f6f0ae5270b8bf3e098a029ad420fd822d7c9185f6857ea3adc02033ced39936` |

### Kho lưu adapter (không GPU)

- `kaggle_b6/kho_luu_manifest/b6_luu/MANIFEST.json`: 21 file kèm sha256 trong tệp manifest.
- `kaggle_b78/kho_luu_manifest/MANIFEST_B78.json`: 11 file kèm sha256 trong tệp manifest.
- `kaggle_b8f2/kho_luu_manifest/MANIFEST_B8F2.json`: 5 file kèm sha256 trong tệp manifest.
- `kaggle_cuoi/kho_luu_manifest/MANIFEST_CUOI.json`: 7 file kèm sha256 trong tệp manifest.
- `kaggle_tl/kho_luu_manifest/MANIFEST_TL.json`: 17 file kèm sha256 trong tệp manifest.
- `kaggle_tl2/kho_luu_manifest/MANIFEST_TL2.json`: 13 file kèm sha256 trong tệp manifest.
- `kaggle_tl3/kho_luu_manifest/MANIFEST_TL3.json`: 9 file kèm sha256 trong tệp manifest.
- Notebook kho lưu trên Kaggle (tài khoản `trinhnhatanh`): `ft-qwen3-4b-b6-luu`, `ft-qwen3-4b-b78-luu`, `ft-qwen3-4b-tl-luu`, `ft-qwen3-4b-b8f1-luu`, `ft-qwen3-4b-b8f2-luu`, `ft-qwen3-4b-tl2-luu`, `ft-qwen3-4b-tl3-luu`, `ft-qwen3-4b-cuoi-luu` (adapter bản cuối). Dataset riêng tư `ft-qwen3-4b-pool` (mẫu pool) và `ft-qwen3-4b-do-khoa` (tập test, chỉ gắn vào phiên đo test). Đầu ra mọi phiên huấn luyện đã được lưu sang kho riêng trước khi đẩy phiên kế; bản đo test (bản 41) không có adapter, bản dịch và điểm đã tải về máy.

## 9. Đo test MỘT lần (bản 41): fine-tune adapter fold 0 ~11% title so với zero-shot

Luật khóa trước khi đo (`tong_hop_do_test.py`, WALKTHROUGH Prompt 806): mỗi nhánh, mỗi hướng, fine-tune thắng khi cận dưới khoảng 95% (bootstrap theo sản phẩm, seed 42) của chênh chrF theo mục > 0. Mẫu: 5.116 mục / 2.558 sản phẩm mỗi nhánh, greedy, cùng mẫu cho cả hai mô hình. chrF chỉ đo độ giống bản dịch máy dùng làm tham chiếu nên khoảng cách với zero-shot bị phóng đại so với chất lượng thật.

| Nhánh | Hướng | chrF zero-shot | chrF fine-tune | Chênh | Thấp | Cao | Thắng |
|---|---|---|---|---|---|---|---|
| cum | vi->zh | 41.49 | 75.63 | 34.14 | 33.30 | 34.95 | True |
| cum | zh->vi | 47.22 | 77.36 | 30.14 | 29.66 | 30.63 | True |
| pair | vi->zh | 43.92 | 75.41 | 31.50 | 30.73 | 32.24 | True |
| pair | zh->vi | 46.34 | 77.65 | 31.31 | 30.81 | 31.83 | True |

Cảnh báo theo luật: không có. Adapter dùng: `cum_f0_p11` (ce352c70…), `pair_f0_p11` (4d06b578…), kiểm hash theo manifest trước khi đo.


## 10. Bản cuối trên toàn pool (bản 43)

Huấn luyện trên toàn pool (23.017 sản phẩm), hai lần, cùng cấu hình đã chốt: ~11% title (cụm 10,74%, pair 10,86%), lr 2e-4, rank 64, 2 epoch. Cách xử lý title (Nhật Anh chọn 07/11/2026): LOẠI mẫu title của 6.386/23.017 sản phẩm bị `loc_title.py` gắn cờ, GIỮ mẫu mô tả; `w_title` tính lại để giữ đúng tỉ trọng. Không còn fold giữ lại nên không chấm chất lượng; chỉ sinh thử 100 mẫu của chính tập huấn luyện để bắt hỏng (kết quả trong bảng mục 3 và 4 và trạng thái `cuoi_trang_thai.json`). Adapter ở `ft-qwen3-4b-cuoi-luu` và đã tải về `ft_run/ban_cuoi/`, hash khớp manifest. Bản cuối KHÔNG đo lại trên test.

