# Mục lục kết quả Kaggle

Mỗi phiên chạy thật một file. Các lần chạy khói/thử gom ở bảng cuối, không dùng làm kết quả.

## 1. Các phiên có kết quả

| # | Phiên | File | Ghi chú |
|---|---|---|---|
| 00 | Dịch dữ liệu song ngữ (song_ngu 1 → 4) | [`00_dich_du_lieu_song_ngu.md`](00_dich_du_lieu_song_ngu.md) | Đây là đầu vào của toàn bộ fine-tune; không phải phiên huấn luyện. |
| 01 | Kiểm môi trường và đo tốc độ (B5) | [`01_moi_truong_b5.md`](01_moi_truong_b5.md) | Không dùng làm kết quả; chỉ để chốt cấu hình chạy. |
| 02 | B6 — dò siêu tham số (fold 0, bản cụm) | [`02_b6_do_sieu_tham_so.md`](02_b6_do_sieu_tham_so.md) | Phiên 1–2 dừng ở trần 2,5 giờ; phiên 3 chạy nốt. |
| 03 | B7 + B8 — pair fold 0, cụm fold 1 | [`03_b78_pair_f0_cum_f1.md`](03_b78_pair_f0_cum_f1.md) | Dừng ở trần 2,5 giờ; bản 26 nối tiếp bị hủy, không có báo cáo. |
| 04 | Thử tỉ trọng title fold 0 (cụm và pair) | [`04_tl_ti_le_fold0.md`](04_tl_ti_le_fold0.md) | Phiên 5,67 giờ; kết quả là cơ sở chọn ~11%. |
| 05 | B8 fold 1 — mốc 1:1 và cụm ~11% | [`05_b8f1_fold1.md`](05_b8f1_fold1.md) | Nối cum_f1 từ bước 1.102 của bản 25. |
| 06 | B8 fold 2 — cụm ~11% | [`06_b8f2_fold2.md`](06_b8f2_fold2.md) | Hoàn tất luật 3 fold cho bản cụm. |
| 07 | Kiểm lr 1e-4 (cụm) và pair fold 1 | [`07_tl2_lr_va_pair_f1.md`](07_tl2_lr_va_pair_f1.md) | Lr 1e-4 không đạt; giữ 2e-4. |
| 08 | Pair fold 2 — mốc 1:1 và ~11% | [`08_tl3_pair_f2.md`](08_tl3_pair_f2.md) | Bản 37 là khói. |
| 09 | Đo test khóa MỘT lần | [`09_do_test_mot_lan.md`](09_do_test_mot_lan.md) | Không huấn luyện. Bản 39–40 là khói. Không đo lại sau lần này. |
| 10 | Bản cuối trên toàn pool (cum_cuoi, pair_cuoi) | [`10_cuoi_toan_pool.md`](10_cuoi_toan_pool.md) | Không còn fold giữ lại nên không có điểm chất lượng; không đo test. |
| 11 | Thử chéo dạng pair ↔ cụm | [`11_cheo_dang_pair_cum.md`](11_cheo_dang_pair_cum.md) | Dùng adapter fold 0 cum_f0_p11 / pair_f0_p11. |
| 12 | Đo tốc độ trên T4 fp16 (đóng gói máy chủ) | [`12_dong6_toc_do_t4.md`](12_dong6_toc_do_t4.md) | Chỉ đo tốc độ, không đo chất lượng. |
| 13 | OpenNMT — học từ đầu, đủ dữ liệu | [`13_onmt_hoc_du.md`](13_onmt_hoc_du.md) | Dùng để so trong bảng năm mô hình. |
| 14 | Moses — phrase-based, hai hướng | [`14_moses_hai_huong.md`](14_moses_hai_huong.md) | Moses pair mạnh vì ~87% dòng thuộc tính đã có nguyên văn trong train. |
| 15 | So sánh năm mô hình (eval fold 0) | [`15_nam_mo_hinh.md`](15_nam_mo_hinh.md) | Cấu hình sàng lọc rank 16, 4M token; không phải công thức cuối. |
| 16 | Quota screen 960 (bản cũ, có lỗi bỏ mẫu) | [`16_quota_screen_960.md`](16_quota_screen_960.md) | Có lỗi: trainer bỏ mẫu dài >960 trong khi ngạch coi như cắt. Không dùng để chốt. |
| 17 | Quota screen 2.048 (đã sửa lỗi độ dài) | [`17_quota_screen_2048.md`](17_quota_screen_2048.md) | Bản sửa đầu tiên; phát hiện metric 'Chính' sai trọng số ở pair, đã sửa ở bước sau. |
| 18 | Quota bước 1 — 3 cấu hình × 2 bộ × 2 seed | [`18_quota_step1_2seed.md`](18_quota_step1_2seed.md) | Kết quả dùng để chốt natural; xem docs/QUOTA_KET_LUAN.md. |
| 19 | C0 — chấm adapter official trên dev | [`19_c0_baseline.md`](19_c0_baseline.md) | Dùng để so candidate với bản đang dùng. |
| 20 | Demo — máy chủ dịch và đo lượng tử hoá | [`20_demo_may_chu.md`](20_demo_may_chu.md) | Q5_K_M đạt cổng; Q4_K_M bị loại. |
| 21 | Dry run ngạch và chọn mẫu | [`21_dry_run_ngach.md`](21_dry_run_ngach.md) | 0 GPU cho phần lớn; dùng để chặn lỗi trước khi chạy thật. |

## 2. Các lần chạy khói / thử (không dùng làm kết quả)

| Thư mục | Ghi chú |
|---|---|
| `kaggle_b6_thu` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_b6_thu2` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_b78_thu` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_cheo_khoi2` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_cuoi8b_khoi` | 8B bản cuối: T4 không đủ VRAM, dừng ở chốt chặn, không có kết quả |
| `kaggle_cuoi8b_khoi2` | 8B bản cuối: T4 không đủ VRAM, dừng ở chốt chặn, không có kết quả |
| `kaggle_cuoi_thu` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_do_test_thu` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_moses_build` | thử/dò, không dùng làm kết quả |
| `kaggle_moses_diag` | thử/dò, không dùng làm kết quả |
| `kaggle_moses_do` | thử/dò, không dùng làm kết quả |
| `kaggle_moses_khoi` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_moses_vizh` | thử/dò, không dùng làm kết quả |
| `kaggle_moses_zhvi` | thử/dò, không dùng làm kết quả |
| `kaggle_nguyen_nhan` | thử/dò, không dùng làm kết quả |
| `kaggle_onmt_cum` | thử/dò, không dùng làm kết quả |
| `kaggle_onmt_khoi` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_smoke_mh` | thử/dò, không dùng làm kết quả |
| `kaggle_tl3_thu` | chạy khói/thử, không dùng làm kết quả |
| `kaggle_tl_thu` | chạy khói/thử, không dùng làm kết quả |
