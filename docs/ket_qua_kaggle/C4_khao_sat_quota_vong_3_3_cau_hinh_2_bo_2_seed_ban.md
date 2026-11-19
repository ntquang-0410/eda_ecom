# C4. Khảo sát quota vòng 3: 3 cấu hình × 2 bộ × 2 seed (bảng đã chốt)

[← Mục lục](INDEX.md) · [← C3](C3_khao_sat_quota_vong_2_ngan_sach_token_co_dinh_ma.md) · [C5 →](C5_tai_lap_o_tai_khoan_2_check1.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | C4 |
| Giai đoạn | C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 55 |
| Thời gian | 10/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_quota_step1` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_quota_step1/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |

## Mục đích

Chốt bảng tỉ trọng: natural, balanced, title_heavy (cụm) và natural, balanced, attr_heavy (pair), seed 42 và 123, cùng ngân sách token.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_quota_step1/quota_report.json`**

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 16 | 2.048 | 241 | 1.725.697 | 0 | 10.79% | 10.79% | 54.60 | 90.50 | 86.91 | 72.05 | 43.95 | 72.55 | 71.55 | 0.0975 | 5834 | – |
| cum | natural | 123 | 16 | 2.048 | 240 | 1.726.199 | 0 | 10.79% | 10.79% | 53.77 | 90.38 | 86.72 | 71.70 | 43.95 | 72.07 | 71.33 | 0.2274 | 5847 | – |
| cum | balanced | 42 | 16 | 2.048 | 234 | 1.725.777 | 0 | 50.00% | 50.00% | 58.27 | 88.10 | 85.12 | 73.19 | 43.95 | 73.19 | 73.19 | 0.5582 | 6326 | – |
| cum | balanced | 123 | 16 | 2.048 | 234 | 1.725.796 | 0 | 50.00% | 50.00% | 57.69 | 87.99 | 84.96 | 73.03 | 43.95 | 72.84 | 73.22 | 0.4264 | 6329 | – |
| cum | title_heavy | 42 | 16 | 2.048 | 232 | 1.726.273 | 0 | 80.00% | 79.97% | 59.22 | 84.13 | 81.64 | 72.36 | 43.95 | 71.67 | 73.04 | 0.3939 | 6550 | – |
| cum | title_heavy | 123 | 16 | 2.048 | 233 | 1.726.006 | 0 | 80.00% | 79.99% | 59.02 | 84.59 | 82.03 | 72.50 | 43.95 | 71.80 | 73.20 | 0.4844 | 6540 | – |
| pair | natural | 42 | 16 | 2.048 | 107 | 817.899 | 0 | 10.34% | 10.34% | 49.80 | 84.24 | 80.79 | 68.55 | 41.97 | 67.02 | 70.08 | 0.2044 | 6708 | – |
| pair | natural | 123 | 16 | 2.048 | 107 | 817.932 | 0 | 10.34% | 10.34% | 50.16 | 82.38 | 79.16 | 68.49 | 41.97 | 66.27 | 70.70 | 0.1646 | 6729 | – |
| pair | balanced | 42 | 16 | 2.048 | 107 | 817.864 | 0 | 50.00% | 50.00% | 53.46 | 77.10 | 74.74 | 68.75 | 41.97 | 65.28 | 72.22 | 0.4671 | 6728 | – |
| pair | balanced | 123 | 16 | 2.048 | 107 | 817.912 | 0 | 50.00% | 50.00% | 54.53 | 78.54 | 76.14 | 68.45 | 41.97 | 66.54 | 70.36 | 0.4714 | 6729 | – |
| pair | attr_heavy | 42 | 16 | 2.048 | 107 | 817.931 | 0 | 15.00% | 15.00% | 51.58 | 81.14 | 78.18 | 67.66 | 41.97 | 66.36 | 68.97 | 0.2460 | 6721 | – |
| pair | attr_heavy | 123 | 16 | 2.048 | 107 | 817.903 | 0 | 15.00% | 15.00% | 51.19 | 82.69 | 79.54 | 68.71 | 41.97 | 66.94 | 70.49 | 0.1282 | 6710 | – |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| cum | 2.048 | 37.123 | 37.123 | 0 | 2.047 | 5.613.023 | 1.725.636 |
| pair | 2.048 | 331.296 | 331.296 | 0 | 1.907 | 5.861.464 | 817.840 |

- Đây là **vòng sàng lọc** (rank 16, một lượt, khoảng 4M token tính toán), không phải công thức cuối (rank 64, 2 epoch). Điểm chỉ dùng để so các cấu hình hoặc mô hình với nhau.
- Bộ `pair` trong khảo sát quota là **từng dòng** (title hoặc một dòng thuộc tính mỗi mẫu, 331.296 mẫu), khác adapter `pair_*` của công thức cuối (cả khối thuộc tính một lượt).

## Kết luận của phiên

- **natural** (title ~11%) cao nhất ở điểm Chính cho cả hai bộ **theo trung bình 2 seed**; ở pair, attr_heavy seed 123 nhỉnh +0,38 nhưng trung bình vẫn thấp hơn natural. Balanced chỉ hơn ở macro 50/50 bản cụm.
- Diễn giải và w hòa vốn: `docs/QUOTA_KET_LUAN.md`, `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md`. Dòng natural của bảng này là dòng Qwen3-4B trong bảng năm mô hình (D6).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_step1/nlp-group3.log` | 14.948 | `ccbc3e7c5a764aec` |
| `ft_run/kaggle_quota_step1/quota/cum_balanced_s123/sinh.jsonl` | 1.101.463 | `0a124e80c895122d` |
| `ft_run/kaggle_quota_step1/quota/cum_balanced_s42/sinh.jsonl` | 1.095.025 | `265d52ffac1ac0fd` |
| `ft_run/kaggle_quota_step1/quota/cum_natural_s123/sinh.jsonl` | 1.101.809 | `b2fdf289f3127b42` |
| `ft_run/kaggle_quota_step1/quota/cum_natural_s42/sinh.jsonl` | 1.096.023 | `2ab477222f1ebe72` |
| `ft_run/kaggle_quota_step1/quota/cum_title_heavy_s123/sinh.jsonl` | 1.108.555 | `9675f9b8d43a2ec0` |
| `ft_run/kaggle_quota_step1/quota/cum_title_heavy_s42/sinh.jsonl` | 1.103.413 | `c0d26acf67d86929` |
| `ft_run/kaggle_quota_step1/quota/pair_attr_heavy_s123/sinh.jsonl` | 254.131 | `e349c8218fcdfa48` |
| `ft_run/kaggle_quota_step1/quota/pair_attr_heavy_s42/sinh.jsonl` | 256.275 | `747ff2466c12857d` |
| `ft_run/kaggle_quota_step1/quota/pair_balanced_s123/sinh.jsonl` | 247.823 | `e72d49bd17dffbb3` |
| `ft_run/kaggle_quota_step1/quota/pair_balanced_s42/sinh.jsonl` | 245.191 | `6c923b7699c85164` |
| `ft_run/kaggle_quota_step1/quota/pair_natural_s123/sinh.jsonl` | 251.435 | `03cabbe06ea3d1a8` |
| `ft_run/kaggle_quota_step1/quota/pair_natural_s42/sinh.jsonl` | 253.265 | `798d356a8627347b` |
| `ft_run/kaggle_quota_step1/quota_report.json` | 90.297 | `fd2bbc880ba3051c` |
