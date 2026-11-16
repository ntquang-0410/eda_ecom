# C3. Khảo sát quota vòng 2 (ngân sách token cố định, max_len 2.048)

[← Mục lục](INDEX.md) · [← C2](C2_c0_adapter_chinh_thuc_cham_tren_eval_cua_vong_sa.md) · [C4 →](C4_khao_sat_quota_vong_3_3_cau_hinh_2_bo_2_seed_ban.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | C3 |
| Giai đoạn | C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 53 |
| Thời gian | 10/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_quota_screen_v2`, `data/processed/ft_run/kernel_v53` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_quota_screen_v2/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |

## Mục đích

Làm lại vòng 1 với cùng ngân sách token đích cho mọi cấu hình (~8.000 token đích mỗi bước) và `max_len` 2.048 để không bỏ mẫu.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_quota_screen_v2/quota_report.json`**

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 16 | 2.048 | 241 | 1.725.697 | 0 | 10.79% | 10.79% | – | – | – | 72.12 | 43.95 | 72.68 | 71.55 | 0.0984 | 5820 | – |
| cum | balanced | 42 | 16 | 2.048 | 234 | 1.725.777 | 0 | 50.00% | 50.00% | – | – | – | 72.99 | 43.95 | 72.92 | 73.06 | 0.5578 | 6309 | – |
| cum | title_heavy | 42 | 16 | 2.048 | 232 | 1.726.273 | 0 | 80.00% | 79.97% | – | – | – | 72.35 | 43.95 | 71.82 | 72.89 | 0.3934 | 6529 | – |
| cum | desc_heavy | 42 | 16 | 2.048 | 238 | 1.726.110 | 0 | 5.00% | 5.00% | – | – | – | 71.36 | 43.95 | 69.75 | 72.97 | 0.1115 | 5990 | – |
| cum | tiki_heavy | 42 | 16 | 2.048 | 233 | 1.725.824 | 0 | 45.00% | 45.00% | – | – | – | 72.55 | 43.95 | 71.01 | 74.09 | 0.4068 | 6378 | – |
| pair | natural | 42 | 16 | 2.048 | 107 | 817.899 | 0 | 10.34% | 10.34% | – | – | – | 68.48 | 41.97 | 66.92 | 70.05 | 0.2049 | 6699 | – |
| pair | balanced | 42 | 16 | 2.048 | 107 | 817.864 | 0 | 50.00% | 50.00% | – | – | – | 68.82 | 41.97 | 65.35 | 72.29 | 0.4671 | 6715 | – |
| pair | title_heavy | 42 | 16 | 2.048 | 108 | 817.864 | 0 | 65.00% | 65.00% | – | – | – | 67.84 | 41.97 | 66.05 | 69.63 | 0.6415 | 6697 | – |
| pair | attr_heavy | 42 | 16 | 2.048 | 107 | 817.931 | 0 | 15.00% | 15.00% | – | – | – | 68.84 | 41.97 | 66.60 | 71.08 | 0.2463 | 6705 | – |
| pair | tiki_heavy | 42 | 16 | 2.048 | 107 | 817.877 | 0 | 40.00% | 40.00% | – | – | – | 66.84 | 41.97 | 62.84 | 70.85 | 0.4421 | 6715 | – |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| cum | 2.048 | 37.123 | 37.123 | 0 | 2.047 | 5.613.023 | 1.725.636 |
| pair | 2.048 | 331.296 | 331.296 | 0 | 1.907 | 5.861.464 | 817.840 |

- Đây là **vòng sàng lọc** (rank 16, một lượt, khoảng 4M token tính toán), không phải công thức cuối (rank 64, 2 epoch). Điểm chỉ dùng để so các cấu hình hoặc mô hình với nhau.
- Bộ `pair` trong khảo sát quota là **từng dòng** (title hoặc một dòng thuộc tính mỗi mẫu, 331.296 mẫu), khác adapter `pair_*` của công thức cuối (cả khối thuộc tính một lượt).

## Kết luận của phiên

- Một seed (42). C4 chạy lại 3 cấu hình mỗi bộ với 2 seed để chốt.

## Ghi chú, giới hạn, sự cố

- Báo cáo bản này chưa ghi điểm theo nhóm, nên không có cột title/mô tả và cột Chính.
- `kernel_v53/nlp-group3.ipynb` là notebook đúng bản 53, lưu để tái lập.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_screen_v2/nlp-group3.log` | 13.353 | `4229efaa8b86e3bf` |
| `ft_run/kaggle_quota_screen_v2/quota_report.json` | 27.742 | `6be5661410912e2a` |
