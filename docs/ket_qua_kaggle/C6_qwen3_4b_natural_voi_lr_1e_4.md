# C6. Qwen3-4B natural với lr 1e-4

[← Mục lục](INDEX.md) · [← C5](C5_tai_lap_o_tai_khoan_2_check1.md) · [D0 →](D0_danh_sach_mau_natural_dung_chung_cho_cac_llm_cha.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | C6 |
| Giai đoạn | C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-group3` (cụm), `honghuynanh/nlp-group3-pair` (pair) |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_quota_4b_lr1e4_acc2` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_quota_4b_lr1e4_acc2/cum/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |
| `ft_run/kaggle_quota_4b_lr1e4_acc2/pair/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |

## Mục đích

Thử lr 1e-4 (lr chung trong cấu hình của Quang) cho 4B ở vòng sàng lọc, so với lr 2e-4 của C4.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_quota_4b_lr1e4_acc2/cum/quota_report.json`**

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 16 | 2.048 | 241 | 1.725.697 | 0 | 10.79% | 10.79% | 52.30 | 89.56 | 85.84 | 70.76 | 43.95 | 70.93 | 70.58 | 0.1143 | 5835 | – |
| cum | natural | 123 | 16 | 2.048 | 240 | 1.726.199 | 0 | 10.79% | 10.79% | 51.89 | 89.13 | 85.40 | 70.31 | 43.95 | 70.51 | 70.10 | 0.2522 | 5848 | – |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| cum | 2.048 | 37.123 | 37.123 | 0 | 2.047 | 5.613.023 | 1.725.636 |

**`ft_run/kaggle_quota_4b_lr1e4_acc2/pair/quota_report.json`**

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | natural | 42 | 16 | 2.048 | 107 | 817.899 | 0 | 10.34% | 10.34% | 46.02 | 78.85 | 75.56 | 63.78 | 41.97 | 62.43 | 65.12 | 0.2458 | 6708 | – |
| pair | natural | 123 | 16 | 2.048 | 107 | 817.932 | 0 | 10.34% | 10.34% | 45.36 | 78.98 | 75.62 | 63.52 | 41.97 | 62.17 | 64.87 | 0.1945 | 6736 | – |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| pair | 2.048 | 331.296 | 331.296 | 0 | 1.907 | 5.861.464 | 817.840 |

## Kết luận của phiên

- lr 1e-4 kém lr 2e-4 rõ ở cả hai bộ (so với dòng natural của C4), vì vòng sàng lọc chỉ có 107–241 bước; giữ lr 2e-4 cho các LLM.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_4b_lr1e4_acc2/cum/nlp-group3.log` | 5.769 | `e85aaf472c294a0c` |
| `ft_run/kaggle_quota_4b_lr1e4_acc2/cum/quota_report.json` | 15.959 | `a192f5a2aaeae00a` |
| `ft_run/kaggle_quota_4b_lr1e4_acc2/pair/nlp-group3-pair.log` | 6.380 | `c8838744f34368ed` |
| `ft_run/kaggle_quota_4b_lr1e4_acc2/pair/quota_report.json` | 15.344 | `409f863607b7efed` |
