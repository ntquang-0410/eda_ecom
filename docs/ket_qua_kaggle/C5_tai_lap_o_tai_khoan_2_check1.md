# C5. Tái lập ở tài khoản 2 (check1)

[← Mục lục](INDEX.md) · [← C4](C4_khao_sat_quota_vong_3_3_cau_hinh_2_bo_2_seed_ban.md) · [C6 →](C6_qwen3_4b_natural_voi_lr_1e_4.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | C5 |
| Giai đoạn | C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-group3` |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 11/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_quota_check1_acc2` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_quota_check1_acc2/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |

## Mục đích

Kiểm môi trường tài khoản 2 bằng cách chạy lại đúng cụm natural seed 42 của C4 và so điểm.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_quota_check1_acc2/quota_report.json`**

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 16 | 2.048 | 241 | 1.725.697 | 0 | 10.79% | 10.79% | 54.79 | 90.50 | 86.93 | 72.14 | 43.95 | 72.64 | 71.63 | 0.0971 | 5832 | – |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| cum | 2.048 | 37.123 | 37.123 | 0 | 2.047 | 5.613.023 | 1.725.636 |

## Đối chiếu với C4 (cùng cấu hình, cùng seed)

| Chỉ số | C4, tài khoản 1 (v55) | C5, tài khoản 2 | Chênh |
|---|---|---|---|
| Số bước | 241 | 241 | 0 |
| Token đích | 1.725.697 | 1.725.697 | 0 |
| chrF macro | 72.05 | 72.14 | 0.09 |
| Title zh→vi | 54.60 | 54.79 | 0.19 |
| Mô tả zh→vi | 90.50 | 90.50 | -0.01 |
| Token/giây | 5833.70 | 5831.90 | -1.80 |

## Kết luận của phiên

- Tái lập được C4: cùng số bước, cùng token đích, chênh chrF dưới 0,2 điểm (bảng đối chiếu), nên môi trường tài khoản 2 dùng được cho các phiên sau.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_check1_acc2/nlp-group3.log` | 4.881 | `5f90de6f4305785a` |
| `ft_run/kaggle_quota_check1_acc2/quota_report.json` | 8.260 | `3d01a7b9f938fa13` |
