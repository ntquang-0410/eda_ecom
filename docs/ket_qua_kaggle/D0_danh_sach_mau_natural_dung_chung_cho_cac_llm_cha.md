# D0. Danh sách mẫu natural dùng chung cho các LLM (chạy khô, CPU)

[← Mục lục](INDEX.md) · [← C6](C6_qwen3_4b_natural_voi_lr_1e_4.md) · [D1 →](D1_qwen3_8b_vong_sang_loc_cum_pair_2_seed.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | D0 |
| Giai đoạn | D. So sánh năm mô hình (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-dry-chon` |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_dry_chon` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_dry_chon/quota_dry.json` | cpu | – | 2.11.0+cpu | 5.20.2 | – | – | – | – |

## Mục đích

Xuất danh sách mẫu natural (seed 42 và 123, cụm và pair) bằng tokenizer Qwen để Qwen3-4B, Qwen3-8B và EcomGPT-7B học đúng cùng danh sách mẫu.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_dry_chon/quota_dry.json`** (mô hình Qwen3-4B, khuôn `qwen`, attention `sdpa`, gradient checkpointing không)

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | – | 2.048 | 216 | 1.725.697 | 0 | 10.79% | 10.79% | – | – | – | – | – | – | – | – | – | – |
| cum | natural | 123 | – | 2.048 | 216 | 1.726.199 | 0 | 10.79% | 10.79% | – | – | – | – | – | – | – | – | – | – |
| pair | natural | 42 | – | 2.048 | 103 | 817.899 | 0 | 10.34% | 10.34% | – | – | – | – | – | – | – | – | – | – |
| pair | natural | 123 | – | 2.048 | 103 | 817.932 | 0 | 10.34% | 10.34% | – | – | – | – | – | – | – | – | – | – |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| cum | 2.048 | 37.123 | 37.123 | 0 | 2.047 | 5.613.023 | 1.725.636 |
| pair | 2.048 | 331.296 | 331.296 | 0 | 1.907 | 5.861.464 | 817.840 |

## Kết luận của phiên

- Các file `chon_*_natural_s*.json` là danh sách mẫu được dùng ở D1, D2.

## Ghi chú, giới hạn, sự cố

- Chạy khô nên không có điểm; cột Bước là số bước ước tính.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_dry_chon/chon_cum_natural_s123.json` | 364.448 | `9040cb9205799fba` |
| `ft_run/kaggle_dry_chon/chon_cum_natural_s42.json` | 365.236 | `09f858265784e2a8` |
| `ft_run/kaggle_dry_chon/chon_pair_natural_s123.json` | 1.601.268 | `41eff75aa1a9f9e8` |
| `ft_run/kaggle_dry_chon/chon_pair_natural_s42.json` | 1.604.671 | `e8cdfb1e4c2918eb` |
| `ft_run/kaggle_dry_chon/nlp-dry-chon.log` | 6.125 | `62248b6aeb5facb4` |
| `ft_run/kaggle_dry_chon/quota_dry.json` | 12.743 | `f1a635f165260112` |
