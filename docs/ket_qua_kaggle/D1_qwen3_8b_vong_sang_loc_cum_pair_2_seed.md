# D1. Qwen3-8B, vòng sàng lọc (cụm + pair, 2 seed)

[← Mục lục](INDEX.md) · [← D0](D0_danh_sach_mau_natural_dung_chung_cho_cac_llm_cha.md) · [D2 →](D2_ecomgpt_7b_vong_sang_loc_cum_pair_2_seed.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | D1 |
| Giai đoạn | D. So sánh năm mô hình (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-8b` |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_quota_nlp-8b` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_quota_nlp-8b/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |

## Mục đích

Fine-tune Qwen3-8B (revision `b968826d`) cùng cấu hình sàng lọc và cùng danh sách mẫu với 4B.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_quota_nlp-8b/quota_report.json`** (mô hình Qwen3-8B, khuôn `qwen`, attention `sdpa`, gradient checkpointing có)

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 16 | 2.048 | 241 | 1.725.697 | 0 | 10.79% | 10.79% | 56.76 | 91.33 | 87.87 | 73.99 | 51.65 | 74.05 | 73.93 | 0.0788 | 2998 | 25.2 |
| cum | natural | 123 | 16 | 2.048 | 240 | 1.726.199 | 0 | 10.79% | 10.79% | 56.27 | 91.35 | 87.84 | 73.85 | 51.65 | 73.81 | 73.89 | 0.1883 | 3001 | 25.2 |
| pair | natural | 42 | 16 | 2.048 | 107 | 817.899 | 0 | 10.34% | 10.34% | 51.99 | 79.72 | 76.95 | 66.12 | 45.19 | 65.86 | 66.38 | 0.1668 | 3240 | 25.2 |
| pair | natural | 123 | 16 | 2.048 | 107 | 817.932 | 0 | 10.34% | 10.34% | 52.06 | 80.00 | 77.21 | 67.09 | 45.19 | 66.03 | 68.15 | 0.1398 | 3244 | 25.2 |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| cum | 2.048 | 37.123 | 37.123 | 0 | 2.047 | 5.613.023 | 1.725.636 |
| pair | 2.048 | 331.296 | 331.296 | 0 | 1.907 | 5.861.464 | 817.840 |

## Điểm trong bảng năm mô hình (tổng hợp trên máy, `tong_hop_5mh.py`)

Eval fold 0. Pair: 1.198 mẫu (từng dòng); cụm: 1.200 mẫu. chrF corpus; `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi; `Macro` = trung bình đều hai hướng. Chênh so với Qwen3-4B (vòng sàng lọc) tính trên chrF câu trung bình, bootstrap theo sản phẩm (2.000 lần, cùng mẫu rút cho mọi mô hình). LLM là trung bình 2 seed.

| Bộ | Mô hình | Số lần chạy | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Chính (corpus) | Macro (corpus) | Chính (câu TB) | Chênh chính (câu) so với 4B [95%] | Chênh macro (câu) so với 4B [95%] | Sinh lặp (TB 4 nhóm) | Lệch chính giữa 2 seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | Qwen3-4B | 2 | 49.98 | 83.31 | 54.20 | 86.58 | 79.98 | 68.52 | 90.03 | mốc | mốc | 0.62% | 1.63 |
| pair | Qwen3-8B | 2 | 52.03 | 79.86 | 56.70 | 77.84 | 77.08 | 66.61 | 90.48 | +0.44 [-0.24; +1.14] | +1.24 [+0.69; +1.82] | 0.75% | 0.26 |
| cum | Qwen3-4B | 2 | 54.19 | 90.44 | 56.33 | 86.54 | 86.81 | 71.87 | 87.65 | mốc | mốc | 0.42% | 0.20 |
| cum | Qwen3-8B | 2 | 56.52 | 91.34 | 58.84 | 88.98 | 87.86 | 73.92 | 88.68 | +1.03 [+0.71; +1.36] | +1.74 [+1.26; +2.28] | 0.38% | 0.03 |

SacreBLEU / TER (trung bình các lần chạy; sacrebleu 2.6.0, zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`; BLEU kiểu mteval-v13a trùng SacreBLEU):

| Bộ | Mô hình | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả |
|---|---|---|---|---|---|
| pair | Qwen3-4B | 25.70 / 65.42 | 76.04 / 20.18 | 35.68 / 56.77 | 84.86 / 9.97 |
| pair | Qwen3-8B | 27.70 / 63.05 | 72.78 / 22.35 | 39.29 / 50.52 | 83.00 / 13.87 |
| cum | Qwen3-4B | 31.35 / 59.12 | 86.73 / 12.30 | 39.04 / 48.14 | 87.58 / 9.87 |
| cum | Qwen3-8B | 33.10 / 54.30 | 87.78 / 11.16 | 43.94 / 46.44 | 89.67 / 7.58 |

## Kết luận của phiên

- Cụm: 8B hơn 4B rõ (khoảng tin cậy xa 0). Pair: 8B và 4B hòa theo chrF câu; chrF corpus của 8B thấp hơn do vài dòng dài (tên thành phần Latin mà 8B dịch sang chữ Hán).

## Ghi chú, giới hạn, sự cố

- Gradient checkpointing bật; VRAM đỉnh khoảng 25 GB (cột VRAM), nên không vừa một card 24 GB.
- Bảng tổng của cả năm mô hình ở phiên D6.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_nlp-8b/nlp-8b.log` | 9.104 | `50b11099586f22f8` |
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s123/adapter_config.json` | 1.095 | `5c5306468b2b51fd` |
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s123/README.md` | 5.282 | `31e164ec6bc146d1` |
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s123/sinh.jsonl` | 1.100.713 | `a43f9bc438e29f5d` |
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s42/adapter_config.json` | 1.095 | `5c5306468b2b51fd` |
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s42/README.md` | 5.282 | `31e164ec6bc146d1` |
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s42/sinh.jsonl` | 1.095.409 | `0d5f7ad4153d7e7f` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s123/adapter_config.json` | 1.095 | `5c5306468b2b51fd` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s123/README.md` | 5.282 | `31e164ec6bc146d1` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s123/sinh.jsonl` | 255.141 | `971b85d448e98f12` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s42/adapter_config.json` | 1.095 | `5c5306468b2b51fd` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s42/README.md` | 5.282 | `31e164ec6bc146d1` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s42/sinh.jsonl` | 249.825 | `5881f060a1f97edd` |
| `ft_run/kaggle_quota_nlp-8b/quota_report.json` | 31.894 | `b448b0d7a7e6f1a3` |

Adapter LoRA (trọng số) có trong output đã tải:

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s123/adapter_model.safetensors` | 174.655.536 | `d18e002d0fe4c513` |
| `ft_run/kaggle_quota_nlp-8b/quota/cum_natural_s42/adapter_model.safetensors` | 174.655.536 | `82a4df4aa771b22c` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s123/adapter_model.safetensors` | 174.655.536 | `cefe2c4df219eb1e` |
| `ft_run/kaggle_quota_nlp-8b/quota/pair_natural_s42/adapter_model.safetensors` | 174.655.536 | `bc45d77abbddb14f` |
