# D2. EcomGPT-7B, vòng sàng lọc (cụm + pair, 2 seed)

[← Mục lục](INDEX.md) · [← D1](D1_qwen3_8b_vong_sang_loc_cum_pair_2_seed.md) · [D3 →](D3_opennmt_transformer_nho_hoc_tu_dau_pair_dich_ban.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | D2 |
| Giai đoạn | D. So sánh năm mô hình (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-ecom` |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_quota_nlp-ecom` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_quota_nlp-ecom/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |

## Mục đích

Fine-tune EcomGPT-7B (ModelScope `damo/nlp_ecomgpt_multilingual-7B-ecom`, nền BLOOMZ-7B1, đổi sang bf16 safetensors) với khuôn Alpaca, attention eager, cùng danh sách mẫu.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_quota_nlp-ecom/quota_report.json`** (mô hình EcomGPT-7B, khuôn `ecomgpt`, attention `eager`, gradient checkpointing có)

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 16 | 2.048 | 191 | 1.386.960 | 0 | 10.79% | 10.62% | 45.13 | 82.52 | 78.78 | 64.12 | 12.78 | 63.82 | 64.41 | 0.4919 | 3070 | 30.0 |
| cum | natural | 123 | 16 | 2.048 | 192 | 1.386.154 | 0 | 10.79% | 10.62% | 46.58 | 82.46 | 78.87 | 64.77 | 12.78 | 64.52 | 65.02 | 0.3004 | 3066 | 30.0 |
| pair | natural | 42 | 16 | 2.048 | 81 | 626.553 | 0 | 10.34% | 10.64% | 38.12 | 69.80 | 66.64 | 57.33 | 13.69 | 53.96 | 60.70 | 0.5322 | 3425 | 30.0 |
| pair | natural | 123 | 16 | 2.048 | 81 | 626.750 | 0 | 10.34% | 10.70% | 38.78 | 73.09 | 69.66 | 56.23 | 13.69 | 55.94 | 56.52 | 0.4046 | 3426 | 30.0 |

Kiểm dữ liệu từng bộ trước khi lấy mẫu:

| Bộ | Trần token | Mẫu tất cả | Mẫu giữ | Mẫu bỏ | Dài nhất | Token đích toàn bộ | Ngân sách token đích |
|---|---|---|---|---|---|---|---|
| cum | 2.048 | 37.123 | 37.123 | 0 | 2.047 | 5.613.023 | 1.725.636 |
| pair | 2.048 | 331.296 | 331.296 | 0 | 1.907 | 5.861.464 | 817.840 |

## Điểm trong bảng năm mô hình (tổng hợp trên máy, `tong_hop_5mh.py`)

Eval fold 0 (tập đã dùng để chọn cấu hình, train = fold 1–4), cấu hình sàng lọc r16, **chưa đo trên test**. Pair: 1.198 mẫu (từng dòng); cụm: 1.200 mẫu. chrF corpus; `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi; `Macro` = trung bình đều hai hướng. Chênh so với Qwen3-4B (vòng sàng lọc) tính trên chrF câu trung bình, bootstrap theo sản phẩm (2.000 lần, cùng mẫu rút cho mọi mô hình). LLM là trung bình 2 seed. Số trên fold 0 bị ảnh hưởng bởi việc chọn.

| Bộ | Mô hình | Số lần chạy | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Chính (corpus) | Macro (corpus) | Chính (câu TB) | Chênh chính (câu) so với 4B [95%] | Chênh macro (câu) so với 4B [95%] | Sinh lặp (TB 4 nhóm) | Lệch chính giữa 2 seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | Qwen3-4B | 2 | 49.98 | 83.31 | 54.20 | 86.58 | 79.98 | 68.52 | 90.03 | mốc | mốc | 0.62% | 1.63 |
| pair | EcomGPT-7B | 2 | 38.45 | 71.45 | 44.18 | 73.03 | 68.15 | 56.78 | 85.23 | -4.81 [-5.95; -3.79] | -6.89 [-7.61; -6.15] | 1.75% | 3.03 |
| cum | Qwen3-4B | 2 | 54.19 | 90.44 | 56.33 | 86.54 | 86.81 | 71.87 | 87.65 | mốc | mốc | 0.42% | 0.20 |
| cum | EcomGPT-7B | 2 | 45.85 | 82.49 | 46.88 | 82.55 | 78.83 | 64.44 | 81.24 | -6.42 [-7.36; -5.55] | -6.34 [-6.98; -5.72] | 2.50% | 0.09 |

SacreBLEU / TER (trung bình các lần chạy; sacrebleu 2.6.0, zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`; BLEU kiểu mteval-v13a trùng SacreBLEU):

| Bộ | Mô hình | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả |
|---|---|---|---|---|---|
| pair | Qwen3-4B | 25.70 / 65.42 | 76.04 / 20.18 | 35.68 / 56.77 | 84.86 / 9.97 |
| pair | EcomGPT-7B | 15.32 / 83.72 | 58.81 / 45.18 | 25.28 / 72.99 | 68.94 / 31.19 |
| cum | Qwen3-4B | 31.35 / 59.12 | 86.73 / 12.30 | 39.04 / 48.14 | 87.58 / 9.87 |
| cum | EcomGPT-7B | 21.92 / 72.47 | 74.61 / 32.03 | 27.55 / 69.61 | 81.91 / 17.60 |

## Kết luận của phiên

- Kém Qwen3-4B rõ ở cả hai bộ (khoảng tin cậy xa 0); tỉ lệ sinh lặp cao hơn 4B (cột Sinh lặp).

## Ghi chú, giới hạn, sự cố

- Tokenizer BLOOM cho ít token hơn nên cùng danh sách mẫu lại ít bước hơn (cột Bước); chưa tách được lỗi do mô hình hay do ít bước và khuôn.
- Bảng tổng của cả năm mô hình ở phiên D6.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_nlp-ecom/nlp-ecom.log` | 11.760 | `88fef3a32a1e2409` |
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s123/adapter_config.json` | 1.072 | `083f42a7986bf2d3` |
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s123/README.md` | 5.290 | `2d994d21d4076b6b` |
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s123/sinh.jsonl` | 1.162.061 | `553a223519aafb28` |
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s42/adapter_config.json` | 1.072 | `083f42a7986bf2d3` |
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s42/README.md` | 5.290 | `2d994d21d4076b6b` |
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s42/sinh.jsonl` | 1.166.569 | `5531456b35075a52` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s123/adapter_config.json` | 1.072 | `083f42a7986bf2d3` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s123/README.md` | 5.290 | `2d994d21d4076b6b` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s123/sinh.jsonl` | 262.055 | `a0dcd21eb74861e1` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s42/adapter_config.json` | 1.072 | `083f42a7986bf2d3` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s42/README.md` | 5.290 | `2d994d21d4076b6b` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s42/sinh.jsonl` | 267.101 | `8a972cf822ba43c4` |
| `ft_run/kaggle_quota_nlp-ecom/quota_report.json` | 32.036 | `d1f70058781beb52` |

Adapter LoRA (trọng số) có trong output đã tải:

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s123/adapter_model.safetensors` | 125.863.064 | `4e3613e4e99321bb` |
| `ft_run/kaggle_quota_nlp-ecom/quota/cum_natural_s42/adapter_model.safetensors` | 125.863.064 | `d6da7246b96c6e3d` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s123/adapter_model.safetensors` | 125.863.064 | `35bfe6363a652d60` |
| `ft_run/kaggle_quota_nlp-ecom/quota/pair_natural_s42/adapter_model.safetensors` | 125.863.064 | `73e096324b9b5a91` |
