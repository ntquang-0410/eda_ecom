# B2. B7 + B8 phần đầu: LoRA attention, pair fold 0, cụm fold 1 (dở)

[← Mục lục](INDEX.md) · [← B1](B1_b6_do_sieu_tham_so_qwen3_4b_lr_rank.md) · [B3 →](B3_thu_ti_trong_title_fold_0_11_va_75_cum_va_pair.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B2 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 25 (bản 26 nối tiếp bị hủy) |
| Thời gian | 05/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_b78` |

## Mục đích

Chấm lại LoRA attention so với LoRA mọi lớp; huấn luyện bản pair fold 0 với cấu hình đã chọn; bắt đầu cụm fold 1.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| B7 + B8 phần đầu | 25 | 05/11/2026 18:13:55 | 05/11/2026 20:43:57 | 2.50 | 2.50 | không | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair_f0 | pair | 0 | 0.00020 | 64 | 132.120.576 | 2 | 1.843 | 37.114 | 12.395.741 | – | – | – | ep1: 5614 / ep2: 5619 |
| lr1e-4_r16_attn | cum | 0 | 0.00010 | 16 | – | 1 | – | – | – | – | – | – | – |
| lr1e-4_r16 | cum | 0 | 0.00010 | 16 | – | 1 | – | – | – | – | – | – | – |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pair_f0 | 0.3798 | 0.3587 | 75.26 | 76.47 | 60.35 | 90.17 | 76.17 | 74.35 | 849 / 76.36 | 454 / 75.12 | 0 |
| lr1e-4_r16_attn | – | – | 71.54 | 72.77 | 55.77 | 87.31 | 71.12 | 71.95 | – / – | – / – | 4 |
| lr1e-4_r16 | – | – | 73.89 | 74.98 | 58.34 | 89.43 | 73.50 | 74.28 | – / – | – / – | 2 |

## So sánh theo cặp (bootstrap, khoảng 95%)

Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

| Nhóm | Cặp / chỉ số | Chênh [95%] |
|---|---|---|
| B7 chrF chung, epoch 1, fold 0 | LoRA attention (q,k,v,o) − LoRA mọi lớp, cùng lr1e-4_r16 | -2.20 [-2.75; -1.65] |

## Kết luận của phiên

- LoRA chỉ attention kém LoRA mọi lớp rõ, nên giữ `target_modules = all-linear`.
- `cum_f1` dừng ở bước 1.102/1.800 do hết giờ; được nối ở B4 từ checkpoint.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_b78/kho_luu_manifest/ft-qwen3-4b-b78-luu.log` | 2.921 | `f26c35c6e660112f` |
| `ft_run/kaggle_b78/kho_luu_manifest/MANIFEST_B78.json` | 1.501 | `bf55ed7795c1f469` |
| `ft_run/kaggle_b78/phien_1_20261105_181337/b78/attn/lr1e-4_r16_attn_sinh.jsonl` | 937.506 | `d69ac06f33eca9fd` |
| `ft_run/kaggle_b78/phien_1_20261105_181337/b78/attn/lr1e-4_r16_sinh.jsonl` | 934.128 | `e1146fbb2bed2f9a` |
| `ft_run/kaggle_b78/phien_1_20261105_181337/b78/b78_trang_thai.json` | 38.719 | `9d151c5547b666ba` |
| `ft_run/kaggle_b78/phien_1_20261105_181337/b78/pair_f0/sinh_ep2.jsonl` | 917.530 | `7ef70d0ce4c9d004` |
| `ft_run/kaggle_b78/phien_1_20261105_181337/bao_cao_b78.json` | 50.284 | `5db0da1ba64ce5fb` |
| `ft_run/kaggle_b78/phien_1_20261105_181337/nlp-group3.log` | 13.279 | `ab3d552126824269` |
| `ft_run/kaggle_b78/tu_dong.log` | 1.487 | `204da4e6997837f6` |
