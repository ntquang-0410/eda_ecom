# B5. Cụm ~11% fold 2 + luật 3 fold bản cụm

[← Mục lục](INDEX.md) · [← B4](B4_b8_fold_1_2_moc_1_1_cum_11_fold_1.md) · [B6 →](B6_kiem_lr_1e_4_cum_pair_fold_1.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B5 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 34 |
| Thời gian | 06/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_b8f2` |

## Mục đích

Chạy `cum_f2_p11` rồi áp luật 3 fold đã khóa trước để quyết đổi tỉ trọng cho bản cụm.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| Cụm ~11% fold 2 | 34 | 06/11/2026 16:51:18 | 06/11/2026 18:20:43 | 1.49 | 3.00 | có | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f2_p11 | cum | 2 | 0.00020 | 64 | 132.120.576 | 2 | 1.839 | 37.134 | 12.975.641 | 10.74% | 49.72% | 0.11966 | ep1: 5534 / ep2: 5534 |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f2_p11 | 0.3687 | 0.3478 | 75.31 | 76.33 | 60.67 | 89.96 | 76.87 | 73.76 | 911 / 76.79 | 473 / 73.04 | 1 |

Số bước bị cắt gradient (clip 1,0) / số bước, theo loại lô (toàn title / trộn / toàn mô tả):

| Việc | Title / trộn / mô tả |
|---|---|
| cum_f2_p11 | 3/256 / 5/1016 / 4/567 |

## So sánh theo cặp (bootstrap, khoảng 95%)

Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

| Nhóm | Cặp / chỉ số | Chênh [95%] |
|---|---|---|
| Fold 2: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +2.69 [+1.76; +3.69] |
| Fold 2: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | +1.23 [+0.23; +2.35] |
| Fold 2: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | +0.33 [-0.46; +1.16] |
| Fold 2: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | +0.20 [-0.61; +1.11] |
| Fold 2: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | -0.38 [-1.35; +0.50] |
| Fold 2: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +1.07 [+0.75; +1.41] |
| Bản cụm, fold 0: mức ~11% − mốc 1 : 1 | chrF chung theo câu | +0.11 [-0.39; +0.61] |
| Bản cụm, fold 1: mức ~11% − mốc 1 : 1 | chrF chung theo câu | +0.40 [-0.05; +0.85] |
| Bản cụm, fold 2: mức ~11% − mốc 1 : 1 | chrF chung theo câu | +0.59 [+0.16; +1.05] |

## Luật 3 fold của bản cụm (khóa trước khi chạy)

| Mục | Giá trị |
|---|---|
| Chênh chrF chung (corpus), ~11% − 1 : 1, fold 0 / 1 / 2 | +0.381, +0.402, +1.150 |
| Trung bình | 0.644 |
| Độ lệch chuẩn mẫu | 0.438 |
| Không học lệch (title, mô tả, zh→vi không tụt quá 1,0) | có |
| Kết luận đổi tỉ lệ | có |
| Lý do | cả 3 fold dương, trung bình 0.64 lớn hơn độ lệch chuẩn mẫu 0.44, không học lệch |

## Kết luận của phiên

- Bản cụm **đạt** luật 3 fold, chốt tỉ trọng title 10,74%.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_b8f2/kho_luu_manifest/ft-qwen3-4b-b8f2-luu.log` | 2.927 | `28a8a15627e3aa2a` |
| `ft_run/kaggle_b8f2/kho_luu_manifest/MANIFEST_B8F2.json` | 682 | `9c48454170a4144d` |
| `ft_run/kaggle_b8f2/phien_1_20261106_165058/b8f2/b8f2_trang_thai.json` | 9.637 | `39af5749aafaf725` |
| `ft_run/kaggle_b8f2/phien_1_20261106_165058/b8f2/cum_f2_p11/sinh_ep2.jsonl` | 999.139 | `78f0f06b6d64964b` |
| `ft_run/kaggle_b8f2/phien_1_20261106_165058/bao_cao_b8f2.json` | 13.366 | `15dd537e3bfacc97` |
| `ft_run/kaggle_b8f2/phien_1_20261106_165058/nlp-group3.log` | 9.623 | `ec3e28a584e5eaad` |
| `ft_run/kaggle_b8f2/phien_1_20261106_165058/tong_hop/kiem_3fold.json` | 4.361 | `c1b0bef30c9f5f5d` |
| `ft_run/kaggle_b8f2/phien_1_20261106_165058/tong_hop/kiem_fold2.json` | 29.546 | `0408b79fef4382b0` |
| `ft_run/kaggle_b8f2/phien_1_20261106_165058/tong_hop/kiem_fold2.md` | 1.362 | `aa286916f99d5c16` |
| `ft_run/kaggle_b8f2/tu_dong.log` | 439 | `9db541be78dc8dfc` |
