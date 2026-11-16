# B3. Thử tỉ trọng title fold 0 (~11% và 75%, cụm và pair)

[← Mục lục](INDEX.md) · [← B2](B2_b7_b8_phan_dau_lora_attention_pair_fold_0_cum_fo.md) · [B4 →](B4_b8_fold_1_2_moc_1_1_cum_11_fold_1.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B3 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 29 |
| Thời gian | 05–06/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_tl` |

## Mục đích

Đổi tỉ trọng title trong loss từ mốc 1 : 1 theo mẫu (~50% loss) sang ~11% (bằng tỉ lệ token đích) và 75%, bằng hệ số `w_title`; đo trên fold 0 cho cả hai bộ.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| Thử tỉ lệ fold 0 | 29 | 05/11/2026 23:20:00 | 06/11/2026 05:00:04 | 5.67 | 7.50 | có | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0_p11 | cum | 0 | 0.00020 | 64 | 132.120.576 | 2 | 1.837 | 37.123 | 13.061.056 | 10.74% | 49.65% | 0.12029 | ep1: 5530 / ep2: 5532 |
| cum_f0_p75 | cum | 0 | 0.00020 | 64 | 132.120.576 | 2 | 1.837 | 37.123 | 13.061.056 | 75.00% | 49.65% | 3.03912 | ep1: 5530 / ep2: 5532 |
| pair_f0_p11 | pair | 0 | 0.00020 | 64 | 132.120.576 | 2 | 1.843 | 37.114 | 12.395.741 | 10.86% | 50.30% | 0.11818 | ep1: 5567 / ep2: 5572 |
| pair_f0_p75 | pair | 0 | 0.00020 | 64 | 132.120.576 | 2 | 1.843 | 37.114 | 12.395.741 | 75.00% | 50.30% | 2.95906 | ep1: 5567 / ep2: 5574 |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0_p11 | 0.3668 | 0.3487 | 76.14 | 76.73 | 60.36 | 91.92 | 76.43 | 75.86 | 895 / 79.05 | 456 / 78.20 | 1 |
| cum_f0_p75 | 0.3806 | 0.3573 | 75.66 | 76.56 | 60.67 | 90.65 | 75.65 | 75.67 | 895 / 74.54 | 456 / 77.06 | 3 |
| pair_f0_p11 | 0.3723 | 0.3514 | 75.16 | 76.39 | 59.39 | 90.94 | 76.00 | 74.33 | 849 / 78.68 | 454 / 75.91 | 1 |
| pair_f0_p75 | 0.3921 | 0.3643 | 74.60 | 75.89 | 59.75 | 89.46 | 75.49 | 73.72 | 844 / 74.07 | 454 / 75.13 | 1 |

Số bước bị cắt gradient (clip 1,0) / số bước, theo loại lô (toàn title / trộn / toàn mô tả):

| Việc | Title / trộn / mô tả |
|---|---|
| cum_f0_p11 | 1/263 / 5/1011 / 6/563 |
| cum_f0_p75 | 262/263 / 456/1011 / 3/563 |
| pair_f0_p11 | 1/263 / 5/1054 / 6/526 |
| pair_f0_p75 | 263/263 / 494/1054 / 5/526 |

## So sánh theo cặp (bootstrap, khoảng 95%)

Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

| Nhóm | Cặp / chỉ số | Chênh [95%] |
|---|---|---|
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản cum | ~11% − mốc ~50% | +0.11 [-0.39; +0.61] |
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản cum | 75% − mốc ~50% | -0.06 [-0.51; +0.42] |
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản pair | ~11% − mốc ~50% | -0.08 [-0.54; +0.36] |
| Thử tỉ lệ, fold 0, chrF chung theo câu, bản pair | 75% − mốc ~50% | -0.58 [-0.99; -0.19] |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +1.43 [+0.54; +2.32] |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | +0.79 [-0.27; +1.93] |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | +0.14 [-0.99; +1.34] |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | -0.31 [-1.24; +0.64] |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | +0.24 [-0.86; +1.35] |
| Dòng mới fold 0 (kiem_dong_moi), bản cum: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +0.69 [+0.39; +1.00] |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +1.36 [+0.57; +2.19] |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | +0.84 [-0.54; +2.15] |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | -0.96 [-1.81; -0.15] |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | -0.72 [-1.57; +0.13] |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | -1.19 [-2.20; -0.25] |
| Dòng mới fold 0 (kiem_dong_moi), bản pair: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +0.82 [+0.52; +1.16] |

## Kết luận của phiên

- Mức ~11% giúp rõ ở dòng mô tả mới hướng zh→vi (cả cụm và pair); chrF chung hòa.
- Mức 75% thua ở pair, hòa ở cụm, và bị cắt gradient gần như mọi bước title nên bỏ.

## Ghi chú, giới hạn, sự cố

- Fold 0 cũng là nơi phát hiện tín hiệu dòng mới, nên chỉ fold 1, 2 (B4–B7) là kiểm độc lập.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_tl/kho_luu_manifest/ft-qwen3-4b-tl-luu.log` | 3.222 | `90c40147fc37f3bd` |
| `ft_run/kaggle_tl/kho_luu_manifest/MANIFEST_TL.json` | 2.351 | `39f06fc75b7462cb` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/bao_cao_tl.json` | 49.211 | `e1541a1d29c23ee9` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/nlp-group3.log` | 26.699 | `4877264215034781` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tl/cum_f0_p11/sinh_ep2.jsonl` | 934.312 | `37136af7a0042805` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tl/cum_f0_p75/sinh_ep2.jsonl` | 933.754 | `7523021c40aa4845` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tl/pair_f0_p11/sinh_ep2.jsonl` | 918.798 | `905aba16a13af802` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tl/pair_f0_p75/sinh_ep2.jsonl` | 921.440 | `e3f173187be2187d` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tl/tl_trang_thai.json` | 37.170 | `852b51ab2d592596` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tong_hop/bao_cao_ti_le.json` | 11.669 | `2d7a74114b6f3df1` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tong_hop/bao_cao_ti_le.md` | 5.074 | `405ea2c8178c75c1` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tong_hop/kiem_dong_moi.json` | 43.764 | `62d7c2c742b91e7c` |
| `ft_run/kaggle_tl/phien_1_20261105_231941/tong_hop/kiem_dong_moi.md` | 4.196 | `08b23e7de7a4f463` |
| `ft_run/kaggle_tl/tu_dong.log` | 924 | `89a017308cbf2c67` |
