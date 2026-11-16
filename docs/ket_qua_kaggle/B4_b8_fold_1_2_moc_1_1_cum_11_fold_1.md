# B4. B8 fold 1, 2 (mốc 1 : 1) + cụm ~11% fold 1

[← Mục lục](INDEX.md) · [← B3](B3_thu_ti_trong_title_fold_0_11_va_75_cum_va_pair.md) · [B5 →](B5_cum_11_fold_2_luat_3_fold_ban_cum.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B4 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 32 |
| Thời gian | 06/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_b8f1` |

## Mục đích

Nối `cum_f1` từ checkpoint bản 25; chạy `cum_f1_p11` (~11%) và `cum_f2` (mốc) để kiểm tỉ trọng trên fold khác.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| B8 fold 1, 2 mốc 1 : 1 + cụm ~11% fold 1 | 32 | 06/11/2026 10:28:00 | 06/11/2026 14:05:59 | 3.63 | 7.50 | có | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f1 | cum | 1 | 0.00020 | 64 | 132.120.576 | 2 | 1.800 | 35.938 | 12.853.300 | – | – | – | ep1: 5569 / ep2: 5563 |
| cum_f1_p11 | cum | 1 | 0.00020 | 64 | 132.120.576 | 2 | 1.800 | 35.938 | 12.853.300 | 10.74% | 49.88% | 0.11859 | ep1: 5533 / ep2: 5531 |
| cum_f2 | cum | 2 | 0.00020 | 64 | 132.120.576 | 2 | 1.839 | 37.134 | 12.975.641 | – | – | – | ep1: 5594 / ep2: 5592 |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f1 | 0.3881 | 0.3661 | 74.74 | 75.41 | 58.56 | 90.92 | 77.37 | 72.12 | 902 / 78.32 | 416 / 76.27 | 1 |
| cum_f1_p11 | 0.3851 | 0.3592 | 75.14 | 75.81 | 58.89 | 91.40 | 78.24 | 72.05 | 907 / 80.45 | 416 / 76.34 | 0 |
| cum_f2 | 0.3775 | 0.3538 | 74.16 | 75.74 | 60.33 | 87.99 | 76.44 | 71.88 | 911 / 73.40 | 473 / 68.51 | 1 |

Số bước bị cắt gradient (clip 1,0) / số bước, theo loại lô (toàn title / trộn / toàn mô tả):

| Việc | Title / trộn / mô tả |
|---|---|
| cum_f1 | 12/93 / 2/391 / 0/214 |
| cum_f1_p11 | 1/266 / 3/964 / 6/570 |
| cum_f2 | 112/256 / 110/1016 / 3/567 |

## So sánh theo cặp (bootstrap, khoảng 95%)

Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

| Nhóm | Cặp / chỉ số | Chênh [95%] |
|---|---|---|
| Fold 1: mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +1.95 [+1.13; +2.82] |
| Fold 1: mức ~11% − mốc 1 : 1 | dòng mới vi→zh | +0.90 [-0.16; +2.00] |
| Fold 1: mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | +0.33 [-0.64; +1.30] |
| Fold 1: mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | +0.28 [-0.54; +1.14] |
| Fold 1: mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | +0.93 [-0.07; +1.95] |
| Fold 1: mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +0.84 [+0.54; +1.14] |

## Ghi chú tự động từ nhật ký

- cum_f1: bản dở ở 'B7+B8 phần đầu' (epoch 1, đã chạy 1102 bước) được thay bằng bản xong ở 'B8 fold 1, 2 + cụm ~11% fold 1' (tiếp tục từ checkpoint). Thống kê clip gradient của cum_f1 chỉ tính phần chạy ở bản sau (698 bước cuối).

## Kết luận của phiên

- Fold 1: mức ~11% vẫn lợi ở dòng mới mô tả zh→vi, title không tụt.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/b78/attn/lr1e-4_r16_attn_sinh.jsonl` | 937.506 | `d69ac06f33eca9fd` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/b78/attn/lr1e-4_r16_sinh.jsonl` | 934.128 | `e1146fbb2bed2f9a` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/b78/b78_trang_thai.json` | 71.798 | `8e2caf6979bd819b` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/b78/cum_f1/sinh_ep2.jsonl` | 926.385 | `0c5335d801574dd0` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/b78/cum_f1_p11/sinh_ep2.jsonl` | 928.281 | `8f7bd4179fb7cd97` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/b78/cum_f2/sinh_ep2.jsonl` | 996.461 | `cc6c743175fe3849` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/b78/pair_f0/sinh_ep2.jsonl` | 917.530 | `7ef70d0ce4c9d004` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/bao_cao_b78.json` | 93.432 | `4a65cc65113e1e99` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/nlp-group3.log` | 18.573 | `c55653f1cc2bddf0` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/tong_hop/bao_cao_b78.json` | 4.637 | `67cdeac7bc3350b8` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/tong_hop/bao_cao_b78.md` | 5.622 | `a7be1a59ad1d6bfb` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/tong_hop/kiem_fold1.json` | 26.346 | `2a5b38d55a062b94` |
| `ft_run/kaggle_b8f1/phien_1_20261106_102741/tong_hop/kiem_fold1.md` | 1.362 | `e3715fb3db294d2c` |
| `ft_run/kaggle_b8f1/tu_dong.log` | 1.085 | `866179fe8c2c588d` |
