# B6. Kiểm lr 1e-4 (cụm) + pair fold 1

[← Mục lục](INDEX.md) · [← B5](B5_cum_11_fold_2_luat_3_fold_ban_cum.md) · [B7 →](B7_pair_fold_2_luat_3_fold_ban_pair.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B6 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 36 |
| Thời gian | 06–07/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_tl2` |

## Mục đích

Kiểm lại lr ở mức tỉ trọng mới (lr 1e-4 so với 2e-4, cụm fold 0); chạy pair fold 1 ở mốc và ở ~11%.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| Kiểm lr 1e-4 (cụm) + pair fold 1 | 36 | 06/11/2026 21:00:52 | 07/11/2026 01:10:13 | 4.16 | 6.00 | có | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0_p11_lr1e4 | cum | 0 | 0.00010 | 64 | 132.120.576 | 2 | 1.837 | 37.123 | 13.061.056 | 10.74% | 49.65% | 0.12029 | ep1: 5531 / ep2: 5538 |
| pair_f1 | pair | 1 | 0.00020 | 64 | 132.120.576 | 2 | 1.778 | 35.931 | 12.203.089 | – | – | – | ep1: 5618 / ep2: 5615 |
| pair_f1_p11 | pair | 1 | 0.00020 | 64 | 132.120.576 | 2 | 1.778 | 35.931 | 12.203.089 | 10.86% | 49.84% | 0.11964 | ep1: 5568 / ep2: 5571 |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0_p11_lr1e4 | 0.3692 | 0.3571 | 75.97 | 76.76 | 60.29 | 91.65 | 75.97 | 75.97 | 895 / 78.27 | 456 / 77.97 | 1 |
| pair_f1 | 0.3888 | 0.3697 | 74.80 | 75.63 | 59.40 | 90.21 | 77.64 | 71.96 | 846 / 76.98 | 422 / 74.91 | 1 |
| pair_f1_p11 | 0.3846 | 0.3636 | 74.93 | 75.91 | 59.41 | 90.46 | 78.13 | 71.73 | 846 / 79.90 | 422 / 73.78 | 1 |

Số bước bị cắt gradient (clip 1,0) / số bước, theo loại lô (toàn title / trộn / toàn mô tả):

| Việc | Title / trộn / mô tả |
|---|---|
| cum_f0_p11_lr1e4 | 2/263 / 10/1011 / 8/563 |
| pair_f1 | 94/234 / 101/1019 / 5/525 |
| pair_f1_p11 | 0/234 / 9/1019 / 6/525 |

## So sánh theo cặp (bootstrap, khoảng 95%)

Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

| Nhóm | Cặp / chỉ số | Chênh [95%] |
|---|---|---|
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | dòng mới mô tả zh→vi (theo sản phẩm) | -0.29 [-0.97; +0.38] |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | dòng mới vi→zh | +0.02 [-0.79; +0.81] |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | title hai hướng (chrF corpus) | -0.08 [-0.86; +0.67] |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | title hai hướng (theo mục) | +0.24 [-0.58; +1.11] |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | title zh→vi (theo mục) | -0.58 [-1.47; +0.24] |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | mô tả zh→vi toàn bộ (theo mục) | -0.27 [-0.55; +0.02] |
| Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11% | chrF chung (theo câu) | +0.03 [-0.43; +0.47] |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +2.10 [+1.23; +2.94] |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | -0.00 [-1.27; +1.12] |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | +0.01 [-0.96; +1.00] |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | +0.20 [-0.77; +1.22] |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | +0.12 [-0.90; +1.13] |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +0.88 [+0.62; +1.14] |
| Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | +0.28 [-0.26; +0.80] |

## Kết luận của phiên

- lr 1e-4 **không đạt** luật lr, giữ lr 2e-4.
- Pair fold 1 đạt cổng dòng mới.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_tl2/cham/kiem_lr_cum_f0.json` | 29.326 | `8aab39bc41baf89e` |
| `ft_run/kaggle_tl2/cham/kiem_lr_cum_f0.md` | 1.634 | `b917473301699865` |
| `ft_run/kaggle_tl2/cham/kiem_pair_f1.json` | 24.351 | `5f8702accf370601` |
| `ft_run/kaggle_tl2/cham/kiem_pair_f1.md` | 1.561 | `560ff1f85a737a7a` |
| `ft_run/kaggle_tl2/kho_luu_manifest/ft-qwen3-4b-tl2-luu.log` | 3.135 | `1083fe7b511318a9` |
| `ft_run/kaggle_tl2/kho_luu_manifest/MANIFEST_TL2.json` | 1.806 | `aad1355ee316184e` |
| `ft_run/kaggle_tl2/phien_1_20261106_210034/bao_cao_tl2.json` | 37.102 | `3aeb0ceceed84efe` |
| `ft_run/kaggle_tl2/phien_1_20261106_210034/nlp-group3.log` | 20.440 | `1d2175652e835a5a` |
| `ft_run/kaggle_tl2/phien_1_20261106_210034/tl2/cum_f0_p11_lr1e4/sinh_ep2.jsonl` | 933.464 | `228b5c15b6dba209` |
| `ft_run/kaggle_tl2/phien_1_20261106_210034/tl2/pair_f1/sinh_ep2.jsonl` | 905.243 | `2b6c49eb5175c808` |
| `ft_run/kaggle_tl2/phien_1_20261106_210034/tl2/pair_f1_p11/sinh_ep2.jsonl` | 905.277 | `dbb30d1d0a63ec3d` |
| `ft_run/kaggle_tl2/phien_1_20261106_210034/tl2/tl2_trang_thai.json` | 27.855 | `b523bb87b4e457ad` |
| `ft_run/kaggle_tl2/tu_dong.log` | 1.504 | `f60b69dc4ce603cf` |
