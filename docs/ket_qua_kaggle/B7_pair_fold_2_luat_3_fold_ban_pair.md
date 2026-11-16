# B7. Pair fold 2 + luật 3 fold bản pair

[← Mục lục](INDEX.md) · [← B6](B6_kiem_lr_1e_4_cum_pair_fold_1.md) · [B8 →](B8_do_test_khoa_mot_lan.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B7 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 38 (bản 37 là khói) |
| Thời gian | 07/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_tl3` |

## Mục đích

Chạy `pair_f2`, `pair_f2_p11` rồi áp luật 3 fold riêng của pair.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| Pair fold 2 | 38 | 07/11/2026 02:31:18 | 07/11/2026 05:17:23 | 2.77 | 3.50 | có | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair_f2 | pair | 2 | 0.00020 | 64 | 132.120.576 | 2 | 1.859 | 37.126 | 12.314.160 | – | – | – | ep1: 5634 / ep2: 5633 |
| pair_f2_p11 | pair | 2 | 0.00020 | 64 | 132.120.576 | 2 | 1.859 | 37.126 | 12.314.160 | 10.86% | 50.35% | 0.11740 | ep1: 5590 / ep2: 5587 |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pair_f2 | 0.3856 | 0.3593 | 73.70 | 75.72 | 60.41 | 86.99 | 76.24 | 71.16 | 856 / 74.57 | 479 / 66.77 | 3 |
| pair_f2_p11 | 0.3771 | 0.3536 | 74.66 | 76.19 | 60.74 | 88.58 | 77.25 | 72.08 | 858 / 76.76 | 479 / 69.82 | 1 |

Số bước bị cắt gradient (clip 1,0) / số bước, theo loại lô (toàn title / trộn / toàn mô tả):

| Việc | Title / trộn / mô tả |
|---|---|
| pair_f2 | 132/288 / 131/1027 / 3/544 |
| pair_f2_p11 | 2/288 / 7/1027 / 3/544 |

## So sánh theo cặp (bootstrap, khoảng 95%)

Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

| Nhóm | Cặp / chỉ số | Chênh [95%] |
|---|---|---|
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +1.36 [+0.57; +2.19] |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | +0.84 [-0.54; +2.15] |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | -0.96 [-1.81; -0.15] |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | -0.72 [-1.57; +0.13] |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | -1.19 [-2.20; -0.25] |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +0.82 [+0.52; +1.16] |
| Bản pair, fold 0 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | -0.08 [-0.54; +0.36] |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +2.10 [+1.23; +2.94] |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | -0.00 [-1.27; +1.12] |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | +0.01 [-0.96; +1.00] |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | +0.20 [-0.77; +1.22] |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | +0.12 [-0.90; +1.13] |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +0.88 [+0.62; +1.14] |
| Bản pair, fold 1 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | +0.28 [-0.26; +0.80] |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới mô tả zh→vi (theo sản phẩm) | +1.58 [+0.72; +2.44] |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | dòng mới vi→zh | +0.17 [-1.14; +1.55] |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (chrF corpus) | +0.34 [-0.65; +1.25] |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title hai hướng (theo mục) | +0.14 [-0.74; +1.03] |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | title zh→vi (theo mục) | +0.68 [-0.29; +1.66] |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | mô tả zh→vi toàn bộ (theo mục) | +0.94 [+0.58; +1.36] |
| Bản pair, fold 2 (luật 3 fold pair): mức ~11% − mốc 1 : 1 | chrF chung (theo câu) | +0.47 [+0.03; +0.94] |

## Luật 3 fold của bản pair (khóa trước ở WALKTHROUGH Prompt 804, `kiem_3fold_pair.py`)

Chép nguyên kết luận do `kiem_3fold_pair.py` sinh ra:


Đây là phần đọc số chrF, không phải đánh giá của người. Dòng mới zh→vi đã được xem ở fold 0 nên chỉ fold 1, 2 là kiểm độc lập.

**Kết luận: dat**

- (a) ĐẠT: dòng mới zh->vi theo fold [1.36, 2.1, 1.58], cận dưới fold 2 +0.72
- (b) ĐẠT: chrF chung theo fold [-0.1, 0.13, 0.97], trung bình +0.33
- (c) ĐẠT: title hai hướng theo fold [-0.96, 0.01, 0.34], trung bình -0.21
- (d) ĐẠT: title zh->vi theo fold [-1.19, 0.12, 0.68], trung bình -0.13

## Kết luận của phiên

- Pair **đạt** luật 3 fold riêng; Nhật Anh duyệt chốt 10,86%.
- Lợi chủ yếu ở zh→vi; title fold 0 của pair tụt thật.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_tl3/cham/kiem_3fold_pair.json` | 74.976 | `17752b2fce7c8538` |
| `ft_run/kaggle_tl3/cham/kiem_3fold_pair.md` | 579 | `f651f1ae391c3a6d` |
| `ft_run/kaggle_tl3/kho_luu_manifest/ft-qwen3-4b-tl3-luu.log` | 3.022 | `857324dc3db88305` |
| `ft_run/kaggle_tl3/kho_luu_manifest/MANIFEST_TL3.json` | 1.228 | `e30697ee3ba7c6db` |
| `ft_run/kaggle_tl3/phien_1_20261107_073538/bao_cao_tl3.json` | 25.098 | `d2064b08c30f13f2` |
| `ft_run/kaggle_tl3/phien_1_20261107_073538/nlp-group3.log` | 15.102 | `bfb1508c4fc40381` |
| `ft_run/kaggle_tl3/phien_1_20261107_073538/tl3/pair_f2/sinh_ep2.jsonl` | 985.257 | `c9fc11280c091655` |
| `ft_run/kaggle_tl3/phien_1_20261107_073538/tl3/pair_f2_p11/sinh_ep2.jsonl` | 983.311 | `f6f0ae5270b8bf3e` |
| `ft_run/kaggle_tl3/phien_1_20261107_073538/tl3/tl3_trang_thai.json` | 18.651 | `005d58b8536f7fda` |
| `ft_run/kaggle_tl3/tu_dong.log` | 238 | `3c7f643ff66dd993` |
| `ft_run/kaggle_tl3/tu_dong_that.log` | 238 | `1413a67e1b30f52a` |
