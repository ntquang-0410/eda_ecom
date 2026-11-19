# D5. Moses vòng 2: học đủ dữ liệu và khớp 4M token, kèm eval bản cụm

[← Mục lục](INDEX.md) · [← D4](D4_moses_vong_1_chon_cach_tach_chu_han_tung_chu_hay.md) · [D6 →](D6_tong_hop_nam_mo_hinh_tren_may_khong_gpu.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | D5 |
| Giai đoạn | D. So sánh năm mô hình (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-moses-v2-zhvi`, `honghuynanh/nlp-moses-v2-vizh` (CPU) |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_moses_v2_zhvi`, `data/processed/ft_run/kaggle_moses_v2_vizh` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_moses_v2_zhvi/moses_report.json` | – | – | – | – | – | 6.644 giây (1.85 giờ) | – | – |
| `ft_run/kaggle_moses_v2_vizh/moses_report.json` | – | – | – | – | – | 3.897 giây (1.08 giờ) | – | – |

## Mục đích

Hai biến thể: học toàn bộ dòng train, và học số dòng khớp ngân sách 4M token của LLM (seed 42); dịch cả eval bản cụm từng mảnh.

## Huấn luyện (`moses_report.json`)

**`ft_run/kaggle_moses_v2_zhvi/moses_report.json`**: cấu hình `{"huong": ["zh-vi"], "seg": ["char"], "mert_lap": 10, "bien_the": ["train", "khop4m_s42"], "eval_cum": true}`, tổng 6.644 giây (1.85 giờ).

| Hướng | Tách chữ Hán | Biến thể | Dòng train | Dòng sau lọc | MERT | chrF dev | n eval | Bản rỗng | Mảnh cụm (n mẫu) | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| zh-vi | char | train | 269.895 | 266.957 | có | 75.14 | 598 | 0 | 7.467 (600) | 3598 |
| zh-vi | char | khop4m_s42 | 37.528 | 37.113 | có | 71.82 | 598 | 0 | 7.467 (600) | 2974 |

**`ft_run/kaggle_moses_v2_vizh/moses_report.json`**: cấu hình `{"huong": ["vi-zh"], "seg": ["char"], "mert_lap": 10, "bien_the": ["train", "khop4m_s42"], "eval_cum": true}`, tổng 3.897 giây (1.08 giờ).

| Hướng | Tách chữ Hán | Biến thể | Dòng train | Dòng sau lọc | MERT | chrF dev | n eval | Bản rỗng | Mảnh cụm (n mẫu) | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| vi-zh | char | train | 61.401 | 60.674 | có | 71.46 | 600 | 0 | 2.407 (600) | 2270 |
| vi-zh | char | khop4m_s42 | 9.006 | 8.912 | có | 66.11 | 600 | 0 | 2.407 (600) | 1544 |

## Điểm trong bảng năm mô hình (tổng hợp trên máy, `tong_hop_5mh.py`)

Eval fold 0 (tập đã dùng để chọn cấu hình, train = fold 1–4), cấu hình sàng lọc r16, **chưa đo trên test**. Pair: 1.198 mẫu (từng dòng); cụm: 1.200 mẫu. chrF corpus; `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi; `Macro` = trung bình đều hai hướng. Chênh so với Qwen3-4B (vòng sàng lọc) tính trên chrF câu trung bình, bootstrap theo sản phẩm (2.000 lần, cùng mẫu rút cho mọi mô hình). LLM là trung bình 2 seed. Số trên fold 0 bị ảnh hưởng bởi việc chọn.

| Bộ | Mô hình | Số lần chạy | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Chính (corpus) | Macro (corpus) | Chính (câu TB) | Chênh chính (câu) so với 4B [95%] | Chênh macro (câu) so với 4B [95%] | Sinh lặp (TB 4 nhóm) | Lệch chính giữa 2 seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | Qwen3-4B | 2 | 49.98 | 83.31 | 54.20 | 86.58 | 79.98 | 68.52 | 90.03 | mốc | mốc | 0.62% | 1.63 |
| pair | Moses (train day du) | 1 | 54.93 | 83.65 | 51.66 | 82.07 | 80.78 | 68.08 | 91.90 | +1.86 [+0.67; +3.04] | +0.75 [+0.01; +1.50] | 0.00% | – |
| pair | Moses (khop 4M token) | 1 | 51.97 | 82.34 | 47.58 | 79.55 | 79.30 | 65.36 | 90.10 | +0.05 [-1.14; +1.23] | -1.82 [-2.60; -1.05] | 0.00% | – |
| cum | Qwen3-4B | 2 | 54.19 | 90.44 | 56.33 | 86.54 | 86.81 | 71.87 | 87.65 | mốc | mốc | 0.42% | 0.20 |
| cum | Moses (train day du) | 1 | 54.93 | 88.48 | 51.66 | 84.02 | 85.12 | 69.77 | 86.14 | -1.51 [-2.06; -1.02] | -2.04 [-2.69; -1.39] | 0.00% | – |
| cum | Moses (khop 4M token) | 1 | 51.97 | 85.71 | 47.58 | 80.68 | 82.33 | 66.48 | 83.48 | -4.17 [-4.79; -3.60] | -5.01 [-5.66; -4.38] | 0.08% | – |

SacreBLEU / TER (trung bình các lần chạy; sacrebleu 2.6.0, zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`; BLEU kiểu mteval-v13a trùng SacreBLEU):

| Bộ | Mô hình | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả |
|---|---|---|---|---|---|
| pair | Qwen3-4B | 25.70 / 65.42 | 76.04 / 20.18 | 35.68 / 56.77 | 84.86 / 9.97 |
| pair | Moses (train day du) | 28.83 / 60.94 | 69.78 / 27.81 | 28.99 / 61.67 | 81.12 / 25.20 |
| pair | Moses (khop 4M token) | 24.81 / 64.66 | 69.73 / 28.92 | 20.68 / 72.03 | 77.85 / 27.08 |
| cum | Qwen3-4B | 31.35 / 59.12 | 86.73 / 12.30 | 39.04 / 48.14 | 87.58 / 9.87 |
| cum | Moses (train day du) | 28.83 / 60.94 | 80.69 / 21.13 | 28.99 / 61.67 | 83.45 / 22.58 |
| cum | Moses (khop 4M token) | 24.81 / 64.66 | 77.78 / 24.19 | 20.68 / 72.03 | 79.97 / 25.49 |

## Kết luận của phiên

- Pair: Moses học đủ hơn 4B; khớp 4M token thì hòa. Cụm: Moses kém 4B ở cả hai biến thể.

## Ghi chú, giới hạn, sự cố

- MERT không tất định: hai lần chạy Moses học đủ lệch khoảng 0,25 điểm chính.
- Khoảng 87% dòng thuộc tính của eval pair đã có nguyên văn trong train, có lợi cho Moses.
- Bảng tổng của cả năm mô hình ở phiên D6.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_khop4m_s42/mert.log` | 45.469 | `5fb2c543587a1815` |
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_khop4m_s42/sinh.jsonl` | 105.167 | `dc859b670e841277` |
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_khop4m_s42/sinh_cum.jsonl` | 477.900 | `881a345719f60685` |
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_khop4m_s42/train_model.log` | 8.515 | `3d34952059e1f85e` |
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_train/mert.log` | 44.710 | `cd6ecc3c9a9e2098` |
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_train/sinh.jsonl` | 105.705 | `21728619890365cb` |
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_train/sinh_cum.jsonl` | 476.970 | `5cac356f02fd1274` |
| `ft_run/kaggle_moses_v2_zhvi/moses/zh-vi_char_train/train_model.log` | 8.497 | `ffd9bcc3f3c05d08` |
| `ft_run/kaggle_moses_v2_zhvi/moses_report.json` | 731 | `0726a08c005577e9` |
| `ft_run/kaggle_moses_v2_zhvi/nlp-moses-v2-zhvi.log` | 28.934 | `9c4323a24d00a7d1` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_khop4m_s42/mert.log` | 40.061 | `ea319ec6f3b522de` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_khop4m_s42/sinh.jsonl` | 68.963 | `df796d8429959d29` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_khop4m_s42/sinh_cum.jsonl` | 122.171 | `246dbe270922a98c` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_khop4m_s42/train_model.log` | 8.465 | `5df238ef22fb24b0` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_train/mert.log` | 45.180 | `78e5b1cb4accf139` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_train/sinh.jsonl` | 68.389 | `b9885db57fe14739` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_train/sinh_cum.jsonl` | 121.058 | `4e907dedef98f2ba` |
| `ft_run/kaggle_moses_v2_vizh/moses/vi-zh_char_train/train_model.log` | 12.714 | `96245ab1bd2668e0` |
| `ft_run/kaggle_moses_v2_vizh/moses_report.json` | 727 | `ca87c047fa867350` |
| `ft_run/kaggle_moses_v2_vizh/nlp-moses-v2-vizh.log` | 28.915 | `d04f7367d644df09` |
