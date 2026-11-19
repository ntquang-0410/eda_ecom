# D3. OpenNMT Transformer nhỏ, học từ đầu (pair + dịch bản cụm)

[← Mục lục](INDEX.md) · [← D2](D2_ecomgpt_7b_vong_sang_loc_cum_pair_2_seed.md) · [D4 →](D4_moses_vong_1_chon_cach_tach_chu_han_tung_chu_hay.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | D3 |
| Giai đoạn | D. So sánh năm mô hình (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-onmt`, `honghuynanh/nlp-onmt-cum` (T4, có Internet) |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_onmt_that`, `data/processed/ft_run/kaggle_onmt_cum` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_onmt_that/onmt_report.json` | – | – | – | – | – | 4.007 giây (1.11 giờ) | – | – |
| `ft_run/kaggle_onmt_cum/onmt_cum_report.json` | – | – | – | – | – | 194 giây (0.05 giờ) | – | – |

## Mục đích

Mốc NMT cổ điển: Transformer d_model 256, 4 lớp, 8 đầu, ff 1024, SentencePiece unigram 16k mỗi ngôn ngữ, Noam lr đỉnh 2e-3, tối đa 20 epoch + dừng sớm, trung bình 5 checkpoint, beam 5; học trên dòng của bản pair.

## Huấn luyện (`kaggle_onmt_that/onmt_report.json`)

Cấu hình chung: `{"huong": ["zh-vi", "vi-zh"], "bien_the": ["train"], "so_epoch": 20, "seed": 42}`. Tổng thời gian phiên: 4.007 giây (1.11 giờ).

| Hướng | Biến thể | Dòng train | Token SP train | Bước | Bước/epoch | Warmup | lr đỉnh | learning_rate Noam | Checkpoint trung bình | Bước đã chạy | Giây train | n eval |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zh-vi | train | 269.895 | 3.778.854 | 22.740 | 1.137 | 2.000 | 0.0020 | 1.4311 | 5 | 10.233 | 2263 | 598 |
| vi-zh | train | 61.401 | 934.205 | 5.620 | 281 | 562 | 0.0020 | 0.7586 | 5 | 5.620 | 1617 | 600 |

Chấm ngay trên Kaggle, bản pair (`cham_onmt.json`, chrF corpus):

| zh→vi title | zh→vi mô tả | Chính (0,1/0,9) | vi→zh title | vi→zh mô tả | Macro |
|---|---|---|---|---|---|
| 48.27 | 77.28 | 74.38 | 29.64 | 64.38 | 54.89 |

## Dịch eval bản cụm (`kaggle_onmt_cum/onmt_cum_report.json`)

Không huấn luyện thêm: dùng checkpoint trung bình ở trên, dịch từng mảnh thuộc tính rồi nối bằng "; ".

| Hướng | Số mảnh | Số mẫu | Checkpoint |
|---|---|---|---|
| zh-vi | 7.467 | 600 | `/kaggle/input/notebooks/honghuynanh/nlp-onmt/onmt/zh-vi_train/avg.pt` |
| vi-zh | 2.407 | 600 | `/kaggle/input/notebooks/honghuynanh/nlp-onmt/onmt/vi-zh_train/avg.pt` |

Thời gian phiên cụm: 194 giây (0.05 giờ).

## Điểm trong bảng năm mô hình (tổng hợp trên máy, `tong_hop_5mh.py`)

Eval fold 0 (tập đã dùng để chọn cấu hình, train = fold 1–4), cấu hình sàng lọc r16, **chưa đo trên test**. Pair: 1.198 mẫu (từng dòng); cụm: 1.200 mẫu. chrF corpus; `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi; `Macro` = trung bình đều hai hướng. Chênh so với Qwen3-4B (vòng sàng lọc) tính trên chrF câu trung bình, bootstrap theo sản phẩm (2.000 lần, cùng mẫu rút cho mọi mô hình). LLM là trung bình 2 seed. Số trên fold 0 bị ảnh hưởng bởi việc chọn.

| Bộ | Mô hình | Số lần chạy | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Chính (corpus) | Macro (corpus) | Chính (câu TB) | Chênh chính (câu) so với 4B [95%] | Chênh macro (câu) so với 4B [95%] | Sinh lặp (TB 4 nhóm) | Lệch chính giữa 2 seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | Qwen3-4B | 2 | 49.98 | 83.31 | 54.20 | 86.58 | 79.98 | 68.52 | 90.03 | mốc | mốc | 0.62% | 1.63 |
| pair | OpenNMT (tu dau, train day du) | 1 | 48.27 | 77.28 | 29.64 | 64.38 | 74.38 | 54.89 | 90.43 | +0.39 [-1.09; +1.90] | -8.33 [-9.40; -7.26] | 0.25% | – |
| cum | Qwen3-4B | 2 | 54.19 | 90.44 | 56.33 | 86.54 | 86.81 | 71.87 | 87.65 | mốc | mốc | 0.42% | 0.20 |
| cum | OpenNMT (tu dau, train day du) | 1 | 47.01 | 84.34 | 29.64 | 73.89 | 80.61 | 58.72 | 82.08 | -5.57 [-6.42; -4.85] | -12.42 [-13.29; -11.57] | 3.33% | – |

SacreBLEU / TER (trung bình các lần chạy; sacrebleu 2.6.0, zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`; BLEU kiểu mteval-v13a trùng SacreBLEU):

| Bộ | Mô hình | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả |
|---|---|---|---|---|---|
| pair | Qwen3-4B | 25.70 / 65.42 | 76.04 / 20.18 | 35.68 / 56.77 | 84.86 / 9.97 |
| pair | OpenNMT (tu dau, train day du) | 27.13 / 59.64 | 72.48 / 25.68 | 16.51 / 75.20 | 68.12 / 26.71 |
| cum | Qwen3-4B | 31.35 / 59.12 | 86.73 / 12.30 | 39.04 / 48.14 | 87.58 / 9.87 |
| cum | OpenNMT (tu dau, train day du) | 26.08 / 60.18 | 80.02 / 21.91 | 16.51 / 75.20 | 76.03 / 19.75 |

## Kết luận của phiên

- Pair: hòa Qwen3-4B theo chrF câu trung bình (+0,39 [−1,09; +1,90]) nhưng kém theo chrF corpus chính (74,38 so 79,98) và macro câu (−8,33 [−9,40; −7,26]); cụm kém −5,57 [−6,42; −4,85].
- Trên **dòng mô tả mới** (P1, eval fold 0), OpenNMT kém Qwen3-4B −8,29 [−15,49; −0,88] (pair zh→vi) và −10,07 [−11,68; −8,30] (cụm zh→vi); trên **dòng khuôn** là +2,12 (pair) và −1,32 (cụm).
- Kết quả cho thấy độ nhạy theo cấu hình và theo loại dòng, **không** kết luận về paradigm NMT.

## Ghi chú, giới hạn, sự cố

- Chạy trên T4 vì cần Internet để cài thư viện và torch 2.2.2 không hỗ trợ GPU Blackwell.
- Bản cụm không huấn luyện riêng: dịch từng mảnh thuộc tính rồi nối.
- Eval fold 0 là tập đã dùng để chọn cấu hình (B6, tỉ trọng title, 5 mô hình, adapter demo); số trên fold 0 bị ảnh hưởng bởi việc chọn, chưa đo test.
- Bảng tổng của cả năm mô hình ở phiên D6.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_onmt_that/cham_onmt.json` | 6.946 | `ed018d51fee4cc87` |
| `ft_run/kaggle_onmt_that/nlp-onmt.log` | 36.535 | `b7171870b61db3da` |
| `ft_run/kaggle_onmt_that/onmt/spm_data/vi.txt` | 18.992.405 | `78ce6a77eddc39c9` |
| `ft_run/kaggle_onmt_that/onmt/spm_data/zh.txt` | 11.772.634 | `bdb841d829af5b25` |
| `ft_run/kaggle_onmt_that/onmt/vi-zh_train/dev.json` | 39.374 | `2cf75710dcc29057` |
| `ft_run/kaggle_onmt_that/onmt/vi-zh_train/eval.json` | 43.189 | `96e54e2a312dc37c` |
| `ft_run/kaggle_onmt_that/onmt/vi-zh_train/sinh.jsonl` | 65.868 | `88d87d1950e41e90` |
| `ft_run/kaggle_onmt_that/onmt/vi-zh_train/train.json` | 2.550.448 | `5b1f3ec480884304` |
| `ft_run/kaggle_onmt_that/onmt/vi-zh_train/train.log` | 111.058 | `56f3e71e99e60d32` |
| `ft_run/kaggle_onmt_that/onmt/vi-zh_train_train.log` | 111.058 | `56f3e71e99e60d32` |
| `ft_run/kaggle_onmt_that/onmt/zh-vi_train/dev.json` | 71.661 | `9bf19c6c3f5a8c60` |
| `ft_run/kaggle_onmt_that/onmt/zh-vi_train/eval.json` | 38.548 | `81413dc37dc556eb` |
| `ft_run/kaggle_onmt_that/onmt/zh-vi_train/sinh.jsonl` | 95.748 | `f8ecfba050a9b414` |
| `ft_run/kaggle_onmt_that/onmt/zh-vi_train/train.json` | 16.205.280 | `cd9078fe3b367507` |
| `ft_run/kaggle_onmt_that/onmt/zh-vi_train/train.log` | 30.237 | `0507ad2ae2e48f75` |
| `ft_run/kaggle_onmt_that/onmt/zh-vi_train_train.log` | 30.237 | `0507ad2ae2e48f75` |
| `ft_run/kaggle_onmt_that/onmt_report.json` | 6.205 | `ad760c20bec8cf84` |
| `ft_run/kaggle_onmt_cum/nlp-onmt-cum.log` | 8.167 | `5561559d51835c06` |
| `ft_run/kaggle_onmt_cum/onmt_cum/vi-zh/cum.json` | 131.602 | `70ec81fe7fe5bef2` |
| `ft_run/kaggle_onmt_cum/onmt_cum/vi-zh/sinh_cum.jsonl` | 118.555 | `b2708a570739bc11` |
| `ft_run/kaggle_onmt_cum/onmt_cum/zh-vi/cum.json` | 281.541 | `7cc84c48aa57d48a` |
| `ft_run/kaggle_onmt_cum/onmt_cum/zh-vi/sinh_cum.jsonl` | 454.451 | `d98bf3d3d241eb00` |
| `ft_run/kaggle_onmt_cum/onmt_cum_report.json` | 492 | `b94354e673c5bff2` |
