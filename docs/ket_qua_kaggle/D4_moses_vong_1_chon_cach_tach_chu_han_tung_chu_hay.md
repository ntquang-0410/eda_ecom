# D4. Moses vòng 1: chọn cách tách chữ Hán (từng chữ hay jieba)

[← Mục lục](INDEX.md) · [← D3](D3_opennmt_transformer_nho_hoc_tu_dau_pair_dich_ban.md) · [D5 →](D5_moses_vong_2_hoc_du_du_lieu_va_khop_4m_token_kem.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | D4 |
| Giai đoạn | D. So sánh năm mô hình (vòng sàng lọc) |
| Notebook Kaggle | `honghuynanh/nlp-moses-zhvi`, `honghuynanh/nlp-moses-vizh` (CPU) |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_moses_zhvi`, `data/processed/ft_run/kaggle_moses_vizh` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_moses_zhvi/moses_report.json` | – | – | – | – | – | 3.847 giây (1.07 giờ) | – | – |
| `ft_run/kaggle_moses_vizh/moses_report.json` | – | – | – | – | – | 3.806 giây (1.06 giờ) | – | – |

## Mục đích

Phrase-based SMT: eflomal + grow-diag-final-and, KenLM 5-gram, cụm tối đa 7, msd-bidirectional-fe, MERT trên dev 1.000 dòng; thử tách chữ Hán từng chữ và bằng jieba, chọn theo chrF dev.

## Huấn luyện (`moses_report.json`)

**`ft_run/kaggle_moses_zhvi/moses_report.json`**: cấu hình `{"huong": ["zh-vi"], "seg": ["char", "jieba"], "mert_lap": 10}`, tổng 3.847 giây (1.07 giờ).

| Hướng | Tách chữ Hán | Biến thể | Dòng train | Dòng sau lọc | MERT | chrF dev | n eval | Bản rỗng | Mảnh cụm (n mẫu) | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| zh-vi | char | train | 269.895 | 266.957 | có | 74.98 | 598 | 0 | – | 3138 |
| zh-vi | jieba | train | 269.895 | 267.074 | có | 74.51 | 598 | 0 | – | 639 |

**`ft_run/kaggle_moses_vizh/moses_report.json`**: cấu hình `{"huong": ["vi-zh"], "seg": ["char", "jieba"], "mert_lap": 10}`, tổng 3.806 giây (1.06 giờ).

| Hướng | Tách chữ Hán | Biến thể | Dòng train | Dòng sau lọc | MERT | chrF dev | n eval | Bản rỗng | Mảnh cụm (n mẫu) | Giây |
|---|---|---|---|---|---|---|---|---|---|---|
| vi-zh | char | train | 61.401 | 60.674 | có | 71.02 | 600 | 0 | – | 2046 |
| vi-zh | jieba | train | 61.401 | 60.798 | có | 70.49 | 600 | 0 | – | 1678 |

## Kết luận của phiên

- Chọn tách **từng chữ** theo chrF dev (bảng).

## Ghi chú, giới hạn, sự cố

- Moses biên dịch ở notebook `honghuynanh/nlp-moses-build` (`./bjam -j4 --with-boost=/usr -q`), gói `moses_bin_scripts.tgz` (xem INDEX, nhóm khói).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_moses_zhvi/moses/zh-vi_char/mert.log` | 44.988 | `bc6564001951e428` |
| `ft_run/kaggle_moses_zhvi/moses/zh-vi_char/sinh.jsonl` | 104.768 | `7735f5892bf04691` |
| `ft_run/kaggle_moses_zhvi/moses/zh-vi_char/train_model.log` | 7.987 | `4393924b98ab8593` |
| `ft_run/kaggle_moses_zhvi/moses/zh-vi_jieba/mert.log` | 13.788 | `6d15a3bc3deaa514` |
| `ft_run/kaggle_moses_zhvi/moses/zh-vi_jieba/sinh.jsonl` | 104.221 | `a1e08d52e43fcefb` |
| `ft_run/kaggle_moses_zhvi/moses/zh-vi_jieba/train_model.log` | 8.006 | `2ade6f63785f9484` |
| `ft_run/kaggle_moses_zhvi/moses_report.json` | 533 | `8344e8286e2fab2d` |
| `ft_run/kaggle_moses_zhvi/nlp-moses-zhvi.log` | 31.528 | `42f85b80ab471e92` |
| `ft_run/kaggle_moses_vizh/moses/vi-zh_char/mert.log` | 44.934 | `c2f088bc4c5746a3` |
| `ft_run/kaggle_moses_vizh/moses/vi-zh_char/sinh.jsonl` | 68.319 | `a2e78c94eafad385` |
| `ft_run/kaggle_moses_vizh/moses/vi-zh_char/train_model.log` | 12.202 | `d471550a1b8a6267` |
| `ft_run/kaggle_moses_vizh/moses/vi-zh_jieba/mert.log` | 44.957 | `0d2a0dd711e0822a` |
| `ft_run/kaggle_moses_vizh/moses/vi-zh_jieba/sinh.jsonl` | 69.096 | `2d73e9bfa14deb6b` |
| `ft_run/kaggle_moses_vizh/moses/vi-zh_jieba/train_model.log` | 12.032 | `d6029a9e286d9597` |
| `ft_run/kaggle_moses_vizh/moses_report.json` | 530 | `acdc863df9e054ef` |
| `ft_run/kaggle_moses_vizh/nlp-moses-vizh.log` | 31.349 | `5c661d1fdd6ae408` |
