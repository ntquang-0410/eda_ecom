# Kế hoạch và kết quả: fine-tune 3 LLM và huấn luyện 2 mô hình cổ điển (cụm và pair)

Cập nhật 12/11/2026. Nhãn: **[ĐO]** đã đo trên dữ liệu thật; **[ƯỚC]** ước tính; **[CHỜ]** chờ quyết định. Số lẻ dùng dấu chấm trong bảng.

## 1. Mục tiêu
So sánh năm mô hình dịch Trung ↔ Việt thương mại điện tử (chiều chính zh→vi, dùng cho extension trên trang 1688) trên hai bộ dữ liệu đã qua Judge: **bản cụm** (cụm) và **bản pair** (pair), mỗi bộ 25.575 sản phẩm (`ntquang0410/zh-vie_ecom`), để chọn mô hình cho bản cuối.

| Nhóm | Mô hình | Cách huấn luyện |
|---|---|---|
| LLM, fine-tune bằng LoRA | Qwen3-4B, Qwen3-8B, EcomGPT-7B | LoRA r16, lr 2e-4, tất cả lớp tuyến tính |
| Cổ điển, huấn luyện từ đầu | OpenNMT (Transformer nhỏ), Moses (phrase-based SMT) | không có checkpoint tiền huấn luyện để fine-tune |

## 2. Trọng số đóng góp title : mô tả (đã chốt)
Title ≈ 11% token đích, mô tả ≈ 89% (cụm 10.74%, pair 10.86%); hệ số loss `w_title` 0.17247 (cụm) và 0.16915 (pair) chỉ dùng cho công thức cuối. Mọi ô so sánh dùng lấy mẫu "natural" (title ≈ 10.3–10.8%). Chi tiết ở `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md`. Điểm chính zh→vi = 0.10 · title + 0.90 · mô tả.

## 3. Quy trình đo (giống nhau cho cả năm mô hình)
- Huấn luyện trên fold 1–4 của pool; đo trên 1.200 mẫu eval cố định của fold 0 (300 mỗi nhóm hướng × loại), không đụng tập test.
- LLM: lấy mẫu natural đúng 4M token tính toán, một lượt, 2 seed (42, 123). Tokenizer lấy mẫu luôn là Qwen nên cả ba LLM học cùng danh sách mẫu.
- Cổ điển: dữ liệu từng dòng (title và từng dòng thuộc tính) của bản pair; đo trên đúng các mẫu eval (pair: 1.198 mẫu; cụm: dịch từng thuộc tính rồi nối bằng "; ").
- Bộ chấm chung `cham_diem.py` (chrF++ cho zh→vi, chrF ký tự cho vi→zh), bootstrap theo sản phẩm trên chrF câu trung bình (`tong_hop_5mh.py`).

## 4. Cấu hình
**LLM** (khác cấu hình chung của Quang, đã thống nhất): lr 2e-4 (lr 1e-4 kém hẳn vì mình chỉ có 107–241 bước, **[ĐO]** macro cụm 70.53 so với 71.87), `max_len` 2.048 (16–20% mô tả vượt 960 token). EcomGPT dùng khuôn Alpaca của Quang, attention eager.
**OpenNMT**: Transformer d_model 256, 4 lớp, 8 đầu, ff 1024; batch 4096 × 2; Noam với lr đỉnh cố định 2e-3; tối đa 20 epoch, dừng sớm theo valid ppl, trung bình 5 checkpoint cuối; SentencePiece unigram 16k mỗi ngôn ngữ; beam 5.
**Moses**: tách zh từng chữ Hán (thắng `jieba` trên dev), căn chỉnh eflomal (model 3, 4-4-4) + grow-diag-final-and, KenLM 5-gram, cụm tối đa 7, msd-bidirectional-fe, MERT trên dev 1.000 dòng.
Chạy trên Kaggle: LLM ở RTX Pro 6000 (qua cuộc thi ARC Prize, nhóm đã được ban tổ chức đồng ý), OpenNMT ở T4 (cần Internet để cài, và torch 2.2.2 không hỗ trợ Blackwell), Moses ở CPU.

## 5. Kết quả [ĐO]
### Bản pair (eval từng dòng, 1.198 mẫu)
| Mô hình | Title | Mô tả | Chính | Macro | Chênh chính (câu) so với 4B [95%] |
|---|---|---|---|---|---|
| Qwen3-4B | 49.98 | 83.31 | 79.98 | 68.52 | mốc |
| Qwen3-8B | 52.03 | 79.86 | 77.08 | 66.61 | +0.44 [−0.24; +1.14] |
| EcomGPT-7B | 38.45 | 71.45 | 68.15 | 56.78 | −4.81 [−5.95; −3.79] |
| OpenNMT | 48.27 | 77.28 | 74.38 | 54.89 | +0.39 [−1.09; +1.90] |
| Moses (train đủ 270k dòng) | 54.93 | 83.65 | 80.78 | 68.08 | +1.86 [+0.67; +3.04] |
| Moses (khớp 4M token) | 51.97 | 82.34 | 79.30 | 65.36 | +0.05 [−1.14; +1.23] |

### Bản cụm (eval mẫu đầy đủ, 1.200 mẫu)
| Mô hình | Title | Mô tả | Chính | Macro | Chênh chính (câu) so với 4B [95%] |
|---|---|---|---|---|---|
| Qwen3-4B | 54.19 | 90.44 | 86.81 | 71.87 | mốc |
| **Qwen3-8B** | 56.52 | 91.34 | 87.86 | 73.92 | **+1.03 [+0.71; +1.36]** |
| EcomGPT-7B | 45.85 | 82.49 | 78.83 | 64.44 | −6.42 [−7.36; −5.55] |
| OpenNMT | 47.01 | 84.34 | 80.61 | 58.72 | −5.57 [−6.42; −4.85] |
| Moses (train đủ) | 54.93 | 88.48 | 85.12 | 69.77 | −1.51 [−2.06; −1.02] |
| Moses (khớp 4M token) | 51.97 | 85.71 | 82.33 | 66.48 | −4.17 [−4.79; −3.60] |

Zero-shot macro (pair / cụm): 4B 41.97 / 43.95; 8B 45.19 / 51.65; EcomGPT 13.69 / 12.78.

## 6. Nhận xét
1. **Cụm (mẫu đầy đủ):** Qwen3-8B tốt nhất, hơn 4B rõ (khoảng tin cậy xa 0), 4B hơn Moses, EcomGPT và OpenNMT kém xa.
2. **Pair (từng dòng):** Moses train đủ nhỉnh hơn 4B, nhưng khi khớp ngân sách 4M token thì ngang 4B; 8B ngang 4B (chrF corpus của 8B thấp hơn vì vài dòng dài, ví dụ danh sách thành phần mà tham chiếu giữ nguyên tên Latin còn 8B dịch sang tiếng Trung).
3. Moses mạnh ở dòng ngắn lặp nhiều (khoảng 87% dòng thuộc tính eval đã có nguyên văn trong train) nhưng mất lợi thế khi dịch cả mô tả có ngữ cảnh.
4. OpenNMT yếu ở vi→zh (title 29.6) vì chỉ có 61 nghìn dòng; EcomGPT yếu cả hai hướng và sinh lặp 2–5% (chưa tách được lỗi do mô hình hay do ít bước và khuôn Alpaca).

## 7. Giới hạn
Tham chiếu là bản dịch máy; eval 1.200 mẫu cố định, một fold; hai mô hình cổ điển chỉ một lần chạy và học trên dòng của bản pair (bản cụm khác pair khoảng 1% mảnh); MERT không tất định (hai lần chạy Moses train đủ lệch 0.25 điểm chính); khoảng tin cậy chỉ bao gồm dao động chọn mẫu, không gồm dao động huấn luyện lại; EcomGPT chạy ít bước hơn do token BLOOM ít hơn (cụm 191, pair 81 bước so với 241 và 107); chưa đánh giá bằng người hay LLM-judge (B9); tập test chưa dùng cho năm mô hình này.

## 8. Việc tiếp theo
- **[CHỜ]** Chạy công thức cuối (rank 64, 2 epoch, `w_title`) cho Qwen3-8B bản cụm. Đề xuất: bản cụm dùng 8B, bản pair giữ 4B (đã có adapter cuối 07/11/2026). Ước ~5–6 giờ RTX Pro 6000 cho 8B [ƯỚC].
- **[CHỜ]** Đo test một lần cho mô hình được chọn.
- **[CHỜ]** B9: chấm so cặp bằng LLM.
- Quang xác nhận vi→zh chỉ và nhóm quyết bản cụm cho mô hình cổ điển có cần huấn luyện riêng không.
- Thu hồi token Kaggle tài khoản 2 khi xong.

## 9. Nơi lưu
Kết quả và bản dịch: `data/processed/ft_run/` (`tong_hop_5mh/`, `tong_hop_5mh_cum/`, `kaggle_quota_nlp-8b/`, `kaggle_quota_nlp-ecom/`, `kaggle_onmt_that/`, `kaggle_moses_v2_*`). Mã: `scripts/ft_train/` (`kaggle_quota.py`, `dong_goi_quota.py`, `kaggle_onmt.py`, `kaggle_moses.py`, `cham_co_dien.py`, `tong_hop_5mh.py`).
