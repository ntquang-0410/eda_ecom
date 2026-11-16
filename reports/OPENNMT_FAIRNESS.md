# OpenNMT fairness — bảng compute và phạm vi kết luận

**Ngày:** 2026-11-16  
**Mục tiêu:** tách "cấu hình NMT from-scratch này yếu" khỏi "paradigm NMT yếu hơn LLM".

---

## 1. Năm trục không công bằng

| Trục | OpenNMT | Qwen3-8B / EcomGPT-7B | Hệ quả |
|---|---|---|---|
| Tham số | ~37M (4 layer, 256 hidden, 32k vocab) | 8,2B / 7B | Lệch ~190–220× tham số |
| Pretraining | from-scratch | pretrained đa ngữ / e-commerce | Đây là khác biệt lớn nhất, không đo "kiến trúc" |
| Token-pass fine-tune | A: 245,8M; B: 188,3M | 4,0M mỗi cell | NMT thấy **61,4× / 47,1×** token |
| Epoch | A: 48,3; B: 10 | A: 0,885; B: 0,253 | A/B của NMT cũng lệch 4,8× |
| Decode | beam 5, max 512, `replace_unk` | greedy, max_new 64/256/768 | Khác hệ tokenization/độ dài |

## 2. Bảng compute (ước tính)

FLOPs fine-tune ≈ `6 × params × token_pass` (không gồm pretraining).

| model | params | token-pass | FLOPs fine-tune | epoch | ghi chú |
|---|---:|---:|---:|---:|---|
| Qwen3-8B A/B | 8,2B | 4,0M | 1,97e17 | 0,885 / 0,253 | LoRA, chỉ adapter |
| Qwen3-4B A/B | 4,0B | 4,0M | 9,6e16 | 0,885 / 0,253 | LoRA |
| EcomGPT-7B A/B | 7,0B | 4,0M | 1,68e17 | 0,885 / 0,253 | LoRA |
| OpenNMT A | ~37M | 245,8M | 5,46e16 | 48,3 | full fine-tune/from-scratch |
| OpenNMT B (cũ) | ~37M | 188,3M | 4,18e16 | 10 | **B-natural, không phải B-balanced** |
| Moses A/B | phrase-based | 4,0M sample | — | — | SMT, không epoch |

**Đọc đúng:** FLOPs fine-tune của OpenNMT A **không** lớn hơn Qwen3-8B (5,5e16 vs 1,97e17). Cái lệch là **pretraining** (Qwen đã học hàng nghìn tỷ token) và **số token-pass** (NMT 61×). Vì vậy không được viết "compute-matched" cho NMT; phải ghi rõ "LLM compute-matched trong nội bộ LLM; NMT là baseline from-scratch với 48 epoch".

## 3. Lỗi nhãn `OpenNMT-B`

- Kernel cũ đọc toàn bộ `units_B` split train: **49.774 dòng / 18,83M token** = phân phối **B-natural**.
- LLM/Moses dùng quota balanced: **52.131 draw / 25.453 unique** (25% token mỗi nhóm).
- Vì vậy cột "B-balanced" của OpenNMT là **B-natural**; mọi so sánh B−A của OpenNMT là nhiễu giữa data và epoch.

**Quyết định:** đổi tên cell thành **OpenNMT B-natural** trong mọi bảng (đã cập nhật `MATRIX_5X2_RESULTS.md`, `MATRIX_5X2_EVAL_F_V2.md`, `MODEL_CAUSE_ANALYSIS.md`). Retrain B-balanced bị chặn bởi môi trường offline (không cài được Python 3.11 + OpenNMT-py); nếu cần bản balanced sẽ chạy ở P2 với wheelhouse tự đóng gói.

## 4. Độ dài / truncation

- 49/1.500 dòng eval có source >512 token, toàn bộ là description dài; OpenNMT cắt ở 512.
- Target char-level ZH, vocab 32k; max_length 512.
- Với description dài, đây là bất lợi hệ thống, không phải lỗi model đơn lẻ. Báo cáo phải tách stratum description và bucket long.

## 5. Kết luận được phép

**Được phép:** "Một Transformer from-scratch ~37M tham số, 4 layer/256 hidden, vocab 32k, train 30k step (~48 epoch A / 10 epoch B-natural), decode beam 512 + replace_unk, 1 seed — sụp mode collapse và không dùng được làm baseline production."

**Không được phép:** "NMT from-scratch không thể cạnh tranh với LLM" như kết luận tổng quát. Muốn claim đó phải có:
1. NMT pretrained (NLLB/mBART) cùng protocol — **đang chạy NLLB-200-distilled-600M**;
2. NMT from-scratch recipe chuẩn (6 layer/512 hidden, 16k BPE, 50–100M token, early stop, checkpoint averaging, beam + length penalty, bỏ `replace_unk`);
3. ≥3 seed và learning curve;
4. sanity check trên IWSLT/WMT.

## 6. Trạng thái

| việc | trạng thái |
|---|---|
| OpenNMT-B balanced 30k step | **không chạy được** — đã relabel B-natural (hợp lệ theo yêu cầu 1) |
| NLLB-200-distilled-600M A/B cùng 4M quota | **đang chạy** L4x4 |
| Bảng compute | **xong** (file này) |
| >=3 seed | **chưa** — plan P1 |
| recipe NMT chuẩn 6/512 + BPE | **chưa** — plan P2 |
