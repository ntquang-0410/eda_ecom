# Matrix 4×2 — 4 trường phái model × 2 bộ dữ liệu

**Ngày:** 2026-11-11  
**Dataset:** `nguynthanhquang/tiki-matrix-data` (đã fix leakage)  
**Budget LLM:** 4.000.000 token / 500 optimizer step, seed 42  
**Eval:** `eval_F` 1.500 dòng (dùng chung), COMET-da 1688 + COMET-QE Tiki

---

## 1. Ma trận 8 cell

| Model | Trường phái | A-natural | B-balanced |
|---|---|---|---|
| Qwen3-8B | LLM (LoRA r16/α32) | cell 1 | cell 2 |
| EcomGPT-7B | LLM chuyên ngành (LoRA r16/α32) | cell 3 | cell 4 |
| OpenNMT-py | Neural MT from-scratch | cell 5 | cell 6 |
| Moses | SMT phrase-based | cell 7 | cell 8 |

## 2. Trạng thái

| cell | trạng thái |
|---|---|
| Qwen3-8B A | **DONE** (4,000M token, 216 phút) |
| Qwen3-8B B | **RUNNING** (từ 11:15) |
| EcomGPT A | đã huỷ (ưu tiên OpenNMT); cần chạy lại với `max_new` đúng |
| EcomGPT B | chưa chạy |
| OpenNMT A/B | kernel đã build, **chờ slot GPU** (giới hạn 2 session) |
| Moses A/B | **DONE** (pipeline Moses + eflomal) |

## 3. Cấu hình OpenNMT

- venv Python 3.11, torch 2.2.2+cu121, OpenNMT-py 3.5.1.
- Transformer: `d_model=256`, `heads=8`, `layers=4`, `transformer_ff=1024`,
  dropout 0.1, label smoothing 0.1, Noam LR, beam 5.
- Batch 4.096 token × accum 2; **10 epoch** (steps = 10 × ceil(train_tokens / 8192)).
- A: 103.626 câu train; B: 49.774 câu train; dev làm valid.
- `src/tgt_seq_length=512`; B có description dài sẽ bị cắt — ghi nhận là hạn chế
  của NMT với trường dài.
- Train A rồi B trong cùng kernel, translate `eval_F` ngay sau mỗi model.

## 4. Cấu hình Moses

- Moses phrase-based: `train-model.perl`, LM 3-gram `lmplz`, phrase length 7,
  `msd-bidirectional-fe`, decode 4 threads.
- Alignment: `eflomal` → sym `grow-diag-final-and` (thay GIZA++ do môi trường Kaggle).
- B description dài có thể làm giảm chất lượng/truncate; cần ghi nhận trong report.

## 5. EcomGPT

- Cùng template prompt và LoRA r16/α32 như Qwen3-8B.
- Pipeline cũ dùng `max_new_tokens=160` cho mọi dòng → **sai với description**;
  bản mới sẽ dùng `eval_frame.max_new` (64/256/768) và chạy A+B sau OpenNMT.

## 6. Blocker hiện tại

- Tài khoản giới hạn **2 GPU session**; hiện Qwen8B-B + một session còn sót
  (EcomGPT vừa huỷ) đang chiếm slot.
- `tiki-matrix-onmt-ab` push tự động retry mỗi 5 phút cho đến khi có slot.
- Quota còn ~10,6h.

## 7. Đánh giá sau khi xong

1. Inference đủ 8 cell trên `eval_F`.
2. COMET-da (1688) + COMET-QE (Tiki).
3. Paired bootstrap 2.000 lần giữa các cặp cell.
4. Bảng 4×2 theo stratum Title / Attribute / Description.
