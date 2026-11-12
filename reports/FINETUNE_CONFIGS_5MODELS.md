# Cấu hình fine-tune / train — Matrix 5×2

**Ngày:** 2026-11-11  
**Mục tiêu:** 5 model × 2 bộ dữ liệu (A-natural, B-balanced), đánh giá trên cùng `eval_F` 1.500 dòng.

---

## 1. Dữ liệu dùng chung

Nguồn: `tiki-matrix-data` (bản đã fix leakage).

| | A | B |
|---|---|---|
| rows | 111.129 | 54.006 |
| token (eff) | 4,518M | 15,825M |
| train / dev | 103.626 / 7.503 | 49.774 / 4.232 |
| nhóm | 1688_title, 1688_attribute, tiki_title, tiki_attribute | 1688_title, 1688_description, tiki_title, tiki_description |
| Tiki target | BT Qwen3-4B | BT Qwen3-4B |
| 1688 target | bản người | bản người |

`eval_F`: 1.500 dòng = 300 title + 300 attribute + 150 description mỗi nguồn 1688/Tiki.
Leakage: `eval_F ∩ units_A = 0`, `eval_F ∩ units_B = 0`.

**Budget LLM:** 4.000.000 token / 500 optimizer step, seed 42.

Epoch thực tế ở 4M token:

```text
A: 4,0M / 4,518M = 0,885 epoch
B: 4,0M / 15,825M = 0,253 epoch
```

Caveat bắt buộc ghi trong báo cáo: B chỉ học ~1/4 epoch so với A; đây là đánh đổi của so sánh compute-matched.

---

## 2. Quota sampling (dùng cho cả 5 model)

```text
B = 4.000.000 token
với mỗi nhóm g:
    T_g = s_g × B
    lấy đều ngẫu nhiên (có hoàn lại) từ pool_g đến khi Σ eff_tokens ≥ T_g
gộp → shuffle → chia 500 step ~8.000 token
```

Config:

```yaml
A_natural:  phân phối token tự nhiên của A
B_balanced: {1688_title: .25, 1688_description: .25,
             tiki_title: .25, tiki_description: .25}
```

Mỗi cell lưu `sample_<config>_seed42.parquet`:
`step, row_index, pair_id, group, eff_tokens`, kèm `epochs_seen` trong run JSON.

---

## 3. Qwen3-8B (LLM đại diện Qwen)

**File:** `configs/matrix_4models/qwen3_8b_lora.yaml`

| thông số | giá trị |
|---|---|
| base | `Qwen/Qwen3-8B` (`qwen-lm/qwen-3/Transformers/8b/1`) |
| dtype | bfloat16 |
| gradient checkpointing | bật |
| max_seq_len | 960 |
| LoRA | r=16, alpha=32, dropout=0,05, bias=none, `all-linear`, causal LM |
| optimizer | AdamW, lr=1e-4, weight_decay=0, betas (0,9; 0,999) |
| scheduler | cosine, warmup 20 step |
| grad clip | 1,0 |
| batch | micro-batch 2, ~8.000 token/step |
| steps | 500 |
| seed | 42 |
| loss | causal LM, token-weighted: `loss × n_tok / step_tokens`; prompt labels = -100 |

**Prompt:**

```text
system: You are a professional Vietnamese-to-Simplified Chinese e-commerce translator.
        Translate the user text into Simplified Chinese. Preserve numbers, units and brand names.
        Return only the translation, no explanation.
user:   source_text
format: tokenizer.apply_chat_template(..., enable_thinking=False)
```

**Inference:** greedy, batch 1 nếu `max_new > 300`, ngược lại 4; `max_new` = 64 title / 256 attribute / 768 description.

**Kết quả đã chạy:**

| cell | token | phút | epochs | hyp |
|---|---:|---:|---:|---:|
| A natural | 4.000.038 | 216,3 | 0,885 | 1.500 |
| B balanced | 4.002.207 | 149,0 | 0,253 | 1.500 |

---

## 3b. Qwen3-4B (LLM nhỏ hơn, cùng recipe)

**File:** `configs/matrix_5models/qwen3_4b_lora.yaml`

| thông số | giá trị |
|---|---|
| base | `Qwen/Qwen3-4B` (`qwen-lm/qwen-3/Transformers/4b/1`) |
| dtype | bfloat16, gradient checkpointing |
| max_seq_len | 960 |
| LoRA | r=16, alpha=32, dropout=0,05, bias=none, `all-linear` |
| optimizer | AdamW, lr=1e-4, cosine, warmup 20 step, 500 step |
| batch | micro-batch **4**, ~8.000 token/step |
| seed | 42 |
| inference | greedy; `max_new` 64/256/768 theo stratum |

**Kết quả đã chạy:**

| cell | token | phút | epochs | hyp | empty |
|---|---:|---:|---:|---:|---:|
| A natural | 4.000.038 | 153,1 | 0,885 | 1.500 | 15 |
| B balanced | 4.002.207 | 116,3 | 0,253 | 1.500 | 15 |

**Nhận xét:** 4B A hòa 8B A (Δ −0,0008, CI chứa 0); 4B B thấp hơn 8B B 0,0072. Ở description, 4B A +0,0156 và 4B B +0,0006 so với 8B → capacity 4B đã đủ cho trường dài; 8B chỉ nhỉnh title/attribute ngắn. Xem `reports/MODEL_CAUSE_ANALYSIS.md`.

---

## 4. EcomGPT-7B (LLM chuyên ngành)

**File:** `configs/matrix_4models/ecomgpt_7b_lora.yaml`

| thông số | giá trị |
|---|---|
| base | `damo/nlp_ecomgpt_multilingual-7B-ecom` |
| dtype | bfloat16, gradient checkpointing |
| max_seq_len | 960 |
| LoRA | r=16, alpha=32, dropout=0,05, bias=none, `all-linear` |
| optimizer | AdamW, lr=1e-4 |
| scheduler | cosine, warmup 20 step, 500 step |
| batch | micro-batch 2, ~8.000 token/step |
| seed | 42 |

**Prompt (Alpaca-style):**

```text
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Translate the following Vietnamese e-commerce text into Simplified Chinese. Return only the translation.

### Input:
{source_text}

### Response:
```

**Inference:** greedy, batch 8, `max_new` theo `eval_F` (64/256/768). Bản cũ dùng cứng 160 token nên đã bị loại; bản mới sửa theo stratum.

**Kết quả đã chạy:**

| cell | token | steps | hyp | empty |
|---|---:|---:|---:|---:|
| A natural | 4.000M | 500 | 1.500 | 0 |
| B balanced | 4.001M | 500 | 1.500 | 21 (toàn bộ ở Tiki attribute) |

---

## 5. OpenNMT-py (Neural MT from-scratch)

**File:** `configs/matrix_4models/opennmt_A.yaml`, `opennmt_B.yaml`

| thông số | giá trị |
|---|---|
| version | OpenNMT-py 3.5.1 |
| env | venv Python 3.11, torch 2.2.2+cu121 |
| kiến trúc | Transformer, `d_model=256`, `heads=8`, `layers=4`, `transformer_ff=1024` |
| dropout | 0,1; attention dropout 0,1 |
| batch | `batch_type=tokens`, 2.048 token × `accum_count=[4]` = 8.192 token/step; `num_workers=0`, `bucket_size=1024` |
| optimizer | Noam, lr=2,0, warmup 2.000, `adam_beta2=0,998` |
| label smoothing | 0,1 |
| epoch | A: **30.000 step ≈ 48,3 epoch** (bản 10 epoch bị degenerate); B: **10 epoch** |
| train_steps | A: **30.000**; B: **22.990** (2.299 step/epoch) |
| valid | dev; valid_steps A 500, B 1.149 |
| checkpoint | mỗi 621 (A) / 2.299 (B) step, giữ 2 |
| decode | beam 5, `max_length=512`, `replace_unk`; batch 32 (B) / batch 4 (A, tránh OOM T4) |
| seed | 42 |

**Vocab:** ép `src_vocab_size=32000`, `tgt_vocab_size=32000` để khớp embedding 32.768. Bản đầu không ép → vocab 48k/75k, target token >32.767 thành `<unk>`, model chỉ sinh EOS (1.500 dòng rỗng). Đây là bug đã fix, không phải kết quả khoa học.

**Kết quả A 30k step:** ppl 823,5; acc 16,0%; hypothesis đủ 1.500 dòng nhưng vẫn lặp cụm (`件套…`) → COMET 1688 0,3581, BLEU 0,45. Kết luận: OpenNMT from-scratch không cạnh tranh được trên dữ liệu 4–16M token này, kể cả khi tăng gấp 3 compute.

**Caveat B:** có **49 dòng eval** (3,3%) dài >512 token; OpenNMT cắt bớt. Đây là hạn chế của NMT với trường description dài, cần ghi nhận.

---

## 6. Moses (SMT phrase-based)

**File:** `configs/matrix_4models/moses_pipeline.sh`

| bước | cấu hình |
|---|---|
| sampling | quota 4M token, cùng config A/B |
| tokenize | tách punctuation; tiếng Trung tách từng chữ Hán |
| alignment | `eflomal`, model=3, iterations=(4,4,4), 4 samplers; sym `grow-diag-final-and` |
| LM | `lmplz -o 3` trên train.zh |
| phrase model | `train-model.perl -do-steps 4-9 -max-phrase-length 7 -reordering msd-bidirectional-fe` |
| decode | `moses -threads 4`, eval_F |
| seed | 42 |

Lưu ý: dùng **eflomal thay GIZA++** do môi trường Kaggle; cùng vai trò alignment. B description dài có thể làm giảm chất lượng/truncate; ghi nhận là hạn chế của Moses với trường dài.

---

## 7. Đánh giá

| | |
|---|---|
| eval | `eval_F` 1.500 dòng |
| 1688 | COMET-da `Unbabel/wmt22-comet-da` + BLEU/chrF |
| Tiki | COMET-QE `Unbabel/wmt20-comet-qe-da` |
| so sánh | paired bootstrap **2.000 lần**, resample theo `product_id` |
| bảng | 5×2 theo stratum Title / Attribute / Description |

15 dòng eval (3 description 1688 + 12 description Tiki) có prompt > 952 token nên Qwen không sinh được hypothesis; EcomGPT B thêm 21 dòng Tiki attribute rỗng. Mọi phép so sánh paired chỉ dùng `eval_id` có score ở cả hai phía, nên n dao động 717–750.

**Bảng kết quả cuối:** xem `reports/MATRIX_5X2_RESULTS.md` (10 cell, stratum, paired bootstrap 2.000 lần).

---

## 8. Môi trường chạy

| model | GPU | ghi chú |
|---|---|---|
| Qwen3-8B / Qwen3-4B | Kaggle RTX Pro 6000 | competition mode, offline; 4B micro-batch 4 |
| EcomGPT-7B | Kaggle RTX Pro 6000 | model offline `nguynthanhquang/ecomgpt-7b` |
| OpenNMT | Kaggle T4 | internet để tạo venv Python 3.11 + torch 2.2.2 |
| Moses | CPU | Moses + eflomal |

---

## 9. Trạng thái

| cell | trạng thái | COMET 1688 | COMET-QE Tiki |
|---|---|---:|---:|
| Qwen3-8B A | **xong** | 0,8560 | 0,0635 |
| Qwen3-8B B | **xong** | 0,8552 | 0,0687 |
| Qwen3-4B A | **xong** | 0,8552 | 0,0647 |
| Qwen3-4B B | **xong** | 0,8480 | 0,0570 |
| EcomGPT-7B A | **xong** | 0,8214 | 0,0102 |
| EcomGPT-7B B | **xong** | 0,7661 | -0,1455 |
| Moses A | **xong** | 0,7947 | -0,2401 |
| Moses B | **xong** | 0,8103 | -0,1466 |
| OpenNMT A (30k step) | **xong** | 0,3581 | -1,3508 |
| OpenNMT B (10 epoch) | **xong** | 0,3424 | -1,3909 |
