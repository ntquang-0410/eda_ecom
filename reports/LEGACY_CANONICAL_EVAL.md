# LEGACY canonical eval — 858 dòng (baseline lịch sử)

> ⚠️ **LEGACY BASELINE — chỉ để đối chiếu lịch sử, KHÔNG dùng xếp hạng chính thức.**
> Sinh từ các run **đơn seed** tại `max_seq=960` (mất 3,2% token A / 6,8% B), index 858, QE bị ẩn cho 12 cell chưa pass provenance. Bảng chính thức sẽ là campaign **≥5 seed, `max_seq=1792`** (xem `EVAL_PROTOCOL.md`, `TRAIN_GUARD_960.md`).

Index: `data/cleaned/eval_paired_858.parquet` (1688 = 231, Tiki = 627, hash trong `.json`).
Quy ước: join theo `eval_id`; hyp rỗng ⇒ 0 điểm, **không lọc dòng**; coverage = % hyp không rỗng.
Metric xếp hạng: **COMET-da (1688)**. COMET-QE (Tiki), BLEU, chrF chỉ là chẩn đoán.
Human eval Tiki: **MISSING** (chưa có file annotation).

## 1. Tổng hợp theo cell (mean ± std qua seed)

| Cell | Seeds | #seed | COMET-da 1688 | Coverage 1688 | COMET-QE Tiki | Coverage Tiki | BLEU (sent) | chrF (sent) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B B-balanced | 42 | 1 | 0.8572 | 100.0% | 0.0470 | 98.6% | 53.81 | 54.56 |
| Qwen3-8B A-natural | 42 | 1 | 0.8567 | 100.0% | 0.0487 | 98.6% | 51.77 | 51.37 |
| Qwen3-8B A-balanced | 42 | 1 | 0.8553 | 100.0% | — | 98.6% | 48.25 | 48.03 |
| Qwen3-4B A-natural | 7,42,123 | 3 | 0.8542 ± 0.0037 | 100.0% | 0.0597 ± 0.0030 | 100.0% | 46.76 | 47.25 |
| Qwen3-8B B-natural | 42 | 1 | 0.8514 | 100.0% | 0.0576 | 98.6% | 51.67 | 52.26 |
| Qwen3-4B A-balanced | 42 | 1 | 0.8477 | 100.0% | — | 98.6% | 46.27 | 46.79 |
| Qwen3-4B B-balanced | 42,123 | 2 | 0.8475 ± 0.0006 | 100.0% | — | 98.6% | 50.66 | 51.83 |
| Qwen3-4B B-natural | 42 | 1 | 0.8436 | 100.0% | — | 98.6% | 49.32 | 50.80 |
| EcomGPT-7B A-natural | 42 | 1 | 0.8250 | 100.0% | — | 100.0% | 39.33 | 41.00 |
| EcomGPT-7B A-balanced | 42 | 1 | 0.8156 | 100.0% | -0.0041 | 100.0% | 37.03 | 39.24 |
| Moses B | 42 | 1 | 0.8037 | 100.0% | — | 100.0% | 49.06 | 50.90 |
| Moses A | 42 | 1 | 0.7862 | 100.0% | — | 100.0% | 42.48 | 44.82 |
| EcomGPT-7B B-balanced | 42 | 1 | 0.7634 | 100.0% | — | 97.1% | 31.10 | 33.86 |
| NLLB-600M A | 7,42 | 2 | 0.6324 ± 0.0029 | 100.0% | — | 100.0% | 18.43 | 24.31 |
| NLLB-600M B | 7,42 | 2 | 0.5938 ± 0.0628 | 99.8% | — | 99.8% | 15.41 | 22.94 |
| OpenNMT A | 42 | 1 | 0.3488 | 100.0% | — | 100.0% | 1.26 | 1.86 |
| OpenNMT B-natural | 42 | 1 | 0.3387 | 100.0% | — | 100.0% | 0.95 | 1.30 |

### 1b. Macro theo stratum

`macro_unw` = trung bình 3 stratum (mỗi stratum trọng số bằng nhau); `macro_w` = trọng số theo số dòng (bằng mean tổng).

| Cell | macro_unw 1688 | macro_w 1688 | macro_unw Tiki | macro_w Tiki |
|---|---:|---:|---:|---:|
| Qwen3-8B B-balanced | 0.8656 | 0.8572 | 0.0785 | 0.0470 |
| Qwen3-8B A-natural | 0.8542 | 0.8567 | 0.0805 | 0.0487 |
| Qwen3-8B A-balanced | 0.8549 | 0.8553 | 0.0861 | 0.0518 |
| Qwen3-4B A-natural | 0.8548 | 0.8542 | 0.0964 | 0.0597 |
| Qwen3-8B B-natural | 0.8615 | 0.8514 | 0.0917 | 0.0576 |
| Qwen3-4B A-balanced | 0.8470 | 0.8477 | 0.0846 | 0.0483 |
| Qwen3-4B B-balanced | 0.8569 | 0.8475 | 0.0822 | 0.0438 |
| Qwen3-4B B-natural | 0.8539 | 0.8436 | 0.0826 | 0.0432 |
| EcomGPT-7B A-natural | 0.8151 | 0.8250 | -0.0011 | -0.0018 |
| EcomGPT-7B A-balanced | 0.8078 | 0.8156 | -0.0072 | -0.0041 |
| Moses B | 0.8081 | 0.8037 | -0.2068 | -0.1663 |
| Moses A | 0.7694 | 0.7862 | -0.3418 | -0.2475 |
| EcomGPT-7B B-balanced | 0.7261 | 0.7634 | -0.1897 | -0.1387 |
| NLLB-600M A | 0.5960 | 0.6324 | -0.6659 | -0.5502 |
| NLLB-600M B | 0.5710 | 0.5938 | -0.7149 | -0.6036 |
| OpenNMT A | 0.3421 | 0.3488 | -1.4814 | -1.4138 |
| OpenNMT B-natural | 0.3433 | 0.3387 | -1.4326 | -1.4059 |

**QE 858 bị ẩn cho 12 cell chưa pass adapter provenance (4.4):** EcomGPT-7B A-natural, EcomGPT-7B B-balanced, Moses A, Moses B, NLLB-600M A, NLLB-600M B, OpenNMT A, OpenNMT B-natural, Qwen3-4B A-balanced, Qwen3-4B B-balanced, Qwen3-4B B-natural, Qwen3-8B A-balanced — xem `reports/ADAPTER_PROVENANCE_AUDIT.md`. COMET-da vẫn hiển thị đầy đủ.


**EcomGPT-7B B (4.6):** attribute rỗng **21/224 (9,4%)**, title 0/291, description 0/112 — B không có group attribute trong train (OOD), nên đọc QE/macro theo stratum, không so tổng; xem `reports/AUDIT_ECOMGPT_B_EMPTY.md`.


## 2. Paired cluster bootstrap (COMET-da 1688, 2.000 lần, cluster = product_id)

| A | B | Δ (A−B) | 95% CI | n |
|---|---|---:|---|---:|
| Qwen3-8B A-natural | Qwen3-8B B-balanced | -0.0005 | [-0.0112; +0.0099] | 231 |
| Qwen3-8B A-natural | Qwen3-8B A-balanced | +0.0013 | [-0.0070; +0.0098] | 231 |
| Qwen3-8B B-balanced | Qwen3-8B B-natural | +0.0057 | [-0.0017; +0.0140] | 231 |
| Qwen3-4B A-natural | Qwen3-4B B-balanced | +0.0067 | [-0.0029; +0.0167] | 231 |
| Qwen3-4B A-natural | Qwen3-4B A-balanced | +0.0065 | [-0.0014; +0.0156] | 231 |
| Qwen3-4B B-balanced | Qwen3-4B B-natural | +0.0039 | [-0.0036; +0.0132] | 231 |
| Qwen3-4B A-natural | Qwen3-8B A-natural | -0.0025 | [-0.0111; +0.0067] | 231 |
| Qwen3-4B B-balanced | Qwen3-8B B-balanced | -0.0097* | [-0.0174; -0.0027] | 231 |
| Qwen3-4B A-natural | EcomGPT-7B A-natural | +0.0292* | [+0.0171; +0.0409] | 231 |
| Qwen3-4B B-balanced | EcomGPT-7B B-balanced | +0.0841* | [+0.0639; +0.1055] | 231 |
| NLLB-600M A | NLLB-600M B | +0.0387 | [+0.0229; +0.0533] | 231 |
| OpenNMT A | OpenNMT B-natural | +0.0102 | [-0.0027; +0.0236] | 231 |
| Qwen3-4B A-natural | NLLB-600M A | +0.2217* | [+0.1951; +0.2501] | 231 |
| Qwen3-4B B-balanced | NLLB-600M B | +0.2537* | [+0.2313; +0.2768] | 231 |

`*` = CI (chỉ phủ variance eval) không chứa 0. **NLLB A vs B không đánh dấu** vì chênh lệch đổi dấu giữa seed 42 và seed 7 — xem ghi chú §3.


## 3. Vi phạm / khoảng trống so với policy

- **Seed:** 17/17 cell có <5 seed. Các cell đã có ≥2 seed: Qwen3-4B A-natural (3), Qwen3-4B B-balanced (2), NLLB-600M A (2), NLLB-600M B (2).
- **Thiếu score từng dòng:** không còn cell nào.
- **Coverage <100%:** 9 cell (Qwen 98,6% do 9 description dài; EcomGPT B 97,1%; NLLB B 99,8%) — đã zero-fill, chi tiết trong artefact.
- **Human eval Tiki:** chưa có; bảng xếp hạng hiện chỉ dựa COMET-da 1688.
- **9 dòng description dài (3,0–3,7k ký tự) bị Qwen bỏ vì prompt >952 token** (guard `max_seq=960`), không phải model fail — xem `reports/AUDIT_EMPTY_QWEN4B.md`.
- **NLLB A vs B là seed-dependent:** seed 42 cho A 0,6344 > B 0,5494; seed 7 cho A 0,6304 < B 0,6382 → không kết luận A/B từ control này. Epochs@4M: A = 0,83 (tập 4,84 M token), B = 0,26 (tập 15,71 M token); mỗi seed B chỉ thấy ~1/4 dữ liệu nên run-variance lớn (Δ seed42−seed7 = −0,0888 [−0,1076; −0,0698]).
- **NLLB B:** seed 42 = 0,5494 vs seed 7 = 0,6382; run hợp lệ — xem `reports/AUDIT_NLLB_B_SEED.md`.
- **Nguồn gốc:** các artefact này là retrofit từ score cũ; run mới phải dùng `tools/score_canonical.py` để log manifest đầy đủ.
- **Training guard 960:** các run hiện tại mất 3,2% (A) và 6,8% (B) token do bỏ dòng prompt >952 — xem `reports/TRAIN_GUARD_960.md`; campaign 5-seed mới dùng `--max-seq 1792`.
