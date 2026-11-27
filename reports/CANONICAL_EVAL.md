# Canonical eval — 831 dòng cố định, zero-fill + coverage

Index: `data/cleaned/eval_paired_831.parquet` (1688 = 231, Tiki = 600, hash trong `.json`).
Quy ước: join theo `eval_id`; hyp rỗng ⇒ 0 điểm, **không lọc dòng**; coverage = % hyp không rỗng.
Metric xếp hạng: **COMET-da (1688)**. COMET-QE (Tiki), BLEU, chrF chỉ là chẩn đoán.
Human eval Tiki: **MISSING** (chưa có file annotation).

## 1. Tổng hợp theo cell (mean ± std qua seed)

| Cell | Seeds | #seed | COMET-da 1688 | Coverage 1688 | COMET-QE Tiki | Coverage Tiki | BLEU (sent) | chrF (sent) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B B-balanced | 42 | 1 | 0.8572 | 100.0% | 0.0509 | 100.0% | 53.81 | 54.56 |
| Qwen3-8B A-natural | 42 | 1 | 0.8567 | 100.0% | 0.0521 | 100.0% | 51.77 | 51.37 |
| Qwen3-4B A-natural | 7,42,123 | 3 | 0.8542 ± 0.0037 | 100.0% | 0.0608 ± 0.0032 | 100.0% | 46.76 | 47.25 |
| Qwen3-8B B-natural | 42 | 1 | 0.8514 | 100.0% | 0.0624 | 100.0% | 51.67 | 52.26 |
| Qwen3-4B A-balanced | 42 | 1 | 0.8477 | 100.0% | 0.0524 | 100.0% | 46.27 | 46.79 |
| Qwen3-4B B-balanced | 42,123 | 2 | 0.8475 ± 0.0006 | 100.0% | 0.0489 ± 0.0040 | 100.0% | 50.66 | 51.83 |
| Qwen3-4B B-natural | 42 | 1 | 0.8436 | 100.0% | 0.0471 | 100.0% | 49.32 | 50.80 |
| EcomGPT-7B A-natural | 42 | 1 | 0.8250 | 100.0% | 0.0077 | 100.0% | 39.33 | 41.00 |
| EcomGPT-7B A-balanced | 42 | 1 | 0.8156 | 100.0% | 0.0020 | 100.0% | 37.03 | 39.24 |
| Moses B | 42 | 1 | 0.8037 | 100.0% | -0.1593 | 100.0% | 49.06 | 50.90 |
| Moses A | 42 | 1 | 0.7862 | 100.0% | -0.2439 | 100.0% | 42.48 | 44.82 |
| EcomGPT-7B B-balanced | 42 | 1 | 0.7634 | 100.0% | -0.1366 | 100.0% | 31.10 | 33.86 |
| NLLB-600M A | 7,42 | 2 | 0.6324 ± 0.0029 | 100.0% | -0.5403 ± 0.0034 | 100.0% | 18.43 | 24.31 |
| NLLB-600M B | 7,42 | 2 | 0.5938 ± 0.0628 | 99.8% | -0.5956 ± 0.1080 | 99.8% | 15.41 | 22.94 |
| OpenNMT A | 42 | 1 | 0.3488 | 100.0% | -1.4177 | 100.0% | 1.26 | 1.86 |
| OpenNMT B-natural | 42 | 1 | 0.3387 | 100.0% | -1.4074 | 100.0% | 0.95 | 1.30 |

## 2. Paired cluster bootstrap (COMET-da 1688, 2.000 lần, cluster = product_id)

| A | B | Δ (A−B) | 95% CI | n |
|---|---|---:|---|---:|
| Qwen3-8B A-natural | Qwen3-8B B-balanced | -0.0005 | [-0.0112; +0.0099] | 231 |
| Qwen3-8B B-balanced | Qwen3-8B B-natural | +0.0057 | [-0.0017; +0.0140] | 231 |
| Qwen3-4B A-natural | Qwen3-4B B-balanced | +0.0067 | [-0.0029; +0.0167] | 231 |
| Qwen3-4B A-natural | Qwen3-4B A-balanced | +0.0065 | [-0.0014; +0.0156] | 231 |
| Qwen3-4B B-balanced | Qwen3-4B B-natural | +0.0039 | [-0.0036; +0.0132] | 231 |
| Qwen3-4B A-natural | Qwen3-8B A-natural | -0.0025 | [-0.0111; +0.0067] | 231 |
| Qwen3-4B B-balanced | Qwen3-8B B-balanced | -0.0097* | [-0.0174; -0.0027] | 231 |
| Qwen3-4B A-natural | EcomGPT-7B A-natural | +0.0292* | [+0.0171; +0.0409] | 231 |
| Qwen3-4B B-balanced | EcomGPT-7B B-balanced | +0.0841* | [+0.0639; +0.1055] | 231 |
| NLLB-600M A | NLLB-600M B | +0.0387* | [+0.0229; +0.0533] | 231 |
| OpenNMT A | OpenNMT B-natural | +0.0102 | [-0.0027; +0.0236] | 231 |
| Qwen3-4B A-natural | NLLB-600M A | +0.2217* | [+0.1951; +0.2501] | 231 |
| Qwen3-4B B-balanced | NLLB-600M B | +0.2537* | [+0.2313; +0.2768] | 231 |

## 3. Vi phạm / khoảng trống so với policy

- **Không có score từng dòng (loại khỏi bảng xếp hạng):** Qwen3-8B A-balanced.
- **Seed:** 16/16 cell có <5 seed. Các cell đã có ≥2 seed: Qwen3-4B A-natural (3), Qwen3-4B B-balanced (2), NLLB-600M A (2), NLLB-600M B (2).
- **Thiếu score từng dòng:** Qwen3-8B A-balanced (chỉ có điểm tổng) → không vào bootstrap.
- **Coverage <100%:** NLLB-600M B (99.8%/99.8%) — đã zero-fill.
- **Human eval Tiki:** chưa có; bảng xếp hạng hiện chỉ dựa COMET-da 1688.
- **Nguồn gốc:** các artefact này là retrofit từ score cũ; run mới phải dùng `tools/score_canonical.py` để log manifest đầy đủ.
