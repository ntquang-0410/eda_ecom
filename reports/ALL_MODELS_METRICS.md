# Bảng tổng hợp mọi metric (canonical)

> Sinh bởi `tools/all_models_table.py` từ artefact canonical. Số liệu thay thế bảng cũ khi có khác biệt (zero-fill + coverage).

## 1. Tổng hợp

| Model | Data | Recipe | Seeds | COMET-da 1688 | Cov 1688 | COMET-QE Tiki (chẩn đoán) | Cov Tiki | BLEU sent | chrF sent |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | natural | 42 | 0.8567 | 100.0% | 0.0521 | 100.0% | 51.77 | 51.37 |
| Qwen3-8B | B | balanced | 42 | 0.8572 | 100.0% | 0.0509 | 100.0% | 53.81 | 54.56 |
| Qwen3-8B | B | natural | 42 | 0.8514 | 100.0% | 0.0624 | 100.0% | 51.67 | 52.26 |
| Qwen3-4B | A | balanced | 42 | 0.8477 | 100.0% | 0.0524 | 100.0% | 46.27 | 46.79 |
| Qwen3-4B | A | natural | 7,42,123 | 0.8542 ± 0.0037 | 100.0% | 0.0608 ± 0.0032 | 100.0% | 46.76 | 47.25 |
| Qwen3-4B | B | balanced | 42,123 | 0.8475 ± 0.0006 | 100.0% | 0.0489 ± 0.0040 | 100.0% | 50.66 | 51.83 |
| Qwen3-4B | B | natural | 42 | 0.8436 | 100.0% | 0.0471 | 100.0% | 49.32 | 50.80 |
| EcomGPT-7B | A | balanced | 42 | 0.8156 | 100.0% | 0.0020 | 100.0% | 37.03 | 39.24 |
| EcomGPT-7B | A | natural | 42 | 0.8250 | 100.0% | 0.0077 | 100.0% | 39.33 | 41.00 |
| EcomGPT-7B | B | balanced | 42 | 0.7634 | 100.0% | -0.1366 | 100.0% | 31.10 | 33.86 |
| Moses | A | natural | 42 | 0.7862 | 100.0% | -0.2439 | 100.0% | 42.48 | 44.82 |
| Moses | B | balanced | 42 | 0.8037 | 100.0% | -0.1593 | 100.0% | 49.06 | 50.90 |
| OpenNMT | A | natural | 42 | 0.3488 | 100.0% | -1.4177 | 100.0% | 1.26 | 1.86 |
| OpenNMT | B | natural | 42 | 0.3387 | 100.0% | -1.4074 | 100.0% | 0.95 | 1.30 |
| NLLB-600M | A | natural | 7,42 | 0.6324 ± 0.0029 | 100.0% | -0.5403 ± 0.0034 | 100.0% | 18.43 | 24.31 |
| NLLB-600M | B | balanced | 7,42 | 0.5938 ± 0.0628 | 99.8% | -0.5956 ± 0.1080 | 99.8% | 15.41 | 22.94 |

## 2. COMET theo stratum + macro

| Model | Data | Recipe | 1688_title | 1688_attribute | 1688_description | macro_1688 | tiki_title | tiki_attribute | tiki_description | macro_tiki |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EcomGPT-7B | A | natural | 0.8072 | 0.8669 | 0.7711 | — | -0.0623 | 0.1058 | 0.0090 | — |
| EcomGPT-7B | A | balanced | 0.7984 | 0.8513 | 0.7737 | — | -0.0470 | 0.0814 | -0.0182 | — |
| EcomGPT-7B | B | balanced | 0.8079 | 0.8227 | 0.5477 | — | -0.0470 | -0.1438 | -0.3754 | — |
| Moses | A | natural | 0.7763 | 0.8392 | 0.6928 | — | -0.2163 | -0.0035 | -0.8029 | — |
| Moses | B | balanced | 0.7717 | 0.8203 | 0.8322 | — | -0.1872 | 0.0114 | -0.4217 | — |
| NLLB-600M | A | natural | 0.6118 | 0.7471 | 0.4290 | — | -0.5597 | -0.1689 | -1.2282 | — |
| NLLB-600M | B | balanced | 0.5610 | 0.6829 | 0.4693 | — | -0.6361 | -0.2040 | -1.2642 | — |
| OpenNMT | A | natural | 0.2536 | 0.4506 | 0.3222 | — | -1.5449 | -0.9952 | -1.9030 | — |
| OpenNMT | B | natural | 0.2823 | 0.3761 | 0.3716 | — | -1.4532 | -1.2438 | -1.6053 | — |
| Qwen3-4B | A | natural | 0.8242 | 0.8791 | 0.8610 | — | -0.0313 | 0.1093 | 0.2241 | — |
| Qwen3-4B | A | balanced | 0.8250 | 0.8694 | 0.8467 | — | -0.0412 | 0.0954 | 0.2310 | — |
| Qwen3-4B | B | balanced | 0.8242 | 0.8431 | 0.9035 | — | -0.0458 | 0.0870 | 0.2402 | — |
| Qwen3-4B | B | natural | 0.8012 | 0.8536 | 0.9070 | — | -0.0456 | 0.0783 | 0.2469 | — |
| Qwen3-8B | A | natural | 0.8288 | 0.8877 | 0.8462 | — | -0.0289 | 0.0877 | 0.2096 | — |
| Qwen3-8B | B | balanced | 0.8329 | 0.8563 | 0.9076 | — | -0.0449 | 0.1132 | 0.1973 | — |
| Qwen3-8B | B | natural | 0.8139 | 0.8578 | 0.9128 | — | -0.0349 | 0.1187 | 0.2249 | — |

## Ghi chú

- Bảng sinh từ `.artifacts/canonical/*/seed*.parquet` — index 831 dòng cố định, hyp rỗng = 0, **không lọc dòng**. Quy ước đầy đủ: `reports/EVAL_PROTOCOL.md`.
- **Xếp hạng chỉ dùng COMET-da 1688.** COMET-QE, BLEU, chrF là chẩn đoán.
- `COMET-da`/`COMET-QE` là mean ± std qua seed (chỉ có std khi ≥2 seed).
- `BLEU sent`/`chrF sent` = trung bình điểm câu (không phải corpus BLEU).
- Cell không có score từng dòng (loại khỏi bảng): Qwen3-8B A.
- Human eval Tiki: **MISSING** (sheet đã sinh, chưa có người chấm).
- Coverage <100%: NLLB-600M B (99.8%/99.8%) — đã zero-fill.
