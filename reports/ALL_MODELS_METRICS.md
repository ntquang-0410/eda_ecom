# Bảng tổng hợp mọi metric (canonical)

> Sinh bởi `tools/all_models_table.py` từ artefact canonical. Số liệu thay thế bảng cũ khi có khác biệt (zero-fill + coverage).

## 1. Tổng hợp

| Model | Data | Recipe | Seeds | COMET-da 1688 | Cov 1688 | COMET-QE Tiki (chẩn đoán) | Cov Tiki | BLEU sent | chrF sent |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | balanced | 42 | 0.8553 | 100.0% | 0.0518 | 98.6% | 48.25 | 48.03 |
| Qwen3-8B | A | natural | 42 | 0.8567 | 100.0% | 0.0487 | 98.6% | 51.77 | 51.37 |
| Qwen3-8B | B | balanced | 42 | 0.8572 | 100.0% | 0.0470 | 98.6% | 53.81 | 54.56 |
| Qwen3-8B | B | natural | 42 | 0.8514 | 100.0% | 0.0576 | 98.6% | 51.67 | 52.26 |
| Qwen3-4B | A | balanced | 42 | 0.8477 | 100.0% | 0.0483 | 98.6% | 46.27 | 46.79 |
| Qwen3-4B | B | balanced | 42,123 | 0.8475 ± 0.0006 | 100.0% | 0.0438 ± 0.0031 | 98.6% | 50.66 | 51.83 |
| Qwen3-4B | B | natural | 42 | 0.8436 | 100.0% | 0.0432 | 98.6% | 49.32 | 50.80 |
| EcomGPT-7B | A | balanced | 42 | 0.8156 | 100.0% | -0.0041 | 100.0% | 37.03 | 39.24 |
| EcomGPT-7B | A | natural | 42 | 0.8250 | 100.0% | -0.0018 | 100.0% | 39.33 | 41.00 |
| EcomGPT-7B | B | balanced | 42 | 0.7634 | 100.0% | -0.1387 | 97.1% | 31.10 | 33.86 |
| Moses | A | natural | 42 | 0.7862 | 100.0% | -0.2475 | 100.0% | 42.48 | 44.82 |
| Moses | B | balanced | 42 | 0.8037 | 100.0% | -0.1663 | 100.0% | 49.06 | 50.90 |
| OpenNMT | A | natural | 42 | 0.3488 | 100.0% | -1.4138 | 100.0% | 1.26 | 1.86 |
| OpenNMT | B | natural | 42 | 0.3387 | 100.0% | -1.4059 | 100.0% | 0.95 | 1.30 |
| NLLB-600M | A | natural | 7,42 | 0.6324 ± 0.0029 | 100.0% | -0.5502 ± 0.0030 | 100.0% | 18.43 | 24.31 |
| NLLB-600M | B | balanced | 7,42 | 0.5938 ± 0.0628 | 99.8% | -0.6036 ± 0.1042 | 99.8% | 15.41 | 22.94 |
| qwen4b_a |  |  | 7,42,123 | 0.8542 ± 0.0037 | 100.0% | 0.0597 ± 0.0030 | 100.0% | 46.76 | 47.25 |

## 2. COMET theo stratum + macro

`macro_unw` = trung bình 3 stratum; `macro_w` = trọng số theo số dòng (bằng mean tổng).

| Model | Data | Recipe | 1688_title | 1688_attribute | 1688_description | macro_unw_1688 | macro_w_1688 | tiki_title | tiki_attribute | tiki_description | macro_unw_tiki | macro_w_tiki |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EcomGPT-7B | A | natural | 0.8072 | 0.8669 | 0.7711 | 0.8151 | 0.8250 | -0.0623 | 0.0927 | -0.0337 | -0.0011 | -0.0018 |
| EcomGPT-7B | A | balanced | 0.7984 | 0.8513 | 0.7737 | 0.8078 | 0.8156 | -0.0470 | 0.0737 | -0.0484 | -0.0072 | -0.0041 |
| EcomGPT-7B | B | balanced | 0.8079 | 0.8227 | 0.5477 | 0.7261 | 0.7634 | -0.0470 | -0.1322 | -0.3898 | -0.1897 | -0.1387 |
| Moses | A | natural | 0.7763 | 0.8392 | 0.6928 | 0.7694 | 0.7862 | -0.2163 | -0.0145 | -0.7947 | -0.3418 | -0.2475 |
| Moses | B | balanced | 0.7717 | 0.8203 | 0.8322 | 0.8081 | 0.8037 | -0.1872 | -0.0110 | -0.4223 | -0.2068 | -0.1663 |
| NLLB-600M | A | natural | 0.6118 | 0.7471 | 0.4290 | 0.5960 | 0.6324 | -0.5597 | -0.1882 | -1.2497 | -0.6659 | -0.5502 |
| NLLB-600M | B | balanced | 0.5610 | 0.6829 | 0.4693 | 0.5710 | 0.5938 | -0.6361 | -0.2177 | -1.2911 | -0.7149 | -0.6036 |
| OpenNMT | A | natural | 0.2536 | 0.4506 | 0.3222 | 0.3421 | 0.3488 | -1.5449 | -1.0014 | -1.8979 | -1.4814 | -1.4138 |
| OpenNMT | B | natural | 0.2823 | 0.3761 | 0.3716 | 0.3433 | 0.3387 | -1.4532 | -1.2503 | -1.5942 | -1.4326 | -1.4059 |
| qwen4b_a |  |  | 0.8242 | 0.8791 | 0.8610 | 0.8548 | 0.8542 | -0.0313 | 0.0953 | 0.2254 | 0.0964 | 0.0597 |
| Qwen3-4B | A | balanced | 0.8250 | 0.8694 | 0.8467 | 0.8470 | 0.8477 | -0.0412 | 0.0826 | 0.2124 | 0.0846 | 0.0483 |
| Qwen3-4B | B | balanced | 0.8242 | 0.8431 | 0.9035 | 0.8569 | 0.8475 | -0.0458 | 0.0716 | 0.2209 | 0.0822 | 0.0438 |
| Qwen3-4B | B | natural | 0.8012 | 0.8536 | 0.9070 | 0.8539 | 0.8436 | -0.0456 | 0.0665 | 0.2270 | 0.0826 | 0.0432 |
| Qwen3-8B | A | natural | 0.8288 | 0.8877 | 0.8462 | 0.8542 | 0.8567 | -0.0289 | 0.0775 | 0.1928 | 0.0805 | 0.0487 |
| Qwen3-8B | A | balanced | 0.8220 | 0.8858 | 0.8569 | 0.8549 | 0.8553 | -0.0352 | 0.0877 | 0.2059 | 0.0861 | 0.0518 |
| Qwen3-8B | B | balanced | 0.8329 | 0.8563 | 0.9076 | 0.8656 | 0.8572 | -0.0449 | 0.0991 | 0.1814 | 0.0785 | 0.0470 |
| Qwen3-8B | B | natural | 0.8139 | 0.8578 | 0.9128 | 0.8615 | 0.8514 | -0.0349 | 0.1032 | 0.2069 | 0.0917 | 0.0576 |

## Ghi chú

- Bảng sinh từ `.artifacts/canonical/*/seed*.parquet` — index 858 dòng cố định, hyp rỗng = 0, **không lọc dòng**. Quy ước đầy đủ: `reports/EVAL_PROTOCOL.md`.
- **Xếp hạng chỉ dùng COMET-da 1688.** COMET-QE, BLEU, chrF là chẩn đoán.
- `COMET-da`/`COMET-QE` là mean ± std qua seed (chỉ có std khi ≥2 seed).
- `BLEU sent`/`chrF sent` = trung bình điểm câu (không phải corpus BLEU).
- Cell không có score từng dòng (loại khỏi bảng): không.
- Human eval Tiki: **MISSING** (sheet đã sinh, chưa có người chấm).
- Coverage <100%: Qwen3-8B A (100.0%/98.6%), Qwen3-8B A (100.0%/98.6%), Qwen3-8B B (100.0%/98.6%), Qwen3-8B B (100.0%/98.6%), Qwen3-4B A (100.0%/98.6%), Qwen3-4B B (100.0%/98.6%), Qwen3-4B B (100.0%/98.6%), EcomGPT-7B B (100.0%/97.1%), NLLB-600M B (99.8%/99.8%) — đã zero-fill.
