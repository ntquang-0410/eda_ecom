# Bảng tổng hợp mọi metric (canonical)

> Sinh bởi `tools/all_models_table.py` từ artefact canonical. Số liệu thay thế bảng cũ khi có khác biệt (zero-fill + coverage).

## 1. Tổng hợp

| Model | Data | Recipe | Seeds | COMET-da 1688 | Cov 1688 | COMET-QE Tiki (chẩn đoán) | Cov Tiki | BLEU sent | chrF sent |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | balanced | 42 | 0.8553 | 100.0% | — | 98.6% | 48.25 | 48.03 |
| Qwen3-8B | A | natural | 42 | 0.8567 | 100.0% | 0.0487 | 98.6% | 51.77 | 51.37 |
| Qwen3-8B | B | balanced | 42 | 0.8572 | 100.0% | 0.0470 | 98.6% | 53.81 | 54.56 |
| Qwen3-8B | B | natural | 42 | 0.8514 | 100.0% | 0.0576 | 98.6% | 51.67 | 52.26 |
| Qwen3-4B | A | balanced | 42 | 0.8477 | 100.0% | — | 98.6% | 46.27 | 46.79 |
| Qwen3-4B | B | balanced | 42,123 | 0.8475 ± 0.0006 | 100.0% | — | 98.6% | 50.66 | 51.83 |
| Qwen3-4B | B | natural | 42 | 0.8436 | 100.0% | — | 98.6% | 49.32 | 50.80 |
| EcomGPT-7B | A | balanced | 42 | 0.8156 | 100.0% | -0.0041 | 100.0% | 37.03 | 39.24 |
| EcomGPT-7B | A | natural | 42 | 0.8250 | 100.0% | — | 100.0% | 39.33 | 41.00 |
| EcomGPT-7B | B | balanced | 42 | 0.7634 | 100.0% | — | 97.1% | 31.10 | 33.86 |
| Moses | A | natural | 42 | 0.7862 | 100.0% | — | 100.0% | 42.48 | 44.82 |
| Moses | B | balanced | 42 | 0.8037 | 100.0% | — | 100.0% | 49.06 | 50.90 |
| OpenNMT | A | natural | 42 | 0.3488 | 100.0% | — | 100.0% | 1.26 | 1.86 |
| OpenNMT | B | natural | 42 | 0.3387 | 100.0% | — | 100.0% | 0.95 | 1.30 |
| NLLB-600M | A | natural | 7,42 | 0.6324 ± 0.0029 | 100.0% | — | 100.0% | 18.43 | 24.31 |
| NLLB-600M | B | balanced | 7,42 | 0.5938 ± 0.0628 | 99.8% | — | 99.8% | 15.41 | 22.94 |
| qwen4b_a |  |  | 7,42,123 | 0.8542 ± 0.0037 | 100.0% | 0.0597 ± 0.0030 | 100.0% | 46.76 | 47.25 |

## 2. COMET theo stratum + macro

`macro_unw` = trung bình 3 stratum; `macro_w` = trọng số theo số dòng (bằng mean tổng).

| Model | Data | Recipe | 1688_title | 1688_attribute | 1688_description | macro_unw_1688 | macro_w_1688 | tiki_title | tiki_attribute | tiki_description | macro_unw_tiki | macro_w_tiki |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EcomGPT-7B | A | balanced | 0.7984 | 0.8513 | 0.7737 | 0.8078 | 0.8156 | -0.0470 | 0.0737 | -0.0484 | -0.0072 | -0.0041 |
| qwen4b_a |  |  | 0.8242 | 0.8791 | 0.8610 | 0.8548 | 0.8542 | -0.0313 | 0.0953 | 0.2254 | 0.0964 | 0.0597 |
| Qwen3-8B | A | natural | 0.8288 | 0.8877 | 0.8462 | 0.8542 | 0.8567 | -0.0289 | 0.0775 | 0.1928 | 0.0805 | 0.0487 |
| Qwen3-8B | B | balanced | 0.8329 | 0.8563 | 0.9076 | 0.8656 | 0.8572 | -0.0449 | 0.0991 | 0.1814 | 0.0785 | 0.0470 |
| Qwen3-8B | B | natural | 0.8139 | 0.8578 | 0.9128 | 0.8615 | 0.8514 | -0.0349 | 0.1032 | 0.2069 | 0.0917 | 0.0576 |

## Ghi chú

- Bảng sinh từ `.artifacts/canonical/*/seed*.parquet` — index 858 dòng cố định, hyp rỗng = 0, **không lọc dòng**. Quy ước đầy đủ: `reports/EVAL_PROTOCOL.md`.
- **Xếp hạng chỉ dùng COMET-da 1688.** COMET-QE, BLEU, chrF là chẩn đoán.
- `COMET-da`/`COMET-QE` là mean ± std qua seed (chỉ có std khi ≥2 seed).
- `BLEU sent`/`chrF sent` = trung bình điểm câu (không phải corpus BLEU).
- Cell không có score từng dòng (loại khỏi bảng): không.
- Human eval Tiki: **MISSING** (sheet đã sinh, chưa có người chấm).
- **QE ẩn cho cell chưa pass adapter provenance (4.4)** — xem `reports/ADAPTER_PROVENANCE_AUDIT.md`; COMET-da vẫn đầy đủ.
- Coverage <100%: Qwen3-8B A (100.0%/98.6%), Qwen3-8B A (100.0%/98.6%), Qwen3-8B B (100.0%/98.6%), Qwen3-8B B (100.0%/98.6%), Qwen3-4B A (100.0%/98.6%), Qwen3-4B B (100.0%/98.6%), Qwen3-4B B (100.0%/98.6%), EcomGPT-7B B (100.0%/97.1%), NLLB-600M B (99.8%/99.8%) — đã zero-fill.
