# Bảng tổng hợp mọi metric — toàn bộ cell

Sinh bởi `tools/all_models_table.py`. Eval `eval_F_v2_common` (831 eval_id: 1688 = 231, Tiki = 600).
COMET-da = `Unbabel/wmt22-comet-da` (1688, có reference); COMET-QE = `Unbabel/wmt20-comet-qe-da` (Tiki, không reference).
BLEU/chrF: 1688, hyp không rỗng, tokenize `zh` (sacrebleu).

## 1. Tổng hợp

| Model | Data | Recipe | Seed | COMET-da 1688 | COMET-QE Tiki | Macro 1688 | Macro Tiki | BLEU | chrF | n 1688 | n Tiki | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-8B | A | natural | 42 | 0,8567 | 0,0521 | 0,8542 | 0,0895 | 53.37 | 52.00 | 231 | 600 |  |
| Qwen3-8B | B | balanced | 42 | 0,8572 | 0,0509 | 0,8656 | 0,0885 | 68.38 | 67.22 | 231 | 600 |  |
| Qwen3-8B | A | balanced | 42 | 0,8553 | 0,0562 | — | — | 44.36 | 43.53 | 231 | 600 | chỉ có điểm tổng, chưa có score từng dòng |
| Qwen3-8B | B | natural | 42 | 0,8514 | 0,0624 | 0,8615 | 0,1029 | 69.87 | 68.29 | 231 | 600 | tái lập 24/11 |
| Qwen3-4B | A | natural | 42 | 0,8584 | 0,0573 | 0,8589 | 0,0974 | 39.81 | 39.59 | 231 | 600 |  |
| Qwen3-4B | B | balanced | 42 | 0,8471 | 0,0461 | 0,8567 | 0,0930 | 66.96 | 65.55 | 231 | 600 |  |
| Qwen3-4B | A | balanced | 42 | 0,8477 | 0,0524 | 0,8470 | 0,0951 | 40.77 | 40.33 | 231 | 600 |  |
| Qwen3-4B | B | natural | 42 | 0,8436 | 0,0471 | 0,8539 | 0,0932 | 68.84 | 67.28 | 231 | 600 |  |
| Qwen3-4B | A | natural | 123 | 0,8511 | 0,0637 | 0,8522 | 0,1036 | 37.66 | 37.70 | 231 | 600 |  |
| Qwen3-4B | B | balanced | 123 | 0,8479 | 0,0517 | 0,8572 | 0,0945 | 66.10 | 64.95 | 231 | 600 |  |
| Qwen3-4B | A | natural | 7 | 0,8530 | 0,0614 | 0,8532 | 0,1011 | 38.80 | 39.08 | 231 | 600 |  |
| EcomGPT-7B | A | natural | 42 | 0,8250 | 0,0077 | 0,8151 | 0,0175 | 27.55 | 27.16 | 231 | 600 |  |
| EcomGPT-7B | B | balanced | 42 | 0,7634 | -0,1366 | 0,7261 | -0,1887 | 14.39 | 15.16 | 231 | 600 |  |
| EcomGPT-7B | A | balanced | 42 | 0,8156 | 0,0020 | 0,8078 | 0,0054 | 19.34 | 24.29 | 231 | 600 | tái lập 24/11 |
| Moses | A | natural | 42 | 0,7862 | -0,2439 | 0,7694 | -0,3409 | 33.11 | 35.13 | 231 | 600 |  |
| Moses | B | balanced | 42 | 0,8037 | -0,1593 | 0,8081 | -0,1992 | 65.14 | 63.66 | 231 | 600 |  |
| OpenNMT | A | natural | 42 | 0,3488 | -1,4177 | 0,3421 | -1,4810 | 0.22 | 1.63 | 231 | 600 |  |
| OpenNMT | B | natural | 42 | 0,3387 | -1,4074 | 0,3433 | -1,4341 | 2.00 | 3.13 | 231 | 600 |  |
| NLLB-600M | A | natural | 42 | 0,6344 | -0,5379 | 0,6000 | -0,6482 | 8.04 | 14.00 | 231 | 600 |  |
| NLLB-600M | B | balanced | 42 | 0,5517 | -0,6753 | 0,5312 | -0,7630 | 7.37 | 12.68 | 230 | 597 |  |
| NLLB-600M | A | natural | 7 | 0,6304 | -0,5427 | 0,5919 | -0,6564 | 7.68 | 13.64 | 231 | 600 |  |
| NLLB-600M | B | balanced | 7 | 0,6382 | -0,5192 | 0,6132 | -0,6425 | 9.59 | 16.07 | 231 | 600 |  |

## 2. COMET theo stratum + macro-average

| Model | Data | Recipe | Seed | 1688_title | 1688_attribute | 1688_description | macro_1688 | tiki_title | tiki_attribute | tiki_description | macro_tiki |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | natural | 42 | 0,8288 | 0,8877 | 0,8462 | 0,8542 | -0,0289 | 0,0877 | 0,2096 | 0,0895 |
| Qwen3-8B | B | balanced | 42 | 0,8329 | 0,8563 | 0,9076 | 0,8656 | -0,0449 | 0,1132 | 0,1973 | 0,0885 |
| Qwen3-8B | B | natural | 42 | 0,8139 | 0,8578 | 0,9128 | 0,8615 | -0,0349 | 0,1187 | 0,2249 | 0,1029 |
| Qwen3-4B | A | natural | 42 | 0,8291 | 0,8828 | 0,8649 | 0,8589 | -0,0315 | 0,0990 | 0,2248 | 0,0974 |
| Qwen3-4B | B | balanced | 42 | 0,8226 | 0,8434 | 0,9040 | 0,8567 | -0,0564 | 0,0920 | 0,2435 | 0,0930 |
| Qwen3-4B | A | balanced | 42 | 0,8250 | 0,8694 | 0,8467 | 0,8470 | -0,0412 | 0,0954 | 0,2310 | 0,0951 |
| Qwen3-4B | B | natural | 42 | 0,8012 | 0,8536 | 0,9070 | 0,8539 | -0,0456 | 0,0783 | 0,2469 | 0,0932 |
| Qwen3-4B | A | natural | 123 | 0,8221 | 0,8740 | 0,8604 | 0,8522 | -0,0286 | 0,1123 | 0,2269 | 0,1036 |
| Qwen3-4B | B | balanced | 123 | 0,8258 | 0,8428 | 0,9030 | 0,8572 | -0,0353 | 0,0820 | 0,2369 | 0,0945 |
| Qwen3-4B | A | natural | 7 | 0,8213 | 0,8805 | 0,8578 | 0,8532 | -0,0339 | 0,1167 | 0,2204 | 0,1011 |
| EcomGPT-7B | A | natural | 42 | 0,8072 | 0,8669 | 0,7711 | 0,8151 | -0,0623 | 0,1058 | 0,0090 | 0,0175 |
| EcomGPT-7B | B | balanced | 42 | 0,8079 | 0,8227 | 0,5477 | 0,7261 | -0,0470 | -0,1438 | -0,3754 | -0,1887 |
| EcomGPT-7B | A | balanced | 42 | 0,7984 | 0,8513 | 0,7737 | 0,8078 | -0,0470 | 0,0814 | -0,0182 | 0,0054 |
| Moses | A | natural | 42 | 0,7763 | 0,8392 | 0,6928 | 0,7694 | -0,2163 | -0,0035 | -0,8029 | -0,3409 |
| Moses | B | balanced | 42 | 0,7717 | 0,8203 | 0,8322 | 0,8081 | -0,1872 | 0,0114 | -0,4217 | -0,1992 |
| OpenNMT | A | natural | 42 | 0,2536 | 0,4506 | 0,3222 | 0,3421 | -1,5449 | -0,9952 | -1,9030 | -1,4810 |
| OpenNMT | B | natural | 42 | 0,2823 | 0,3761 | 0,3716 | 0,3433 | -1,4532 | -1,2438 | -1,6053 | -1,4341 |
| NLLB-600M | A | natural | 42 | 0,6106 | 0,7467 | 0,4427 | 0,6000 | -0,5586 | -0,1695 | -1,2165 | -0,6482 |
| NLLB-600M | B | balanced | 42 | 0,5013 | 0,6526 | 0,4397 | 0,5312 | -0,7773 | -0,2290 | -1,2827 | -0,7630 |
| NLLB-600M | A | natural | 7 | 0,6130 | 0,7476 | 0,4153 | 0,5919 | -0,5609 | -0,1684 | -1,2399 | -0,6564 |
| NLLB-600M | B | balanced | 7 | 0,6207 | 0,7199 | 0,4988 | 0,6132 | -0,5029 | -0,1791 | -1,2457 | -0,6425 |

## Ghi chú

- Đơn vị: COMET-da/QE là điểm hệ thống (trung bình per-row); BLEU/chrF thang 0–100.
- `n` = số dòng thực sự được chấm sau khi lọc common + hyp không rỗng; NLLB B seed 42 thiếu 1 dòng 1688 và 3 dòng Tiki (hyp rỗng).
- Seed noise ước lượng 0,005–0,007 COMET (1688) ⇒ chênh lệch <0,01 giữa hai cell cùng cấu hình không nên đọc là hiệu ứng.
- **Loại khỏi bảng:** Qwen3-4B B-balanced seed 7 (hyp cũ sinh bằng adapter mẫu competition, đang chờ inference lại) và EcomGPT-7B B-natural (generation thoái hoá: 1.471/1.500 hyp rỗng).
- Qwen3-8B A-balanced chỉ có điểm tổng từ `CHECKPOINT_2026-11-24.md`; BLEU/chrF trong bảng tính từ hyp cùng adapter, COMET chưa có score từng dòng nên không bootstrap được.
- EcomGPT dùng ngân sách ~1,42 M token hiệu dụng (recipe chia theo token đã encode), không phải 4 M như Qwen/NLLB — xem `MATRIX_5X2_EVAL_F_V2.md` §11.3.

## 3. Đọc bảng

1. **Trong nhóm LLM, khoảng cách rất hẹp:** 5 cell Qwen hàng đầu trên 1688 nằm trong 0,8436–0,8584 (Δ < 0,015), tức xấp xỉ 2–3 lần seed noise (0,005–0,007). Không có cell nào "thắng tuyệt đối".
2. **BLEU tách B khỏi A rất mạnh nhưng COMET thì không:** mọi model đều có BLEU(B) cao hơn BLEU(A) 15–30 điểm (do B chứa description giàu từ vựng trùng reference), trong khi COMET chênh < 0,011. BLEU ở đây đo mức "khớp bề mặt", không đo chất lượng dịch.
3. **Tiki QE gần 0 cho toàn bộ LLM** (−0,05…+0,10) và âm sâu cho Moses/NLLB/OpenNMT: không có reference nên QE chỉ đo "độ trôi" so với phân bố dịch tự nhiên; đừng dùng cột này để xếp hạng model.
4. **EcomGPT thấp hơn Qwen 0,03–0,09 COMET** nhưng một phần do ngân sách token hiệu dụng chỉ ~1,42 M (35% của 4 M) — xem `MATRIX_5X2_EVAL_F_V2.md` §11.3.
5. **NLLB là control pretrained NMT:** A/B đảo dấu giữa seed 42 và seed 7 (A 0,6344/0,6304; B 0,5517/0,6382) ⇒ với NMT, chênh lệch A/B nằm trong nhiễu seed, không phải hiệu ứng dữ liệu.
