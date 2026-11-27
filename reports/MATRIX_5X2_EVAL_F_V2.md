# Matrix 5 model × 2 data — COMET

Eval: `eval_F_v2` — 858 dòng sạch nội dung; tập common cho mọi cell = 831 dòng (1688: 231, Tiki: 600). 1688 = COMET-da có reference; Tiki = COMET-QE không reference. BLEU/chrF tính trên phần 1688 có reference và hypothesis không rỗng.

## 1. Điểm tổng theo cell

| model | data | 1688 da | Tiki QE | BLEU 1688 | chrF 1688 | n 1688 | n Tiki |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | 0.8567 | 0.0521 | 53.37 | 52.00 | 231 | 600 |
| Qwen3-8B | B | 0.8572 | 0.0509 | 68.38 | 67.22 | 231 | 600 |
| Qwen3-4B | A | 0.8584 | 0.0573 | 39.81 | 39.59 | 231 | 600 |
| Qwen3-4B | B | 0.8471 | 0.0461 | 66.96 | 65.55 | 231 | 600 |
| EcomGPT-7B | A | 0.8250 | 0.0077 | 27.55 | 27.16 | 231 | 600 |
| EcomGPT-7B | B | 0.7634 | -0.1366 | 14.39 | 15.16 | 231 | 600 |
| Moses | A | 0.7862 | -0.2439 | 33.11 | 35.13 | 231 | 600 |
| Moses | B | 0.8037 | -0.1593 | 65.14 | 63.66 | 231 | 600 |
| OpenNMT | A | 0.3488 | -1.4177 | 0.22 | 1.63 | 231 | 600 |
| OpenNMT | B-natural | 0.3387 | -1.4074 | 2.00 | 3.13 | 231 | 600 |

## 2. Theo stratum

| model | data | 1688 Title | 1688 Attr | 1688 Desc | Tiki Title | Tiki Attr | Tiki Desc |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | 0.829 | 0.888 | 0.846 | -0.029 | 0.088 | 0.210 |
| Qwen3-8B | B | 0.833 | 0.856 | 0.908 | -0.045 | 0.113 | 0.197 |
| Qwen3-4B | A | 0.829 | 0.883 | 0.865 | -0.032 | 0.099 | 0.225 |
| Qwen3-4B | B | 0.823 | 0.843 | 0.904 | -0.056 | 0.092 | 0.243 |
| EcomGPT-7B | A | 0.807 | 0.867 | 0.771 | -0.062 | 0.106 | 0.009 |
| EcomGPT-7B | B | 0.808 | 0.823 | 0.548 | -0.047 | -0.144 | -0.375 |
| Moses | A | 0.776 | 0.839 | 0.693 | -0.216 | -0.004 | -0.803 |
| Moses | B | 0.772 | 0.820 | 0.832 | -0.187 | 0.011 | -0.422 |
| OpenNMT | A | 0.254 | 0.451 | 0.322 | -1.545 | -0.995 | -1.903 |
| OpenNMT | B-natural | 0.282 | 0.376 | 0.372 | -1.453 | -1.244 | -1.605 |

## 3. So sánh model trong cùng recipe (paired bootstrap, 2.000 lần, cluster theo sản phẩm)

Δ = model − Qwen cùng recipe. `*` = CI 95% không chứa 0.

| recipe | model | Δ 1688 da | Δ Tiki QE | n 1688 | n Tiki |
|---|---|---:|---:|---:|---:|
| A | Qwen3-4B | +0.0017 [-0.0073; +0.0116] | +0.0052 [-0.0122; +0.0226] | 231 | 600 |
| A | EcomGPT-7B | -0.0317 [-0.0452; -0.0179]* | -0.0444 [-0.0662; -0.0224]* | 231 | 600 |
| A | Moses | -0.0705 [-0.0858; -0.0550]* | -0.2960 [-0.3378; -0.2583]* | 231 | 600 |
| A | OpenNMT | -0.5078 [-0.5255; -0.4887]* | -1.4697 [-1.5329; -1.4034]* | 231 | 600 |
| B | Qwen3-4B | -0.0101 [-0.0174; -0.0035]* | -0.0049 [-0.0221; +0.0127] | 231 | 600 |
| B | EcomGPT-7B | -0.0938 [-0.1147; -0.0742]* | -0.1876 [-0.2270; -0.1490]* | 231 | 600 |
| B | Moses | -0.0534 [-0.0663; -0.0418]* | -0.2102 [-0.2427; -0.1777]* | 231 | 600 |
| B | OpenNMT | -0.5185 [-0.5320; -0.5036]* | -1.4584 [-1.5056; -1.4056]* | 231 | 600 |

## 4. Hiệu ứng dữ liệu B − A trong cùng model

Δ = B − A. `*` = CI 95% không chứa 0.

| model | Δ 1688 da | Δ Tiki QE | Δ 1688 Title | Δ 1688 Attr | Δ 1688 Desc | Δ Tiki Title | Δ Tiki Attr | Δ Tiki Desc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | +0.0005 [-0.0099; +0.0112] | -0.0011 [-0.0186; +0.0160] | +0.0041 [-0.0021; +0.0117] | -0.0314 [-0.0496; -0.0140]* | +0.0614 [+0.0411; +0.0907]* | -0.0161 [-0.0400; +0.0058] | +0.0255 [-0.0024; +0.0562] | -0.0124 [-0.0652; +0.0400] |
| Qwen3-4B | -0.0113 [-0.0217; -0.0014]* | -0.0112 [-0.0262; +0.0046] | -0.0065 [-0.0133; +0.0002] | -0.0394 [-0.0622; -0.0177]* | +0.0391 [+0.0313; +0.0464]* | -0.0248 [-0.0468; -0.0042]* | -0.0070 [-0.0399; +0.0268] | +0.0186 [+0.0032; +0.0392]* |
| EcomGPT-7B | -0.0616 [-0.0805; -0.0425]* | -0.1443 [-0.1810; -0.1076]* | +0.0007 [-0.0096; +0.0105] | -0.0442 [-0.0668; -0.0230]* | -0.2234 [-0.2945; -0.1532]* | +0.0152 [-0.0054; +0.0352] | -0.2496 [-0.3349; -0.1715]* | -0.3844 [-0.5000; -0.2706]* |
| Moses | +0.0176 [+0.0054; +0.0297]* | +0.0847 [+0.0593; +0.1115]* | -0.0045 [-0.0152; +0.0060] | -0.0189 [-0.0389; -0.0009]* | +0.1394 [+0.1175; +0.1613]* | +0.0291 [+0.0069; +0.0508]* | +0.0150 [-0.0399; +0.0701] | +0.3812 [+0.3495; +0.4139]* |
| OpenNMT (B-natural) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |


**Lưu ý:** OpenNMT-B là **B-natural** (49.774 dòng, 18,83M token), không phải B-balanced (52.131 draw/25.453 unique); dòng B−A của OpenNMT bị loại khỏi bảng hiệu ứng dữ liệu. Xem `reports/OPENNMT_FAIRNESS.md`.

## 5. Đọc kết quả

- **Xếp hạng giữ nguyên:** Qwen3-4B A (0,8584) ≈ Qwen3-8B B (0,8572) ≈ Qwen3-8B A (0,8567) > Qwen3-4B B (0,8471) > EcomGPT A (0,8250) > Moses B (0,8037) > Moses A (0,7862) > EcomGPT B (0,7634) >> OpenNMT.
- **Qwen3-8B A vs B:** B−A = +0,0005 [−0,0099; +0,0112] trên 1688 và −0,0011 [−0,0186; +0,0160] trên Tiki, đều không có ý nghĩa. Đánh đổi stratum vẫn rõ: description +0,0614* nhưng attribute −0,0314*.
- **Qwen3-4B vs Qwen3-8B:** A +0,0017 [−0,0073; +0,0116] → hòa; B −0,0101 [−0,0174; −0,0035] → 8B nhỉnh hơn. Description: A +0,0187 [−0,0007; +0,0490] (4B nhỉnh, chưa đủ mạnh), B −0,0036* (8B nhỉnh rất nhẹ) trên 1688; Tiki description B +0,0462* (4B nhỉnh). Kết luận "4B đủ capacity cho description" vẫn đứng, nhưng không còn là "bằng tuyệt đối" ở mọi stratum.
- **Qwen3-4B A vs B:** B−A = −0,0113* trên 1688; description +0,0391* nhưng attribute −0,0394*. Cùng kiểu đánh đổi với 8B.
- **BLEU vs COMET:** Qwen3-4B A có BLEU 39,81 so với 53,37 của 8B A, nhưng COMET A +0,0017 (hòa). 4B sinh description đủ trường hơn; 8B đôi khi co về `其他`, nên BLEU phạt 4B trong khi COMET không.
- **EcomGPT-7B:** B kém A −0,0616* trên 1688 và −0,1443* trên Tiki; description B 0,548 so với A 0,771.
- **Moses:** B hơn A +0,0176* trên 1688 và +0,0847* trên Tiki, chủ yếu nhờ description (+0,1394*); nhưng QE Tiki vẫn âm.
- **OpenNMT (B-natural):** vẫn là outlier (0,349/0,339 trên 1688; −1,42/−1,41 trên Tiki), không đổi kết luận.
- **Tiki QE:** chỉ Qwen dương; EcomGPT A +0,0077 (gần 0), EcomGPT B −0,1366, Moses âm, OpenNMT rất âm.

## 6. Caveat

- `eval_F_v2` = **858 dòng** (1688: 90 title + 96 attribute + 45 description; Tiki: 291 title + 224 attribute + 112 description). Tập common cho mọi cell = **831 dòng** (1688: 231; Tiki: 600), do Qwen thiếu 15 prompt dài và EcomGPT-B thiếu 21 Tiki attribute.
- Vì 1688 description chỉ còn **45 dòng**, CI của stratum này rộng; mọi kết luận riêng cho 1688-description phải đọc kèm n.
- So với `eval_F` gốc, chênh lệch tuyệt đối lớn nhất là **0,018 COMET** (Qwen-B Tiki), xếp hạng model không đổi. Điểm trên tập gốc cao hơn ở các dòng leaked, đặc biệt Tiki QE (Qwen +0,09…+0,13), nên báo cáo cũ có xu hướng lạc quan nhẹ.
- Vẫn là 1 seed training; paired bootstrap chỉ phủ variance của eval, không phủ variance khởi tạo/seed.
- `eval_F_v2` được lọc theo 5 khoá: `product_id`, `source_hash`, `source_hash_strong`, `pair_hash`, `near_dup_cluster_id` (gồm `nhom_trung` 1688 + title-cluster Tiki + MinHash LSH 0,8). Chi tiết: `review_audit.json`.
- Moses dùng eflomal thay GIZA++; OpenNMT A/B không cùng compute (48,3 vs 10 epoch) — xem cảnh báo fairness riêng.
## 7. Macro-average theo stratum (mỗi stratum trọng số bằng nhau)

| model | 1688 macro | Tiki macro |
|---|---:|---:|
| Qwen3-8B A | 0,8542 | 0,0895 |
| Qwen3-8B B | **0,8656** | 0,0885 |
| Qwen3-4B A | 0,8589 | **0,0974** |
| Qwen3-4B B | 0,8567 | 0,0930 |
| EcomGPT-7B A | 0,8151 | 0,0175 |
| EcomGPT-7B B | 0,7261 | -0,1887 |
| Moses A | 0,7694 | -0,3409 |
| Moses B | 0,8081 | -0,1992 |
| OpenNMT A | 0,3421 | -1,4810 |
| OpenNMT B | 0,3433 | -1,4341 |

Macro-average đảo nhẹ thứ hạng: Qwen3-8B B dẫn 1688, Qwen3-4B A dẫn Tiki. Chênh lệch giữa 4 cell Qwen vẫn nhỏ (<0,01 trên 1688), không đủ để tuyên bố một cell thắng tuyệt đối.

## 8. Điểm theo bucket độ dài nguồn

Bucket: short <80 ký tự; medium 80–300; long >300.

### 1688

| model | short | medium | long |
|---|---:|---:|---:|
| Qwen3-8B A | **0,8874** | 0,8369 | 0,8476 |
| Qwen3-8B B | 0,8624 | 0,8329 | 0,8947 |
| Qwen3-4B A | 0,8843 | 0,8358 | 0,8619 |
| Qwen3-4B B | 0,8501 | 0,8209 | 0,8915 |
| EcomGPT A | 0,8713 | 0,8139 | 0,7767 |
| EcomGPT B | 0,8429 | 0,8011 | 0,5744 |
| Moses A | 0,8522 | 0,7786 | 0,7018 |
| Moses B | 0,8394 | 0,7706 | 0,8125 |
| OpenNMT A | 0,4685 | 0,2659 | 0,3254 |
| OpenNMT B | 0,3910 | 0,2858 | 0,3594 |

### Tiki

| model | short | medium | long |
|---|---:|---:|---:|
| Qwen3-8B A | 0,0005 | 0,0560 | 0,2147 |
| Qwen3-8B B | 0,0180 | 0,0151 | 0,2048 |
| Qwen3-4B A | 0,0089 | 0,0453 | **0,2301** |
| Qwen3-4B B | -0,0087 | 0,0267 | 0,2492 |
| EcomGPT A | 0,0086 | -0,0041 | 0,0201 |
| EcomGPT B | -0,0982 | -0,0488 | -0,3762 |
| Moses A | -0,0871 | -0,2233 | -0,7806 |
| Moses B | -0,0605 | -0,2060 | -0,4193 |
| OpenNMT A | -1,1593 | -1,6919 | -1,8992 |
| OpenNMT B | -1,2766 | -1,5756 | -1,6129 |

Điểm **tăng theo độ dài** ở Qwen (đặc biệt Tiki), nghĩa là mô hình xử lý description dài tốt hơn title ngắn; EcomGPT B và Moses sụp ở long bucket. Đây là lý do phải báo cáo theo bucket thay vì chỉ trung bình toàn cục.

## 9. Quyết định sử dụng

- **Số liệu chính thức:** `eval_F_v2_common` (831 dòng) — dùng cho mọi so sánh model/recipe.
- **Số liệu phụ:** `eval_F` gốc (1.500) chỉ để đối chiếu với các báo cáo trước; mọi chênh lệch >0,018 phải kiểm tra lại.
- **Không** dùng `eval_F` gốc để tuyên bố "B giúp description" mà không kèm bảng v2; mặc dù kết luận định tính vẫn giữ, mức chênh thay đổi.

## 10. Control bổ sung (2×2, seed 123, NLLB)

Chi tiết: [`CONTROLS_2X2.md`](CONTROLS_2X2.md).

### 10.1. Qwen3-4B — tách quota khỏi field mix (seed 42, eval_F_v2_common)

| Cell | Field | Sampling | 1688 da | Tiki QE |
|---|---|---|---:|---:|
| `qwen4b_a` | A | natural | **0,8584** | **0,0573** |
| `qwen4b_a_balanced` | A | balanced | 0,8477 | 0,0524 |
| `qwen4b_b` | B | balanced | 0,8471 | 0,0461 |
| `qwen4b_b_natural` | B | natural | 0,8436 | 0,0471 |

- A-natural hơn 3 cell còn lại 0,011–0,015 trên 1688 (CI 95% không chứa 0).
- Quota balanced **giảm** A 0,0107 (có ý nghĩa), **không cải thiện** B (+0,0035, không ý nghĩa).
- Khác biệt field (A vs B) nhỏ hơn khác biệt sampling: B-natural ≈ A-balanced.
- Trên Tiki QE mọi khác biệt giữa 4 cell đều không ý nghĩa.

### 10.2. Seed 42 vs 123

| Cell | Seed 42 | Seed 123 | Δ (123−42) |
|---|---:|---:|---:|
| A-natural 1688 | 0,8584 | 0,8511 | **−0,0072** [−0,0144; −0,0009]* |
| A-natural Tiki QE | 0,0573 | 0,0637 | +0,0064 [−0,0040; +0,0174] |
| B-balanced 1688 | 0,8471 | 0,8479 | +0,0008 [−0,0030; +0,0048] |
| B-balanced Tiki QE | 0,0461 | 0,0517 | +0,0056 [−0,0054; +0,0171] |

Dao động seed ước lượng **0,005–0,007 COMET trên 1688** (A-natural −0,0072; B-balanced +0,0008). Vì vậy:
- Khoảng cách A-natural vs A-balanced (0,0107) chỉ ~1,5 lần seed noise → **chưa đủ mạnh để kết luận chắc chắn**; A-balanced thấp hơn cả hai seed của A-natural nên hiệu ứng có khả năng thật, nhưng cần ≥3 seed.
- Khoảng cách giữa model (Qwen vs EcomGPT 0,03; vs Moses 0,07; vs NLLB 0,22; vs OpenNMT 0,51) đều lớn hơn seed noise nhiều lần.
- CI của paired bootstrap chỉ phủ variance eval, **không** phủ seed; phải ghi rõ khi trích dẫn.

### 10.3. NLLB-200-distilled-600M (pretrained NMT control)

| Cell | 1688 da | Tiki QE |
|---|---:|---:|
| NLLB A | 0,6344 | −0,5379 |
| NLLB B | 0,5517 | −0,6753 |
| OpenNMT from-scratch A | 0,3488 | −1,4177 |
| OpenNMT from-scratch B-natural | 0,3387 | −1,4074 |

NLLB hơn OpenNMT from-scratch ~0,29/0,21 nhưng vẫn kém Qwen3-4B ~0,22 trên 1688 → kết luận "OpenNMT yếu do pretraining/capacity/recipe" được xác nhận, không phải "paradigm NMT".

## 11. Tái lập & seed bổ sung (2026-11-24)

### 11.1. 8B B-natural — artefact tái lập

Bản chạy 23/11 bị mất hyp cùng VM A100. Đã inference lại từ `nguynthanhquang/qwen8b-b-ckpt` step 500:

| Cell | 1688 da | Tiki QE | BLEU 1688 | n |
|---|---:|---:|---:|---|
| Qwen3-8B B-natural (tái lập) | 0,8514 | 0,0624 | 69,87 | 231 / 600 |

Trùng khớp tuyệt đối với điểm đã ghi hôm 23/11 ⇒ cell 8B B giờ **có artefact đầy đủ** (hyp 1.500 dòng + score từng dòng).

### 11.2. Seed 7 — Qwen3-4B và NLLB

| Cell | 1688 da | Tiki QE | Artefact |
|---|---:|---:|---|
| Qwen3-4B A-natural seed 7 | **0,8530** | **0,0614** | ✅ hyp 1.500 dòng (15 rỗng); adapter step 500 `qwen4b-a-s7-ckpt` |
| Qwen3-4B B-balanced seed 7 | ⏳ chờ inference lại (hyp cũ sai adapter) | ⏳ | adapter thật step 500 đã backup: `nguynthanhquang/qwen4b-b-balanced-s7-adapter` |
| NLLB A seed 7 | 0,6304 | −0,5427 | ✅ |
| NLLB B seed 7 | 0,6382 | −0,5192 | ✅ |
| EcomGPT A-balanced (tái lập) | 0,8156 | 0,0020 | ✅ adapter step 500 trên Kaggle |

**Đọc kết quả:** A-natural seed 7 = 0,8530 so với seed 42 = 0,8584 (Δ −0,0054), Tiki 0,0614 so với 0,0573 (Δ +0,0041) — đúng bằng bậc seed noise ước lượng 0,005–0,007. Với Qwen3-4B, khoảng cách A-natural − A-balanced seed 42 là 0,0107; NLLB cho thấy chênh lệch A/B đảo dấu giữa seed 42 và seed 7 (0,6344 vs 0,6304 cho A; 0,5517 vs 0,6382 cho B) ⇒ dao động seed của control NMT lớn hơn hiệu ứng dữ liệu. Vì vậy mọi so sánh A/B giữa các cấu hình phải kèm seed noise, không chỉ CI bootstrap.

### 11.3. Hạn chế công bằng của cell EcomGPT

Recipe EcomGPT chia 8.000 token/step theo **độ dài chuỗi đã encode** (prompt + source + target), trong khi Qwen/NLLB chia theo **`eff_tokens`** (source + target). Đo trên A-balanced: 4,0 M token encoded ≈ **1,42 M token hiệu dụng** ⇒ EcomGPT chỉ thấy ~35% ngân sách 4 M token của các model khác. Đây là hạn chế của recipe, cần nêu khi so sánh EcomGPT với Qwen/NLLB.
