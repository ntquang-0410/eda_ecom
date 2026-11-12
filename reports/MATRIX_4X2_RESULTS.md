# Matrix 4 model × 2 data — COMET

Eval: `eval_F` 1.500 dòng dùng chung cho cả 8 cell (cùng `eval_id`, cùng reference). 1688 = COMET-da có reference; Tiki = COMET-QE không reference. BLEU/chrF tính trên phần 1688 có reference và hypothesis không rỗng.

## 1. Điểm tổng theo cell

| model | data | 1688 da | Tiki QE | BLEU 1688 | chrF 1688 | n 1688 | n Tiki |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | 0.8560 | 0.0635 | 55.19 | 54.02 | 747 | 738 |
| Qwen3-8B | B | 0.8552 | 0.0687 | 70.95 | 69.29 | 747 | 738 |
| EcomGPT-7B | A | 0.8214 | 0.0102 | 26.37 | 27.43 | 750 | 750 |
| EcomGPT-7B | B | 0.7661 | -0.1455 | 14.39 | 15.20 | 750 | 729 |
| Moses | A | 0.7947 | -0.2401 | 33.96 | 36.03 | 750 | 750 |
| Moses | B | 0.8103 | -0.1466 | 67.82 | 65.86 | 750 | 750 |
| OpenNMT | A | 0.3581 | -1.3508 | 0.45 | 1.79 | 750 | 750 |
| OpenNMT | B | 0.3424 | -1.3909 | 2.05 | 3.08 | 750 | 750 |

## 2. Theo stratum

| model | data | 1688 Title | 1688 Attr | 1688 Desc | Tiki Title | Tiki Attr | Tiki Desc |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | 0.835 | 0.877 | 0.855 | -0.021 | 0.076 | 0.220 |
| Qwen3-8B | B | 0.836 | 0.849 | 0.908 | -0.037 | 0.107 | 0.215 |
| EcomGPT-7B | A | 0.808 | 0.852 | 0.788 | -0.057 | 0.104 | -0.042 |
| EcomGPT-7B | B | 0.817 | 0.820 | 0.557 | -0.038 | -0.136 | -0.377 |
| Moses | A | 0.797 | 0.843 | 0.694 | -0.208 | 0.006 | -0.797 |
| Moses | B | 0.790 | 0.817 | 0.836 | -0.180 | 0.020 | -0.415 |
| OpenNMT | A | 0.269 | 0.463 | 0.328 | -1.545 | -0.892 | -1.880 |
| OpenNMT | B | 0.283 | 0.386 | 0.374 | -1.452 | -1.237 | -1.577 |

## 3. So sánh model trong cùng recipe (paired bootstrap, 2.000 lần, cluster theo sản phẩm)

Δ = model − Qwen cùng recipe. `*` = CI 95% không chứa 0.

| recipe | model | Δ 1688 da | Δ Tiki QE | n 1688 | n Tiki |
|---|---|---:|---:|---:|---:|
| A | EcomGPT-7B | -0.0346 [-0.0417; -0.0273]* | -0.0446 [-0.0651; -0.0230]* | 747 | 738 |
| A | Moses | -0.0613 [-0.0690; -0.0539]* | -0.2956 [-0.3301; -0.2603]* | 747 | 738 |
| A | OpenNMT | -0.4979 [-0.5091; -0.4872]* | -1.4066 [-1.4710; -1.3418]* | 747 | 738 |
| B | EcomGPT-7B | -0.0876 [-0.1003; -0.0764]* | -0.2118 [-0.2460; -0.1747]* | 747 | 717 |
| B | Moses | -0.0451 [-0.0513; -0.0388]* | -0.2100 [-0.2440; -0.1750]* | 747 | 738 |
| B | OpenNMT | -0.5130 [-0.5219; -0.5047]* | -1.4585 [-1.5021; -1.4117]* | 747 | 738 |

## 4. Hiệu ứng dữ liệu B − A trong cùng model

Δ = B − A. `*` = CI 95% không chứa 0.

| model | Δ 1688 da | Δ Tiki QE | Δ 1688 Title | Δ 1688 Attr | Δ 1688 Desc | Δ Tiki Title | Δ Tiki Attr | Δ Tiki Desc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | -0.0008 [-0.0057; +0.0045] | +0.0052 [-0.0106; +0.0208] | +0.0002 [-0.0041; +0.0044] | -0.0283 [-0.0363; -0.0205]* | +0.0534 [+0.0414; +0.0674]* | -0.0157 [-0.0376; +0.0039] | +0.0306 [+0.0066; +0.0579]* | -0.0046 [-0.0446; +0.0340] |
| EcomGPT-7B | -0.0553 [-0.0670; -0.0441]* | -0.1572 [-0.1928; -0.1229]* | +0.0094 [+0.0042; +0.0149]* | -0.0318 [-0.0442; -0.0201]* | -0.2315 [-0.2701; -0.1920]* | +0.0188 [-0.0020; +0.0409] | -0.2507 [-0.3259; -0.1775]* | -0.3355 [-0.4255; -0.2462]* |
| Moses | +0.0156 [+0.0089; +0.0223]* | +0.0935 [+0.0681; +0.1192]* | -0.0066 [-0.0115; -0.0015]* | -0.0257 [-0.0362; -0.0151]* | +0.1424 [+0.1307; +0.1543]* | +0.0280 [+0.0061; +0.0498]* | +0.0147 [-0.0327; +0.0665] | +0.3824 [+0.3545; +0.4123]* |
| OpenNMT | -0.0156 [-0.0236; -0.0078]* | -0.0401 [-0.0812; +0.0035] | +0.0144 [+0.0067; +0.0214]* | -0.0767 [-0.0924; -0.0614]* | +0.0463 [+0.0389; +0.0545]* | +0.0931 [+0.0521; +0.1344]* | -0.3444 [-0.4304; -0.2613]* | +0.3021 [+0.2498; +0.3552]* |

## 5. Đọc kết quả

- **Model:** Qwen3-8B > EcomGPT-7B > Moses > OpenNMT trên 1688. Qwen hơn cell mạnh thứ nhì (EcomGPT A) 0,0346 COMET, CI [0,0273; 0,0417].
- **Qwen A vs B:** B − A = -0.0008 [-0.0057; +0.0045] trên 1688 và +0.0052 [-0.0106; +0.0208] trên Tiki; cả hai CI đều chứa 0 → không có khác biệt tổng thể có ý nghĩa.
- **Đánh đổi theo stratum (Qwen):** B − A = **+0,053** description 1688 (CI [+0,041; +0,067]) nhưng **−0,028** attribute (CI [−0,036; −0,021]); Tiki attribute +0,031. Recipe B dạy description tốt hơn nhưng mất attribute, nên A/B hòa về tổng.
- **EcomGPT-7B:** B kém A rõ rệt ở cả tổng (Δ −0,055 da; −0,157 QE) lẫn attribute và description.
- **Moses:** B hơn A ở 1688 (+0,0156, CI [+0,009; +0,022]) và Tiki QE (+0,0935), chủ yếu nhờ description; nhưng cả hai vẫn có COMET-QE Tiki âm.
- **OpenNMT from-scratch:** thất bại trên dữ liệu này (COMET 1688 ≈ 0,35; BLEU < 2,1; output lặp cụm). Bản A 30k step vẫn degenerate, không phải lỗi huấn luyện đơn thuần.
- **Tiki QE:** chỉ Qwen dương; EcomGPT/Moses/OpenNMT đều âm. Trong matrix này, chỉ LLM fine-tune là khả thi cho miền Tiki.

## 6. Caveat

- Recipe B có 15,825M token nhưng LLM chỉ dùng 4,000M token → khoảng **0,253 epoch**; A có 4,518M token → khoảng 0,885 epoch. So sánh A/B là so sánh *coverage* dưới cùng ngân sách token, không phải so sánh cùng số epoch.
- Moses dùng eflomal thay GIZA++ (hạn chế của pipeline SMT).
- OpenNMT A chạy 30.000 step (~48 epoch trên 4,5M token), B chạy 22.990 step (10 epoch); hai baseline NMT không bị ràng buộc ngân sách 4M token như LLM.
- COMET-QE Tiki chấm không reference nên chỉ đo chất lượng bề mặt/ngữ nghĩa theo model QE, không thay thế đánh giá người.

