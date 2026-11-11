# Matrix 4 model × 2 data — COMET

Eval: `eval_F` 1.500 dòng. 1688 = COMET-da, Tiki = COMET-QE.

## Điểm theo model × data

| cell | 1688 da | Tiki QE | BLEU 1688 | n 1688 | n Tiki |
|---|---:|---:|---:|---:|---:|
| ecomgpt_a | 0.8214 | 0.0102 | - | 750 | 750 |
| ecomgpt_b | 0.7661 | -0.1455 | - | 750 | 729 |
| moses_a | 0.7947 | -0.2401 | - | 750 | 750 |
| moses_b | 0.8103 | -0.1466 | - | 750 | 750 |
| onmt_a_bad | 0.3162 | -1.1485 | - | 553 | 631 |
| onmt_b | 0.3424 | -1.3909 | - | 750 | 750 |
| qwen_a | 0.8560 | 0.0635 | - | 747 | 738 |
| qwen_b | 0.8552 | 0.0687 | - | 747 | 738 |

## Theo stratum

| cell | 1688_title | 1688_attribute | 1688_description | tiki_title | tiki_attribute | tiki_description |
|---|---:|---:|---:|---:|---:|---:|
| ecomgpt_a | 0.808 | 0.852 | 0.788 | -0.057 | 0.104 | -0.042 |
| ecomgpt_b | 0.817 | 0.820 | 0.557 | -0.038 | -0.136 | -0.377 |
| moses_a | 0.797 | 0.843 | 0.694 | -0.208 | 0.006 | -0.797 |
| moses_b | 0.790 | 0.817 | 0.836 | -0.180 | 0.020 | -0.415 |
| onmt_a_bad | 0.286 | 0.404 | 0.314 | -1.220 | -1.041 | -1.157 |
| onmt_b | 0.283 | 0.386 | 0.374 | -1.452 | -1.237 | -1.577 |
| qwen_a | 0.835 | 0.877 | 0.855 | -0.021 | 0.076 | 0.220 |
| qwen_b | 0.836 | 0.849 | 0.908 | -0.037 | 0.107 | 0.215 |

## Paired bootstrap vs `qwen_a`

| cell | Δ | CI 95% | n sản phẩm |
|---|---:|---|---:|
| ecomgpt_a | -0.0396 | [-0.0504; -0.0286] | 1485 |
| ecomgpt_b | -0.1457 | [-0.1662; -0.1277] | 1464 |
| moses_a | -0.1777 | [-0.1959; -0.1587] | 1485 |
| moses_b | -0.1248 | [-0.1416; -0.1066] | 1485 |
| onmt_a_bad | -0.8920 | [-0.9238; -0.8621] | 1169 |
| onmt_b | -0.9807 | [-1.0197; -0.9397] | 1485 |
| qwen_b | +0.0022 | [-0.0062; +0.0108] | 1485 |

## Đọc kết quả

- Qwen3-8B là baseline mạnh nhất trên cả 1688 và Tiki.
- Moses B đạt BLEU 1688 cao nhưng COMET-QE Tiki rất thấp.
- OpenNMT A/B hiện rất yếu; A bản 10 epoch bị degenerate, bản 30k step đang chạy.

