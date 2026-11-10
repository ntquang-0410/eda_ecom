# Weight survey B — quota sampler

Baseline: `natural`. 1688 = COMET-da, Tiki = COMET-QE.
Hai thang điểm khác nhau nên báo cáo riêng, không lấy trung bình gộp.
Paired bootstrap 4.000 lần trên eval_id dùng chung.

## COMET

| config | n | 1688 (da) | Tiki (qe) | 1688_description | 1688_title | tiki_description | tiki_title |
|---|---:|---:|---:|---:|---:|---:|---:|
| title_heavy | 885 | **0.857** | **0.053** | 0.899 | 0.836 | 0.240 | -0.033 |
| balanced | 885 | **0.855** | **0.057** | 0.900 | 0.833 | 0.247 | -0.031 |
| tiki_heavy | 885 | **0.845** | **0.052** | 0.898 | 0.819 | 0.241 | -0.035 |
| natural | 885 | **0.841** | **0.056** | 0.911 | 0.806 | 0.250 | -0.032 |
| desc_heavy | 885 | **0.835** | **0.054** | 0.907 | 0.800 | 0.248 | -0.036 |

## Paired bootstrap vs baseline

| config | Δ1688 | CI 95% | P(win) | ΔTiki | CI 95% | P(win) |
|---|---:|---|---:|---:|---|---:|
| title_heavy::seed42 | +0.0160 | [+0.0099, +0.0229] | 1.000 | -0.0034 | [-0.0178, +0.0108] | 0.321 |
| balanced::seed42 | +0.0144 | [+0.0081, +0.0211] | 1.000 | +0.0002 | [-0.0146, +0.0145] | 0.527 |
| tiki_heavy::seed42 | +0.0047 | [-0.0007, +0.0103] | 0.956 | -0.0045 | [-0.0208, +0.0111] | 0.303 |
| desc_heavy::seed42 | -0.0054 | [-0.0113, +0.0010] | 0.047 | -0.0026 | [-0.0121, +0.0071] | 0.303 |

## Credit score (1..9) — gemini-3.8-flash (một phần, quota)

| config | mean | n |
|---|---:|---:|
| balanced | 6.68 | 146 |
| natural | 6.64 | 140 |
| desc_heavy | 6.59 | 138 |
| title_heavy | 6.31 | 135 |
| tiki_heavy | 6.15 | 121 |

### COMET system score (raw)

| key | system | BLEU | chrF |
|---|---:|---:|---:|
| balanced::42::1688 | 0.855 | 71.668 | 70.166 |
| balanced::42::tiki | 0.057 | - | - |
| desc_heavy::42::1688 | 0.835 | 71.326 | 71.121 |
| desc_heavy::42::tiki | 0.054 | - | - |
| natural::42::1688 | 0.841 | 74.263 | 72.352 |
| natural::42::tiki | 0.057 | - | - |
| tiki_heavy::42::1688 | 0.845 | 69.380 | 67.647 |
| tiki_heavy::42::tiki | 0.052 | - | - |
| title_heavy::42::1688 | 0.857 | 68.509 | 66.343 |
| title_heavy::42::tiki | 0.053 | - | - |
