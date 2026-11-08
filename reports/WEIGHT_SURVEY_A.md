# Weight survey A — quota sampler

Baseline: `natural`. 1688 = COMET-da, Tiki = COMET-QE.
Hai thang điểm khác nhau nên báo cáo riêng, không lấy trung bình gộp.
Paired bootstrap 4.000 lần trên eval_id dùng chung.

## COMET

| config | n | 1688 (da) | Tiki (qe) | 1688_attribute | 1688_title | tiki_attribute | tiki_title |
|---|---:|---:|---:|---:|---:|---:|---:|
| natural | 1200 | **0.856** | **0.025** | 0.883 | 0.830 | 0.081 | -0.032 |
| attr_heavy | 1200 | **0.851** | **0.029** | 0.879 | 0.823 | 0.081 | -0.022 |
| title_heavy | 1200 | **0.849** | **0.028** | 0.861 | 0.837 | 0.081 | -0.026 |
| balanced | 1200 | **0.849** | **0.025** | 0.866 | 0.832 | 0.079 | -0.029 |
| tiki_heavy | 1200 | **0.841** | **0.022** | 0.865 | 0.816 | 0.085 | -0.040 |

## Paired bootstrap vs baseline

| config | Δ1688 | CI 95% | P(win) | ΔTiki | CI 95% | P(win) |
|---|---:|---|---:|---:|---|---:|
| attr_heavy::seed42 | -0.0049 | [-0.0084, -0.0014] | 0.003 | +0.0048 | [-0.0082, +0.0173] | 0.772 |
| title_heavy::seed42 | -0.0069 | [-0.0120, -0.0019] | 0.004 | +0.0033 | [-0.0089, +0.0159] | 0.696 |
| balanced::seed42 | -0.0074 | [-0.0125, -0.0025] | 0.002 | +0.0005 | [-0.0142, +0.0146] | 0.529 |
| tiki_heavy::seed42 | -0.0155 | [-0.0203, -0.0107] | 0.000 | -0.0025 | [-0.0195, +0.0134] | 0.377 |

## Credit score (1..9) — gemini-3.8-flash (một phần, quota)

| config | mean | n |
|---|---:|---:|
| natural | 7.75 | 125 |
| attr_heavy | 7.45 | 122 |
| balanced | 7.36 | 125 |
| title_heavy | 7.21 | 107 |
| tiki_heavy | 7.06 | 101 |

### COMET system score (raw)

| key | system | BLEU | chrF |
|---|---:|---:|---:|
| attr_heavy::42::1688 | 0.851 | 46.215 | 46.854 |
| attr_heavy::42::tiki | 0.029 | - | - |
| balanced::42::1688 | 0.849 | 45.871 | 46.663 |
| balanced::42::tiki | 0.025 | - | - |
| natural::42::1688 | 0.856 | 47.502 | 48.407 |
| natural::42::tiki | 0.025 | - | - |
| tiki_heavy::42::1688 | 0.841 | 42.054 | 43.530 |
| tiki_heavy::42::tiki | 0.022 | - | - |
| title_heavy::42::1688 | 0.849 | 46.983 | 47.622 |
| title_heavy::42::tiki | 0.028 | - | - |
