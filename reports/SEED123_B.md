# Seed 42 vs 123 — tập B

Cluster bootstrap theo `product_id`, 4.000 lần. Δ so với `natural` cùng seed.

## 1688

| config | Δ seed 42 | CI 42 | Δ seed 123 | CI 123 | Δ trung bình | cùng dấu |
|---|---:|---|---:|---|---:|---|
| balanced | +0.0144 [+0.0081; +0.0213] | +0.0147 [+0.0102; +0.0195] | +0.0146 | có |
| desc_heavy | -0.0054 [-0.0113; +0.0006] | - | -0.0054 | có |
| tiki_heavy | +0.0047 [-0.0008; +0.0102] | - | +0.0047 | có |
| title_heavy | +0.0160 [+0.0097; +0.0227] | +0.0162 [+0.0112; +0.0215] | +0.0161 | có |

## tiki

| config | Δ seed 42 | CI 42 | Δ seed 123 | CI 123 | Δ trung bình | cùng dấu |
|---|---:|---|---:|---|---:|---|
| balanced | +0.0002 [-0.0141; +0.0144] | -0.0090 [-0.0234; +0.0051] | -0.0044 | không |
| desc_heavy | -0.0026 [-0.0126; +0.0068] | - | -0.0026 | có |
| tiki_heavy | -0.0045 [-0.0211; +0.0113] | - | -0.0045 | có |
| title_heavy | -0.0034 [-0.0173; +0.0105] | +0.0052 [-0.0113; +0.0217] | +0.0009 | không |

## Cách đọc

- Config chỉ được coi là tái lập nếu Δ cùng dấu ở cả 2 seed và CI từng seed không chứa 0.
- Nếu seed 123 đảo dấu hoặc CI chứa 0, gain seed 42 là nhiễu → giữ `natural`.

