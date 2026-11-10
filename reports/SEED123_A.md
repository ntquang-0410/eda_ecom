# Seed 42 vs 123 — tập A

Cluster bootstrap theo `product_id`, 4.000 lần. Δ so với `natural` cùng seed.

## 1688

| config | Δ seed 42 | CI 42 | Δ seed 123 | CI 123 | Δ trung bình | cùng dấu |
|---|---:|---|---:|---|---:|---|
| attr_heavy | -0.0049 [-0.0083; -0.0013] | -0.0040 [-0.0077; -0.0004] | -0.0044 | có |
| balanced | -0.0074 [-0.0124; -0.0025] | - | -0.0074 | có |
| tiki_heavy | -0.0155 [-0.0205; -0.0109] | - | -0.0155 | có |
| title_heavy | -0.0069 [-0.0118; -0.0021] | - | -0.0069 | có |

## tiki

| config | Δ seed 42 | CI 42 | Δ seed 123 | CI 123 | Δ trung bình | cùng dấu |
|---|---:|---|---:|---|---:|---|
| attr_heavy | +0.0048 [-0.0083; +0.0176] | -0.0194 [-0.0335; -0.0064] | -0.0073 | không |
| balanced | +0.0005 [-0.0151; +0.0151] | - | +0.0005 | có |
| tiki_heavy | -0.0025 [-0.0187; +0.0134] | - | -0.0025 | có |
| title_heavy | +0.0033 [-0.0098; +0.0159] | - | +0.0033 | có |

## Cách đọc

- Config chỉ được coi là tái lập nếu Δ cùng dấu ở cả 2 seed và CI từng seed không chứa 0.
- Nếu seed 123 đảo dấu hoặc CI chứa 0, gain seed 42 là nhiễu → giữ `natural`.

