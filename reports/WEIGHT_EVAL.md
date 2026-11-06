# Trọng số desc / title / attribute có làm tăng chất lượng?

## Giao thức

- Qwen3-4B LoRA, 5M token/run, 625 optimizer step
- 5 cấu hình × 2 seed (42, 123)
- eval cố định: 1688 title 300 / attr 300 / desc 150; Tiki title 300 / attr 300
- 1688: COMET-da có reference; Tiki: COMET-QE không reference
- paired bootstrap 2.000 lần theo eval_id

## Điểm COMET theo stratum

| config | 1688_title | 1688_attribute | 1688_description | tiki_title | tiki_attribute | tiki_macro | macro_1688 | all_macro |
|---|---|---|---|---|---|---|---|---|
| natural | 0.7920 | 0.8657 | 0.9143 | -0.0878 | 0.0645 | -0.0117 | 0.8573 | 0.5097 |
| balanced | 0.8122 | 0.8677 | 0.9131 | -0.0350 | 0.0926 | 0.0288 | 0.8643 | 0.5301 |
| no_desc | 0.8247 | 0.8621 | 0.8638 | -0.0370 | 0.0811 | 0.0221 | 0.8502 | 0.5189 |
| tiki_heavy | 0.8008 | 0.8635 | 0.9141 | -0.0331 | 0.0836 | 0.0252 | 0.8595 | 0.5258 |
| title_heavy | 0.8234 | 0.8658 | 0.9144 | -0.0326 | 0.0806 | 0.0240 | 0.8679 | 0.5303 |

## So với natural (paired bootstrap, gộp 2 seed)

| cấu hình | stratum | diff | CI 95% | P(win) |
|---|---|---:|---|---:|
| balanced | 1688/title | +0.0202 | [+0.0124, +0.0286] | 1.0 |
| balanced | 1688/attribute | +0.0020 | [-0.0045, +0.0087] | 0.704 |
| balanced | 1688/description | -0.0012 | [-0.0048, +0.0031] | 0.291 |
| balanced | tiki/title | +0.0528 | [+0.0293, +0.0774] | 1.0 |
| balanced | tiki/attribute | +0.0281 | [+0.0068, +0.0497] | 0.998 |
| no_desc | 1688/title | +0.0327 | [+0.0249, +0.0407] | 1.0 |
| no_desc | 1688/attribute | -0.0037 | [-0.0118, +0.0038] | 0.195 |
| no_desc | 1688/description | -0.0505 | [-0.0563, -0.0439] | 0.0 |
| no_desc | tiki/title | +0.0508 | [+0.0232, +0.0802] | 1.0 |
| no_desc | tiki/attribute | +0.0166 | [-0.0055, +0.0380] | 0.926 |
| tiki_heavy | 1688/title | +0.0088 | [-0.0002, +0.0180] | 0.973 |
| tiki_heavy | 1688/attribute | -0.0022 | [-0.0087, +0.0043] | 0.244 |
| tiki_heavy | 1688/description | -0.0002 | [-0.0061, +0.0054] | 0.457 |
| tiki_heavy | tiki/title | +0.0547 | [+0.0287, +0.0808] | 1.0 |
| tiki_heavy | tiki/attribute | +0.0191 | [-0.0019, +0.0419] | 0.968 |
| title_heavy | 1688/title | +0.0315 | [+0.0235, +0.0396] | 1.0 |
| title_heavy | 1688/attribute | +0.0001 | [-0.0062, +0.0061] | 0.499 |
| title_heavy | 1688/description | +0.0001 | [-0.0042, +0.0051] | 0.497 |
| title_heavy | tiki/title | +0.0552 | [+0.0319, +0.0791] | 1.0 |
| title_heavy | tiki/attribute | +0.0161 | [-0.0014, +0.0331] | 0.964 |

## Tiki macro

| cấu hình | diff | CI 95% | P(win) |
|---|---:|---|---:|
| balanced | +0.0401 | [+0.0240, +0.0566] | 1.0 |
| no_desc | +0.0337 | [+0.0160, +0.0501] | 1.0 |
| tiki_heavy | +0.0367 | [+0.0204, +0.0531] | 1.0 |
| title_heavy | +0.0356 | [+0.0210, +0.0502] | 1.0 |

## Kết luận

**natural** — 2 seed không cùng dấu

Không có bằng chứng đủ mạnh rằng gán weight làm tăng chất lượng.
Giữ phân bố token tự nhiên (desc ~64%).

## Đọc kết quả

- Nếu một cấu hình tăng đúng stratum mà nó được ưu tiên (ví dụ title_heavy tăng 1688/title)
  và không làm giảm stratum khác quá 0,01, đó là bằng chứng weight có tác dụng.
- Nếu mọi chênh lệch đều <0,005 hoặc CI chứa 0, kết luận là natural.
- Gemma chỉ là tín hiệu phụ; ưu tiên COMET paired.
