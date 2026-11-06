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
| natural | 0.7947 | 0.8657 | 0.9160 | -0.0838 | 0.0660 | -0.0089 | 0.8588 | 0.5117 |
| balanced | 0.8122 | 0.8677 | 0.9131 | -0.0350 | 0.0926 | 0.0288 | 0.8643 | 0.5301 |
| no_desc | 0.8247 | 0.8621 | 0.8638 | -0.0370 | 0.0811 | 0.0221 | 0.8502 | 0.5189 |
| tiki_heavy | 0.8008 | 0.8635 | 0.9141 | -0.0331 | 0.0836 | 0.0252 | 0.8595 | 0.5258 |
| title_heavy | 0.8246 | 0.8660 | 0.9148 | -0.0264 | 0.1001 | 0.0368 | 0.8685 | 0.5358 |

## So với natural (paired bootstrap, gộp 2 seed)

| cấu hình | stratum | diff | CI 95% | P(win) |
|---|---|---:|---|---:|
| balanced | 1688/title | +0.0175 | [+0.0095, +0.0268] | 1.0 |
| balanced | 1688/attribute | +0.0020 | [-0.0050, +0.0088] | 0.697 |
| balanced | 1688/description | -0.0029 | [-0.0066, +0.0010] | 0.071 |
| balanced | tiki/title | +0.0488 | [+0.0216, +0.0776] | 1.0 |
| balanced | tiki/attribute | +0.0266 | [+0.0046, +0.0491] | 0.989 |
| no_desc | 1688/title | +0.0300 | [+0.0219, +0.0391] | 1.0 |
| no_desc | 1688/attribute | -0.0037 | [-0.0119, +0.0044] | 0.191 |
| no_desc | 1688/description | -0.0522 | [-0.0575, -0.0464] | 0.0 |
| no_desc | tiki/title | +0.0468 | [+0.0162, +0.0785] | 0.999 |
| no_desc | tiki/attribute | +0.0151 | [-0.0067, +0.0361] | 0.909 |
| tiki_heavy | 1688/title | +0.0061 | [-0.0034, +0.0162] | 0.901 |
| tiki_heavy | 1688/attribute | -0.0022 | [-0.0090, +0.0041] | 0.252 |
| tiki_heavy | 1688/description | -0.0019 | [-0.0077, +0.0031] | 0.227 |
| tiki_heavy | tiki/title | +0.0507 | [+0.0216, +0.0805] | 1.0 |
| tiki_heavy | tiki/attribute | +0.0176 | [-0.0050, +0.0417] | 0.936 |
| title_heavy | 1688/title | +0.0299 | [+0.0214, +0.0390] | 1.0 |
| title_heavy | 1688/attribute | +0.0002 | [-0.0079, +0.0082] | 0.525 |
| title_heavy | 1688/description | -0.0012 | [-0.0055, +0.0033] | 0.298 |
| title_heavy | tiki/title | +0.0574 | [+0.0302, +0.0849] | 1.0 |
| title_heavy | tiki/attribute | +0.0341 | [+0.0091, +0.0591] | 0.999 |

## Tiki macro

| cấu hình | diff | CI 95% | P(win) |
|---|---:|---|---:|
| balanced | +0.0375 | [+0.0203, +0.0564] | 1.0 |
| no_desc | +0.0310 | [+0.0119, +0.0494] | 1.0 |
| tiki_heavy | +0.0342 | [+0.0157, +0.0525] | 1.0 |
| title_heavy | +0.0455 | [+0.0271, +0.0649] | 1.0 |

## Gemma credit (1-9, blind)

| cấu hình | dataset | pair_type | mean | rating | n |
|---|---|---|---:|---|---:|
| balanced | 1688 | attribute | 9.0 | AAA | 1 |
| balanced | 1688 | description | 6.0 | BBB | 1 |
| balanced | 1688 | title | 6.0 | BBB | 1 |
| balanced | tiki | attribute | 9.0 | AAA | 2 |
| balanced | tiki | title | 9.0 | AAA | 2 |
| natural | 1688 | description | 9.0 | AAA | 1 |
| natural | 1688 | title | 6.0 | BBB | 1 |
| natural | tiki | attribute | 8.667 | AAA | 3 |
| natural | tiki | title | 9.0 | AAA | 1 |
| no_desc | 1688 | description | 4.5 | B | 2 |
| no_desc | 1688 | title | 8.0 | AA | 1 |
| no_desc | tiki | attribute | 7.5 | AA | 2 |
| no_desc | tiki | title | 6.5 | BBB | 2 |
| tiki_heavy | 1688 | attribute | 9.0 | AAA | 1 |
| tiki_heavy | 1688 | description | 7.0 | A | 2 |
| tiki_heavy | 1688 | title | 6.5 | BBB | 2 |
| tiki_heavy | tiki | attribute | 2.0 | CC | 1 |
| tiki_heavy | tiki | title | 7.333 | A | 3 |
| title_heavy | 1688 | attribute | 7.0 | A | 2 |
| title_heavy | 1688 | description | 8.25 | AA | 4 |
| title_heavy | 1688 | title | 6.0 | BBB | 1 |
| title_heavy | tiki | attribute | 8.5 | AA | 2 |
| title_heavy | tiki | title | 4.5 | B | 2 |

## Kết luận

**natural** — 2 seed không cùng dấu

Không có bằng chứng đủ mạnh rằng gán weight làm tăng chất lượng.
Giữ phân bố token tự nhiên (desc ~64%).

## Đọc kết quả

- Nếu một cấu hình tăng đúng stratum mà nó được ưu tiên (ví dụ title_heavy tăng 1688/title)
  và không làm giảm stratum khác quá 0,01, đó là bằng chứng weight có tác dụng.
- Nếu mọi chênh lệch đều <0,005 hoặc CI chứa 0, kết luận là natural.
- Gemma chỉ là tín hiệu phụ; ưu tiên COMET paired.
