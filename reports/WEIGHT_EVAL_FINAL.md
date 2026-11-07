# Kết luận thực nghiệm weight — Tiki + 1688

> **CẢNH BÁO KỸ THUẬT (2026-11-07):** báo cáo này dùng sampler cũ
> `p_i ∝ w_g * t_i`; token share thực tế bị lệch do `sum(t_i²)`.
> Ví dụ A natural: attr 88,6% thay vì 65,1%; title 3,3% thay vì 16%.
> Kết quả chỉ nên xem là tham khảo; A/B sẽ chạy lại bằng quota-based sampling.


## Thiết kế

```text
model        : Qwen3-4B + LoRA
budget       : 5.000.000 token/run, 625 optimizer step
seed         : 42 (5 cấu hình) + 123 (natural, balanced, title_heavy)
eval 1688    : 300 title + 300 attribute + 150 description, COMET-da
eval Tiki    : 300 title + 300 attribute, COMET-QE
judge        : Gemma 4 31B, blind, credit AAA..C
bootstrap    : 2.000 lần paired theo eval_id
```

## COMET (2 seed)

| config | 1688 macro | Tiki macro | Tiki title | Tiki attr |
|---|---:|---:|---:|---:|
| natural | 0.8573 | -0.0117 | -0.0878 | 0.0645 |
| balanced | 0.8652 | 0.0221 | -0.0312 | 0.0754 |
| no_desc | 0.8502 | 0.0221 | -0.0370 | 0.0811 |
| tiki_heavy | 0.8595 | 0.0252 | -0.0331 | 0.0836 |
| title_heavy | **0.8679** | **0.0240** | **-0.0326** | **0.0806** |

So với natural:

```text
balanced     : Tiki +0.0337 CI [+0.0201,+0.0480] | 1688 +0.0092 CI [+0.0057,+0.0134]
title_heavy  : Tiki +0.0356 CI [+0.0210,+0.0502] | 1688 +0.0126 CI [+0.0086,+0.0169]
```

`title_heavy` vs `balanced`:

```text
Tiki      +0.0019  CI [-0.0086,+0.0116]   -> không phân biệt
1688      +0.0034  CI [+0.0008,+0.0059]   -> title_heavy nhỉnh nhẹ
```

## Gemma credit (2 seed, blind AAA..C)

| config | mean | grade | n |
|---|---:|---|---:|
| balanced | **7.868** | AA | 500 |
| title_heavy | 7.846 | AA | 500 |
| tiki_heavy | 7.704 | AA | 250 |
| no_desc | 7.700 | AA | 250 |
| natural | 7.680 | AA | 500 |

Theo seed:

```text
seed 42 : balanced 7.872 > title_heavy 7.744 > tiki_heavy 7.704 > no_desc 7.700 > natural 7.600
seed 123: title_heavy 7.948 > balanced 7.864 > natural 7.760
```

## Kết luận

**Có, gán weight làm tăng chất lượng so với natural.** Cả `balanced` và
`title_heavy` đều tăng Tiki và 1688 với CI 95% không chứa 0.

Hai cấu hình đồng hạng:

```text
balanced
  - tăng đều 1688 title +0.0208, attr +0.0018, desc +0.0010
  - Tiki title +0.0566, attr +0.0109
  - Gemma credit 2 seed 7.868 (nhỉnh nhất)
  - lựa chọn mặc định nếu ưu tiên cân bằng/ổn định

title_heavy
  - 1688 title +0.0315 (mạnh nhất), attr/desc gần như đi ngang
  - Tiki title +0.0552, attr +0.0161
  - Gemma credit 2 seed 7.846
  - chọn nếu ưu tiên 1688 title
```

`no_desc` bị loại vì làm 1688 description giảm **-0.0505**.
`tiki_heavy` không vượt được hai cấu hình trên.

## Đề xuất

```text
Mặc định : balanced
Nếu trọng tâm là 1688 title : title_heavy
Không dùng : no_desc
```

## Hạn chế

- COMET-QE Tiki dùng `wmt20-comet-qe-da`; điểm tuyệt đối thấp, nhưng so sánh paired vẫn hợp lệ.
- Tiki không có reference người; target Tiki là synthetic.
- `title_heavy` và `balanced` chênh nhau dưới ngưỡng 0,005 trên Tiki; không nên tuyên bố một cái hơn tuyệt đối.