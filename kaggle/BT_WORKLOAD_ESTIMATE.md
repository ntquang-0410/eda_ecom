# Ước lượng workload BT — Tiki Plan A (đo thật trên dataset)

Đo bằng Qwen3 tokenizer trên 3.000 mẫu ngẫu nhiên của `tiki_translation_cache.parquet`
(route `mt`), prompt đúng như notebook BT.

## Data

```text
MT cache entry          29.446
duplicate cache_key     0
null source_text        0
value_class             100% 'text'  -> luôn là translate_value
min_priority            1: 29.251 | 2: 195
source_chars            p50 33 | p95 152 | max 3.273
```

## Prompt / output token

```text
prompt tokens           p50 133 | p90 164 | p95 184 | p99 314 | max 1.074
cap 256 -> 98.10%   cap 384 -> 99.57%   cap 512 -> 99.73%   cap 768 -> 99.93%
expected_zh_chars       p50 30 | p95 38 | p99 230 | tổng 728.474 ký tự
output tokens (ước)     1.38 M  (zh_chars + 22 token JSON overhead)
input tokens (ước)      4.19 M
```

Vì p95 = 184 token, `max_input_tokens = 1024` trong notebook là dư an toàn;
với 8B dùng 768 vẫn phủ 99,93%.

## Ước lượng thời gian (TPU v5e, 1 chip; 8B sharded 8 chip)

Giả định hiệu suất thực tế 20% peak, decode memory-bound theo kích thước weights:

| model | weights | batch | tok/s (ước) | tổng thời gian |
|---|---|---|---|---|
| 0.6B | 1.2 GB | 8 | ~1.200 | ~30 phút |
| 1.7B | 3.4 GB | 2 | ~300 | ~1,4 h |
| 4B | 8.0 GB | 2 | ~150 | ~2,8 h |
| 8B sharded | 4.2 GB/chip | 2 | ~200 | ~2 h |
| **tổng** | | | | **~6,5 h** |

Quota TPU còn 19,7h (refresh 2026-10-31) → đủ cho cả 4 model, kể cả chạy lại 1 lần.

Đây là ước lượng để lập kế hoạch, **không phải cam kết**; con số thật sẽ có từ
`state_slice_*.json` (`entries_per_min`) và `run_state.json` sau session đầu.

## Nghĩa là

- 3 model 0.6B/1.7B/4B vừa **một** session TPU nếu không bị queue quá lâu.
- 8B nên chạy session riêng, 2-4 slice.
- Không cần tăng `max_new_tokens` quá 96: p99 output chỉ 230 ký tự.

## Validate trước khi chạy

Đã kiểm tra local với chính dataset private `ntquang0410/tiki-plan-a-bt-input`
(bản trên Kaggle, 43 MB):

```text
cache       40.646 x 11   -> MT 29.446
meta       120.884 x 87
merge      29.446 row, 0 duplicate, 0 null  (khớp plan A stats)
```
