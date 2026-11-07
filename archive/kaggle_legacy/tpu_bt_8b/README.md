# Qwen3-8B BT — TPU v5e-8 sharded

Notebook full BT cho **Qwen3-8B** trên `TpuV5E8`, chạy sharded 8 chip.

Chỉ push/ chạy **sau khi** `kaggle/tpu_shard/` (spike) xác nhận:

```text
n_dev = 8
validation_match = 3/3
```

## Điểm khác notebook 0.6B/1.7B/4B

| | 0.6B/1.7B/4B | 8B |
|---|---|---|
| device | `xla:0` | 8 chip, SPMD tensor-parallel |
| batch | 8 / 2 / 2 | 2 |
| max_input | 1024 | 768 |
| max_new | 128 | 96 |
| đơn vị lưu | 1 file/model | 1 file/slice (`N_SLICES=4`) |
| dừng | hết data | hết data hoặc `TIME_BUDGET_MIN=450` |

## Chia slice

Round-robin theo thứ tự `(min_priority, source_chars, cache_key)`:

```python
df['slice_index'] = np.arange(len(df)) % N_SLICES
```

Mỗi slice trải đều độ dài/priority, nên 1 slice xong là có mẫu đại diện,
không bị dồn toàn câu dài vào slice cuối.

File:

```text
/kaggle/working/plan_b_outputs/qwen3-8b/slice_00of04.parquet
/kaggle/working/plan_b_outputs/qwen3-8b/slice_01of04.parquet
...
/kaggle/working/plan_b_outputs/qwen3-8b/state_slice_00.json
```

## Time budget

Session TPU bị cắt cứng. Notebook tự dừng ở `TIME_BUDGET_MIN=450` (7.5h),
ghi checkpoint, in `next_slice_start`. Session sau chỉ cần đổi:

```python
SLICE_START = 1   # hoặc next_slice_start trong summary
```

## Tải kết quả

```bash
kaggle kernels output ntquang0410/tiki-plan-b-bt-tpu-8b -p /tmp/opencode/tpu_8b
```

Ghép:

```python
pd.concat([pd.read_parquet(p) for p in sorted(Path('.').glob('slice_*.parquet'))])
```

## Quota

TPU quota tuần này: 20h, còn ~19.7h (refresh 2026-10-31).
Vì vậy 8B **không** chạy cùng lúc với 0.6B/1.7B/4B, mà chạy sau khi 3 model kia xong.
