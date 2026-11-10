# TPU sharding spike — Qwen3-8B trên TpuV5E8

Mục tiêu: chạy Qwen3-8B (bf16, ~16 GB weights) trên 8 chip TPU v5e (16 GB/chip)
bằng PyTorch/XLA SPMD, thay vì phải chờ RTX Pro 6000.

## Vì sao không chạy single chip

`TpuV5E8` = 8 chip × 16 GB HBM. Notebook BT hiện tại chỉ dùng `xla:0`, nên trần là
**16 GB** — 8B bf16 đã 16 GB weights, chưa tính activation/KV cache.

## Thiết kế sharding

Tensor-parallel kiểu Megatron, 1 all-reduce mỗi block:

| module | partition spec | ghi chú |
|---|---|---|
| `q_proj`, `k_proj`, `v_proj` | `('x', None)` | chia head theo chip |
| `o_proj` | `(None, 'x')` | all-reduce output |
| `gate_proj`, `up_proj` | `('x', None)` | chia intermediate dim |
| `down_proj` | `(None, 'x')` | all-reduce output |
| `embed_tokens`, `lm_head`, norm | replicated | tránh all-gather vocab mỗi token |

Với Qwen3-8B: 32 Q heads / 8 = 4 head/chip, 8 KV heads / 8 = 1 head/chip,
intermediate 12288 / 8 = 1536/chip. Chia hết, không lẻ.

Memory/chip (đã tính từ config thật, bf16):

```text
transformer weights   1.74 GB  (13.89 GB / 8)
embed_tokens          1.24 GB  (replicated, 151936 x 4096)
lm_head               1.24 GB  (replicated, tie_word_embeddings=False)
KV cache (batch 2, 864 token)  0.25 GB
activation + workspace         ~0.5-1 GB
-----------------------------------------
                      ~4.2-4.5 GB / 16 GB
```

## Đã kiểm tra chia hết (không lẻ head)

| model | q out/chip | kv out/chip | intermediate/chip |
|---|---|---|---|
| 0.6B | 2048/8 = 256 | 1024/8 = 128 | 3072/8 = 384 |
| 1.7B | 2048/8 = 256 | 1024/8 = 128 | 6144/8 = 768 |
| 4B   | 4096/8 = 512 | 1024/8 = 128 | 9728/8 = 1216 |
| 8B   | 4096/8 = 512 | 1024/8 = 128 | 12288/8 = 1536 |

`Qwen3RMSNorm` per-head (`q_norm`/`k_norm`) là 1-D → replicated, đúng vì norm chạy
trên `head_dim` nằm gọn trong từng head sau khi chia.

## Cách nạp model không OOM

`shard_and_move()` nạp **từng parameter**:

```text
CPU param -> .to(device) -> xs.mark_sharding -> xm.mark_step -> gắn lại vào module
```

Peak memory = shard đã nạp + parameter lớn nhất (~1.24 GB), không bao giờ là 16 GB.
Có memory guard dừng ở 90% HBM và ghi lại trạng thái.

## Notebook làm gì

`shard_spike.ipynb`

1. Probe: số XLA device, API SPMD, memory/chip.
2. Validate trên **Qwen3-0.6B**: so greedy output single-device vs sharded.
   Kỳ vọng `3/3` khớp text.
3. Smoke **Qwen3-8B**: nạp sharded, generate 2 prompt, đo memory + token/s.
4. Ghi `/kaggle/working/shard_spike_report.json`.

## Đọc kết quả

```bash
kaggle kernels output ntquang0410/tiki-plan-b-tpu-shard-spike -p /tmp/opencode/shard_spike
python -m json.tool /tmp/opencode/shard_spike/shard_spike_report.json
```

Quyết định:

| kết quả | kết luận |
|---|---|
| `n_dev = 8`, `validation_match = 3/3`, 8B chạy | dùng SPMD cho 8B full |
| `n_dev = 1` | Kaggle chỉ cấp 1 core → 8B phải sang RTX |
| 8B OOM giữa chừng | xem `memory_at_error`, cần shard cả `lm_head`/`embed_tokens` |
| text lệch | sharding sai spec, so lại `partition_spec` |

## Trạng thái

- Notebook đã dựng xong, syntax-checked.
- **Chưa push được** vì Kaggle giới hạn 1 TPU batch session:
  `Maximum batch TPU session count of 1 reached` (slot đang bị kernel BT 3 model giữ).
- Sẽ push ngay khi kernel `tiki-plan-b-bt-tpu-v5e-8` kết thúc.

## Bước tiếp theo (sau khi spike pass)

Kernel full 8B sẽ tái dùng `shard_and_move` + `partition_spec`, thêm:

- `SLICE_START`/`SLICE_END` để chia 29.446 entry thành nhiều session;
- checkpoint theo `cache_key` (giống notebook BT hiện tại);
- giới hạn `max_new_tokens` 96 để vừa 1 session TPU.

Ước lượng: nếu smoke cho ~10-20 token/s với batch 1, tổng 29.446 entry × ~40 token
≈ 1.2M token → 17-33 giờ → bắt buộc chia 4-6 slice.
