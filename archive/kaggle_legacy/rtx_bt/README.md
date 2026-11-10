# RTX Pro 6000 — BT 4 model Qwen3 (offline)

Notebook: `plan_b_bt_rtx.ipynb` — kernel `ntquang0410/tiki-plan-b-bt-rtx-4-model`

Chạy **cả 4 model** (0.6B/1.7B/4B/8B) tuần tự trên 1 GPU 96GB, không cần sharding.

## Vì sao offline

`competition_sources: arc-prize-2026-arc-agi-3` là thứ khiến Kaggle cấp RTX Pro 6000
(không có nó thì rơi về T4 16GB). Nhưng notebook có competition source:

- không được để public;
- `enable_internet: true` bị API trả **400 Bad Request**.

Nên notebook phải chạy **hoàn toàn offline**:

- không `pip install`;
- model lấy từ `model_sources` (`qwen-lm/qwen-3/Transformers/{0.6b,1.7b,4b,8b}/1`);
- có preflight check `transformers >= 4.51` + `Qwen3ForCausalLM`; nếu image cũ thì
  fail ngay (2 phút) thay vì chết giữa session.

## Config

| | giá trị |
|---|---|
| models | 0.6B, 1.7B, 4B, 8B |
| dtype | bf16 |
| batch (96GB) | 16 / 16 / 12 / 8 |
| max_input | 768 token |
| max_new | 96 token |
| checkpoint | mỗi 200 entry |
| time budget | 660 phút |

Batch tự hạ theo VRAM (`>=80GB`, `>=40GB`, còn lại), và model bị **skip** nếu VRAM
không đủ (`8B` cần >= 40GB) — nên nếu rơi vào T4 vẫn có 0.6B/1.7B/4B một phần.

## Resolve model offline

1. `kagglehub.model_download('qwen-lm/qwen-3/Transformers/8b/1')`
2. fallback: quét `/kaggle/input/**/config.json`, so `(hidden_size, intermediate_size,
   num_hidden_layers)` với config chuẩn của từng model:

```text
0.6B (1024, 3072, 28)   1.7B (2048, 6144, 28)
4B   (2560, 9728, 36)   8B   (4096, 12288, 36)
```

## Output

```text
/kaggle/working/plan_b_outputs/<model>/predictions.parquet
/kaggle/working/plan_b_outputs/<model>/run_state.json
/kaggle/working/plan_b_rtx_summary.json
```

`run_state.json` có `entries_per_min`, `vram_alloc_gb`, `gpu` để orchestrator đọc.

## Ước lượng

1.38M output token/model. Với RTX Pro 6000 (~1.8 TB/s):

| model | weights | batch | tok/s (ước) | thời gian |
|---|---|---|---|---|
| 0.6B | 1.2 GB | 16 | ~2500 | ~10 phút |
| 1.7B | 3.4 GB | 16 | ~1500 | ~16 phút |
| 4B | 8.0 GB | 12 | ~1200 | ~20 phút |
| 8B | 16 GB | 8 | ~900 | ~26 phút |

→ cả 4 model khoảng **1,5–2h**, nhanh hơn TPU đáng kể.

## Quan hệ với TPU

Orchestrator `kaggle/auto_tpu_pipeline.py` cho 2 kernel chạy đua:

- RTX được cấp máy trước → chạy 4 model ở đây, **huỷ** kernel TPU đang xếp hàng.
- TPU được cấp máy trước → chạy 3 model + sharding 8B, **huỷ** kernel này.

Nếu notebook này fail sớm (preflight transformers), orchestrator tự quay lại nhánh TPU.
