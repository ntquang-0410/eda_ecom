# data/bt/raw/description/ — output gốc BT description

| đường dẫn | rows | trạng thái |
|---|---:|---|
| `qwen3-4b/predictions_full.parquet` | 13.200 | completed, good 88,73% |
| `qwen3-4b/predictions_partial_5900.parquet` | 5.900 | bản cũ, dừng do time budget |
| `qwen3-4b/summary_4b8b.json` | — | summary kernel 4B-8B |
| `qwen3-4b/summary_4models.json` | — | summary kernel 4-models |
| `qwen3-8b/predictions.parquet` | 11.900 | dừng do time budget, good 95,58% |
| `qwen3-8b/summary.json` | — | summary riêng 8B |
| `partial_old/qwen3-0.6b/` | 13.200 | partial cũ |
| `partial_old/qwen3-1.7b/` | 13.200 | partial cũ |
| `partial_old/qwen3-4b/` | 5.900 | partial cũ (trùng `predictions_partial_5900`) |
| `logs/tiki-description-bt-4b-8b.log` | — | log kernel 4B-8B |

## Cột

`pair_id`, `product_id`, `split`, `source_text`, `tokens_est`, `target_text_bt`,
`raw_output`, `json_ok`, `good`, `n_new_tokens`, `input_truncated`, `latency_ms`,
`model_id`, `prompt_version`, `attempt`, `qc_status`.

`split` lấy theo `product_id` từ pair table Tiki: train 10.491 / dev 1.386 / test 1.323.
