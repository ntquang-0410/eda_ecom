# partial_old — BT description partial cũ

Output của kernel `nadsa091/tiki-description-bt-4-models`, trước khi chạy lại 4B/8B.

| model | rows | good | ghi chú |
|---|---:|---:|---|
| `qwen3-0.6b` | 13.200 | 66,8% | completed |
| `qwen3-1.7b` | 13.200 | 72,3% | completed |
| `qwen3-4b` | 5.900 | 93,0% | dừng do time budget |

Bản 4B ở đây trùng với `../qwen3-4b/predictions_partial_5900.parquet`.
Bản đã process của 0.6B/1.7B nằm ở `data/bt/description/processed/`.
