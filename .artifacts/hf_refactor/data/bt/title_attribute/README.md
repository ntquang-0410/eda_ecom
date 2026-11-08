# data/bt/title_attribute/ — BT title + attribute (v5)

Prompt `plan_b_rtx_v5` (brand-aware), gate loại `target == key_zh`, sinh cho toàn bộ
29.446 cache MT của Tiki. Bốn model chỉ khác `target_text_bt`.

## Quy mô

| model | full | train (qc ok) | units | good rate |
|---|---:|---:|---:|---:|
| qwen3-0.6b | 29.446 | 27.535 | 45.709 | 93,51% |
| qwen3-1.7b | 29.446 | 29.029 | 45.709 | 98,58% |
| qwen3-4b | 29.446 | 29.353 | 45.709 | 99,68% |
| qwen3-8b | 29.446 | 29.393 | 45.709 | 99,82% |

Split unit: train 36.434 / dev 4.824 / test 4.451.

## File mỗi model

| file | nội dung |
|---|---|
| `full.parquet` | cache-level, giữ `raw_output`, `qc_status` |
| `train.parquet` | chỉ dòng `qc_status = ok` |
| `units.parquet` | unit-level, có `split`, `qty_weight` |

`units.parquet` là bảng nên dùng để train/eval:

- `source_text_bt`: source đưa vào model
- `target_text_bt`: bản dịch synthetic
- `qty_status`, `qty_decision`, `qty_repaired`, `qty_weight`
- `split`: train/dev/test

`manifest.json` chứa bytes + sha256 + rows cho từng file.
