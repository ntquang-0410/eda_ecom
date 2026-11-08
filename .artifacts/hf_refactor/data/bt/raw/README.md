# data/bt/raw/ — output gốc từ Kaggle

Giữ nguyên output của model, gồm cả dòng lỗi (`good = False`) để audit.
Đây là nguồn để chạy lại `code/process_bt_desc.py`.

```text
title_attribute/qwen3-4b/     BT title+attribute v5 (29.446)
description/
  qwen3-4b/                   full 13.200 + partial 5.900
  qwen3-8b/                   11.900 (dừng do time budget)
  partial_old/                0.6B/1.7B/4B bản partial cũ
  logs/                       log kernel 4B-8B
```

| thư mục | rows | nguồn |
|---|---:|---|
| `title_attribute/qwen3-4b/` | 29.446 | kernel BT v5 |
| `description/qwen3-4b/predictions_full.parquet` | 13.200 | `nadsa091/tiki-description-bt-4b-8b` |
| `description/qwen3-4b/predictions_partial_5900.parquet` | 5.900 | `nadsa091/tiki-description-bt-4-models` |
| `description/qwen3-8b/predictions.parquet` | 11.900 | `nadsa091/tiki-description-bt-4b-8b` |
| `description/partial_old/qwen3-*/` | 13.200 / 13.200 / 5.900 | kernel 4-models |

Xem README trong từng thư mục con để biết cột và trạng thái.
