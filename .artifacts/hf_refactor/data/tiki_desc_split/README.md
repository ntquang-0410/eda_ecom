# data/tiki_desc_split/ — attribute vs description

Tách Tiki thành 2 loại, tương ứng 1688:

```text
attribute   : attributes_vi / desc_vi_cleaned   (key-value)
description : desc_prose_vi_cleaned             (prose)
```

| file | rows | nội dung |
|---|---:|---|
| `attribute.parquet` | 89.325 | mỗi dòng một attribute name + values |
| `description.parquet` | 13.200 | mỗi dòng một description prose |
| `tiki_desc_split.parquet` | 102.525 | gộp cả hai, cột `pair_type` |

Split theo `product_id`: train 10.491 / dev 1.386 / test 1.323 (description).

## Target tiếng Trung

Tiki **không có** `description_zh`, nên `target_text` để trống. Đây là **source-only
split** để audit, tạo eval QE và làm input cho BT.

BT description đã hoàn tất:

```text
data/bt/description/processed/qwen3-4b/   13.200 dòng gốc → 11.404 unit
data/bt/description/processed/qwen3-8b/   11.900 dòng gốc → 10.342 unit (partial)
```

`manifest_tiki_desc.json` chứa sha256/bytes của bộ split gốc.
