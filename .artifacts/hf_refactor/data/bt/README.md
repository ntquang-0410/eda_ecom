# data/bt/ — back-translation Tiki bằng Qwen3

Hai mảng BT độc lập:

```text
title_attribute/          BT title + attribute, prompt plan_b_rtx_v5
description/
  processed/              BT description đã QC/chunk (dùng để train)
  raw/                    output gốc Kaggle (4B, 8B, partial cũ)
```

| mảng | model | prompt | full | đã process |
|---|---|---|---:|---|
| title/attribute | 0.6B/1.7B/4B/8B | `plan_b_rtx_v5` | 29.446 | 45.709 unit/model |
| description | 4B | `plan_b_desc_v1` | 13.200 | 11.404 unit |
| description | 8B | `plan_b_desc_v1` | 11.900 | 10.342 unit |

> Tất cả `target_text_bt` là bản dịch synthetic. Không dùng làm reference gold.

- Title/attribute: xem `title_attribute/README.md`.
- Description: xem `description/README.md`.
- Output gốc: xem `raw/README.md`.
