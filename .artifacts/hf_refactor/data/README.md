# data/ — tổng quan

| thư mục | nội dung | quy mô | README |
|---|---|---|---|
| `1688/` | train/dev/test 1688 | 80.811 / 5.254 / 5.306 | `1688/README.md` |
| `pairs/` | pair table dedup | 106.528 + 26.864…29.093 | `pairs/README.md` |
| `bt/` | back-translation Tiki | 4 model × 2 mảng | `bt/README.md` |
| `weight/` | weight experiment A/B | ~176k unit | `weight/README.md` |
| `tiki_desc_split/` | split attribute/description | 102.525 dòng | `tiki_desc_split/README.md` |

## Quan hệ giữa các bộ

```text
1688 raw ──dedup/chunk──> pairs/1688.parquet ──> weight/eval + weight/splits_v2
Tiki title/attribute ──BT Qwen3 v5──> bt/title_attribute/<model>/
Tiki description     ──BT plan_b_desc_v1──> bt/description/processed/<model>/
bt/title_attribute + bt/description ──merge──> pairs/tiki_qwen3-<model>.parquet
weight/splits_v2 ──quota processing──> weight/processed_A|B ──survey──> weight/survey_ab
```

## Nguyên tắc

- Split theo `product_id`, không để sản phẩm xuất hiện ở cả train và test.
- Tiki target là synthetic; không dùng làm gold.
- Mọi so sánh model/weight phải dùng cùng `eval_id` và cùng token budget.
