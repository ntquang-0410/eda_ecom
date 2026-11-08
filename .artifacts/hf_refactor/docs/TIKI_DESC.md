# Tiki description — split + BT

## Split

```text
data/tiki_desc_split/attribute.parquet        89.325 dòng, key-value
data/tiki_desc_split/description.parquet      13.200 dòng, prose
data/tiki_desc_split/tiki_desc_split.parquet  102.525 dòng, gộp
```

- split theo `product_id`: train 10.491 / dev 1.386 / test 1.323.
- Tiki **không có** `description_zh`; `target_text` để trống.
- Đây là source-only split cho eval QE và BT.

## Eval mở rộng

```text
data/weight/eval/eval_tiki_ext.parquet   750 dòng: 300 title + 300 attr + 150 desc
data/weight/eval/eval_tiki_desc.parquet  150 dòng description
```

Description không có reference người; dùng COMET-QE.

## BT description

```text
data/bt/description/raw/qwen3-4b/predictions_full.parquet   13.200 dòng (good 88,7%)
data/bt/description/raw/qwen3-8b/predictions.parquet        11.900 dòng (good 95,6%, partial)
data/bt/description/processed/qwen3-4b/                     11.404 unit
data/bt/description/processed/qwen3-8b/                     10.342 unit
```

Prompt `plan_b_desc_v1`, source `desc_prose_vi_cleaned`, max input/output 1.536.

8B dừng do time budget ở 11.900/13.200 sản phẩm — chưa phải bộ final.
