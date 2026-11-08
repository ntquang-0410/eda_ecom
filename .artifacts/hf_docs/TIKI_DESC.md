# Tiki description split + BT partial

Bổ sung ngày 2026-11-07.

## Split

```text
data/tiki_desc_split/attribute.parquet        89.325 dòng, key-value
data/tiki_desc_split/description.parquet      13.200 dòng, prose
data/tiki_desc_split/tiki_desc_split.parquet  102.525 dòng, gộp
```

- split theo `product_id`: train 10.491 / dev 1.386 / test 1.323 (description)
- Tiki **không có** `description_zh`; `target_text` để trống.
- Đây là source-only split cho eval QE và BT.

## Eval mở rộng

```text
data/weight_eval/eval_tiki_ext.parquet   750 dòng: 300 title + 300 attr + 150 desc
data/weight_eval/eval_tiki_desc.parquet  150 dòng description
```

Description không có reference người; dùng COMET-QE.

## BT partial

```text
bt_desc_partial/qwen3-0.6b/predictions.parquet  13.200 (good 66.8%)
bt_desc_partial/qwen3-1.7b/predictions.parquet  13.200 (good 72.3%)
bt_desc_partial/qwen3-4b/predictions.parquet     5.900  (good 93.0%, partial)
```

Đây là **partial**, chưa phải bộ final. 4B/8B đang chạy tiếp.

Prompt: `plan_b_desc_v1`, source `desc_prose_vi_cleaned`, max input/output 1.536.
