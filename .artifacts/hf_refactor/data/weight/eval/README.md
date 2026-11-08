# data/weight/eval/ — eval set + train view

## Eval cố định

| file | rows | thành phần | metric |
|---|---:|---|---|
| `eval_1688.parquet` | 750 | 300 title + 300 attribute + 150 description | COMET-da + BLEU/chrF |
| `eval_tiki.parquet` | 600 | 300 title + 300 attribute | COMET-QE |
| `eval_tiki_ext.parquet` | 750 | 300 title + 300 attribute + 150 description | COMET-QE |
| `eval_tiki_desc.parquet` | 150 | description only | COMET-QE |

- 1688 có `reference` người.
- Tiki `reference` để trống; **không** dùng `target_text_bt` làm gold.

## Train view

`train_view_<model>.parquet` = 1688 train + Tiki train của model đó.

| model | rows | token |
|---|---:|---:|
| 0.6B | 115.506 | 10,87M |
| 1.7B | 116.900 | 10,91M |
| 4B | 117.247 | 10,95M |
| 8B | 117.363 | 10,96M |

Đã bỏ `tokens_est > 1024`, thêm `group` và `eff_tokens` để sample theo token budget.

`meta.json` chứa group share, rows và token mass của từng view.
