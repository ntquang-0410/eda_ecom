# Migration map — 2026-11-08

Refactor repo HF để nhóm theo pipeline. Đường dẫn cũ → mới:

## Data

| cũ | mới |
|---|---|
| `data/1688_exp/` | `data/1688/` |
| `data/pairs_dedup/` | `data/pairs/` |
| `data/bt_tiki/` | `data/bt/title_attribute/` |
| `data/bt_tiki_desc/` | `data/bt/description/processed/` |
| `data/bt_qwen3-4b/title_attribute/` | `data/bt/raw/title_attribute/qwen3-4b/` |
| `data/bt_qwen3-4b/description/` | `data/bt/raw/description/qwen3-4b/` |
| `bt_desc_partial/` | `data/bt/raw/description/partial_old/` |
| `data/weight_eval/` | `data/weight/eval/` |
| `data/weight_splits_v2/` | `data/weight/splits_v2/` |
| `data/processed_A/` | `data/weight/processed_A/` |
| `data/processed_B/` | `data/weight/processed_B/` |
| `data/processed_A_B.json` | `data/weight/processed_A_B.json` |
| `data/weight_survey_ab/` | `data/weight/survey_ab/` |

## Docs / code

| cũ | mới |
|---|---|
| `RUN_ON_RTX6000.md` | `docs/RUN_ON_RTX6000.md` |
| `TIKI_DESC.md` | `docs/TIKI_DESC.md` |
| `WEIGHT_SPLITS_A_B.md` | `docs/WEIGHT_SPLITS_A_B.md` |
| `manifest.json` | `docs/manifest_2026-11-07.json` |
| `manifest_tiki_desc.json` | `data/tiki_desc_split/manifest_tiki_desc.json` |
| `requirements.txt` | `code/requirements.txt` |

## Reports

| cũ | mới |
|---|---|
| `reports/figures_A_dedup/` | `reports/figures/eda_a_dedup/` |
| `reports/figures_ab/` | `reports/figures/eda_ab/` |
| `reports/figures_processed_ab/` | `reports/figures/processed_ab/` |
| `reports/figures_desc_bt/` | `reports/figures/desc_bt/` |
| `reports/TIKI_DESC_4B_PROCESSING.md` | `reports/TIKI_DESC_BT_PROCESSING.md` |

Các báo cáo trong `reports/` đã được cập nhật link figure sang đường dẫn mới.
Đường dẫn `data/cleaned/...` trong báo cáo là đường dẫn của **code repo**, không phải
đường dẫn trên HF; dùng bảng trên để map.
