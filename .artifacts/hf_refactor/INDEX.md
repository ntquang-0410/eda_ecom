# INDEX

## Dữ liệu

| đường dẫn | nội dung |
|---|---|
| `data/README.md` | tổng quan toàn bộ dữ liệu |
| `data/1688/README.md` | train/dev/test 1688 |
| `data/pairs/README.md` | pair table 1688 + Tiki |
| `data/bt/README.md` | tổng quan back-translation |
| `data/bt/title_attribute/README.md` | BT title+attribute v5 (4 model) |
| `data/bt/description/README.md` | BT description (processed + raw) |
| `data/bt/description/processed/README.md` | units/full/train đã QC |
| `data/bt/raw/README.md` | output gốc Kaggle 4B/8B |
| `data/weight/README.md` | weight experiment |
| `data/weight/eval/README.md` | eval set + train view |
| `data/weight/splits_v2/README.md` | split A/B trước EDA |
| `data/weight/processed_A/README.md` | units A |
| `data/weight/processed_B/README.md` | units B |
| `data/weight/survey_ab/README.md` | survey quota A/B |
| `data/tiki_desc_split/README.md` | split attribute/description Tiki |

## Code

- `code/README.md` — danh sách script và thứ tự chạy
- `code/WEIGHT_EVAL_PIPELINE.md` — thiết kế pipeline weight cũ

## Báo cáo

- `reports/README.md` — mục lục báo cáo
- `reports/WEIGHT_SURVEY_A_B.md` — **kết luận weight A/B**
- `reports/TIKI_DESC_BT_PROCESSING.md` — EDA/processing BT description
- `reports/EDA_PROCESSED_A_B.md` — EDA dữ liệu A/B sau processing

## Ghi chú kỹ thuật

- `docs/README.md`
- `docs/MIGRATION_2026-11-08.md` — map đường dẫn cũ → mới
- `docs/TIKI_DESC.md` — split description + BT
- `docs/WEIGHT_SPLITS_A_B.md` — split A/B trước EDA
- `docs/RUN_ON_RTX6000.md` — chạy trên GPU RTX Pro 6000
