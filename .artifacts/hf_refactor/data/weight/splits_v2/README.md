# data/weight/splits_v2/ — split A/B trước EDA

Tách từ dữ liệu **trước EDA/processing**, dùng làm đầu vào cho quota survey.

| file | rows | token | ghi chú |
|---|---:|---:|---|
| `A_title_attribute.parquet` | 118.477 | 4,707M | A đã dedup exact pair |
| `A_title_attribute_raw.parquet` | 118.688 | — | A raw trước dedup |
| `B_title_description.parquet` | 59.209 | 14,598M | B raw từ BT 4B |
| `eval_A.parquet` | 1.200 | — | 300 mỗi stratum |
| `eval_B.parquet` | 900 | — | 300 title + 150 desc mỗi nguồn |
| `meta_A.json`, `meta_B.json` | — | — | group share + configs |
| `meta_A_dedup.json` | — | — | log dedup A |
| `meta.json` | — | — | tổng hợp A/B |

## Group share

**A**: 1688_attribute 65,1% · 1688_title 16,0% · tiki_title 10,4% · tiki_attribute 8,6%.

**B**: 1688_description 56,6% · tiki_description 34,9% · 1688_title 5,2% · tiki_title 3,3%.

## Configs

A: `natural`, `balanced` 25/25/25/25, `title_heavy` 40/20/25/15,
`attr_heavy` 10/60/5/25, `tiki_heavy` 10/20/30/40.

B: `natural`, `balanced` 25/25/25/25, `title_heavy` 40/10/40/10,
`desc_heavy` 2/55/3/40, `tiki_heavy` 10/15/35/40.

Thứ tự nhóm trong config: 1688_title / 1688_attribute hoặc 1688_description /
tiki_title / tiki_attribute hoặc tiki_description.
