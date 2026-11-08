# Weight splits A/B (trước EDA)

## A — title + attribute

```text
raw      : 118.688 dòng, 4,707M token
dedup    : 118.477 dòng (bỏ 211 exact pair)
groups   : 1688_title, 1688_attribute, tiki_title, tiki_attribute
eval     : 1.200 dòng (300 mỗi stratum)
```

## B — title + description

```text
rows     : 59.209, 14,598M token
groups   : 1688_title, 1688_description, tiki_title, tiki_description
eval     : 900 dòng (300 title + 150 description mỗi nguồn)
source   : BT 4B description, trước QC/EDA
```

## Sau processing

| | rows | token |
|---|---:|---:|
| A | 118.732 | 4,842M |
| B | 57.406 | 15,710M |

Xem `data/weight/README.md`, `reports/WEIGHT_SURVEY_TECHNIQUE_A_B.md`,
`reports/WEIGHT_SURVEY_A_B.md`.
