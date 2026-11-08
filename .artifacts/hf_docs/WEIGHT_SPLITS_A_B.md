# Weight splits A/B (pre-EDA)

## A — title + attribute

```text
rows 118.688, effective tokens 4,707M
groups: 1688_title, 1688_attribute, tiki_title, tiki_attribute
eval: 1.200 dòng
```

## B — title + description

```text
rows 59.209, effective tokens 14,598M
groups: 1688_title, 1688_description, tiki_title, tiki_description
eval: 900 dòng
Tiki description: raw BT 4B, trước QC/EDA
empty target: 1.150 dòng (giữ nguyên)
```

Xem `reports/EDA_PROCESSING_PLAN_A_B.md` và
`reports/WEIGHT_SURVEY_TECHNIQUE_A_B.md`.
