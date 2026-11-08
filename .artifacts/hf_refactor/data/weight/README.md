# data/weight/ — weight experiment A/B

Thực nghiệm gán trọng số token cho 4 nhóm dữ liệu trên Qwen3-4B.

```text
eval/             eval set cố định + train view cho 4 model
splits_v2/        split A/B trước EDA + eval_A/eval_B
processed_A/      A sau processing: title + attribute
processed_B/      B sau processing: title + description
survey_ab/        quota survey 5 config, seed 42
```

| thư mục | nội dung | rows | token |
|---|---|---:|---:|
| `eval/` | eval 1688/tiki + train view | 750 / 600 / ~117k | ~11M |
| `splits_v2/` | A/B trước EDA | 118.477 / 59.209 | 4,707M / 14,598M |
| `processed_A/` | A sau processing | 118.732 | 4,842M |
| `processed_B/` | B sau processing | 57.406 | 15,710M |
| `survey_ab/` | hyps + scores | 10 run | 4M/run |

## Tập A và B

| | A | B |
|---|---|---|
| Nhóm | 1688_title, 1688_attribute, tiki_title, tiki_attribute | 1688_title, 1688_description, tiki_title, tiki_description |
| Eval | 1.200 dòng | 900 dòng |
| Kết luận | giữ `natural` | chọn `balanced` |

Xem `survey_ab/README.md` và `reports/WEIGHT_SURVEY_A_B.md`.
