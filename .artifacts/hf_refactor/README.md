---
language:
- vi
- zh
multilinguality:
- translation
task_categories:
- translation
tags:
- ecommerce
- tiki
- "1688"
- qwen3
- back-translation
- machine-translation
pretty_name: Tiki / 1688 — 4-model translation + weight study
---

# Tiki + 1688 — dataset cho dịch thương mại điện tử Việt → Trung

Repo gom dữ liệu, output back-translation (BT) và kết quả thực nghiệm cho:

1. **4 model Qwen3**: `0.6B`, `1.7B`, `4B`, `8B`;
2. **BT Tiki** bằng Qwen3: title/attribute (v5) và description (`plan_b_desc_v1`);
3. **Weight survey A/B** trên Qwen3-4B với quota sampling;
4. **Đánh giá**: COMET-da (1688), COMET-QE (Tiki), credit judge (AAA..C).

> **Cảnh báo:** `target_text_bt` của Tiki là bản dịch **synthetic** do Qwen3 sinh ra,
> không phải bản dịch người. Không dùng làm reference gold. Tiki chỉ nên chấm bằng
> COMET-QE hoặc judge theo source.

## Cấu trúc

```text
README.md                     file này
INDEX.md                      mục lục nhanh
code/                         script train / eval / judge / sampler
data/
  1688/                       train/dev/test 1688 + tokenized OpenNMT
  pairs/                      pair table đã dedup cho 1688 và 4 model Tiki
  bt/
    title_attribute/          BT title+attribute v5 (4 model, đã process)
    description/
      processed/              BT description đã QC/chunk (4 model)
      raw/                    output gốc từ Kaggle (4B, 8B, partial cũ)
  weight/
    eval/                     eval set + train view cho weight experiment
    splits_v2/                split A/B trước EDA + eval_A/eval_B
    processed_A/              units A sau processing (quota survey)
    processed_B/              units B sau processing (quota survey)
    survey_ab/                hypothesis, COMET/credit scores, run summary
  tiki_desc_split/            split attribute vs description của Tiki
docs/                         ghi chú kỹ thuật + migration map
reports/                      báo cáo EDA / thực nghiệm + figures
```

Mỗi thư mục dữ liệu có `README.md` riêng, ghi rõ số dòng, cột, nguồn và cách dùng.

## Bộ dữ liệu chính

| thư mục | nội dung | quy mô |
|---|---|---|
| `data/1688/` | train/dev/test 1688 | 80.811 / 5.254 / 5.306 |
| `data/pairs/` | pair table 1688 + Tiki 4 model | 106.528 + 26.864…29.093 |
| `data/bt/title_attribute/` | BT title+attribute v5 | 29.446 full / 45.709 unit mỗi model |
| `data/bt/description/processed/` | BT description đã QC | 7.868…11.404 unit mỗi model |
| `data/weight/eval/` | eval 1688/tiki + train view | 750 / 600 / ~117k |
| `data/weight/splits_v2/` | A/B trước EDA | A 118.477, B 59.209 |
| `data/weight/processed_A/` | A sau processing | 118.732 unit / 4,842M token |
| `data/weight/processed_B/` | B sau processing | 57.406 unit / 15,710M token |
| `data/weight/survey_ab/` | survey quota 5 config × 2 tập | 10 run, seed 42 |

## Kết quả nổi bật

- **BT 4B description**: 13.200 dòng, good 88,73%, QC ok 85,42%, 11.404 unit / 6,363M token.
- **BT 8B description**: 11.900/13.200 dòng, good 95,58%, QC ok 90,35%, 10.342 unit / 5,494M token (dừng do time budget).
- **Weight survey A**: giữ `natural` — mọi reweight đều giảm COMET-da 1688 có ý nghĩa.
- **Weight survey B**: `balanced` tốt hơn `natural` +0,0144 COMET-da (CI [+0,0081; +0,0211]); `title_heavy` tương đương trên COMET nhưng credit thấp hơn.

Chi tiết: `reports/WEIGHT_SURVEY_A_B.md`, `reports/TIKI_DESC_BT_PROCESSING.md`.

## Bắt đầu nhanh

```bash
pip install -r code/requirements.txt

# xem cấu trúc từng bộ
cat data/README.md
cat data/bt/description/README.md

# chấm lại weight survey
python code/score_comet_weight.py \
  --hyp-dir data/weight/survey_ab/hyps --out-dir /tmp/scores --gpus 1
python code/aggregate_weight_ab.py \
  --suite A --scores-dir /tmp/scores \
  --out /tmp/WEIGHT_SURVEY_A.md --summary /tmp/summary_A.json
```

## Nguồn / giấy phép

- 1688: dữ liệu song ngữ thương mại điện tử do người dùng cung cấp.
- Tiki: title/attribute/description sản phẩm; target Tiki là synthetic từ Qwen3.
- Chỉ dùng cho nghiên cứu/dịch thuật nội bộ; kiểm tra lại điều khoản nguồn trước khi
  tái phân phối thương mại.
