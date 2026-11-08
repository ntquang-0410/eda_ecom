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
- machine-translation
pretty_name: Tiki / 1688 4-model translation evaluation
---

# Tiki + 1688 — dataset cho đánh giá 4 model Qwen3

Bộ này gom **toàn bộ dữ liệu cần thiết** để:

1. huấn luyện / đánh giá 4 model Qwen3: `0.6B`, `1.7B`, `4B`, `8B`;
2. chạy lại thực nghiệm weight `desc / title / attribute` trên Qwen3-4B;
3. chấm bằng COMET (1688 reference + Tiki QE) và Gemma judge.

> **Lưu ý quan trọng:** `target_text_bt` của Tiki là bản dịch **synthetic** do từng model Qwen3 sinh ra, không phải bản dịch người. Không dùng nó làm reference gold khi báo cáo chất lượng. Tiki chỉ nên chấm bằng COMET-QE hoặc judge theo source.

## Cấu trúc

```text
data/
  pairs_dedup/
    1688.parquet                    1688 đã dedup/chunk
    tiki_qwen3-0.6b.parquet         Tiki pair table cho 0.6B
    tiki_qwen3-1.7b.parquet
    tiki_qwen3-4b.parquet
    tiki_qwen3-8b.parquet
  weight_eval/
    eval_1688.parquet               750 dòng: 300 title + 300 attr + 150 desc
    eval_tiki.parquet               600 dòng: 300 title + 300 attr, không reference
    train_view_0.6b.parquet         train view cho weight experiment
    train_view_1.7b.parquet
    train_view_4b.parquet
    train_view_8b.parquet
    meta.json                       token mass / group share
  1688_exp/
    train.parquet dev.parquet test.parquet test_small.parquet
    *.vi *.zh *.tok.vi *.tok.zh     bản text / tokenized cho OpenNMT/Moses
  bt_tiki/
    qwen3-0.6b/{full,train,units}.parquet
    qwen3-1.7b/...
    qwen3-4b/...
    qwen3-8b/...
    README.md manifest.json
code/
  build_weight_eval_sets.py
  train_weight_t4.py
  train_weight_gpu.py
  tpu_weight_train.py
  score_comet_weight.py
  build_judge_input_multi.py
  gemma_judge.py
  aggregate_weight_eval.py
  WEIGHT_EVAL_PIPELINE.md
manifest.json                       sha256 + số dòng từng file
```

## 1. `pairs_dedup/`

Dùng để train/eval theo từng model.

Cột chính:

| cột | ý nghĩa |
|---|---|
| `pair_id` | id duy nhất, dạng `1688:<product>:title`, `tiki:<product>:attr:...` |
| `domain` | `1688` / `tiki` |
| `pair_type` | `title` / `attribute` / `description` |
| `product_id` | id sản phẩm, split theo product để tránh leak |
| `split` | `train` / `dev` / `test` |
| `source_text` | tiếng Việt |
| `target_text` | tiếng Trung; với Tiki là synthetic |
| `key_vi`, `key_zh` | key phụ (nếu có) |
| `tokens_est` | ước lượng token VI/3.6 + ZH/1.6 |
| `dup_count` | số lần cặp bị trùng trước dedup |
| `qty_status`, `qty_weight` | trạng thái đại lượng và weight repair |
| `chunk_index`, `n_chunks` | chunk description dài |

Split 1688 và Tiki theo `product_id`; không có product nào nằm ở cả train và test.

## 2. `weight_eval/`

### Eval set cố định

`eval_1688.parquet`:

- 300 title, 300 attribute, 150 description
- có `reference` người
- dùng **COMET-da**

`eval_tiki.parquet`:

- 300 title, 300 attribute
- `reference` để trống
- dùng **COMET-QE**

Mọi cấu hình / seed phải dùng đúng cùng `eval_id`, cùng thứ tự.

### Train view cho weight experiment

`train_view_<model>.parquet` gồm 1688 train + Tiki train của model đó:

- bỏ dòng `tokens_est > 1024`
- thêm `group` = `title_1688` / `attr_1688` / `desc_1688` / `tiki`
- thêm `eff_tokens` (dùng để sample đúng token budget)

Khi train weight, lọc thêm `eff_tokens <= 960` để prompt + target vừa `MAX_SEQ=1024`.

## 3. `1688_exp/`

Bộ train/dev/test 1688 cho track 1688-only và cho so sánh 4 model:

| file | rows |
|---|---:|
| `train.parquet` | 80.811 |
| `dev.parquet` | 5.254 |
| `test.parquet` | 5.306 |
| `test_small.parquet` | 1.000 (500 title + 500 attribute) |

`test_small` **không có description**. Muốn đánh giá description phải dùng `weight_eval/eval_1688.parquet`.

Các file `.tok.vi` / `.tok.zh` là tokenized cho OpenNMT/Moses, không cần cho HuggingFace models.

## 4. `bt_tiki/`

Mỗi model Qwen3 có 3 file:

| file | rows | dùng làm gì |
|---|---:|---|
| `full.parquet` | 29.446 | cache-level, giữ `raw_output`, `qc_status` |
| `train.parquet` | ~27.5k | chỉ dòng `qc_status=ok` |
| `units.parquet` | 45.709 | unit-level, có `split`, `qty_weight` |

`units.parquet` là bảng nên dùng để train/eval theo unit:

- `split`: train 36.434 / dev 4.824 / test 4.451
- `source_text_bt`: source đưa vào model
- `target_text_bt`: bản dịch synthetic của model tương ứng
- `qty_status`, `qty_decision`, `qty_repaired`, `qty_weight`

Bốn bộ **chỉ khác `target_text_bt`**; source/split giống nhau.

## 5. Cách đánh giá 4 model

```text
với mỗi model trong 0.6B / 1.7B / 4B / 8B:
  train trên 1688 train + bt_tiki/<model>/units.parquet split=train
  generate trên:
    - weight_eval/eval_1688.parquet   -> COMET-da + BLEU/chrF
    - weight_eval/eval_tiki.parquet   -> COMET-QE
  judge Gemma blind trên một subset stratified
```

Không dùng `target_text_bt` của Tiki làm reference. Nếu cần so sánh 4 model trên Tiki, dùng cùng một QE model và cùng eval_id.

## 6. Thực nghiệm weight (Qwen3-4B)

```text
5 cấu hình:
  natural      phân bố token tự nhiên (desc ~64%)
  balanced     title 15, attr 35, desc 25, tiki 25
  no_desc      title 25, attr 45, desc 0,  tiki 30
  tiki_heavy   title 10, attr 25, desc 20, tiki 45
  title_heavy  title 30, attr 30, desc 15, tiki 25

2 seed: 42, 123
5M token/run, 625 optimizer step, ~8.000 token/step
paired bootstrap 2.000 lần theo eval_id
guardrail: COMET-da 1688 không giảm quá 0,01
```

Chạy:

```bash
# train + generate (GPU hoặc TPU)
python code/train_weight_t4.py \
  --model /path/Qwen3-4B \
  --train-view data/weight_eval/train_view_4b.parquet \
  --eval-dir data/weight_eval \
  --configs natural balanced no_desc tiki_heavy title_heavy \
  --seeds 42 123 --tokens 5000000 \
  --out /runs/weight

# COMET (1688 da + Tiki QE)
python code/score_comet_weight.py \
  --hyp-dir /runs/weight --out-dir /runs/weight --batch-size 8

# Gemma blind
python code/build_judge_input_multi.py \
  --hyp-dir /runs/weight --out-dir /runs/judge --n 50
python code/gemma_judge_credit.py \
  --input /runs/judge/judge_input.jsonl \
  --map /runs/judge/judge_map.jsonl \
  --keys keys.txt --out-dir /runs/judge --model gemma-4-31b-it --batch-size 10

# tổng hợp
python code/aggregate_weight_eval.py \
  --scores-dir /runs/weight --eval-dir data/weight_eval \
  --gemma /runs/judge/gemma_scores.json \
  --out reports/WEIGHT_EVAL.md
```

`train_weight_gpu.py` là bản GPU dùng đúng 625 step / ~8.000 token mỗi step.
`train_weight_t4.py` là bản GPU đơn giản (sampler theo token, số step thay đổi).
`tpu_weight_train.py` là bản TPU 8 chip với cùng scheduler công bằng.

## 7. Lưu ý phương pháp

- **Không** so sánh COMET-QE Tiki với COMET-da 1688; hai thang khác nhau.
- **Không** dùng Tiki `target_text_bt` làm gold.
- 1688 `test_small` chỉ có title/attribute; description phải dùng `eval_1688`.
- `train_view` đã lọc `<=1024`; khi train với `MAX_SEQ=1024` nên lọc thêm `<=960`.
- Nếu chạy 4 model, mỗi model phải dùng đúng `tiki_qwen3-<model>.parquet` và
  `bt_tiki/qwen3-<model>/` của nó; không trộn target giữa các model.
- Weight experiment chỉ có ý nghĩa khi cùng token budget, cùng optimizer step,
  cùng eval_id; nếu chỉ cố định token mà số step thay đổi, kết quả bị confound.

## 8. Kiểm tra toàn vẹn

`manifest.json` chứa `bytes`, `sha256`, `rows` cho từng file. Kiểm tra:

```bash
python - <<'PY'
import json, hashlib
from pathlib import Path
root=Path('.')
m=json.loads((root/'manifest.json').read_text())
for rel, info in m['files'].items():
    p=root/rel
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert h==info['sha256'], rel
print('OK', len(m['files']), 'files')
PY
```

## Nguồn / giấy phép

- 1688: dữ liệu song ngữ thương mại điện tử do người dùng cung cấp.
- Tiki: title/attribute sản phẩm; bản dịch Tiki là synthetic từ Qwen3.
- Chỉ dùng cho nghiên cứu/dịch thuật nội bộ; không tái phân phối thương mại khi
  chưa kiểm tra lại điều khoản nguồn.
