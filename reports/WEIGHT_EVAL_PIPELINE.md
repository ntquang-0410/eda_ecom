# Pipeline đánh giá weight desc / title / attribute (COMET + Gemma)

Mục tiêu: trả lời **có cần gán weight cho description/title/attribute không**,
dựa trên COMET (reference 1688 + QE Tiki) và Gemma judge.

## Thành phần

```text
kaggle/weight_eval/weight_eval_comet.ipynb   train 5 cấu hình + sinh output + COMET
tools/build_judge_input_multi.py             tạo JSONL chấm Gemma (có tiền tố config)
tools/gemma_judge.py                         chấm Gemma, xoay vòng API key
tools/aggregate_weight_eval.py               gộp COMET + Gemma, bootstrap, kết luận
```

## Bước 1 — chạy kernel COMET

```bash
cd /content/eda_ecom
kaggle kernels push -p kaggle/weight_eval
kaggle kernels status ntquang0410/tiki-weight-eval-comet
kaggle kernels output ntquang0410/tiki-weight-eval-comet -p /tmp/opencode/weight_eval
```

Kernel sẽ:

```text
với mỗi cấu hình natural | balanced | no_desc | tiki_heavy | title_heavy:
  - train Qwen3-4B LoRA, 5M token, sampler theo weight
  - sinh 1.000 output 1688 test_small
  - sinh 1.000 output Tiki test
  - lưu hyp_<config>_1688.parquet / hyp_<config>_tiki.parquet
sau đó:
  - COMET wmt22-comet-da cho 1688 (có reference)
  - COMET wmt22-cometkiwi-da cho Tiki (QE)
  - lưu scores_<config>_{1688,tiki}.parquet + weight_eval_comet.json
```

Thời gian ước tính: ~2h (5 lần train + generate + COMET).

## Bước 2 — tạo input chấm Gemma

```bash
python tools/build_judge_input_multi.py \
  --hyp-dir /tmp/opencode/weight_eval \
  --out-dir data/cleaned/judge \
  --n 100
```

Sinh `judge_input.jsonl` với id dạng `balanced::1688:123:title` để giữ cả
config, dataset và pair_type.

## Bước 3 — chấm Gemma

```bash
cp keys.example.txt keys.txt
nano keys.txt        # dán mỗi API key một dòng

python tools/gemma_judge.py \
  --input data/cleaned/judge/judge_input.jsonl \
  --keys keys.txt \
  --out-dir data/cleaned/judge \
  --model gemma-3-27b-it \
  --batch-size 25
```

Script tự xoay key khi hết quota, resume nếu đứt, và tổng hợp
`data/cleaned/judge/gemma_scores.json` (có `by_config`).

## Bước 4 — tổng hợp và kết luận

```bash
python tools/aggregate_weight_eval.py \
  --comet /tmp/opencode/weight_eval/weight_eval_comet.json \
  --gemma data/cleaned/judge/gemma_scores.json \
  --scores-dir /tmp/opencode/weight_eval \
  --out reports/WEIGHT_EVAL.md
```

Output:

```text
reports/WEIGHT_EVAL.md
data/cleaned/weight_eval_summary.json
```

## Quy tắc quyết định

```text
primary   : COMET-QE trên Tiki (domain đích)
guardrail : COMET 1688 không thấp hơn best quá 0,01
bootstrap : 2.000 lần paired trên per-segment score

nếu best - natural < 0,005 hoặc CI chứa 0
    -> kết luận natural (không cần weight)
ngược lại
    -> chọn cấu hình tốt nhất
```

Gemma dùng làm tín hiệu phụ: nếu COMET và Gemma lệch nhau, ghi chú trong báo cáo.

## Tiêu chí "không cần weight"

Kết luận `natural` nghĩa là phân bố token tự nhiên (desc ~63%) đã tốt nhất;
không cần can thiệp weight cho desc/title/attribute.

Kết luận `balanced`/`no_desc`/... nghĩa là cần weight, kèm mức cụ thể trong báo cáo.
