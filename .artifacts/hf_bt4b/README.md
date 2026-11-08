# BT 4B — toàn bộ output Kaggle (Qwen3-4B)

Back-translation Tiki Việt → Trung bằng **Qwen3-4B**, gồm hai mảng:

1. `title_attribute/` — BT cho title + attribute (cache MT của Tiki).
2. `description/` — BT cho description sản phẩm.

Tất cả file là output gốc từ kernel Kaggle, chưa qua lọc/repair (giữ cả dòng lỗi để audit).

---

## 1. Title + attribute

| | |
|---|---|
| File | `title_attribute/predictions.parquet` |
| Rows | 29.446 (= toàn bộ cache MT) |
| Prompt | `plan_b_rtx_v5`, brand-aware |
| Model | `qwen3-4b` |
| good | 29.353 (**99,68%**) |
| `json_ok` | 99,97% |
| `input_truncated` | 6 dòng |
| Thời gian | 53,5 phút |
| Trạng thái | completed |

Cột: `cache_key`, `source_text_bt`, `key_vi`, `key_zh_effective`, `value_class`,
`target_text_bt`, `raw_output`, `json_ok`, `good`, `n_new_tokens`,
`input_truncated`, `latency_ms`, `attempt`, `prompt_variant`, `model_id`,
`prompt_version`, `qc_status`.

`run_state.json` là state của run 4B; `summary_all_models.json` là summary cả 4 model
(0.6B/1.7B/4B/8B) để so sánh; `log.jsonl` là log kernel.

Bản đã xử lý (full/train/units, có qty repair) nằm ở:

```text
data/bt_tiki/qwen3-4b/{full,train,units}.parquet
```

---

## 2. Description

### 2a. Bản đầy đủ (dùng chính)

| | |
|---|---|
| File | `description/predictions_full.parquet` |
| Rows | 13.200 (= toàn bộ sản phẩm Tiki) |
| Prompt | `plan_b_desc_v1` |
| Model | `qwen3-4b` |
| good | 11.712 (**88,73%**) |
| `json_ok` | 91,29% |
| `input_truncated` | 444 dòng |
| Thời gian | 320,2 phút |
| Kernel | `nadsa091/tiki-description-bt-4b-8b` |
| Trạng thái | completed |

Cột: `pair_id`, `product_id`, `split`, `source_text`, `tokens_est`,
`target_text_bt`, `raw_output`, `json_ok`, `good`, `n_new_tokens`,
`input_truncated`, `latency_ms`, `model_id`, `prompt_version`, `attempt`,
`qc_status`.

### 2b. Bản partial cũ (giữ để audit)

| | |
|---|---|
| File | `description/predictions_partial_5900.parquet` |
| Rows | 5.900 |
| good | 5.489 (93,03%) |
| Thời gian | 74,1 phút |
| Kernel | `nadsa091/tiki-description-bt-4-models` |
| Trạng thái | dừng do `time_budget` |

Bản này bị bản đầy đủ thay thế; giữ lại để đối chiếu. Trên HF còn một bản trùng
ở `bt_desc_partial/qwen3-4b/predictions.parquet`.

`run_state_full.json` / `run_state_partial.json` là state từng run;
`summary_4b8b.json` / `summary_4models.json` là summary của kernel tương ứng;
`logs/` chứa log kernel 4B-8B.

---

## 3. Ghi chú

- Cùng kernel `tiki-description-bt-4b-8b` còn output **8B** (11.900 dòng, good 95,58%,
  dừng do time budget) — không nằm trong thư mục này vì đây là bản 4B.
- Description 4B **đã** được đưa qua `tools/process_bt_desc.py`:
  `data/bt_tiki_desc/qwen3-4b/{full,train,units}.parquet` (11.404 unit, 85,4% QC ok,
  6,363M token). Xem `reports/TIKI_DESC_4B_PROCESSING.md`.
- Các dòng `good = False` vẫn được giữ; lọc theo `qc_status == "ok"` hoặc `good`.
