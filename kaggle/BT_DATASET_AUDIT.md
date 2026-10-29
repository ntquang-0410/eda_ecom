# Audit 4 tập Tiki BT sau 2 vòng repair

Ngày: 2026-10-29. Nguồn: `data/cleaned/bt_tiki/`.

## Kết luận nhanh

4 tập **đầy đủ ở mức cache**: mỗi tập 29.446 dòng, không trùng, không thiếu
`cache_key`, cùng một tập khóa và cùng `source_text` giữa 4 model.

Chưa kiểm chứng được 2 điểm vì mất artifact nguồn + credential Kaggle:
đối chiếu với 29.446 cache route `mt` của Plan A, và mở rộng ra 45.709 unit.

## A. Cấu trúc từng file

| model | rows | dup key | null key | source rỗng | target rỗng | json_ok |
|---|---|---|---|---|---|---|
| 0.6B | 29.446 | 0 | 0 | 0 | 2 | 99,99% |
| 1.7B | 29.446 | 0 | 0 | 0 | 1 | 100,00% |
| 4B | 29.446 | 0 | 0 | 0 | 1 | 99,99% |
| 8B | 29.446 | 0 | 0 | 0 | 2 | 99,99% |

Target rỗng đều nằm trong nhóm `still_untranslated`, không lọt vào `_train`.

## B. Đồng nhất giữa 4 model

```text
key set      : giống hệt, 29.446 khóa mỗi tập (thiếu 0, thừa 0)
source_text  : lệch 0 / 29.446 ở cả 3 cặp so với 0.6B
train ⊂ full : True
target khớp  : True
```

## C. Trạng thái QC

| model | ok | repaired v1 | repaired v2 | cứu JSON | still | phủ |
|---|---|---|---|---|---|---|
| 0.6B | 15.708 | 12.345 | 1.233 | 2 | 158 | 99,46% |
| 1.7B | 20.729 | 8.086 | 582 | 0 | 49 | 99,83% |
| 4B | 29.193 | 210 | 31 | 1 | 11 | 99,96% |
| 8B | 29.154 | 250 | 31 | 1 | 10 | 99,97% |

Không có dòng `qc_status != still` mà target rỗng hoặc còn tiếng Việt.

## D. Vệ sinh dữ liệu

```text
control char        : 0
mojibake target     : 0
mojibake source     : 0
leading/trailing ws : 0
raw_output ↔ target : lệch 4 dòng / 117.784 (0,003%)
```

`target == source` (3.188 / 3.615 / 1.875 / 1.873) chủ yếu là brand/model Latin
hợp lệ: `iPhone 13`, `Type C`, `Z Fold 3`.

## E. Chất lượng ngữ nghĩa (heuristic)

| model | lệch số/đơn vị | ratio < 0,2 | p95 ratio |
|---|---|---|---|
| 0.6B | 3.345 | 3.443 | 1,00 |
| 1.7B | 1.919 | 1.214 | 1,00 |
| 4B | 1.909 | 978 | 1,00 |
| 8B | 1.730 | 497 | 1,00 |

Đây là **cờ**, không phải kết luận: một phần là model code (`S23` → `三星`) hoặc
quy đổi đơn vị. Cần COMET-QE để chốt.

## F. Đối chiếu nguồn Plan A — ĐẠT 100%

Sau khi khôi phục credential và tải lại `tiki-plan-a-bt-input`:

```text
cache Plan A            : 40.646 dòng, 29.446 khóa route 'mt'
dataset key set         : 29.446
missing_in_dataset      : 0
extra_in_dataset        : 0
```

Mở rộng ra unit (recompute `cache_key` bằng đúng hàm Plan A):

| kiểm tra | kết quả |
|---|---|
| unit MT | 45.709 |
| map được | 45.709 (100%) |
| không map được | 0 |
| `cache_key` recompute lệch Plan A | 0 |
| `pair_id` trùng | 0 |
| product phủ | 13.200 / 13.200 |
| split unit | train 36.434 / dev 4.824 / test 4.451 |

## G. Dataset unit-level

`tiki_mining/bt_add_units.py` xuất thêm 4 file:

```text
<model>/units.parquet   45.709 dòng
  cột: pair_id, product_id, split, unit_type, key_vi, key_zh,
       source_text_bt, target_text_bt, model_id, prompt_version,
       json_ok, qc_status, repair_pass, attempt, latency_ms
```

| model | units | usable | còn cờ | train | dev | test | target rỗng |
|---|---|---|---|---|---|---|---|
| 0.6B | 45.709 | 45.547 | 162 | 36.308 | 4.801 | 4.438 | 0 |
| 1.7B | 45.709 | 45.659 | 50 | 36.398 | 4.813 | 4.448 | 0 |
| 4B | 45.709 | 45.698 | 11 | 36.424 | 4.823 | 4.451 | 0 |
| 8B | 45.709 | 45.699 | 10 | 36.424 | 4.824 | 4.451 | 0 |

Không có dòng usable nào có target rỗng.

## H. Việc còn lại

- Lưu `.artifacts/plan_a` vào nơi bền vững (đã nằm trong repo, không dùng `/tmp`).
- Merge với 1688: chốt weight / cap attribute / gold set.
- Thêm COMET-QE nếu muốn lọc ngữ nghĩa trước khi train.
