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

## F. Chưa kiểm chứng được (blocked)

1. `29.446 key` có đúng bằng tập cache route `mt` của Plan A không.
   Số lượng khớp số đã biết nhưng không còn file nguồn để so.
2. Mở rộng 29.446 cache → 45.709 unit MT (cần `tiki_bt_units_plan_a.parquet`).
3. Join `product_id` / `split` cho train-test (dataset hiện chỉ có cache-level).

Nguyên nhân: `/tmp/opencode` bị xoá và `/root/.kaggle` mất credential.

## G. Việc cần làm

- Khôi phục `~/.kaggle/access_token` (hoặc `kaggle.json`).
- Tải lại `ntquang0410/tiki-plan-a-bt-input` và lưu vào `.artifacts/` (trong repo,
  không dùng `/tmp`).
- Bổ sung `product_id`, `unit_type`, `split` vào dataset trước khi merge 1688.
