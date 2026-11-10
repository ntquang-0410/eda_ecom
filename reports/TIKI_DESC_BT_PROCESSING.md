# Tiki description BT — QC & processing

**Ngày:** 2026-11-08  
**Script:** `code/process_bt_desc.py`  
**Input:** `data/bt/raw/description/`  
**Output:** `data/bt/description/processed/`

## 1. Funnel processing

| model | rows | good | QC ok | QC ok % | dedup bỏ | units | train | token (M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | 13,200 | 10,156 | 9,467 | 71.7% | 721 | 10,192 | 8,108 | 5.382 |
| qwen3-1.7b | 13,200 | 11,675 | 10,960 | 83.0% | 702 | 11,753 | 9,313 | 6.497 |
| qwen3-4b | 13,200 | 12,772 | 12,339 | 93.5% | 680 | 13,497 | 10,678 | 7.691 |
| qwen3-8b | 13,198 | 12,738 | 12,114 | 91.8% | 626 | 13,418 | 10,637 | 7.673 |

![funnel](figures/desc_bt/desc_bt_funnel.png)

## 2. Lỗi QC

| model | empty | no CJK | untranslated | char ratio | truncated | number mismatch |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | 798 | 3,086 | 2,723 | 1,115 | 66 | 5,618 |
| qwen3-1.7b | 626 | 1,823 | 1,454 | 813 | 102 | 5,061 |
| qwen3-4b | 393 | 710 | 417 | 488 | 26 | 4,192 |
| qwen3-8b | 621 | 957 | 418 | 819 | 0 | 4,876 |

![flags](figures/desc_bt/desc_bt_qc_flags.png)

Nhận xét:

- QC pass: 71.7% → 83.0% → 93.5% → 91.8%.
- `empty` còn 798, 626, 393, 621 dòng; phần lớn là target chỉ chứa boilerplate giá/tax.
- `number_mismatch` vẫn là lỗi phổ biến nhất ở mọi model.
- `input_truncated`: qwen3-0.6b=66, qwen3-1.7b=102, qwen3-4b=26, qwen3-8b=0.
- `f_boilerplate` phản ánh boilerplate phía source (đã strip), không phải lỗi target.

## 3. Sau chunking

| model | units | train/dev/test | token (M) | sản phẩm bị chunk |
|---|---:|---|---:|---:|
| qwen3-0.6b | 10,192 | 8,108 / 1,031 / 1,053 | 5.382 | 689 |
| qwen3-1.7b | 11,753 | 9,313 / 1,252 / 1,188 | 6.497 | 796 |
| qwen3-4b | 13,497 | 10,678 / 1,458 / 1,361 | 7.691 | 985 |
| qwen3-8b | 13,418 | 10,637 / 1,449 / 1,332 | 7.673 | 1,059 |

![length](figures/desc_bt/desc_bt_length.png)

## 4. So sánh công bằng 4B vs 8B

8B hiện có 12,572/13.200 sản phẩm.
Trên cùng 12,572 sản phẩm: 4B QC ok **94.3%**, 8B **92.5%**.
8B đã phủ gần đủ; chênh lệch chủ yếu do chất lượng target, không còn do thiếu dữ liệu.

## 5. Dùng cho train

```text
data/bt/description/processed/qwen3-4b/train.parquet
data/bt/description/processed/qwen3-4b/units.parquet
data/bt/description/processed/qwen3-8b/train.parquet
```
