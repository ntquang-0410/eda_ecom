# data/bt/description/ — BT description Tiki

Description prose (13.200 sản phẩm) không có bản dịch người, được BT bằng Qwen3
với prompt `plan_b_desc_v1`.

```text
processed/     đã QC + dedup + chunk, dùng để train
raw/           output gốc từ Kaggle (giữ cả dòng lỗi)
```

## Trạng thái

| model | rows | good | QC ok | units | token |
|---|---:|---:|---:|---:|---:|
| qwen3-0.6b | 13.200 | 66,8% | 61,3% | 7.868 | 3,881M |
| qwen3-1.7b | 13.200 | 72,3% | 68,1% | 9.125 | 5,259M |
| qwen3-4b | 13.200 | 88,7% | 85,4% | 11.404 | 6,363M |
| qwen3-8b | 11.900 | 95,6% | 90,3% | 10.342 | 5,494M |

**8B là bản partial**: dừng do time budget ở 11.900/13.200 sản phẩm, thiếu 1.300
description dài nhất (median source 1.450 vs 1.563 ký tự). Trên cùng 11.321 sản phẩm,
QC ok của 8B là 91,2% và 4B là 89,9%.

- Chi tiết processed: `processed/README.md`
- Output gốc: `raw/README.md`
- Báo cáo EDA: `reports/TIKI_DESC_BT_PROCESSING.md`
