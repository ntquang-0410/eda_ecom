# NLLB B — subsample cân bằng epoch với A

**Mục tiêu:** budget 4 M token nhưng B lớn gấp 3,2 lần A ⇒ NLLB B chỉ thấy 0,25 epoch, A thấy 0,83.
Cắt ngẫu nhiên B xuống cùng kích thước A để hai bên ~0,83 epoch.

## 1. Kết quả

| Tập | Rows | Eff tokens | Epoch @4 M |
|---|---:|---:|---:|
| A (gốc) | 118.732 | 4.841.703 | 0,826 |
| B (gốc) | 57.406 | 15.710.361 | 0,255 |
| **B subsample** | **17.695** | **4.841.873** | **0,826** |

- Seed cắt: **42**; overshoot +170 token so với A (0,004%).
- File: `data/cleaned/nllb_train_b_subsampled.parquet` (giữ nguyên cột gốc + `subsample_seed`, `subsample_rank`).
- Metadata: `data/cleaned/nllb_train_b_subsampled.json`.

## 2. Phân bố giữ nguyên

| Group | % token B gốc | % token subsample |
|---|---:|---:|
| 1688_description | 52,57% | 52,95% |
| 1688_title | 4,79% | 4,73% |
| tiki_description | 39,54% | 39,18% |
| tiki_title | 3,11% | 3,14% |

Lệch tối đa 0,38 điểm % ⇒ cắt ngẫu nhiên không làm đổi phân bố nhóm.

Thành phần split của subsample: train 15.130, dev 1.340, test 1.225 (giữ đúng tỉ lệ của B).

## 3. Cách dùng

1. Thay `processed_B/units.parquet` bằng file này khi retrain NLLB B (hoặc trỏ pipeline sang file mới).
2. **Build lại sample** `sample_B_balanced_seed42.parquet` từ file subsample (cùng seed 42, cùng budget 4 M,
   cùng `quota_indices`) — không dùng lại sample cũ vì `row_index` trỏ vào bảng gốc.
3. Ghi vào manifest: `train_pool=subsampled_B`, `subsample_seed=42`, `epochs_at_4M=0,826`.
4. Khi so sánh A/B của NLLB, hai bên đã cùng epoch; vẫn phải chạy ≥3 seed vì run-variance B lớn
   (`AUDIT_NLLB_B_SEED.md`).

## 4. Tái lập

```bash
python tools/make_nllb_b_subsample.py
```
