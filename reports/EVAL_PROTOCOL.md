# Eval protocol — chuẩn đánh giá cố định

**Phiên bản:** 2026-11-27  
**Trạng thái:** đang áp dụng; các bảng cũ (`MATRIX_5X2_EVAL_F_V2.md`, `ALL_MODELS_METRICS.md`) được giữ để tham chiếu lịch sử.

---

## 1. Test set paired cố định

| Mục | Giá trị |
|---|---|
| File index | `data/cleaned/eval_paired_858.parquet` |
| Số dòng | **858** = 1688: 231 + Tiki: 627 |
| Khoá join | `eval_id` (string), không đổi thứ tự |
| Hash | `data/cleaned/eval_paired_858.json` → `eval_ids_sha256` |
| Nguồn | `data/cleaned/eval_F_v2_common.parquet` (đã qua leakage audit: 0 product_id trùng train) |

Mọi so sánh **bắt buộc** dùng đúng index này. Không thêm/bớt dòng theo từng model.

### Lịch sử 831 → 858

Bản 831 (1688: 231 + Tiki: 600) là tập *phụ thuộc model*: 27 dòng Tiki (18 attribute + 9 description)
bị loại vì có cell cho hyp rỗng. Từ 2026-11-27 chuyển chính thức sang **858** (thuần nội dung).
So sánh 831 ↔ 858 và lý do: `reports/AUDIT_858.md`; audit rỗng của Qwen4B: `reports/AUDIT_EMPTY_QWEN4B.md`.

- 27 dòng thêm đều là Tiki, đã chấm COMET-QE cho 22 run, zero-fill.
- COMET-da 1688 không đổi; QE Tiki giảm 0,002–0,010; coverage LLM còn 97,1–98,6%.

## 2. Quy ước điểm

1. Join hyp theo `eval_id`; **hyp rỗng ⇒ 0 điểm**, không loại dòng.
2. `coverage` = % hyp không rỗng, báo cáo riêng cho 1688 và Tiki.
3. Không bao giờ lọc theo độ dài, theo stratum, hay theo việc model khác có hyp hay không.
4. Điểm hệ thống = trung bình per-row trên đúng index.
   **Eval không được lọc dòng:** `generate_rows(..., truncate=True)` + `--eval-max-seq 1280`
   (bug 9 dòng description dài bị bỏ âm thầm đã sửa — xem `reports/AUDIT_EMPTY_QWEN4B.md`).
5. **Macro:** `macro_w` = trung bình theo trọng số số dòng mỗi stratum (bằng mean tổng);
   `macro_unw` = trung bình 3 stratum với trọng số bằng nhau. Báo cáo cả hai cho 1688 và Tiki.

## 3. Artefact per-row (bắt buộc cho mọi run)

Đường dẫn: `.artifacts/canonical/<cell>/seed<seed>.parquet`

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `eval_id` | str | khoá |
| `dataset` | str | `1688` / `tiki` |
| `stratum` | str | `title` / `attribute` / `description` |
| `hyp` | str | rỗng nếu model không sinh được |
| `ref` | str | rỗng với Tiki (không reference) |
| `comet_da` | float | 1688; NaN với Tiki |
| `comet_qe` | float | Tiki; NaN với 1688 |
| `bleu` | float | sentence BLEU, 1688; NaN với Tiki |
| `chrf` | float | sentence chrF, 1688; NaN với Tiki |
| `nonempty` | bool | dùng để tính coverage |

Manifest đi kèm: `.artifacts/canonical/<cell>/seed<seed>.manifest.json`.

Tạo artefact:

```bash
# đã có score từng dòng
python tools/score_canonical.py --cell qwen4b_a --seed 7 \
  --hyp .artifacts/hyp/hyp_qwen4b_a_natural_seed7.parquet \
  --da-scores scores_da.parquet --qe-scores scores_qe.parquet

# tự chấm COMET (GPU + internet)
python tools/score_canonical.py --cell qwen4b_a --seed 7 --hyp ... --compute-comet
```

Retrofit các run cũ: `python tools/retrofit_canonical.py` (đã chạy cho 20 run).

## 4. Vai trò metric

| Metric | Vai trò | Dùng để xếp hạng? |
|---|---|---|
| **COMET-da (1688)** | **chính thức** | ✅ |
| **Human eval (Tiki)** | **chính thức** | ✅ |
| COMET-QE (Tiki) | chẩn đoán | ❌ |
| BLEU / chrF (1688) | chẩn đoán bề mặt | ❌ |

Bảng xếp hạng chỉ dùng COMET-da. QE/BLEU chỉ xuất hiện ở cột chẩn đoán kèm nhãn.

## 5. Seed policy

- Cell chính (10 dòng matrix): **≥5 seed** — `7, 42, 123, 2024, 2025`.
- Control: ≥3 seed — `7, 42, 123`.
- Báo cáo **mean ± std** qua seed (ddof=1).
- **Paired bootstrap** 2.000 lần, cluster theo `product_id`, cho Δ giữa hai cell; CI này chỉ phủ variance eval, **không phủ seed** — phải ghi rõ.
- So sánh giữa cell dùng per-row score trung bình qua seed (paired theo `eval_id`).

## 6. Reproducibility — manifest mỗi run

Trainer (`train_weight_quota_gpu.py`, `train_ecomgpt.py`) ghi `run_manifest_<config>_seed<seed>.json`:

```json
{
  "commit_hash": "...", "config_hash": "...", "seed": 42,
  "token_budget": 4000000, "adapter_path": "...", "base_model": "...",
  "cell_config": {"dataset": "A", "config": "natural", "micro_batch": 4, "...": "..."},
  "result": {"steps": 500, "tokens": 4000153, "minutes": 61.6}
}
```

- `commit_hash` = `git rev-parse HEAD` tại thời điểm train.
- `config_hash` = sha256 16 ký tự của dict cấu hình (dataset, config, seed, token, steps, lr, micro-batch, max-seq, dtype, adapter).
- **Leak-check assertion** giữ nguyên trong trainer: `assert not (train.product_id ∩ eval.product_id)`, log số sản phẩm trùng.

## 7. Human eval Tiki (chính thức)

Chưa có dữ liệu người chấm → trạng thái **MISSING**.

Thiết kế bắt buộc:

| Mục | Quy định |
|---|---|
| Mẫu | 120 dòng Tiki, phân tầng title/attribute/description |
| Systems | 4 cell, ẩn danh `S1..S4`, thứ tự ngẫu nhiên theo item |
| Thang | adequacy 1–5, fluency 1–5 (có thể thêm pairwise A/B/tie) |
| Rater | ≥2 người, độc lập, không biết hệ thống |
| Đồng thuận | Krippendorff α (ordinal) ≥0,6 mới dùng |
| Lưu | `data/cleaned/human_eval/tiki_sheet_*.csv` + `tiki_key_*.json` |

Sinh sheet: `python tools/make_human_eval_sheet.py --cells qwen4b_a qwen4b_b ecomgpt_a moses_b --n 120`.
Sheet mẫu đã sinh (chưa có điểm): `data/cleaned/human_eval/tiki_sheet_qwen4b_a_qwen4b_b_ecomgpt_a_moses_b.csv`.

## 8. Trạng thái tuân thủ (2026-11-27)

| Yêu cầu | Trạng thái |
|---|---|
| Index 858 cố định + hash | ✅ (thay 831 từ 27/11) |
| Zero-fill + coverage | ✅ (NLLB B seed 42: 0,5517 → **0,5494**, coverage 99,8%) |
| Artefact per-row cho mọi run | ✅ 21/21 run (Qwen3-8B A-balanced đã chấm đủ 858: DA 0,8553, QE 0,0518); EcomGPT B-natural FAILED (loại) |
| Audit 858 (zero-fill 27 dòng) | ✅ `reports/AUDIT_858.md`; 858 giờ là index chính |
| Seed ≥5 | ❌ 37 run cell chính còn thiếu (~107 giờ GPU); xem `SEED_MATRIX.md` |
| COMET-da là metric xếp hạng | ✅ `CANONICAL_EVAL.md` |
| Eval không lọc dòng dài | ✅ harness đã sửa; qwen4b_a đã sinh lại 9 dòng (coverage 100%); các cell Qwen khác còn 9 zero-fill, QE thấp hơn ~0,003 |
| Macro `macro_w` / `macro_unw` / stratum | ✅ bảng 1b `CANONICAL_EVAL.md` + §2 `ALL_MODELS_METRICS.md` |
| Human eval Tiki | ❌ MISSING |
| Manifest + leak assert | ✅ từ 27/11 (retrofit: commit/config hash có, hyperparams một số run để trống) |

## 9. Lệnh chuẩn

```bash
python tools/retrofit_canonical.py      # build artefact từ score cũ
python tools/aggregate_canonical.py     # bảng mean±std + bootstrap
python tools/seed_matrix.py             # khoảng trống seed + chi phí
python tools/make_human_eval_sheet.py   # sheet chấm Tiki
```
