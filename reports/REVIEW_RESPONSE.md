# Phản hồi 3 biên bản phản biện — trạng thái & kế hoạch

**Ngày:** 2026-11-16  
**Phạm vi:** 3 file `REVIEW_1_DATA_LEAKAGE.md`, `REVIEW_2_OPENNMT_FAIRNESS.md`, `REVIEW_3_QUOTA_LENGTH_BIAS.md`.  
**Nguyên tắc:** không tuyên bố "đã xử lý" nếu chưa có artefact tái lập; các mục chưa xong ghi rõ `PARTIAL`/`PLANNED`.

---

## 0. Tóm tắt

| Phản biện | Mức | Trạng thái | Kết luận ngắn |
|---|---|---|---|
| 1 — Data leakage | BLOCKER | **FIXED** | `eval_F` gốc leak 642/1.500 (42,8%); `eval_F_v2` 858 dòng sạch; re-score 10 cell xong; xếp hạng không đổi, max Δ 0,018 |
| 2 — OpenNMT fairness | MAJOR | **PARTIAL (3/4)** | Đã relabel B-natural, bảng compute, NLLB control xong (0,6344 / 0,5517); thiếu seed 2–3 và recipe chuẩn |
| 3 — Quota/length bias | MAJOR | **PARTIAL (3/4)** | Audit exposure/tokenizer/eval xong; 2×2 đủ cho Qwen3-4B, seed 42/123; thiếu 2×2 cho 4 model còn lại và seed thứ 3 |

---

## 1. Phản biện 1 — Data leakage

| # | Yêu cầu | Trạng thái | Bằng chứng |
|---:|---|---|---|
| 1 | Chuẩn hoá hash + near-dup + nhóm trùng | **DONE** | `tools/verify_review.py`; `review_audit.json` |
| 2 | `eval_F_v2` disjoint 5 khoá + assertion fail cứng | **PARTIAL** | `eval_F_v2` 858 dòng + `verify_review.py` assert 5 khoá; **preflight của kernel vẫn chỉ assert `product_id`** |
| 3 | Bảng overlap theo stratum + re-score 10 cell / sensitivity | **DONE** | `reports/LEAKAGE_AUDIT.md` §2, §4; `reports/MATRIX_5X2_EVAL_F_V2.md` |
| 4 | Báo cáo leakage gốc + audit script + hash trong run JSON | **PARTIAL** | `LEAKAGE_AUDIT.md`; `RUN_HASH_MANIFEST.json`; 3 run JSON đã backfill; **5 run mới chưa có hash** |
| 5 | Nếu giữ eval cũ: bảng leaked rows + điểm loại trừ | **DONE** | `LEAKAGE_AUDIT.md` §4.1; `EVAL_F_V2_SENSITIVITY.csv` |

**Kết quả định lượng:**

- `product_id` overlap = 0 (đúng như tuyên bố cũ), nhưng source hash 76/11, pair hash 14/0, nhóm near-dup **584/511**.
- `eval_F_v2` = 858 dòng; common 831 (1688: 231, Tiki: 600).
- Re-score: Qwen3-8B A 0,8567; B 0,8572; Qwen3-4B A 0,8584; B 0,8471; EcomGPT A 0,8250; B 0,7634; Moses A 0,7862; B 0,8037; OpenNMT A 0,3488; B 0,3387.
- Max |Δ| so với eval gốc = 0,018 (Qwen-B Tiki); xếp hạng giữ nguyên.
- Leaked rows được chấm cao hơn trên Tiki QE (Qwen +0,087…+0,130), nên báo cáo cũ lạc quan nhẹ.

**Còn lại:** `eval_F_v3` thiết kế trước khi inference, tách near-dup group ngay từ split; hiện `eval_F_v2` là lọc hậu hoc trên hypothesis đã có (hợp lệ cho sensitivity, chưa phải benchmark độc lập hoàn hảo).

## 2. Phản biện 2 — OpenNMT fairness

| # | Yêu cầu | Trạng thái | Bằng chứng |
|---:|---|---|---|
| 1 | Sửa nhãn/recipe: retrain B-balanced 30k hoặc đổi tên B-natural | **DONE (relabel)** | Đã đổi cell thành **B-natural** trong mọi bảng; retrain B-balanced bị chặn bởi offline env |
| 2 | Thêm baseline NMT pretrained (mBART/NLLB) cùng protocol | **DONE** | NLLB-600M A 0,6344 / −0,5379; B 0,5517 / −0,6753 (`CONTROLS_2X2.md`) |
| 3 | Hoặc recipe NMT chuẩn + ≥3 seed + benchmark | **PLANNED (P2)** | chưa chạy; cần 6 layer/512, BPE 16k, 50–100M token |
| 4 | Bảng compute công bằng, sửa "compute-matched" | **DONE** | `reports/OPENNMT_FAIRNESS.md` §2 |
| 5 | Xử lý độ dài giống nhau, báo cáo stratum dài | **PARTIAL** | length buckets trong `MATRIX_5X2_EVAL_F_V2.md` §8; OpenNMT vẫn cắt 49 dòng >512 |

**Điểm quan trọng:** FLOPs fine-tune OpenNMT A (5,46e16) **không lớn hơn** Qwen3-8B (1,97e17); cái lệch là pretraining và token-pass 61,4×. Vì vậy không được dùng "compute-matched" cho NMT, nhưng cũng không được nói "NMT tốn compute hơn". Kết luận đúng: **một cấu hình from-scratch 37M/4 layer/256 hidden sụp**, không phải "NMT paradigm yếu".

## 3. Phản biện 3 — Quota & length bias

| # | Yêu cầu | Trạng thái | Bằng chứng |
|---:|---|---|---|
| 1 | Per-group epoch, coverage, draws/row trong run JSON + bảng | **DONE** | `review_audit.json` `quota`; `QUOTA_LENGTH_AUDIT.md` §2.1 |
| 2 | Tokenizer thật, bỏ cap 960, báo cáo truncation | **DONE (Qwen)** | A 6,560M / B 5,926M Qwen tokens; epoch 0,8855 / 0,2477; truncation 0,1% / 0,6% (uncapped 30,4% tiki desc) |
| 3 | Eval cố định + sensitivity + bucket + macro-average | **DONE** | `eval_F_v2_common` 831; `MATRIX_5X2_EVAL_F_V2.md` §7–8 |
| 4 | 2×2 field × quota, hoặc B-natural + A-balanced cho 5 model | **PARTIAL (Qwen3-4B xong)** | 2×2 đủ cho Qwen3-4B; 4 model còn lại chưa có (`DEBATE_GAP_AUDIT.md` §3) |
| 5 | ≥3 seed + CI theo seed + sampling seed 123 | **PARTIAL** | 2 seed cho Qwen3-4B: A-natural 0,8584/0,8511; B-balanced 0,8471/0,8479; seed variance 0,005–0,007; thiếu seed thứ 3 |

**Đã sửa đúng:** quota trong nội bộ nhóm (manifest median/P90 khớp population).  
**Chưa sửa end-to-end:** exposure lệch 18×, 87% description không học, target dài bị cắt, eval cũ khác n theo cell.

---

## 4. Kế hoạch xử lý từng vấn đề

### P0 — Chặn bảo vệ (đã xong)

1. `eval_F_v2` + audit 5 khoá + báo cáo leakage.
2. Re-score 10 cell trên `eval_F_v2_common` + sensitivity + macro/length.
3. Bảng compute OpenNMT + xác nhận lỗi nhãn B.
4. Quota exposure + tokenizer thật + hash manifest.

### P1 — Đang chạy / tuần này

5. **OpenNMT-B balanced 30k** (L4x4) → thay cell B; cập nhật `MATRIX_5X2_EVAL_F_V2.md`.
6. **NLLB-200-distilled-600M A/B** cùng 4M quota, cùng max_new → control pretrained NMT (đang chạy L4x4, LoRA r16).
7. **Qwen3-4B 2×2 control:** A-balanced (đang chạy L4x4) + B-natural (chờ slot), so với A-natural/B-balanced đã có.
8. **Qwen3-4B seed 123 A/B** + sampling seed 123 → ước lượng seed variance tối thiểu.
9. Cập nhật mọi `run_*.json` mới với `dataset_hash`, `sample_hash`, `eval_hash`, per-group exposure.

### P2 — Trước khi nộp tiểu luận

10. **2×2 đầy đủ cho 5 model** (A-natural, A-balanced, B-natural, B-balanced) — ưu tiên Qwen3-8B, EcomGPT, Moses, OpenNMT.
11. **≥3 seed** cho ít nhất các cell Qwen (4B/8B) và OpenNMT/NLLB; báo cáo CI theo seed.
12. **NMT recipe chuẩn** (6 layer/512, BPE 16k, 50–100M token, early stop, averaging, beam + length penalty, bỏ `replace_unk`) + IWSLT/WMT sanity.
13. **`eval_F_v3`** thiết kế trước inference, split theo `nhom_trung`/near-dup group; không lọc hậu hoc.
14. Viết lại mọi kết luận chỉ dùng `eval_F_v2_common`/v3; bỏ các câu "NMT thất bại", "quota đã khắc phục length bias" nếu chưa đủ.

### P3 — Đóng gói

15. Re-upload HF: báo cáo mới, `eval_F_v2`, scores v2, audit JSON, manifest hash.
16. Cập nhật `README`, `docs/INDEX.md`, `FINETUNE_CONFIGS_5MODELS.md` với caveat và trạng thái mới.

---

## 5. Artefact

| Nội dung | Đường dẫn |
|---|---|
| Audit tái lập | `tools/verify_review.py` → `review_audit.json` |
| Eval sạch | `data/cleaned/eval_F_v2.parquet`, `eval_F_v2_common.parquet` |
| Leakage report | `reports/LEAKAGE_AUDIT.md` |
| Re-score v2 | `reports/MATRIX_5X2_EVAL_F_V2.md` |
| Sensitivity | `reports/EVAL_F_V2_SENSITIVITY.csv` |
| Quota | `reports/QUOTA_LENGTH_AUDIT.md` |
| OpenNMT | `reports/OPENNMT_FAIRNESS.md` |
| Gap audit 3 phản biện | `reports/DEBATE_GAP_AUDIT.md` |
| Hash manifest | `reports/RUN_HASH_MANIFEST.json/.csv` |

**Lưu ý GPU:** các kernel mới dùng `machine_shape: NvidiaL4x4`, `competition_sources: ["gemma-4-developer-agent"]` và dataset `metric/gemma-4-developer-agent-wheelhouse`; internet off.
