# Báo cáo leakage gốc + audit nội dung `eval_F` → `eval_F_v2`

**Ngày:** 2026-11-16  
**Phạm vi:** `eval_F` 1.500 dòng, `units_A/B` post-fix, upstream `nhom_trung` 1688, Tiki title clusters, MinHash near-dup.  
**Tái lập:** `python tools/verify_review.py --out review_audit.json` → `review_audit.json`, `data/cleaned/eval_F_v2.parquet`, `data/cleaned/eval_F_v2_meta.json`.

---

## 1. Cơ chế lỗi cũ (đã xác nhận)

| Chỉ số | Trước fix (`data/cleaned/processed_A`) | Sau fix (`tiki-matrix-data`) |
|---|---:|---:|
| `units_A` rows / splits | 118.732 = train 103.883 + dev 7.536 + **test 7.313** | 111.129 = train 103.626 + dev 7.503 (**không test**) |
| `eval_A` cũ | 1.200 dòng / 915 sản phẩm | — |
| 915 sản phẩm eval cũ nằm trong `units.test` | **915/915 = 100%** | — |
| `eval_F` mới / `units` product overlap | — | **0** (A và B) |

→ Bản fix đã loại đúng test khỏi train và giữ `product_id` disjoint. Đây là phần **đúng** và được giữ.

## 2. Lỗi còn lại: disjoint theo `product_id` không đủ

### 2.1. Overlap nội dung `eval_F` với `units` train+dev

Chuẩn hoá: NFKC + casefold + gộp whitespace; `strong` bỏ thêm punctuation/space. Hash SHA-256 cho source, pair (source+target) và source-strong.

| Tiêu chí | A | B |
|---|---:|---:|
| `product_id` | 0 | 0 |
| Source trùng **nguyên văn** | 43 (2,87%) | 10 (0,67%) |
| Source trùng sau NFKC+casefold | 76 (5,07%) | 11 (0,73%) |
| Source trùng sau chuẩn hoá mạnh | 110 (7,33%) | 11 (0,73%) |
| Cặp `(source,target)` trùng khít | 14 | 0 |
| `near_dup_cluster_id` (nhom_trung + title + MinHash) | **584 (38,93%)** | **511 (34,07%)** |
| MinHash LSH bổ sung ngoài exact/group | 18 | 33 |
| **Tổng leak (union 5 khoá)** | **584** | **511** |

### 2.2. Theo stratum

| Stratum | A leak / n | B leak / n |
|---|---:|---:|
| 1688/title | **210 / 300** | **210 / 300** |
| 1688/attribute | **196 / 300** | **176 / 300** |
| 1688/description | **102 / 150** | **102 / 150** |
| tiki/title | 1 / 300 | 2 / 300 |
| tiki/attribute | 72 / 300 | 7 / 300 |
| tiki/description | 3 / 150 | 14 / 150 |

**Nguyên nhân:** 16.348 sản phẩm 1688 chỉ có 15.863 title ZH unique; upstream `nhom_trung` có 6.566 nhóm near-duplicate (cùng title/shop/description hoặc MinHash Jaccard ≥ 0,8), nhưng split cũ chỉ theo `product_id`. Vì vậy 2/3 title 1688 trong eval có cùng nhóm near-dup với train. Đây là kênh leak mạnh nhất, không phải source trùng nguyên văn.

### 2.3. Dedup file A vẫn lỗi chuẩn hoá

`data/cleaned/weight_splits_v2/A_title_attribute.parquet` (118.477 dòng, `dedup_rule=exact_pair_train_first`):

- **583 nhóm cặp khít** vẫn nằm ở ≥2 split;
- **3.191 dòng trùng dư** ngoài bản đầu;
- khớp đúng con số hội đồng nêu. Nguyên nhân: `tools/dedup_weight_A.py` chỉ gộp whitespace, phân biệt hoa/thường, nên `lớn → 均码` và `Lớn → 均码` không bị coi là trùng.

## 3. `eval_F_v2` — tập sạch nội dung

**Định nghĩa:** loại một dòng eval nếu khớp **bất kỳ** khoá nào với `units` train+dev của A hoặc B:

1. `product_id`
2. `source_hash` (NFKC + casefold + whitespace)
3. `source_hash_strong` (bỏ punctuation/space)
4. `pair_hash` (source+target)
5. `near_dup_cluster_id`: `nhom_trung` 1688 + title-cluster Tiki + MinHash LSH 0,8/32 perm (5-char shingles)

**Kết quả:** 1.500 → **858 dòng**.

| Stratum | Số dòng |
|---|---:|
| 1688/title | 90 |
| 1688/attribute | 96 |
| 1688/description | 45 |
| tiki/title | 291 |
| tiki/attribute | 224 |
| tiki/description | 112 |

Tập **common** cho mọi cell (loại thêm 15 prompt Qwen >952 token và 21 Tiki-attribute rỗng của EcomGPT-B):
**831 dòng** — 1688: 231; Tiki: 600.

## 4. Tác động điểm số

### 4.1. Điểm trên leaked vs clean

| cell | COMET leaked | COMET clean | Δ clean−leak |
|---|---:|---:|---:|
| Qwen3-8B A Tiki | +0,1361 | +0,0494 | **−0,0867** |
| Qwen3-8B B Tiki | +0,1772 | +0,0477 | **−0,1295** |
| Qwen3-4B A Tiki | +0,1211 | +0,0537 | −0,0674 |
| Qwen3-4B B Tiki | +0,1334 | +0,0422 | −0,0912 |
| EcomGPT A Tiki | +0,0712 | −0,0018 | −0,0730 |
| Moses B Tiki | −0,0462 | −0,1663 | −0,1200 |
| OpenNMT A Tiki | −1,0299 | −1,4138 | −0,3839 |

Trên 1688, chênh lệch nhỏ hơn (|Δ| ≤ 0,013), vì leak chủ yếu là title/attribute ngắn, dễ với mọi model. Trên Tiki QE, leak làm điểm **cao hơn rõ** — báo cáo cũ lạc quan nhẹ.

### 4.2. Re-score 10 cell trên `eval_F_v2_common`

| Model | Data | COMET 1688 | COMET-QE Tiki | BLEU | chrF |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | A | 0,8567 | 0,0521 | 53,37 | 52,00 |
| Qwen3-8B | B | **0,8572** | **0,0509** | 68,38 | 67,22 |
| Qwen3-4B | A | **0,8584** | 0,0573 | 39,81 | 39,59 |
| Qwen3-4B | B | 0,8471 | 0,0461 | 66,96 | 65,55 |
| EcomGPT-7B | A | 0,8250 | 0,0077 | 27,55 | 27,16 |
| EcomGPT-7B | B | 0,7634 | −0,1366 | 14,39 | 15,16 |
| Moses | A | 0,7862 | −0,2439 | 33,11 | 35,13 |
| Moses | B | 0,8037 | −0,1593 | 65,14 | 63,66 |
| OpenNMT | A | 0,3488 | −1,4177 | 0,22 | 1,63 |
| OpenNMT | B | 0,3387 | −1,4074 | 2,00 | 3,13 |

**Kết luận:** xếp hạng model **không đổi**. Chênh lệch tuyệt đối lớn nhất so với `eval_F` gốc là **0,018 COMET** (Qwen-B Tiki); 1688 gần như bất biến (≤0,009). Các kết luận chính vẫn đứng, nhưng mọi con số nên trích từ `eval_F_v2_common` để tránh leak.

## 5. Yêu cầu bắt buộc — trạng thái

| # | Yêu cầu | Trạng thái | Bằng chứng |
|---:|---|---|---|
| 1 | Chuẩn hoá hash + near-dup + nhóm trùng | **xong** | `tools/verify_review.py`; `review_audit.json` |
| 2 | `eval_F_v2` disjoint 5 khoá + assert fail cứng | **xong** | `data/cleaned/eval_F_v2.parquet`; meta |
| 3 | Bảng overlap theo stratum + re-score 10 cell / sensitivity | **xong** | báo cáo này §2, §4; `MATRIX_5X2_EVAL_F_V2.md` |
| 4 | Báo cáo leakage gốc + audit script + hash trong run JSON | **xong một phần** | báo cáo này; script; hash đã có trong `RUN_HASH_MANIFEST.json`; 3 run JSON matrix đã backfill, các cell còn lại dùng manifest. |
| 5 | Nếu giữ eval cũ: bảng leaked rows + điểm loại trừ | **xong** | §4.1; `EVAL_F_V2_SENSITIVITY.csv` |

**Caveat:** `eval_F_v2` chỉ còn 231 dòng 1688 (45 description), nên CI của 1688 rộng; v2 dùng để kiểm tra độ bền, không thay thế một eval lớn hơn. Vẫn cần một `eval_F_v3` được thiết kế **trước** khi sinh hypothesis, với nhóm near-dup tách train/test ngay từ đầu.
