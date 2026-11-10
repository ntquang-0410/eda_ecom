# Seed 42 + 123 — xác nhận weight survey A/B

**Ngày:** 2026-11-10  
**Model:** Qwen3-4B + LoRA, 4M token, 500 step, quota sampler  
**Seed:** 42 và 123  
**Eval:** A 1.200 dòng, B 900 dòng (885 có hypothesis)

---

## 1. Tập A — title + attribute

| config | Δ1688 seed 42 | Δ1688 seed 123 | Kết luận |
|---|---:|---:|---|
| attr_heavy | -0,0049 [-0,0083; -0,0013] | -0,0040 [-0,0077; -0,0004] | giảm, tái lập |
| title_heavy | -0,0069 [-0,0118; -0,0021] | chưa chạy | giảm |
| balanced | -0,0074 [-0,0124; -0,0025] | chưa chạy | giảm |
| tiki_heavy | -0,0155 [-0,0205; -0,0109] | chưa chạy | giảm mạnh nhất |

Tiki (QE): `attr_heavy` đổi dấu (+0,0048 seed 42 → -0,0194 seed 123) → không có lợi ích Tiki ổn định.

**Kết luận A: giữ `natural`.** Cả 2 seed đều cho thấy reweight làm 1688 giảm; không có config nào cải thiện Tiki một cách nhất quán.

---

## 2. Tập B — title + description

| config | Δ1688 seed 42 | Δ1688 seed 123 | Trung bình | Tái lập |
|---|---:|---:|---:|---|
| **balanced** | +0,0144 [+0,0081; +0,0213] | +0,0147 [+0,0102; +0,0195] | **+0,0146** | **có** |
| title_heavy | +0,0160 [+0,0097; +0,0227] | +0,0162 [+0,0112; +0,0215] | +0,0161 | **có** |
| tiki_heavy | +0,0047 [-0,0008; +0,0102] | chưa chạy | +0,0047 | chưa |
| desc_heavy | -0,0054 [-0,0113; +0,0006] | chưa chạy | -0,0054 | chưa |

Tiki (QE):

| config | ΔTiki seed 42 | ΔTiki seed 123 |
|---|---:|---:|
| balanced | +0,0002 | -0,0090 (ns) |
| title_heavy | -0,0034 (ns) | +0,0052 (ns) |

**Kết luận B: `balanced` được xác nhận.** Gain +0,0146 COMET-da trên 1688 lặp lại ở cả 2 seed, CI không chứa 0. Tiki không đổi. `title_heavy` cũng tái lập (+0,0161) nhưng credit thấp hơn và cực đoan hơn.

---

## 3. Credit score (một phần do quota API)

| suite | n | tốt nhất | kém nhất |
|---|---:|---|---|
| A | 840/1.000 | natural 7,65 | tiki_heavy 7,07 |
| B | 680/1.000 | balanced 6,68 | tiki_heavy 6,15 |

Credit xác nhận A `natural`, B `balanced`; loại `tiki_heavy`.

Phần còn lại: A 160 dòng, B 320 dòng — chấm tiếp khi quota reset.

---

## 4. Chốt

| tập | config | căn cứ |
|---|---|---|
| A | **natural** | 2 seed đều giảm khi reweight; credit natural cao nhất |
| B | **balanced** | +0,0146 COMET-da 1688, tái lập 2 seed; credit balanced cao nhất |

Lưu ý phạm vi: với B, gain chỉ vững trên **1688**; khi gộp Tiki, CI chứa 0 (p ≈ 0,94–0,96). Không báo cáo "B tốt hơn overall".

---

## 5. File

```text
reports/SEED123_A.md / SEED123_B.md      bảng chi tiết từng seed
reports/CLUSTER_BOOTSTRAP.md             CI theo product cho seed 42
data/weight/survey_ab/scores_A_seed123/  score seed 123
data/weight/survey_ab/scores_B_seed123/
data/weight/survey_ab/hyps/              thêm hyp_*_seed123
```

Tái lập:

```bash
python code/aggregate_seed123.py --suite B \
  --scores42  data/weight/survey_ab/scores_B \
  --scores123 data/weight/survey_ab/scores_B_seed123 \
  --eval      data/weight/splits_v2/eval_B.parquet \
  --out /tmp/SEED123_B.md --summary /tmp/seed123_B.json
```
