---
language:
- vi
- zh
multilinguality:
- translation
task_categories:
- translation
tags:
- ecommerce
- tiki
- vietnamese
- chinese
- machine-translation
- qwen3
pretty_name: Tiki BT (v5 main, v3 archive)
---

# Tiki BT — v5 là bộ chính

Mỗi model Qwen3 tự sinh bản dịch cho cùng 29.446 cache MT của Tiki.
Bốn bộ **chỉ khác `target_text_bt`**.

## Phiên bản

| phiên bản | prompt | ghi chú |
|---|---|---|
| **v5 (chính)** | `plan_b_rtx_v5` brand-aware | gate loại `target == key_zh`; tốt nhất |
| v3 (archive) | `plan_b_rtx_v3` + 2 vòng repair | giữ để đối chiếu |

## Cấu trúc

```text
qwen3-0.6b/
  full.parquet        v5, 29.446 dòng (cache-level) + qc_status
  train.parquet       v5, chỉ dòng qc_status=ok
  units.parquet       v5, 45.709 dòng (unit-level, có product_id + split)
  v3/                 bản cũ: full/train/units
qwen3-1.7b/  qwen3-4b/  qwen3-8b/   (cùng cấu trúc)
reports/              báo cáo repair/audit
manifest.json         số dòng + sha256 từng file
```

## Chất lượng v5 (gate chặt)

| model | good | unit usable |
|---|---|---|
| 0.6B | 93,5% | 43.598 / 45.709 |
| 1.7B | 98,6% | 45.219 |
| 4B | 99,7% | 45.612 |
| 8B | 99,8% | 45.652 |

Lỗi `target == key_zh` từng chiếm 31% ở 0.6B (v3) đã giảm còn 4 dòng.

## Nguồn

13.200 sản phẩm Tiki → 120.884 unit; 45.709 unit route MT
(13.200 title + 32.509 attribute). Split theo `product_id`:
train 36.434 / dev 4.824 / test 4.451.

Bản dịch là **synthetic**, không phải gold.
