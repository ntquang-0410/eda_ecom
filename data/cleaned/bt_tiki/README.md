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
pretty_name: Tiki BT (Qwen3 0.6B / 1.7B / 4B / 8B)
---

# Tiki BT — bản dịch máy Việt → Trung theo từng model

Mỗi model Qwen3 tự sinh bản dịch cho **cùng một tập unit** của Tiki, sau đó được
repair 2 vòng. Bốn bộ **chỉ khác cột `target_text_bt`**, mọi thứ khác dùng chung.

## Cấu trúc

```text
qwen3-0.6b/
  full.parquet    29.446 dòng (cache-level) + audit qc_status
  train.parquet   chỉ dòng dùng được để fine-tune
  units.parquet   45.709 dòng (unit-level, có product_id + split)
qwen3-1.7b/  qwen3-4b/  qwen3-8b/   (cùng cấu trúc)
reports/          bt_repair_report.json, bt_units_report.json, bt_audit.json
manifest.json     số dòng + sha256 từng file
```

## Nguồn

- 13.200 sản phẩm Tiki → 120.884 unit, trong đó **45.709 unit route MT**
  (13.200 title + 32.509 attribute).
- Split theo `product_id`: train 36.434 / dev 4.824 / test 4.451.
- `cache_key` là sha1 của (source, key, value_class, route) — 29.446 khóa MT.

## Cột chính

| cột | ý nghĩa |
|---|---|
| `pair_id` | id unit (`tiki:<product>:title` / `tiki:<product>:attr:<i>:<j>`) |
| `product_id`, `split`, `unit_type` | ngữ cảnh sản phẩm |
| `source_text_bt` | câu nguồn tiếng Việt |
| `target_text_bt` | bản dịch tiếng Trung của model tương ứng |
| `key_vi`, `key_zh_effective` | tên thuộc tính (chỉ attribute) |
| `qc_status` | `ok` / `repaired` / `repaired_v2` / `recovered_json` / `still_untranslated` |
| `repair_pass`, `prompt_version` | vòng sinh và prompt đã dùng |

## Chất lượng

Đã audit: 29.446/29.446 khóa khớp Plan A, 45.709/45.709 unit map được,
0 `pair_id` trùng, 0 lỗi encoding.

Còn 233 unit gắn cờ `still_untranslated` (mã màu, model code, đơn vị thuần).

Lưu ý: bản dịch là **synthetic**, không phải gold. Xem `reports/bt_audit.json`
để biết kiểm tra đầy đủ; EDA và làm sạch nằm ở pipeline code (`tiki_mining/bt_eda_clean.py`).
