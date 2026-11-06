# Tiki description split — attribute vs description

Tách `data/cleaned/tiki_vi_cado.parquet` thành 2 loại, tương ứng 1688:

```text
attribute   : attributes_vi / desc_vi_cleaned   (key-value)
description : desc_prose_vi_cleaned             (prose)
```

## Files

| file | rows | nội dung |
|---|---:|---|
| `attribute.parquet` | 89.325 | mỗi dòng một attribute name + values |
| `description.parquet` | 13.200 | mỗi dòng một description prose |
| `tiki_desc_split.parquet` | 102.525 | gộp cả hai, cột `pair_type` |

Split theo `product_id`, lấy từ `pairs_dedup/tiki_qwen3-4b.parquet`:
`train 10.491 / dev 1.386 / test 1.323` (description).

## Quan trọng: target tiếng Trung

Tiki **không có** `description_zh` (cột rỗng 100%). Vì vậy:

```text
attribute.target_text = ""
description.target_text = ""
```

Đây là **source-only split** để:

- audit / thống kê cấu trúc;
- tạo eval Tiki description chấm COMET-QE;
- làm input cho bước BT tạo target.

Không đưa trực tiếp vào train vì target rỗng.

## Eval mở rộng

Đã tạo:

```text
data/cleaned/weight_eval/eval_tiki_ext.parquet
```

gồm 300 title + 300 attribute + **150 description** (source-only, QE), với
`eval_tiki_desc.parquet` riêng cho description.

## Nếu cần training-ready

Phải chạy BT cho 13.200 description prose:

```text
source : desc_prose_vi_cleaned
target : sinh bằng Qwen3-4B (adapter title_heavy hoặc base)
token  : ~5,23M token ước lượng
```

Sau đó gate `qc_status`, dedup/chunk ≤800 token và merge vào pair table.
