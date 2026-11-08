# data/pairs/ — pair table đã dedup

| file | rows | token ước lượng |
|---|---:|---:|
| `1688.parquet` | 106.528 | 12,33M |
| `tiki_qwen3-0.6b.parquet` | 26.864 | 0,80M |
| `tiki_qwen3-1.7b.parquet` | 28.625 | 0,85M |
| `tiki_qwen3-4b.parquet` | 28.979 | 0,89M |
| `tiki_qwen3-8b.parquet` | 29.093 | 0,90M |

## Cột chính

| cột | ý nghĩa |
|---|---|
| `pair_id` | id duy nhất (`1688:<product>:title`, `tiki:<product>:attr:...`) |
| `domain` | `1688` / `tiki` |
| `pair_type` | `title` / `attribute` / `description` |
| `product_id` | id sản phẩm, dùng để split |
| `split` | `train` / `dev` / `test` |
| `source_text` | tiếng Việt |
| `target_text` | tiếng Trung; Tiki là synthetic |
| `key_vi`, `key_zh` | key phụ (nếu có) |
| `tokens_est` | VI/3,6 + ZH/1,6 |
| `dup_count` | số lần cặp trùng trước dedup |
| `qty_status`, `qty_weight` | trạng thái đại lượng và weight repair |
| `chunk_index`, `n_chunks` | chunk description dài |

## Dedup

| file | trước | sau | bỏ |
|---|---:|---:|---:|
| 1688 | 428.464 | 106.528 | 321.936 |
| tiki 0.6B | 43.598 | 26.864 | 16.734 |
| tiki 1.7B | 45.219 | 28.625 | 16.594 |
| tiki 4B | 45.612 | 28.979 | 16.633 |
| tiki 8B | 45.652 | 29.093 | 16.559 |

Dedup exact pair; description dài được chunk. Bốn bộ Tiki chỉ khác `target_text`.
