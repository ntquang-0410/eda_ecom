# data/bt/description/processed/ — units đã QC

Output của `code/process_bt_desc.py`: strip boilerplate Tiki, QC CJK/tỷ lệ/số,
dedup exact pair, chunk >1.280 token.

## Quy mô

| model | full | units | train/dev/test | token | chunked |
|---|---:|---:|---|---:|---:|
| qwen3-0.6b | 12.471 | 7.868 | 6.292 / 797 / 779 | 3,881M | 164 |
| qwen3-1.7b | 12.486 | 9.125 | 7.238 / 946 / 941 | 5,259M | 433 |
| qwen3-4b | 12.499 | 11.404 | 9.052 / 1.185 / 1.167 | 6,363M | 499 |
| qwen3-8b | 11.321 | 10.342 | 8.243 / 1.064 / 1.035 | 5,494M | 21 |

`full` = sau dedup, trước lọc QC; `units` = chỉ `qc_status = ok`, đã chunk.
`summary.json` tổng hợp 4 model; `eda.json` trong từng model có đầy đủ flag QC.

## Cột

`pair_id`, `product_id`, `split`, `source_text`, `target_text_bt`, `tokens_est`,
`qc_status`, `f_empty`, `f_short`, `f_no_cjk`, `f_untranslated`, `f_ratio`,
`f_identical`, `f_input_truncated`, `f_number_mismatch`, `f_boilerplate`,
`chunk_index`, `n_chunks`, `unit_type` (= `description`).

## Lưu ý

- `source_text` đã strip boilerplate "Giá sản phẩm trên Tiki…".
- `f_boilerplate` phản ánh source gốc có boilerplate, không phải lỗi target.
- 8B chỉ có 11.900 dòng gốc (partial), nên `full`/`units` nhỏ hơn và ít chunk hơn.
