# data/weight/processed_B/ — units B

Processing từ `splits_v2/B_title_description.parquet`: strip boilerplate Tiki,
bỏ target rỗng, QC Tiki description, dedup exact pair, chunk parent >1.280 token.

| | |
|---|---|
| raw rows | 59.209 |
| bỏ target rỗng | 1.532 |
| sau dedup (full) | 57.999 |
| units | 57.406 |
| token | 15,710M |
| split | train 49.166 / dev 4.169 / test 4.071 |
| chunked parents | 626 |

Group share: 1688_description 52,6% · tiki_description 39,5% · 1688_title 4,8% · tiki_title 3,1%.

`full.parquet` giữ cả dòng `needs_review`; `units.parquet` chỉ dòng `qc_status = ok`.
Đây là dữ liệu đầu vào của quota sampler cho survey B.
