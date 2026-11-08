# data/weight/processed_A/ — units A

Processing từ `splits_v2/A_title_attribute.parquet`: giữ toàn bộ dòng đã qua gate,
chunk parent >1.280 token.

| | |
|---|---|
| rows (full = units) | 118.732 |
| token | 4,842M |
| split | train 103.883 / dev 7.536 / test 7.313 |
| chunked parents | 127 |

Group share: 1688_attribute 66,1% · 1688_title 15,5% · tiki_title 10,1% · tiki_attribute 8,3%.

`full.parquet` và `units.parquet` giống nhau vì A không có dòng nào bị QC loại.
Đây là dữ liệu đầu vào của quota sampler cho survey A.
