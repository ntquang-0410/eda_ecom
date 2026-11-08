# EDA tập A — sau dedup exact pair

**Ngày:** 2026-11-07  
**Nguồn:** `data/cleaned/weight_splits_v2/A_title_attribute.parquet`  
**Raw backup:** `A_title_attribute_raw.parquet`

## 1. Dedup đã làm gì

```text
raw rows                 : 118,688
dedup rows               : 118,477
exact pair removed       : 211
cross-split dup groups   : 56
source-dup rows kept     : 6,886
rule                     : exact (source,target), train > dev > test
```

Không dedup source-only: cùng source nhưng target khác có thể là biến thể hợp lệ.
Các cặp trùng đúng nhưng nằm ở nhiều split được giữ ở **train** để tránh leakage.

## 2. Phân bố sau dedup

| group | rows | row % | tokens | token % | median tok | empty | traditional |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1688_title | 16,133 | 13.6% | 752,144 | 16.0% | 46.5 | 0 | 5 |
| 1688_attribute | 73,560 | 62.1% | 3,064,267 | 65.1% | 9.9 | 0 | 2 |
| tiki_title | 13,057 | 11.0% | 487,867 | 10.4% | 34.6 | 0 | 1 |
| tiki_attribute | 15,727 | 13.3% | 402,223 | 8.6% | 9.0 | 0 | 0 |

Split: `{'train': 103653, 'dev': 7534, 'test': 7290}`.
Eval A: `1,200` dòng, không đổi.

![A group](figures/eda_a_dedup/A_group_share.png)

![A length](figures/eda_a_dedup/A_length.png)

![A ratio](figures/eda_a_dedup/A_ratio.png)

![A CJK](figures/eda_a_dedup/A_cjk.png)

![A split](figures/eda_a_dedup/A_split.png)

![A dup flags](figures/eda_a_dedup/A_dup_flags.png)

## 3. Nhận xét

- Chỉ mất **211 / 118,688 dòng** (~0,18%); phân bố gần như không đổi.
- Có **56** nhóm cặp trùng nằm ở nhiều split; đã dồn về train, nên hiện không còn exact duplicate cross-split.
- `1688_attribute` vẫn chiếm ~65% token; title 1688 ~16%; Tiki ~19%.
- Tiki attribute giảm nhẹ do các cặp generic (`1 mét`/`1米`, `1 ngày`/`1天`, ...) bị dedup với 1688.
- Còn **6,886** dòng source trùng nhưng target khác; giữ lại vì có thể là biến thể/ngữ cảnh khác. Cần theo dõi khi train.
- Không có target rỗng; A vẫn sạch hơn B.

## 4. Top source trùng (giữ lại, target khác nhau)

| source | rows | số target khác |
|---|---:|---:|
| `Kiểu nút bấm` | 64 | 64 |
| `Loại nút` | 48 | 48 |
| `Khác` | 47 | 47 |
| `Không` | 44 | 44 |
| `Là` | 40 | 40 |
| `áo thun1` | 37 | 37 |
| `Hàng sẵn có` | 35 | 35 |
| `giữa` | 32 | 32 |

## 5. Dùng cho weight survey

```text
Dùng A_title_attribute.parquet (đã dedup exact pair) cho khảo sát weight.
Không dùng A_title_attribute_raw.parquet trừ khi cần audit.
eval_A.parquet giữ nguyên.
```