# EDA sau Processing — A/B

**Ngày:** 2026-11-07  
**Input:** `data/cleaned/processed_A/units.parquet`, `processed_B/units.parquet`.

## 1. Tóm tắt processing

| bộ | raw rows | empty dropped | dedup removed | rows full | rows units | chunked parents | token (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 118,477 | 0 | 0 | 118,732 | 118,732 | 127 | 4.842 |
| B | 59,209 | 1,532 | 485 | 57,999 | 57,406 | 626 | 15.71 |

- A: giữ toàn bộ dòng người/BT đã qua gate; chunk 127 parent >1.280 token.
- B: bỏ 1.532 target rỗng, dedup 485 cặp, chunk 626 parent; Tiki description raw 4B được QC riêng.

## 2. A sau processing

| group | rows | token % | median token | empty | CJK mean |
|---|---:|---:|---:|---:|---:|
| 1688_title | 16,133 | 15.5% | 46.5 | 0 | - |
| 1688_attribute | 73,815 | 66.1% | 9.9 | 0 | - |
| tiki_title | 13,057 | 10.1% | 34.6 | 0 | - |
| tiki_attribute | 15,727 | 8.3% | 9.0 | 0 | - |

Split: `{'train': 103883, 'dev': 7536, 'test': 7313}`. Eval A: **1,200** dòng.

![A group](figures/processed_ab/A_group_share.png)

![A length](figures/processed_ab/A_length.png)

![A ratio](figures/processed_ab/A_ratio.png)

![A CJK](figures/processed_ab/A_cjk.png)

![A split](figures/processed_ab/A_split.png)

### Nhận xét A

- `1688_attribute` vẫn ~66% token; title 1688 ~15,5%; Tiki ~18%.
- Tất cả dòng A được giữ (`qc_ok = rows`), không bị lọc bởi CJK/ratio vì đây là dữ liệu người/BT đã gate.
- Chunk 127 parent làm tăng nhẹ số dòng; token tăng nhẹ do cách pack chunk.
- A sẵn sàng cho quota sampling; không còn exact duplicate.

## 3. B sau processing

| group | rows | token % | median token | empty | CJK mean |
|---|---:|---:|---:|---:|---:|
| 1688_title | 16,133 | 4.8% | 46.5 | 0 | - |
| 1688_description | 16,819 | 52.6% | 472.5 | 0 | - |
| tiki_title | 13,057 | 3.1% | 34.6 | 0 | - |
| tiki_description | 11,397 | 39.5% | 535.7 | 0 | - |

Split: `{'train': 49166, 'dev': 4169, 'test': 4071}`. Eval B: **900** dòng.

![B group](figures/processed_ab/B_group_share.png)

![B length](figures/processed_ab/B_length.png)

![B ratio](figures/processed_ab/B_ratio.png)

![B CJK](figures/processed_ab/B_cjk.png)

![B split](figures/processed_ab/B_split.png)

### Nhận xét B

- Description vẫn ~92% token; title ~8%.
- `tiki_description` còn 11.397 dòng sau khi bỏ 1.532 target rỗng và QC raw 4B.
- 626 parent description >1.280 token được chunk; đây là lý do rows units > rows full.
- Boilerplate Tiki đã được strip khỏi source/target trước khi tính lại token.
- B vẫn description-heavy; `balanced`/`title_heavy` sẽ giảm desc mạnh, `desc_heavy` gần natural.

## 4. Dùng cho survey

```text
A survey: data/cleaned/processed_A/units.parquet
B survey: data/cleaned/processed_B/units.parquet
eval A/B giữ nguyên
quota sampling: tools/quota_sampling.py
budget 4M, 500 step
```

## 5. Kiểm chứng quota sampling giữ phân bố độ dài

Mô phỏng quota 4M token, seed 42, cấu hình `natural`:

| tập | nhóm | dòng quần thể | dòng được lấy | median gốc | median mẫu | p10 mẫu | p90 mẫu | dòng khác nhau |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1688_attribute | 73.815 | 60.961 | 9,9 | 9,9 | 3,5 | 96,9 | 41.441 |
| A | 1688_title | 16.133 | 13.329 | 46,5 | 46,5 | 40,3 | 53,2 | 9.099 |
| A | tiki_attribute | 15.727 | 12.777 | 9,0 | 8,9 | 3,8 | 55,9 | 8.760 |
| A | tiki_title | 13.057 | 10.856 | 34,6 | 34,6 | 18,8 | 58,1 | 7.401 |
| B | 1688_description | 16.819 | 4.252 | 472,5 | 480,0 | 241,9 | 766,5 | 3.772 |
| B | 1688_title | 16.133 | 4.105 | 46,5 | 46,6 | 40,5 | 53,2 | 3.651 |
| B | tiki_description | 11.397 | 2.911 | 535,7 | 533,1 | 194,5 | 960,0 | 2.580 |
| B | tiki_title | 13.057 | 3.318 | 34,6 | 34,6 | 19,1 | 59,8 | 2.946 |

Nhận xét:

- Median của mẫu trùng median quần thể ở cả 8 nhóm → không còn hiện tượng "chỉ lấy câu dài".
- Đuôi dài vẫn còn (A attribute p90 ~97 token; B tiki description p90 = 960 token do trần cắt).
- Tỷ lệ dòng khác nhau / dòng được lấy của A attribute là 68%; sample có hoàn lại nhưng vẫn phủ phần lớn quần thể.
- Với B, mỗi nhóm chỉ dùng 20–25% số dòng, phù hợp ngân sách 4M token trên corpus 15,7M token.
