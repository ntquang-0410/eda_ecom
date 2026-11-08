# EDA — bộ A (title+attribute) và B (title+description)

> **Cập nhật:** A trong file này là bản trước dedup. Bản dedup + EDA lại A:
> `reports/EDA_A_DEDUP.md`. B không đổi.

**Ngày:** 2026-11-07  
**Nguồn:** dữ liệu trước EDA/processing; B dùng raw BT 4B cho Tiki description.

## 0. Tóm tắt nhanh

- A: **118,688 dòng**, **4.71M token**, eval **1,200** dòng.
- B: **59,209 dòng**, **14.60M token**, eval **900** dòng.
- B có **1,150 target rỗng** và **13,199** dòng còn boilerplate Tiki.
- A không có description; B không có attribute.

## 1. Bộ A — title + attribute

| group | rows | row % | tokens | token % | median tok | empty | boilerplate | traditional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1688_title | 16,133 | 13.6% | 752,144 | 16.0% | 46.5 | 0 | 0 | 5 |
| 1688_attribute | 73,576 | 62.0% | 3,064,320 | 65.1% | 9.9 | 0 | 0 | 2 |
| tiki_title | 13,057 | 11.0% | 487,867 | 10.4% | 34.6 | 0 | 0 | 1 |
| tiki_attribute | 15,922 | 13.4% | 402,847 | 8.6% | 8.8 | 0 | 0 | 0 |

Split: `{'train': 103807, 'dev': 7561, 'test': 7320}`; 0 product overlap train/test.

![A group share](figures/eda_ab/A_group_share.png)

![A length](figures/eda_ab/A_length.png)

![A ratio](figures/eda_ab/A_ratio.png)

![A CJK](figures/eda_ab/A_cjk.png)

![A split](figures/eda_ab/A_split.png)

### Nhận xét A

- `1688_attribute` chiếm ~65% token nhưng chỉ ~62% số dòng → attribute là nhóm lớn nhất.
- `1688_title` ~16% token; title bị underweight tự nhiên so với attribute.
- Tiki title + attr ~19% token; Tiki attr rất ngắn, dễ bị chìm nếu sampling theo dòng.
- Không có target rỗng; chất lượng A sạch hơn B.
- Vì vậy A nên khảo sát `title_heavy` và `attr_heavy`; `balanced` sẽ kéo Tiki lên mạnh.

## 2. Bộ B — title + description

| group | rows | row % | tokens | token % | median tok | empty | boilerplate | traditional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1688_title | 16,133 | 27.3% | 752,144 | 5.1% | 46.5 | 0 | 0 | 5 |
| 1688_description | 16,819 | 28.4% | 8,258,465 | 56.6% | 472.5 | 0 | 0 | 2 |
| tiki_title | 13,057 | 22.1% | 487,867 | 3.3% | 34.6 | 0 | 0 | 1 |
| tiki_description | 13,200 | 22.3% | 5,099,780 | 34.9% | 340.6 | 1,150 | 13,199 | 12 |

Split: `{'train': 50544, 'dev': 4408, 'test': 4257}`; 0 product overlap train/test.

![B group share](figures/eda_ab/B_group_share.png)

![B length](figures/eda_ab/B_length.png)

![B ratio](figures/eda_ab/B_ratio.png)

![B CJK](figures/eda_ab/B_cjk.png)

![B split](figures/eda_ab/B_split.png)

![B flags](figures/eda_ab/B_flags.png)

### Nhận xét B

- Description chiếm ~91,5% token; title chỉ ~8,5%.
- `tiki_description` ~35% token dù chỉ ~22% số dòng → Tiki desc rất dài.
- `1688_description` ~57% token.
- Có **1,150** target rỗng, chủ yếu ở Tiki description raw 4B.
- **13,199** dòng còn boilerplate Tiki; đây là lý do phải tách `B_raw` và `B_processed`.
- Vì natural đã description-heavy, `title_heavy`/`balanced` thực chất là **giảm description**; `desc_heavy` chỉ tăng nhẹ.
- Cần strip boilerplate, bỏ target rỗng, normalize Traditional → Simplified trước khi train final.

## 3. Eval

### A

| dataset | pair_type | rows |
|---|---|---:|
| 1688 | attribute | 300 |
| 1688 | title | 300 |
| tiki | attribute | 300 |
| tiki | title | 300 |

### B

| dataset | pair_type | rows |
|---|---|---:|
| 1688 | description | 150 |
| 1688 | title | 300 |
| tiki | description | 150 |
| tiki | title | 300 |

Tiki description không có reference người; chỉ dùng COMET-QE. 1688 dùng COMET-da.

## 4. Kết luận cho processing

### A

```text
bỏ target rỗng (không có)
dedup exact pair
cap eff_tokens <= 960
giữ split product-level
quantity normalization
```

### B

```text
tách B_raw (weight survey) và B_processed (train final)
strip boilerplate Tiki ở source/target
bỏ 1.150 target rỗng
normalize Traditional -> Simplified nếu có opencc
dedup exact pair
cap eff_tokens <= 960
chunk description >960 token
```

## 5. Đề xuất weight survey

```text
A: natural | balanced | title_heavy | attr_heavy | tiki_heavy
B: natural | balanced | title_heavy | desc_heavy | tiki_heavy
budget 4M token, 500 optimizer step, seed 42 trước
```

Lưu ý: A chỉ có 4,707M token; 4M là budget an toàn. B có 14,598M token.