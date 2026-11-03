# Pha 0 — Token share

Đo trên pair table 1688 + Tiki 8B, token ước lượng VI/3,6 + ZH/1,6 ký tự.

## 1. Tự nhiên (chưa cap, chưa weight)

Tổng: **15.23M token**

| nguồn | token | share | rows |
|---|---:|---:|---:|
| desc_1688 | 8.37M | 55.0% | 16,133 |
| attr_1688 | 5.05M | 33.1% | 395,493 |
| tiki | 1.06M | 6.9% | 45,652 |
| title_1688 | 0.75M | 4.9% | 16,133 |

## 2. Sau cap 15 attribute/sản phẩm

Tổng: **13.48M token**

| nguồn | token | share | rows |
|---|---:|---:|---:|
| desc_1688 | 8.37M | 62.1% | 16,133 |
| attr_1688 | 3.30M | 24.4% | 238,310 |
| tiki | 1.06M | 7.9% | 45,652 |
| title_1688 | 0.75M | 5.6% | 16,133 |

## 3. Sau base weight + cap 15

Tổng: **10.44M token**

| nguồn | token | share | rows |
|---|---:|---:|---:|
| desc_1688 | 6.70M | 64.2% | 16,133 |
| attr_1688 | 1.98M | 18.9% | 238,310 |
| tiki | 1.04M | 9.9% | 45,652 |
| title_1688 | 0.72M | 6.9% | 16,133 |

## 4. Weight cần để đạt share mục tiêu

| cấu hình | title | attr | desc | tiki |
|---|---:|---:|---:|---:|
| balanced | 1.40 | 0.74 | 0.21 | 1.65 |
| no_desc | 1.77 | 0.72 | 0.00 | 1.50 |
| tiki_heavy | 0.81 | 0.46 | 0.14 | 2.58 |
| title_heavy | 2.15 | 0.49 | 0.10 | 1.27 |

## 5. Description và max_seq

- p50 **472** token | p95 **1006** | p99 **1605** | max **6640**
- vượt 1.280 token: **358** dòng (2,2%)
- vượt 512 token: **6,881** dòng

Đề xuất: chunk description ở ~800 token cho các dòng vượt 1.280, hoặc truncate 358 dòng.

## 6. Kết luận

```text
description chiếm 55% token tự nhiên, 62% sau cap 15, 64% sau base weight
title 1688 chỉ 4,9–6,9% — thấp hơn nhiều so với mục tiêu 15%
Tiki chỉ 6,9–9,9% — muốn 25% phải tăng weight ~1,65x
cap 15 giảm attribute từ 5,05M xuống 3,30M token (-35%)
tổng token/epoch: 13,48M (cap 15)
```

## 7. Cấu hình đề xuất mặc định

```yaml
share: title 15% | attr 35% | desc 25% | tiki 25%
weight: title 1.40 | attr 0.74 | desc 0.21 | tiki 1.65
cap: 15 attribute/sản phẩm
desc: chunk nếu >1280 token
```