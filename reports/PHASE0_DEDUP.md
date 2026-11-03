# Pha 0 (sau dedup + chunk)

Áp dụng: dedup `(source,target)`, giữ `dup_count`; 358 description dài được cắt
theo câu/đoạn, không overlap, mỗi chunk ≤800 token (max thực tế 1.279,5).

## 1. Token mass

Tổng **13.23M token**

| nguồn | token | share | rows |
|---|---:|---:|---:|
| desc_1688 | 8.36M | 63.2% | 16,819 |
| attr_1688 | 3.22M | 24.3% | 73,576 |
| tiki | 0.90M | 6.8% | 29,093 |
| title_1688 | 0.75M | 5.7% | 16,133 |

## 2. Sau cap 15

Tổng **13.23M token**

| nguồn | token | share | rows |
|---|---:|---:|---:|
| desc_1688 | 8.36M | 63.2% | 16,819 |
| attr_1688 | 3.21M | 24.3% | 73,382 |
| tiki | 0.90M | 6.8% | 29,093 |
| title_1688 | 0.75M | 5.7% | 16,133 |

## 3. Weight cho cấu hình balanced

| nguồn | share mục tiêu | weight |
|---|---:|---:|
| title_1688 | 15% | 1.297 |
| attr_1688 | 35% | 0.707 |
| desc_1688 | 25% | 0.194 |
| tiki | 25% | 1.802 |

## 4. Description

- p50 **472** | p95 **936** | p99 **1191** | max **1280** token
- vượt 1.280 token: **0** (đã xử lý)
- vượt 512 token: **7,214**

## 5. Kết luận

```text
desc chiếm 63,2% token (dưới ngưỡng 70%)
weight balanced: title 1,30 | attr 0,71 | desc 0,19 | tiki 1,80
desc weight ~0,19 — nếu cấu hình nào đẩy desc >70%, hạ về 0,18
quantity weight vẫn tạm thời, chờ khảo sát
```