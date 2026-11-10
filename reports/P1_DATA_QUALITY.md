# P1 — Chất lượng dữ liệu & dataset C

**Ngày:** 2026-11-10

---

## 1. Number mismatch

`f_number_mismatch` hiện có trong units BT description nhưng **không blocking**.
Để dùng cho train, tách 2 mức:

```text
soft  : missing_numbers + extra_numbers = 1..2
hard  : missing_numbers + extra_numbers >= 3
```

| model | units | soft | hard | hard % | token hard (M) | token total (M) |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | 7.868 | 871 | 2.505 | 31,8% | 1,53 | 3,81 |
| qwen3-1.7b | 9.125 | 1.137 | 2.918 | 32,0% | 1,86 | 5,13 |
| qwen3-4b | 11.404 | 1.270 | 2.533 | 22,2% | 1,68 | 6,19 |
| qwen3-8b | 10.342 | 1.095 | 2.415 | 23,4% | 1,52 | 5,37 |

**Chính sách đề xuất cho train final:**

- `hard` → weight `0.5` (không xoá, vì vẫn giữ được văn phong/ngữ cảnh).
- `soft` → weight `1.0`.
- Không dùng `number_mismatch` làm blocking cứng, tránh mất 22–32% description.

Nếu sau này cần chính xác số tuyệt đối, chạy thêm một vòng BT có prompt "copy all numbers exactly" cho riêng nhóm hard.

---

## 2. Empty / boilerplate

Sau retry, các dòng còn `empty` chủ yếu là target chỉ chứa boilerplate giá/tax
hoặc source rỗng/số:

| model | empty còn lại |
|---|---:|
| qwen3-0.6b | 798 |
| qwen3-1.7b | 626 |
| qwen3-4b | 393 |
| qwen3-8b | 619 |

Các dòng này đã nằm ngoài `units` (qc_status = needs_review). **Không retry thêm**;
nếu cần, chỉ giữ flag để audit.

---

## 3. Dataset C (production mix)

Gộp 6 nhóm:

```text
1688_title, 1688_attribute, 1688_description
tiki_title, tiki_attribute, tiki_description
```

| | |
|---|---|
| rows | 146.948 |
| token | 19,312M |
| split | train 128.153 / dev 9.512 / test 9.283 |
| eval C | 900 dòng, 150 mỗi stratum, lấy từ **dev** |

Token share tự nhiên:

| nhóm | share |
|---|---:|
| 1688_description | 42,8% |
| tiki_description | 32,2% |
| 1688_attribute | 16,6% |
| 1688_title | 3,9% |
| tiki_title | 2,5% |
| tiki_attribute | 2,1% |

### Config C

| config | ý tưởng |
|---|---|
| `natural` | giữ phân phối token tự nhiên |
| `balanced_B` | attribute giữ natural; title/description chia đều 4 nhóm |
| `title_boost` | attribute natural; title 40 / desc 10 mỗi nguồn |
| `attr_boost` | attribute 50%; title/desc chia theo natural |

`eval_C` lấy từ dev nên không trùng sản phẩm với train và không phải eval đã dùng
để chọn seed 42/123 → holdout sạch hơn.

---

## 4. Trạng thái chạy

| job | trạng thái |
|---|---|
| `tiki-bt-fix-8b-full` | đang chạy 1.300 description |
| `tiki-weight-c1` (natural, balanced_B) | đang chạy |
| `tiki-weight-c2` (title_boost, attr_boost) | chờ slot GPU |

Sau khi xong: chấm COMET trên `eval_A`, `eval_B`, `eval_C` và chốt recipe production.
