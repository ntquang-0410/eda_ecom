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
| qwen3-8b | 13.418 | 1.447 | 3.831 | 28,6% | 2,50 | 7,49 |

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
| qwen3-8b | 621 |

Các dòng này đã nằm ngoài `units` (qc_status = needs_review). **Không retry thêm**;
nếu cần, chỉ giữ flag để audit.

---

## 3. Dataset C — đã bỏ

Theo yêu cầu, không chạy thực nghiệm C (6 nhóm). Đã huỷ kernel, xoá dataset/kết quả C.
Recipe production sẽ lấy trực tiếp từ kết luận A/B:

- attribute: giữ natural (A)
- title / description: tái cân bằng theo B (`balanced`)

## 4. Trạng thái chạy

| job | trạng thái |
|---|---|
| `tiki-bt-fix-8b-full` | xong: 13.198 dòng, 13.418 unit |
| C1/C2 | đã huỷ, xoá dataset C |

Bước tiếp theo: train final 4B với recipe A+B, đánh giá trên `eval_A` + `eval_B`.
