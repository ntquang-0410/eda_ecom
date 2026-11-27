# Audit empty hypothesis — Qwen3-4B A-natural trên 858 (đã xác minh + sửa + chạy lại)

**Ngày:** 2026-11-27  
**Trạng thái:** ✅ đã xác minh nguyên nhân, ✅ đã sửa harness, ✅ đã sinh lại 9 dòng × 3 seed.

---

## 1. Xác minh tiền đề "27/27 rỗng"

**Sai.** Chỉ **9/27 rỗng**, và đúng bằng **9/9 dòng description**; 18/18 attribute có hyp.

| Nhóm | Số dòng | Rỗng | Prompt tokens |
|---|---:|---:|---|
| Attribute | 18 | **0** | 57–532 |
| Description | 9 | **9** | 971–1.196 |
| Tổng | 27 | 9 | — |

Kiểm tra trên cả 3 seed (42, 123, 7):

| Seed | Rỗng/27 | Trùng tập bị guard loại | Rỗng ngoài guard |
|---|---:|---:|---:|
| 42 | 9 | 9 | 0 |
| 123 | 9 | 9 | 0 |
| 7 | 9 | 9 | 0 |

⇒ Không phải lỗi decoding: các dòng này **không bao giờ được đưa vào `generate`**.

## 2. Nguyên nhân — bug config của harness (không phải model)

`generate_rows()` dùng **cùng `max_seq = 960` cho train và eval**, rồi lọc:

```python
part = part[part["prompt_tokens"] < max_seq - 8]   # ngưỡng 952
```

9 dòng description có prompt 971–1.196 token (chat template Qwen + source 3,0–3,7k ký tự) → bị **bỏ âm thầm**, không log, không đưa vào generate. Đây là **config bug**: eval không được lấy giới hạn của train để loại dòng test; đúng ra phải truncate hoặc nâng `max_seq` cho eval.

Ba giả thuyết được loại trừ:

| Giả thuyết | Kết luận |
|---|---|
| `max-seq-len` | ✅ đúng là nguyên nhân (guard 952) |
| decoding config (max_new, greedy, EOS) | ❌ không liên quan — dòng bị loại trước generate; 0 rỗng ngoài guard |
| prompt template | ❌ template giống hệt 618 dòng còn lại; chỉ độ dài khác |

## 3. Sửa harness

`tools/tpu_weight_train.py` — `generate_rows(..., truncate=False)`:
- `truncate=True` (dùng cho eval): **không lọc**, mọi dòng đều được sinh; bucket mở rộng tới `max_seq` để không cắt prompt.
- `truncate=False` (train/TPU cũ): giữ hành vi cũ nhưng **log rõ số dòng bị bỏ + eval_id + max token**.

`tools/train_weight_quota_gpu.py`:
- thêm `--eval-max-seq` (0 = dùng `--max-seq`);
- eval gọi `generate_rows(..., eval_max, truncate=True)`.

Khuyến nghị cho run mới: `--eval-max-seq 1280` (bao trùm prompt dài nhất 1.196 token).

## 4. Chạy lại qwen4b_a

Thiết bị: Colab T4. Adapter lấy lại từ output kernel cũ (seed 42, 123) và `qwen4b-a-s7-ckpt` (seed 7). Sinh đúng 9 dòng bị thiếu với `eval_max_seq=1280`, rồi chấm lại COMET-QE.

| Seed | Trước: QE 858 (zero-fill) | Sau: QE 858 | Δ | Coverage trước → sau |
|---:|---:|---:|---:|---|
| 42 | 0,0530 | **0,0563** | +0,0033 | 98,6% → **100%** |
| 123 | 0,0587 | **0,0621** | +0,0034 | 98,6% → **100%** |
| 7 | 0,0573 | **0,0608** | +0,0035 | 98,6% → **100%** |
| **Trung bình** | 0,0563 | **0,0597 ± 0,0030** | +0,0034 | 98,6% → **100%** |

- 9/9 dòng sinh được hyp (0 rỗng) ở cả 3 seed.
- **COMET-da 1688 không đổi**: 0,8584 / 0,8511 / 0,8530 (231 dòng vốn đã đủ, không nằm trong guard).

## 5. Kết luận về ảnh hưởng điểm 858

1. **Metric xếp hạng (COMET-da 1688) KHÔNG bị ảnh hưởng** — các dòng bị lỗi đều là Tiki.
2. **COMET-QE Tiki bị lệch −0,0034** do zero-fill 9 dòng; đã sửa bằng cách sinh lại.
3. **Coverage Tiki 98,6% → 100%** sau khi sửa.
4. Đây là bug config thật, nên **qwen4b_a đã được chạy lại trước khi dùng so sánh**; artefact canonical đã cập nhật (`.artifacts/canonical/qwen4b_a/seed{42,123,7}.parquet`), log sinh lại ở `.artifacts/reinfer_long/`.

## 6. Phạm vi còn lại

Cùng bug này ảnh hưởng **mọi cell Qwen** (9 dòng description dài bị bỏ):

| Cell | Rỗng/27 | Adapter sẵn có? | Ghi chú |
|---|---:|---|---|
| qwen4b_a | 9 → **0** | ✅ | đã sửa hôm nay |
| qwen4b_a_balanced | 9 | cần lấy từ kernel output | sửa được trên T4 |
| qwen4b_b (42, 123) | 9 | kernel output | sửa được trên T4 |
| qwen4b_b_natural | 9 | kernel output | sửa được trên T4 |
| qwen8b_a / qwen8b_a_balanced / qwen8b_b_natural | 9 | qwen8b*-ckpt | cần ≥24 GB (L4/A100), T4 16 GB không đủ |

Cho đến khi sửa hết, bảng QE phải ghi chú: *"các cell Qwen còn lại bị zero-fill 9 dòng description dài, QE thấp hơn ~0,003 so với khi sửa"*. COMET-da không bị ảnh hưởng nên so sánh chính thức vẫn hợp lệ.
