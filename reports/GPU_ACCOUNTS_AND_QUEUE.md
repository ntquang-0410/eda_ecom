# Tài khoản Kaggle, quota GPU và hàng đợi thực nghiệm

**Cập nhật:** 2026-11-16 22:55 UTC

## 1. Tài khoản và quota

| Account | Token | Đã dùng | Còn lại | Đang chạy |
|---|---|---:|---:|---|
| `nguynthanhquang` | `/root/.kaggle_01/access_token` | 28,18h | **1,82h** | `tiki-nllb-b-infer` (T4x2) |
| `nadsa091` | `/root/.kaggle_02/access_token` | 11,48h | **18,52h** | `abal` + `bnat` (2×L4x4) |
| `anhtamxinhdep` | `/root/.kaggle_03/access_token` | 7,68h | **22,32h** | `a-s123` + `b-s123` (2×L4x4) |

## 2. `machine_shape`

- `"NvidiaL4"` = **4× L4 24 GB** (đúng). `"NvidiaL4x4"` bị hạ xuống T4.
- L4x4 chiếm **1 slot** trong giới hạn 2 GPU session/account → **chạy được 2 kernel L4x4 song song**.
- Lỗi `Maximum batch GPU session count of 2 reached` trước đó là do phiên T4 cũ chưa nhả slot.

## 3. Định mức quota

| Máy | Quota / 1h wall |
|---|---:|
| T4x2 | ~1,0h |
| 1× L4x4 | ~3,0h |
| 2× L4x4 song song | ~6,0h |

## 4. Trạng thái kernel

| Kernel | Account | Máy | Bắt đầu | Ghi chú |
|---|---|---|---|---|
| `tiki-m4b-abal` | nadsa091 | L4x4 | 18:10 | A-balanced, seed 42 |
| `tiki-m4b-bnat` | nadsa091 | L4x4 | 22:44 | B-natural, seed 42 |
| `tiki-m4b-a-s123` | anhtamxinhdep | L4x4 | 19:11 | A-natural, seed 123 |
| `tiki-m4b-b-s123` | anhtamxinhdep | L4x4 | 22:44 | B-balanced, seed 123 |
| `tiki-nllb-b-infer` | nguynthanhquang | T4x2 | 22:51 | inference-only từ adapter NLLB-B |
| `tiki-nllb-control` | nguynthanhquang | T4x2 | 09:01 | **A xong** (482,5 phút, 1500 hyp, 2 rỗng); bị cắt ở inference B do chạm trần 12h |

## 5. Ngân sách và rủi ro

- `nadsa091`: 18,52h ≈ **3,1h wall** với 2×L4x4. Cần (abal còn ~1–3h) + (bnat ~3h) → sát trần.
- `anhtamxinhdep`: 22,32h ≈ **3,7h wall**. Cần (a-s123 còn ~2–3h) + (b-s123 ~3h) → sát trần.
- **Quy tắc bảo vệ:** nếu quota một account giảm xuống dưới thời gian còn lại của kernel chạy trước, huỷ kernel thứ hai để bảo toàn output kernel thứ nhất.

## 6. Sau khi xong

1. Pull hypothesis 4 cell Qwen + NLLB A/B.
2. Chấm COMET CPU (`/content/cometenv`) trên `eval_F_v2_common` (831 dòng).
3. Cập nhật bảng 2×2, seed variance, NLLB.
