# Training-side guard — số dòng bị loại bởi max_seq=960 (ngưỡng prompt 952)

Guard: `encode_row()` trả `None` khi `prompt_tokens >= max_seq - 8` (Qwen chat template).
Ảnh hưởng thật đo trên **tập mẫu của từng config/seed** (draws có hoàn lại).

## Dataset A

- Units: **118,732** dòng; bị loại: **172** (**0.14%** dòng, **3.22%** token); prompt max 1439.

### A · natural

| seed | draws | draws bị loại | % draws | % eff_tokens mất | unique rows bị loại |
|---:|---:|---:|---:|---:|---:|
| 42 | 97,923 | 148 | 0.15% | 3.36% | 96 |
| 123 | 98,320 | 148 | 0.15% | 3.34% | 108 |
| 7 | 99,283 | 120 | 0.12% | 2.76% | 95 |

### A · balanced

| seed | draws | draws bị loại | % draws | % eff_tokens mất | unique rows bị loại |
|---:|---:|---:|---:|---:|---:|
| 42 | 110,723 | 58 | 0.05% | 1.29% | 50 |
| 123 | 109,750 | 58 | 0.05% | 1.30% | 48 |
| 7 | 111,148 | 57 | 0.05% | 1.29% | 52 |

## Dataset B

- Units: **57,406** dòng; bị loại: **1,134** (**1.98%** dòng, **6.81%** token); prompt max 1625.

### B · natural

| seed | draws | draws bị loại | % draws | % eff_tokens mất | unique rows bị loại |
|---:|---:|---:|---:|---:|---:|
| 42 | 14,586 | 265 | 1.82% | 6.23% | 232 |
| 123 | 14,643 | 285 | 1.95% | 6.74% | 245 |
| 7 | 14,600 | 300 | 2.05% | 7.08% | 264 |

### B · balanced

| seed | draws | draws bị loại | % draws | % eff_tokens mất | unique rows bị loại |
|---:|---:|---:|---:|---:|---:|
| 42 | 52,085 | 131 | 0.25% | 3.08% | 123 |
| 123 | 52,139 | 146 | 0.28% | 3.47% | 138 |
| 7 | 52,240 | 161 | 0.31% | 3.83% | 150 |


## Kết luận

| Tập | % dòng bị bỏ | **% eff_tokens bị bỏ** | Exposure thực tế (budget 4 M) |
|---|---:|---:|---:|
| A | 0,14% | **3,22%** | ~3,87 M |
| B | 1,98% | **6,81%** | ~3,73 M |

- Theo **token**, cả A và B đều vượt ngưỡng 1% (A 3,2%, B 6,8%); theo số dòng chỉ B vượt (1,98%).
- Các dòng bị bỏ là **các dòng dài nhất** (prompt max 1.439 token với A, 1.625 với B) — đúng nhóm
  description dài, nên thiên lệch có hệ thống chứ không ngẫu nhiên.
- Guard loại **âm thầm**: `step_tokens` vẫn tính các dòng bị bỏ, nên loss của step bị chia nhỏ hơn
  (gradient bị down-weight) thay vì renormalize — thêm một nguồn lệch nhỏ giữa các step.

## Khuyến nghị

1. **Các run 5-seed mới dùng `--max-seq 1792`** (và `--eval-max-seq 1792`): prompt max hiện tại là 1.625
   < 1.784 ⇒ **0% dòng bị loại**.
2. Giữ các run 960 hiện có làm **legacy/ablation**; khi so sánh phải ghi rõ chênh exposure 3–7%.
3. Chi phí: số token/step không đổi (8.000), nhưng padding dài hơn ⇒ ước tính **+15–25% thời gian/step**.
4. EcomGPT dùng cùng kiểu guard (`maxlen=960`) — cần đo lại bằng tokenizer BLOOM trước khi chốt
   ngân sách; nếu tương tự thì áp dụng `maxlen=1792` cho campaign sau.
