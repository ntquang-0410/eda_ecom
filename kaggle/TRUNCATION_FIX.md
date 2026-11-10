# Lỗi cắt cụt output BT — nguyên nhân và cách sửa

Ghi lại để không tái phạm, áp dụng cho cả nhánh GPU và TPU.

## Triệu chứng

`predictions.parquet` có một phần dòng mà `target_text_bt` là tiếng Trung/Việt
**cụt giữa câu**, không phải bản dịch hoàn chỉnh.

## Tái hiện (local, Tesla T4, Qwen3-0.6B, đúng prompt của pipeline)

Lấy 3 entry dài nhất của tập MT rồi generate với `max_new_tokens=96` (cấu hình cũ):

| `expected_zh_chars` | `max_new` | token sinh ra | JSON đóng | kết quả |
|---|---|---|---|---|
| 1781 | 96 | 96 | ✗ | cụt |
| 1406 | 96 | 96 | ✗ | cụt |
| 1399 | 96 | 96 | ✗ | cụt |

Với `max_new=1024`:

| `expected_zh_chars` | token sinh ra | JSON đóng | kết quả |
|---|---|---|---|
| 1781 | 1024 | ✗ | model lặp lại payload dạng Python repr → hỏng |
| 1406 | 733 | ✓ | xong, parse OK |
| 1399 | 1024 | ✗ | vẫn cần > 1024 token |

## Nguyên nhân (3 lớp)

1. **Hằng số `max_new_tokens=96` không dựa trên phân bố thật.**
   Đo trên 29.446 entry:

   ```text
   expected_zh_value_chars: p50 30 | p99 230 | p99.5 336 | p99.9 684 | max 1781
   ```

   96 token chỉ phủ tới ~74 ký tự Hán → **976 entry (3,31%)** bị cắt cụt.

2. **`max_input_tokens=768` cắt nguồn.**
   Prompt token đo thật: p99 314, max 1074. Có 339 entry dài > 384 ký tự,
   một phần vượt 768 token → nguồn bị cắt, bản dịch sai lệch dù không "cụt".

3. **Không có kiểm tra hoàn chỉnh.** Sau khi generate, code chỉ `json.loads`
   rồi fallback về raw text. Không phân biệt "model trả lời sai" với
   "bị cắt vì hết token", nên lỗi diễn ra âm thầm.

## Cách sửa (notebook v2 / `PROMPT_VERSION=plan_b_rtx_v3`)

```python
MAX_NEW_TOKENS       = 1536   # phủ ~1200 ký tự Hán
MAX_NEW_TOKENS_RETRY = 4096   # repair pass cho entry cực dài
MAX_INPUT_TOKENS     = 1280   # phủ 100% prompt (max đo được 1074)
```

Thêm 3 thứ vào mỗi dòng output:

| cột | ý nghĩa |
|---|---|
| `json_ok` | JSON đóng + có `translation` khác rỗng |
| `n_new_tokens` | số token đã sinh (phát hiện chạm trần) |
| `input_truncated` | nguồn có bị cắt ở `MAX_INPUT_TOKENS` |
| `attempt` | 1 = main pass, 2/3 = repair |

**Repair pass**: sau khi chạy xong main loop, mọi dòng `json_ok=False` được sinh
lại với `batch=1`, `max_new=4096`, tối đa 2 lần. Nếu vẫn hỏng thì
`qc_status='needs_review'` để lọc ra sau, không lặng lẽ đẩy bản cụt vào dataset.

## Bài học cho nhánh TPU

`kaggle/tpu_bt` và `kaggle/tpu_bt_8b` đang dùng `max_new_tokens=96`/`128` với
`max_new` lấy từ config — **cùng lỗi này**. Khi chạy lại TPU phải đổi:

```python
MAX_NEW_TOKENS = 1536
MAX_INPUT_TOKENS = 1280
+ json_ok / repair
```

Ước lượng chi phí cho 976 entry repair: ~2 phút/model trên RTX Pro 6000.
