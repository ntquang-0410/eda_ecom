# RTX Pro 6000 — BT riêng Qwen3-8B (chạy song song)

Kernel `ntquang0410/tiki-plan-b-bt-rtx-8b`, notebook `plan_b_bt_rtx_8b.ipynb`.

## Vì sao tách riêng

Kernel 4-model chạy tuần tự 0.6B → 1.7B → 4B → 8B. Đo thực tế cho thấy tốc độ
giảm mạnh về cuối (sort theo độ dài tăng dần):

```text
25% entry đầu  → 13,2% token
50% entry      → 30,9% token
75% entry      → 57,7% token
100% entry     → 100%  token   (tổng 1,38M output token)
```

Nên 8B (model nặng nhất, chạy cuối) là chỗ dễ bị time budget cắt nhất.
Kaggle cho **2 GPU batch session đồng thời**, nên 8B được đẩy sang session thứ hai.

## Cấu hình

Giống hệt kernel 4-model (cùng code, cùng fix cắt cụt), chỉ đổi:

```python
MODEL_KEYS_TO_RUN = ['qwen3-8b']
```

```text
max_new 1536 (retry 4096) | max_input 1280 | batch 8 | bf16 | 102 GiB
```

## Kết quả

```text
/kaggle/working/plan_b_outputs/qwen3-8b/predictions.parquet
```

Cùng `json_ok` / `n_new_tokens` / `attempt` như kernel chính. Lấy bản nào xong
trước cũng được — hai kernel độc lập, không phụ thuộc nhau.

## Lưu ý về trùng lặp

Kernel 4-model vẫn sẽ tới lượt 8B sau khi xong 4B. Nếu lúc đó 8B ở đây đã xong,
có thể huỷ kernel 4-model để tiết kiệm ~2,5h GPU (quota còn ~26h).
