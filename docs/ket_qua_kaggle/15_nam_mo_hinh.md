# So sánh năm mô hình (eval fold 0)

Qwen3-4B, Qwen3-8B, EcomGPT-7B, OpenNMT, Moses trên cùng eval fold 0.

|  |  |
|---|---|
| Notebook | nlp-group3 / nlp-8b / nlp-ecom |
| Tài khoản | trinhnhatanh + honghuynanh |
| Bản Kaggle | không lưu trong file kết quả |
| Thời gian | 10–13/11/2026 |
| Dùng ở đâu | so sánh 5 mô hình trên eval fold 0 |
| Ghi chú | Cấu hình sàng lọc rank 16, 4M token; không phải công thức cuối. |

## 1. Bảng tổng hợp (từ `tong_hop_5mh*`)

### Pair

| Mô hình | Loại | Chính | Macro | Lệch giữa lần |
|---|---|---|---|---|
| Qwen3-4B | LLM | 79.98 | 68.52 | 1.63 |
| Qwen3-8B | LLM | 77.08 | 66.61 | 0.26 |
| EcomGPT-7B | LLM | 68.15 | 56.78 | 3.03 |
| OpenNMT (tu dau, train day du) | co_dien | 74.38 | 54.89 | — |
| Moses (train day du) | co_dien | 80.78 | 68.08 | — |
| Moses (khop 4M token) | co_dien | 79.30 | 65.36 | — |
| Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | LLM_cong_thuc_cuoi | 83.28 | 74.99 | — |

### Cụm

| Mô hình | Loại | Chính | Macro | Lệch giữa lần |
|---|---|---|---|---|
| Qwen3-4B | LLM | 86.81 | 71.87 | 0.20 |
| Qwen3-8B | LLM | 87.86 | 73.92 | 0.03 |
| EcomGPT-7B | LLM | 78.83 | 64.44 | 0.09 |
| OpenNMT (tu dau, train day du) | co_dien | 80.61 | 58.72 | — |
| Moses (train day du) | co_dien | 85.12 | 69.77 | — |
| Moses (khop 4M token) | co_dien | 82.33 | 66.48 | — |
| Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16) | LLM_cong_thuc_cuoi | 90.38 | 76.72 | — |

## 2. Ghi chú

- Ba LLM dùng cấu hình sàng lọc rank 16, 4M token, 2 seed; không phải công thức cuối.
- Dòng `Qwen3-4B công thức cuối` đo trên T4 fp16, lời dẫn khác nên không xếp hạng chung.
- Số chi tiết (BLEU/SacreBLEU/TER) nằm trong `bang_metric_bai_bao.md` của từng thư mục.

## Nguồn

| File (trong `data/processed/...`) | Byte | sha256 (16) |
|---|---|---|
| tong_hop_5mh/ket_qua.json | 28,385 | 23f608da4e4aff14 |
| tong_hop_5mh_cum/ket_qua.json | 28,581 | 5cf76498389ffd88 |
| tong_hop_5mh/bang_metric_bai_bao.md | 5,229 | 6f76855f9ad8a0bc |
| tong_hop_5mh_cum/bang_metric_bai_bao.md | 5,230 | 7bd00ec7f874fd7e |


---

*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*
