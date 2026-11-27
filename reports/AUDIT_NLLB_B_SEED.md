# Audit NLLB-600M B — seed 42 vs seed 7

Câu hỏi: std qua seed = 0,0628 (seed 42 = 0,5494; seed 7 = 0,6382) — đây là seed noise thật, hay một run bị lỗi?

## 1. Chẩn đoán

| Seed | COMET-da | COMET-QE | Cov 1688 | Cov Tiki | len med | rep4 med | BLEU sent | chrF sent |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.5494 | -0.6773 | 99.6% | 99.5% | 28 | 0.026 | 12.84 | 20.97 |
| 7 | 0.6382 | -0.5299 | 100.0% | 100.0% | 25 | 0.026 | 17.98 | 24.90 |

| Seed | 1688 Title | 1688 Attribute | 1688 Description |
|---:|---:|---:|---:|
| 42 | 0.5013 | 0.6458 | 0.4397 |
| 7 | 0.6207 | 0.7199 | 0.4988 |

Paired cluster bootstrap (seed 42 − seed 7, 2.000 lần, cluster = product_id): **-0.0888** [-0.1076; -0.0698], n = 231.

## 2. Đọc kết quả

- Độ dài hyp, tỉ lệ lặp 4-gram và coverage của hai seed gần như nhau ⇒ **không phải run hỏng** (không rỗng, không lặp vô hạn, không cắt cụt).
- Chênh lệch trải đều cả 3 stratum (title −0,119; attribute −0,074; description −0,059) ⇒ không phải lỗi cục bộ một nhóm dữ liệu.
- Với NLLB-600M LoRA, 4 M token trên tập B (~1,5 M token) là ~2,7 epoch; mỗi seed lấy một tập con khác nhau ⇒ phương sai run thật sự lớn. Đây là **seed/run variance**, không phải eval noise: CI của paired bootstrap chỉ phủ variance eval và vẫn cho −0,089.
- Không lấy được log train seed 42 (kernel nằm ở account khác, không còn token), nên không thể xác nhận thêm bằng learning curve; nhưng artefact hyp đầy đủ và hợp lệ.

## 3. Hệ quả báo cáo

- Với n = 2 seed, std 0,0628 **không đủ tin cậy**; phải báo cả hai điểm riêng (0,5494 và 0,6382) thay vì chỉ mean ± std.
- NLLB B là control duy nhất có seed thứ 2 cho đến nay; để kết luận A/B của NLLB cần ≥5 seed.
- Không loại seed 42: nó hợp lệ. Nếu buộc chọn một điểm để so sánh với Qwen, dùng seed 7 (coverage 100%, chạy sau, có log) và ghi rõ.
