# Các tuyên bố phải sửa trước khi bảo vệ

**Ngày:** 2026-11-16  
**Mục đích:** danh sách câu trong báo cáo cũ vượt quá dữ liệu, kèm cách viết lại an toàn.

| # | Câu cũ | Vấn đề | Cách viết mới |
|---:|---|---|---|
| 1 | "Leakage: `eval_F ∩ units_A = 0`, `eval_F ∩ units_B = 0`" | Chỉ đúng theo `product_id`; 642/1.500 dòng trùng source/pair/near-dup | "`product_id` disjoint = 0, nhưng content/near-dup leak 42,8%; `eval_F_v2` 858 dòng sạch dùng cho số liệu chính" |
| 2 | "OpenNMT from-scratch thất bại, NMT không cạnh tranh được" | Chỉ đúng cho cấu hình 37M/4 layer/256 hidden; B cũ là B-natural; pretraining/khác compute | "Cấu hình from-scratch 37M này sụp mode collapse; không kết luận cho paradigm NMT. Cần NLLB control + recipe 6/512" |
| 3 | "Quota sampling đã khắc phục length bias" | Chỉ sửa trong nội bộ nhóm; exposure lệch 18×, 87% description không học, target cắt | "Đã sửa bias chọn dòng trong nhóm; chưa sửa end-to-end; xem per-group exposure" |
| 4 | "B-balanced giúp description" | Đúng với Qwen/Moses; EcomGPT B tệ hơn; B chỉ 0,253 epoch | "B giúp description ở Qwen/Moses; EcomGPT bị truncation/ít mẫu nên tệ hơn; không phải quy luật chung" |
| 5 | "Qwen3-4B bằng Qwen3-8B" | Trên v1 A hòa, B 8B hơn 0,007; v2 B 8B hơn 0,010; desc mixed | "4B tương đương ở A; 8B nhỉnh B ~0,010; desc 4B đủ dùng nhưng không bằng tuyệt đối" |
| 6 | "Qwen hơn EcomGPT 0,0346, CI [0,027;0,042]" | CI chỉ resample eval, 1 seed; leak; v2 0,032 | "Trên eval_F_v2, Qwen hơn EcomGPT A ~0,032; CI chưa gồm seed variance" |
| 7 | "Compute-matched 4M token" | NMT thấy 47–61× token-pass; LLM pretrained | "Compute-matched chỉ trong nội bộ LLM; NMT là baseline from-scratch 48/10 epoch" |
| 8 | "OpenNMT-B = B-balanced" | Sai; kernel đọc full units_B train = B-natural | "OpenNMT-B = B-natural; đã relabel, loại khỏi bảng B−A" |

## Quy tắc trích dẫn

- **Số liệu chính:** `MATRIX_5X2_EVAL_F_V2.md` (common 831), không dùng bảng 1.500 gốc.
- **Chẩn đoán lỗi bề mặt:** `MODEL_DIAGNOSTICS.md` (v1) chỉ để minh họa rep4/rỗng/mất số.
- **OpenNMT:** chỉ trích `OPENNMT_FAIRNESS.md`; không dùng câu "NMT thua LLM".
- **Quota:** chỉ trích `QUOTA_LENGTH_AUDIT.md`; không nói "đã khắc phục hoàn toàn".
- **Seed:** mọi so sánh model là 1 seed; ghi rõ "chưa có seed variance".
