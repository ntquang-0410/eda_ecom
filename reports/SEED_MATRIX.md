# Seed matrix — policy ≥5 seed/cell

Seed đích: cell chính [7, 42, 123, 2024, 2025]; control [7, 42, 123]. Nguồn: `.artifacts/canonical/*/seed*.parquet`.
Ước tính còn thiếu: **49 run ≈ 184 giờ GPU** ở `max_seq=960`; nếu chạy `--max-seq 1792` (khuyến nghị, xem `TRAIN_GUARD_960.md`): **≈ 212–231 giờ** (+15–25%), trong đó 37 run thuộc cell chính (chưa tính thời gian chờ/quota).

| Cell | Vai trò | Seed đã có | # | Seed thiếu | # | Giờ GPU 960 | Giờ GPU 1792 (+15–25%) | Ghi chú |
|---|---|---|---:|---|---:|---:|---:|---|
| EcomGPT-7B A-balanced | control | 42 | 1 | 7,123 | 2 | 4.4 | 5.1–5.5 |  |
| NLLB-600M A | control | 7,42 | 2 | 123 | 1 | 7.0 | 8.0–8.8 | 424 phút + infer |
| NLLB-600M B | control | 7,42 | 2 | 123 | 1 | 5.0 | 5.8–6.2 | 282 phút + infer |
| OpenNMT B-natural | control | 42 | 1 | 7,123 | 2 | 2.0 | 2.3–2.5 |  |
| Qwen3-4B A-balanced | control | 42 | 1 | 7,123 | 2 | 10.0 | 11.5–12.5 |  |
| Qwen3-4B B-natural | control | 42 | 1 | 7,123 | 2 | 9.0 | 10.3–11.2 |  |
| Qwen3-8B A-balanced | control | 42 | 1 | 7,123 | 2 | 14.0 | 16.1–17.5 | thiếu score từng dòng |
| EcomGPT-7B A-natural | main | 42 | 1 | 7,123,2024,2025 | 4 | 8.8 | 10.1–11.0 | recipe token encoded |
| EcomGPT-7B B-balanced | main | 42 | 1 | 7,123,2024,2025 | 4 | 8.8 | 10.1–11.0 |  |
| Moses A | main | 42 | 1 | 7,123,2024,2025 | 4 | 2.0 | 2.3–2.5 | CPU |
| Moses B | main | 42 | 1 | 7,123,2024,2025 | 4 | 2.0 | 2.3–2.5 | CPU |
| OpenNMT A | main | 42 | 1 | 7,123,2024,2025 | 4 | 4.0 | 4.6–5.0 |  |
| Qwen3-4B A-natural | main | 7,42,123 | 3 | 2024,2025 | 2 | 10.0 | 11.5–12.5 | 500 step + infer |
| Qwen3-4B B-balanced | main | 42,123 | 2 | 7,2024,2025 | 3 | 13.5 | 15.5–16.9 | 500 step + infer |
| Qwen3-8B A-natural | main | 42 | 1 | 7,123,2024,2025 | 4 | 28.0 | 32.2–35.0 | 500 step A100 + infer |
| Qwen3-8B B-balanced | main | 42 | 1 | 7,123,2024,2025 | 4 | 28.0 | 32.2–35.0 | 500 step A100 + infer |
| Qwen3-8B B-natural | main | 42 | 1 | 7,123,2024,2025 | 4 | 28.0 | 32.2–35.0 | artefact tái lập 24/11 |

## Thứ tự đề xuất

1. **4 cell Qwen 4B** (A-natural, B-balanced) lên 5 seed — đây là nơi có tuyên bố so sánh A/B. Cần thêm 5 run ≈ 24 h.
2. **Qwen 8B** thêm 4 seed mỗi cell ≈ 56 h — nặng nhất, chạy song song 2 account Kaggle sau khi quota reset.
3. **EcomGPT** thêm 4 seed mỗi cell ≈ 18 h (nhanh, nhưng phải sửa ngân sách token trước — xem MATRIX §11.3).
4. **NLLB** thêm 3 seed mỗi cell ≈ 36 h (control, có thể chạy sau).
5. **Moses/OpenNMT** thêm 4 seed ≈ 12 h (CPU/rẻ, chạy cuối).

## Lưu ý

- Chi phí trên là giờ GPU thuần; thực tế cần thêm ~30% cho preemption, tải model, inference.
- Cột 1792 = chi phí khi dùng `--max-seq 1792` để giữ 100% token (960 làm mất 3,2% A / 6,8% B); ước tính +15–25% thời gian/step do padding dài, số token/step không đổi.
- Kaggle: 2 account × 30 h GPU/tuần (reset thứ Bảy). Colab A100: phiên 12 h, hay bị thu hồi.
- Với 5 seed, báo cáo mean ± std qua seed; paired bootstrap CI chỉ phủ variance eval, phải ghi rõ khi so sánh giữa cell.
- Cell Qwen3-8B A-balanced hiện **không có score từng dòng** → phải chấm lại trước khi tính seed mean.
