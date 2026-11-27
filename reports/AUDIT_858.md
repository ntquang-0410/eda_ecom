> **Cập nhật 27/11:** 858 đã trở thành index chính thức; 831 chỉ còn để tham chiếu lịch sử.

# Audit 858 dòng — zero-fill 27 dòng Tiki bị loại khỏi 831

Index: `data/cleaned/eval_paired_858.parquet` (1688 = 231, Tiki = 627). 27 dòng bổ sung đều là Tiki (18 attribute + 9 description), chấm COMET-QE; 1688 giữ nguyên nên **COMET-da và xếp hạng chính thức không đổi**.

## 1. Theo cell (trung bình qua seed)

| Cell | #seed | QE Tiki 831 | QE Tiki 858 | Δ | Cov 831 | Cov 858 | Hyp rỗng thêm |
|---|---:|---:|---:|---:|---:|---:|---:|
| nllb_a | 2 | -0.5403 | -0.5502 | -0.0099 | 100.0% | 100.0% | 0 |
| ecomgpt_a | 1 | 0.0077 | -0.0018 | -0.0095 | 100.0% | 100.0% | 0 |
| nllb_b | 2 | -0.5956 | -0.6036 | -0.0080 | 99.8% | 99.8% | 0 |
| moses_b | 1 | -0.1593 | -0.1663 | -0.0070 | 100.0% | 100.0% | 0 |
| ecomgpt_a_balanced | 1 | 0.0020 | -0.0041 | -0.0061 | 100.0% | 100.0% | 0 |
| qwen4b_b | 2 | 0.0489 | 0.0438 | -0.0051 | 100.0% | 98.6% | 18 |
| qwen8b_b_natural | 1 | 0.0624 | 0.0576 | -0.0048 | 100.0% | 98.6% | 9 |
| qwen4b_a | 3 | 0.0608 | 0.0563 | -0.0045 | 100.0% | 98.6% | 27 |
| qwen4b_a_balanced | 1 | 0.0524 | 0.0483 | -0.0041 | 100.0% | 98.6% | 9 |
| qwen4b_b_natural | 1 | 0.0471 | 0.0432 | -0.0040 | 100.0% | 98.6% | 9 |
| qwen8b_b | 1 | 0.0509 | 0.0470 | -0.0040 | 100.0% | 98.6% | 9 |
| moses_a | 1 | -0.2439 | -0.2475 | -0.0036 | 100.0% | 100.0% | 0 |
| qwen8b_a | 1 | 0.0521 | 0.0487 | -0.0034 | 100.0% | 98.6% | 9 |
| ecomgpt_b | 1 | -0.1366 | -0.1387 | -0.0021 | 100.0% | 97.1% | 18 |
| onmt_b | 1 | -1.4074 | -1.4059 | +0.0015 | 100.0% | 100.0% | 0 |
| onmt_a | 1 | -1.4177 | -1.4138 | +0.0039 | 100.0% | 100.0% | 0 |
| qwen8b_a_balanced | 1 | nan | nan | +nan | 100.0% | 98.6% | 9 |

## 2. Theo run

| Cell | Seed | QE 831 | QE 858 | Δ | Cov 831 | Cov 858 |
|---|---:|---:|---:|---:|---:|---:|
| ecomgpt_a | 42 | 0.0077 | -0.0018 | -0.0095 | 100.0% | 100.0% |
| ecomgpt_a_balanced | 42 | 0.0020 | -0.0041 | -0.0061 | 100.0% | 100.0% |
| ecomgpt_b | 42 | -0.1366 | -0.1387 | -0.0021 | 100.0% | 97.1% |
| moses_a | 42 | -0.2439 | -0.2475 | -0.0036 | 100.0% | 100.0% |
| moses_b | 42 | -0.1593 | -0.1663 | -0.0070 | 100.0% | 100.0% |
| nllb_a | 7 | -0.5427 | -0.5524 | -0.0097 | 100.0% | 100.0% |
| nllb_a | 42 | -0.5379 | -0.5481 | -0.0102 | 100.0% | 100.0% |
| nllb_b | 7 | -0.5192 | -0.5299 | -0.0107 | 100.0% | 100.0% |
| nllb_b | 42 | -0.6719 | -0.6773 | -0.0054 | 99.5% | 99.5% |
| onmt_a | 42 | -1.4177 | -1.4138 | +0.0039 | 100.0% | 100.0% |
| onmt_b | 42 | -1.4074 | -1.4059 | +0.0015 | 100.0% | 100.0% |
| qwen4b_a | 7 | 0.0614 | 0.0573 | -0.0042 | 100.0% | 98.6% |
| qwen4b_a | 42 | 0.0573 | 0.0530 | -0.0043 | 100.0% | 98.6% |
| qwen4b_a | 123 | 0.0637 | 0.0587 | -0.0050 | 100.0% | 98.6% |
| qwen4b_a_balanced | 42 | 0.0524 | 0.0483 | -0.0041 | 100.0% | 98.6% |
| qwen4b_b | 42 | 0.0461 | 0.0416 | -0.0045 | 100.0% | 98.6% |
| qwen4b_b | 123 | 0.0517 | 0.0459 | -0.0057 | 100.0% | 98.6% |
| qwen4b_b_natural | 42 | 0.0471 | 0.0432 | -0.0040 | 100.0% | 98.6% |
| qwen8b_a | 42 | 0.0521 | 0.0487 | -0.0034 | 100.0% | 98.6% |
| qwen8b_a_balanced | 42 | nan | nan | +nan | 100.0% | 98.6% |
| qwen8b_b | 42 | 0.0509 | 0.0470 | -0.0040 | 100.0% | 98.6% |
| qwen8b_b_natural | 42 | 0.0624 | 0.0576 | -0.0048 | 100.0% | 98.6% |

## 3. Kết luận

- 27 dòng bổ sung làm QE Tiki thay đổi trong khoảng -0.0099 … +0.0039; mức lớn nhất ở **nllb_a** (-0.0099).
- Các cell Qwen có 9/27 hyp rỗng ở nhóm này ⇒ coverage Tiki giảm còn ~98,6%; EcomGPT B-balanced rỗng 18/27 ⇒ ~97,1%. Đây chính là phần bị 831 che mất.
- Vì 1688 không đổi, kết luận xếp hạng theo COMET-da giữ nguyên; nhưng coverage Tiki phải báo cáo trên 858 kể từ nay.
- Qwen3-8B A-balanced bị loại khỏi bảng QE 858 vì artefact 831 không có score từng dòng (chỉ 27 dòng mới được chấm) — cần chấm lại toàn bộ 600 dòng Tiki trước khi so sánh.
- Khuyến nghị: chuyển index chính thức sang 858 (thuần nội dung) nếu chấp nhận đổi số QE; nếu giữ 831 để liên tục, luôn kèm phụ lục 858 này.

---

## 4. Bổ sung 27/11 — Qwen3-8B A-balanced đã chấm đủ 858

Trước đó artefact 831 không có score từng dòng nên bảng §1 để `nan`. Đã chấm lại toàn bộ trên Colab T4
(COMET-da 231 dòng + COMET-QE 618 dòng không rỗng):

| Cell | 1688 da | Tiki QE (858, zero-fill) | Coverage Tiki |
|---|---:|---:|---:|
| Qwen3-8B A-balanced seed 42 | **0,8553** | **0,0518** | 98,6% |

DA trùng khớp điểm tổng cũ 0,8553 ⇒ hyp + adapter hợp lệ. Không còn cell nào thiếu score trong canonical 858.
