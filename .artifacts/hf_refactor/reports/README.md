# reports/ — mục lục báo cáo

## Weight A/B (kết quả chính)

| file | nội dung |
|---|---|
| `WEIGHT_SURVEY_A_B.md` | tổng hợp A/B, khuyến nghị, so sánh sampler cũ |
| `WEIGHT_SURVEY_A.md` | bảng COMET + credit chi tiết A |
| `WEIGHT_SURVEY_B.md` | bảng COMET + credit chi tiết B |
| `WEIGHT_SURVEY_TECHNIQUE_A_B.md` | thiết kế quota sampling + bằng chứng lệch của sampler cũ |
| `EDA_A_B.md` | EDA split A/B trước dedup |
| `EDA_A_DEDUP.md` | EDA A sau dedup |
| `EDA_PROCESSED_A_B.md` | EDA sau processing, kiểm chứng quota |
| `EDA_PROCESSING_PLAN_A_B.md` | kế hoạch processing ban đầu |

## Back-translation

| file | nội dung |
|---|---|
| `TIKI_DESC_BT_PROCESSING.md` | QC/processing description BT 0.6B/1.7B/4B/8B |

## Báo cáo cũ (sampler lỗi, chỉ tham khảo)

| file | ghi chú |
|---|---|
| `WEIGHT_EXPERIMENT_REPORT.md` | dùng `p_i ∝ w_g·t_i`, token share bị lệch |
| `WEIGHT_EVAL_FINAL.md` | kết luận cũ; xem cảnh báo đầu file |

## Figures

```text
figures/eda_ab/          EDA split A/B trước dedup
figures/eda_a_dedup/     EDA A sau dedup
figures/processed_ab/    EDA sau processing
figures/desc_bt/         QC/processing description BT
```
