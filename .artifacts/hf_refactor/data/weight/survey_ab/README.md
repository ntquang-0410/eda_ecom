# data/weight/survey_ab/ — quota survey A/B

5 config × 2 tập × seed 42, Qwen3-4B + LoRA, quota sampling 4M token, 500 step.

## Thiết kế

| | A | B |
|---|---|---|
| Dữ liệu | `processed_A/units.parquet` | `processed_B/units.parquet` |
| Config | natural, balanced, title_heavy, attr_heavy, tiki_heavy | natural, balanced, title_heavy, desc_heavy, tiki_heavy |
| Eval | 1.200 dòng | 900 dòng |
| Hypothesis | 1.200/config | 885/900/config |

Quota sampler: mỗi nhóm nhận `T_g = s_g × 4M` token, sample đều có hoàn lại
trong nhóm đến đủ `T_g`. Token share thực tế khớp target đến 4 chữ số thập phân.

## Kết quả COMET

| tập | config tốt nhất | Δ vs natural |
|---|---|---|
| A | **natural** | mọi reweight giảm 1688 (0,5–1,6 điểm) |
| B | **balanced** | +0,0144 COMET-da [+0,0081; +0,0211] |

Credit score (gemini-3.8-flash, một phần do quota API) xác nhận A `natural`,
B `balanced`; loại `tiki_heavy`.

## File

```text
hyps/            hyp_A_*.parquet, hyp_B_*.parquet
scores_A/        scores_<config>_seed42_<1688|tiki>.parquet + comet_scores.json
scores_B/
credit/          credit_A/B_gemini-3.8-flash_partial.json
runs/A|B/        run_<config>_seed42.json + summary.json
summary_A.json   bảng COMET + bootstrap + credit
summary_B.json
```

Chi tiết: `reports/WEIGHT_SURVEY_A_B.md`, `reports/WEIGHT_SURVEY_A.md`, `reports/WEIGHT_SURVEY_B.md`.
