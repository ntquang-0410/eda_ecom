# Weight survey A/B — quota sampler

**Ngày:** 2026-11-08  
**Model:** Qwen3-4B + LoRA (r=16, all-linear)  
**Dữ liệu:** processed units chia từ dữ liệu **trước EDA**  
**Ngân sách:** 4.000.000 token/config, 500 optimizer step, seed 42  
**Sampler:** quota-based (không dùng `p_i ∝ w_g·t_i`)

---

## 1. Thiết lập

| | A | B |
|---|---|---|
| Nhóm | 1688_title, 1688_attribute, tiki_title, tiki_attribute | 1688_title, 1688_description, tiki_title, tiki_description |
| Units | 118.732 dòng / 4,842M token | 57.406 dòng / 15,710M token |
| Eval | 1.200 dòng (300 mỗi stratum) | 900 dòng (300 title + 150 description mỗi nguồn) |
| Hypothesis | 1.200/1.200 | 885/900 |

B có 15 dòng không sinh được hypothesis (12 tiki description + 3 1688 description) do prompt vượt 952 token. 15 dòng này **giống hệt nhau ở cả 5 config**, nên paired comparison vẫn hợp lệ.

Điểm số:

- 1688: COMET-da (có reference) + BLEU/chrF.
- Tiki: COMET-QE (`wmt20-comet-qe-da`, không reference); điểm tuyệt đối thấp là bình thường, chỉ so sánh paired.
- Paired bootstrap 4.000 lần trên `eval_id` dùng chung.

Token share thực tế khớp target đến 4 chữ số thập phân ở cả 10 run (xem `data/weight_survey_ab/runs/`).

---

## 2. Tập A — title + attribute

| config | 1688 (da) | Tiki (qe) | 1688_title | 1688_attribute | tiki_title | tiki_attribute |
|---|---:|---:|---:|---:|---:|---:|
| **natural** | **0,856** | 0,025 | 0,830 | **0,883** | -0,032 | 0,081 |
| attr_heavy | 0,851 | **0,029** | 0,823 | 0,879 | -0,022 | 0,081 |
| title_heavy | 0,849 | 0,028 | **0,837** | 0,861 | -0,026 | 0,081 |
| balanced | 0,849 | 0,025 | 0,832 | 0,866 | -0,029 | 0,079 |
| tiki_heavy | 0,841 | 0,022 | 0,816 | 0,865 | -0,040 | 0,085 |

Paired bootstrap vs `natural`:

| config | Δ1688 | CI 95% | P(win) | ΔTiki | CI 95% | P(win) |
|---|---:|---|---:|---:|---|---:|
| attr_heavy | -0,0049 | [-0,0084; -0,0014] | 0,003 | +0,0048 | [-0,0082; +0,0173] | 0,772 |
| title_heavy | -0,0069 | [-0,0120; -0,0019] | 0,004 | +0,0033 | [-0,0089; +0,0159] | 0,696 |
| balanced | -0,0074 | [-0,0125; -0,0025] | 0,002 | +0,0005 | [-0,0142; +0,0146] | 0,529 |
| tiki_heavy | -0,0155 | [-0,0203; -0,0107] | 0,000 | -0,0025 | [-0,0195; +0,0134] | 0,377 |

**Kết luận A:** `natural` tốt nhất trên 1688. Mọi cách tăng weight đều làm 1688 giảm có ý nghĩa (0,5–1,6 điểm). Tiki QE không phân biệt được config nào (CI đều chứa 0).

Lý do: `1688_attribute` vừa là nhóm nhiều token nhất (66%), vừa là stratum điểm cao nhất (0,883). Giảm tỷ lệ attribute để tăng title (title_heavy) chỉ cải thiện title +0,007 nhưng mất attribute -0,022 → net âm. `tiki_heavy` tệ nhất vì lấy 70% token cho Tiki trong khi eval 1688 chiếm một nửa.

---

## 3. Tập B — title + description

| config | 1688 (da) | Tiki (qe) | 1688_title | 1688_description | tiki_title | tiki_description |
|---|---:|---:|---:|---:|---:|---:|
| **title_heavy** | **0,857** | 0,053 | **0,836** | 0,899 | -0,033 | 0,240 |
| balanced | 0,855 | **0,057** | 0,833 | 0,900 | -0,031 | **0,247** |
| tiki_heavy | 0,845 | 0,052 | 0,819 | 0,898 | -0,035 | 0,241 |
| natural | 0,841 | 0,056 | 0,806 | **0,911** | -0,032 | 0,250 |
| desc_heavy | 0,835 | 0,054 | 0,800 | 0,907 | -0,036 | 0,248 |

Paired bootstrap vs `natural`:

| config | Δ1688 | CI 95% | P(win) | ΔTiki | CI 95% | P(win) |
|---|---:|---|---:|---:|---|---:|
| title_heavy | +0,0160 | [+0,0099; +0,0229] | 1,000 | -0,0034 | [-0,0178; +0,0108] | 0,321 |
| balanced | +0,0144 | [+0,0081; +0,0211] | 1,000 | +0,0002 | [-0,0146; +0,0145] | 0,527 |
| tiki_heavy | +0,0047 | [-0,0007; +0,0103] | 0,956 | -0,0045 | [-0,0208; +0,0111] | 0,303 |
| desc_heavy | -0,0054 | [-0,0113; +0,0010] | 0,047 | -0,0026 | [-0,0121; +0,0071] | 0,303 |

**Kết luận B:** `title_heavy` và `balanced` đều tăng 1688 có ý nghĩa (+0,016 / +0,014; CI không chứa 0; P(win) = 1,000). Tiki không đổi.

Cơ chế: `natural` dồn 92% token cho description. Chuyển bớt sang title:

- 1688_title: 0,806 → 0,836 (**+0,030**)
- 1688_description: 0,911 → 0,899 (**-0,012**)

Eval B có 2/3 số dòng là title nên net dương. BLEU lại giảm (74,3 → 68,5) vì BLEU là corpus-level và bị chi phối bởi các dòng description dài; COMET tính trung bình theo câu nên phản ánh đúng hơn mức cải thiện title.

`balanced` và `title_heavy` tương đương trên COMET (chênh 0,0016; CI chứa 0). Tuy nhiên credit của `title_heavy` thấp hơn `natural` trên 1688 (-0,25, chưa significant) và thấp hơn `balanced` 0,37 điểm, nên **`balanced` là lựa chọn an toàn hơn**.

---

## 4. Credit score (metric phụ)

Gemma-4 (`gemma-4-31b-it`) hết quota ngày sau 360/1.000 dòng (A) và 420/1.000 dòng (B). Để có metric phụ nhất quán, toàn bộ phần còn lại được chấm bằng **gemini-3.8-flash**, cùng prompt và thang credit AAA..C (1..9). Do quota, flash chấm được **580/1.000** dòng A và **680/1.000** dòng B; các batch được xáo trộn nên mẫu vẫn rải đều trên config/stratum.

**A** (n = 580):

| config | mean | n |
|---|---:|---:|
| natural | **7,70** | 125 |
| attr_heavy | 7,47 | 122 |
| balanced | 7,34 | 125 |
| title_heavy | 7,11 | 107 |
| tiki_heavy | 7,07 | 101 |

Paired trên các `eval_id` được chấm ở cả hai config (1688): `attr_heavy` +0,15 (ns), `balanced` -0,58 (p = 0,018), `title_heavy` -0,71 (p = 0,033), `tiki_heavy` -0,83 (p = 0,003).

**B** (n = 680):

| config | mean | n |
|---|---:|---:|
| balanced | **6,68** | 146 |
| natural | 6,64 | 140 |
| desc_heavy | 6,59 | 138 |
| title_heavy | 6,31 | 135 |
| tiki_heavy | 6,15 | 121 |

Paired 1688 vs `natural`: `balanced` -0,16 (ns), `title_heavy` -0,25 (ns), `desc_heavy` -0,05 (ns), `tiki_heavy` -0,82 (p = 0,000). Tiki: `balanced` +0,32 (ns), `title_heavy` +0,07 (ns).

Nhận xét:

- Credit **xác nhận A**: `natural` cao nhất, mọi reweight thấp hơn; `tiki_heavy` kém cả 1688 lẫn Tiki.
- Credit **không tách được** `balanced`/`title_heavy` khỏi `natural` ở B, nhưng loại `tiki_heavy`.
- `title_heavy` có xu hướng credit âm trên 1688 (dù chưa significant) trong khi COMET lại cao nhất → đây là điểm mâu thuẫn metric cần thận trọng.

## 5. So với survey cũ (sampler lỗi)

Kết quả cũ nói `balanced`/`title_heavy` tốt nhất cho cả hai tập. Điều đó **không còn đúng** sau khi sửa sampler:

- Sampler cũ chọn dòng theo `p_i ∝ w_g·t_i`, token thực tế tỷ lệ với `Σt_i²`, nên câu dài áp đảo.
- A `natural` cũ thực chất cho 88,6% token vào attribute (target 65,1%); A `balanced` cũ cho 53,9% vào attribute (target 25%).
- Vì vậy "natural" cũ không phải natural, và các config bị so sánh trên phân phối khác hẳn thiết kế.

Với quota sampler, phân phối token đúng thiết kế, và kết luận đảo lại ở A: **không tăng weight**, giữ natural.

---

## 6. Khuyến nghị

| tập | chọn | lý do |
|---|---|---|
| A (title + attribute) | **natural** | mọi reweight đều giảm 1688; Tiki không đổi |
| B (title + description) | **balanced** | +0,014 COMET-da trên 1688 (có ý nghĩa), credit trung tính, ít cực đoan |

Tóm lại: chỉ nên tái cân bằng khi một nhóm đang chiếm quá nhiều token so với độ hữu ích của nó (B: description 92%). Với A, attribute nhiều token nhưng cũng là nhóm học tốt nhất, nên giữ nguyên.

---

## 7. Hạn chế

- 1 seed (42); các CI trên là bất định theo câu, chưa gồm bất định do seed.
- B thiếu 15 dòng description dài; ảnh hưởng như nhau ở mọi config.
- COMET-QE cho Tiki có điểm tuyệt đối rất thấp và không tách được config; chưa kết luận được weight nào tốt cho Tiki.
- Credit score mới chấm được một phần (A 580/1.000, B 680/1.000) do quota API; là metric phụ, COMET là metric chính.
- COMET-QE cho Tiki không tách được config; kết luận cho Tiki chỉ là "không thay đổi".

---

## 8. File

```text
data/weight_survey_ab/hyps/        hyp_A_*.parquet, hyp_B_*.parquet
data/weight_survey_ab/scores_A/    scores_*.parquet + comet_scores.json
data/weight_survey_ab/scores_B/
data/weight_survey_ab/runs/        run_*.json + summary.json
reports/WEIGHT_SURVEY_A.md         bảng chi tiết A
reports/WEIGHT_SURVEY_B.md         bảng chi tiết B
```

HF: https://huggingface.co/datasets/ntquang0410/tiki-4model-eval
