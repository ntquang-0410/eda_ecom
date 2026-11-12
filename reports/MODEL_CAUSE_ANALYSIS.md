# Phân tích nguyên nhân — Matrix 5 model × 2 recipe

**Ngày:** 2026-11-12  
**Eval:** `eval_F` 1.500 dòng (300 title + 300 attribute + 150 description cho mỗi nguồn 1688/Tiki).  
**Metric:** COMET-da `Unbabel/wmt22-comet-da` (1688, có reference), COMET-QE `Unbabel/wmt20-comet-qe-da` (Tiki), BLEU/chrF (1688) là metric phụ.

---

## 0. Tóm tắt

1. **Qwen3-8B và Qwen3-4B gần như đồng hạng.** Chênh lệch tổng 1688: 4B A −0,0008 (CI chứa 0), 4B B −0,0072 (CI không chứa 0 nhưng rất nhỏ). Ở **description**, 4B không thua 8B: A +0,0156 có ý nghĩa, B +0,0006. 8B chỉ nhỉnh ở **title/attribute ngắn**.
2. **A/B không phải "tốt hơn/kém hơn" mà là coverage.** A = title + attribute; B = title + description. Vì vậy B tăng description nhưng giảm attribute ở **mọi** model.
3. **B là recipe cực đoan về mặt học:** chỉ 0,253 epoch, description chỉ chiếm 3–4 mẫu/step, 4,9%/13,7% target description bị cắt ở 960 token. Điều này giải thích vì sao B giúp Qwen/Moses nhưng **phá EcomGPT**.
4. **OpenNMT thất bại vì năng lực + mode collapse**, không phải chỉ vì thiếu dữ liệu: 4 layer/256 hidden, 32k vocab, target char-level, ppl 823/561, 52–60% câu có 4-gram lặp, 97–100% mất số.
5. **BLEU và COMET bất đồng rõ ở Qwen3-4B A:** BLEU 41,7 vs 55,2 của 8B nhưng COMET bằng nhau. 4B sinh description đủ trường hơn; 8B đôi khi co về `其他`/chuỗi rỗng. Với mục tiêu dịch thương mại, COMET phản ánh đúng hơn.

---

## 1. Thiết kế dữ liệu và các yếu tố gây nhiễu

| | A — natural | B — balanced |
|---|---|---|
| rows | 111.129 | 54.006 |
| token (eff) | 4,518M | 15,825M |
| epoch ở 4M token | 0,885 | **0,253** |
| nhóm train | 1688_title, 1688_attribute, tiki_title, tiki_attribute | 1688_title, 1688_description, tiki_title, tiki_description |
| stratum không có trong train | **description** | **attribute** |

### 1.1. Ngân sách token ≠ ngân sách ví dụ

Bảng dưới tính từ sample 4M token thực tế (cùng seed 42, cùng quota):

| recipe | nhóm | train rows | train tokens (M) | sampled rows | unique rows | rows/step | target bị cắt >960 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 1688_attribute | 70.114 | 3,03 | 62.491 | 41.252 | 125,0 | 0,2 |
| A | 1688_title | 15.350 | 0,72 | 13.582 | 9.038 | 27,2 | 0,0 |
| A | tiki_attribute | 14.012 | 0,34 | 12.460 | 8.224 | 24,9 | 0,0 |
| A | tiki_title | 11.653 | 0,43 | 10.293 | 6.839 | 20,6 | 0,0 |
| **A** | **TOTAL** | **111.129** | **4,52** | **98.826** | **65.353** | **197,7** | **0,1** |
| B | 1688_description | 15.986 | 7,84 | 2.025 | 1.894 | **4,0** | **4,9** |
| B | 1688_title | 15.350 | 0,72 | 21.449 | 11.558 | 42,9 | 0,0 |
| B | tiki_description | 11.017 | 6,83 | 1.602 | 1.488 | **3,2** | **13,7** |
| B | tiki_title | 11.653 | 0,43 | 27.055 | 10.513 | 54,1 | 0,0 |
| **B** | **TOTAL** | **54.006** | **15,83** | **52.131** | **25.453** | **104,3** | **0,6** |

Hệ quả:

- A cho model **65.353 dòng unique**, gần 198 dòng/step, chủ yếu attribute ngắn.
- B chỉ cho **25.453 dòng unique**, 104 dòng/step; description chỉ **3–4 dòng/step**. Title bị lặp 2,4–2,6 lần.
- B có **1.894 + 1.488 = 3.382** mẫu description unique; A có **0**.
- Target description dài bị cắt ở 960 token: 4,9% (1688) và 13,7% (Tiki). Đây là cắt **target**, không phải cắt prompt.

Đây là lý do gốc khiến B không thể đánh giá công bằng như A: cùng 4M token nhưng B vừa ít epoch, vừa ít ví dụ, vừa bị cắt target.

---

## 2. Kết quả tổng hợp

| model | data | COMET 1688 | COMET-QE Tiki | BLEU | chrF | rỗng % | rep4 | mất số % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | **0,8560** | 0,0635 | 55,19 | 54,02 | 1,0 | 0,046 | 16,9 |
| Qwen3-8B | B | 0,8552 | **0,0687** | **70,95** | **69,29** | 1,0 | 0,046 | 16,1 |
| Qwen3-4B | A | 0,8552 | 0,0647 | 41,66 | 41,68 | 1,0 | 0,042 | 17,0 |
| Qwen3-4B | B | 0,8480 | 0,0570 | 68,77 | 67,36 | 1,0 | 0,044 | 16,3 |
| EcomGPT-7B | A | 0,8214 | 0,0102 | 26,37 | 27,43 | 0,0 | 0,062 | 26,0 |
| EcomGPT-7B | B | 0,7661 | -0,1455 | 14,39 | 15,20 | 1,4 | 0,060 | **41,4** |
| Moses | A | 0,7947 | -0,2401 | 33,96 | 36,03 | 0,0 | 0,036 | 12,8 |
| Moses | B | 0,8103 | -0,1466 | 67,82 | 65,86 | 0,0 | 0,037 | 12,6 |
| OpenNMT | A | 0,3581 | -1,3508 | 0,45 | 1,79 | 0,0 | **0,519** | **96,9** |
| OpenNMT | B | 0,3424 | -1,3909 | 2,05 | 3,08 | 0,0 | **0,389** | **100,0** |

### 2.1. Theo stratum

| model | data | 1688 Title | 1688 Attr | 1688 Desc | Tiki Title | Tiki Attr | Tiki Desc |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | A | 0,835 | **0,877** | 0,855 | -0,021 | 0,076 | 0,220 |
| Qwen3-8B | B | **0,836** | 0,849 | 0,908 | -0,037 | **0,107** | 0,215 |
| Qwen3-4B | A | 0,831 | 0,872 | 0,870 | -0,025 | 0,078 | **0,230** |
| Qwen3-4B | B | 0,827 | 0,839 | **0,909** | -0,048 | 0,075 | **0,245** |
| EcomGPT-7B | A | 0,808 | 0,852 | 0,788 | -0,057 | 0,104 | -0,042 |
| EcomGPT-7B | B | 0,817 | 0,820 | 0,557 | -0,038 | -0,136 | -0,377 |
| Moses | A | 0,797 | 0,843 | 0,694 | -0,208 | 0,006 | -0,797 |
| Moses | B | 0,790 | 0,817 | 0,836 | -0,180 | 0,020 | -0,415 |
| OpenNMT | A | 0,269 | 0,463 | 0,328 | -1,545 | -0,892 | -1,880 |
| OpenNMT | B | 0,283 | 0,386 | 0,374 | -1,452 | -1,237 | -1,577 |

**Đọc nhanh:** title dễ với mọi model neural (0,79–0,84); attribute dễ với LLM (0,84–0,88); description là nơi phân hóa mạnh nhất. Tiki QE chỉ dương ở Qwen; ngay cả Qwen cũng âm ở title.

---

## 3. Nguyên nhân theo từng model

### 3.1. Qwen3-8B — mạnh nhất, ổn định nhất

- **Điểm:** A 0,8560/0,0635; B 0,8552/0,0687.
- **A/B:** tổng hòa (B−A −0,0008 trên 1688; +0,0052 trên Tiki, cả hai CI chứa 0), nhưng đánh đổi rõ:
  - description 1688: **+0,053** (CI [+0,041; +0,067]);
  - attribute 1688: **−0,028** (CI [−0,036; −0,021]);
  - Tiki attribute: +0,031.
- **Vì sao mạnh:** pretraining đa ngữ + long-context đủ tốt để **zero-shot description** dù A không có description. B chỉ cần 4M token (0,253 epoch) đã đạt description 0,908 — mức cao nhất toàn matrix.
- **Vì sao không hơn 4B nhiều:** với dữ liệu 4M token và bài toán ngắn, capacity 8B đã vượt ngưỡng cần thiết. 8B chỉ còn lợi thế ở title/attribute ngắn (precision từng từ) và zero-shot desc.

### 3.2. Qwen3-4B — đủ tốt, là điểm cân bằng compute/quality

- **Điểm:** A 0,8552/0,0647; B 0,8480/0,0570.
- **So với 8B (paired bootstrap):**
  - A: −0,0008 [−0,0055; +0,0041] → **hòa**;
  - B: −0,0072 [−0,0114; −0,0031] → thấp hơn **rất nhẹ**;
  - Tiki: A +0,0012, B −0,0117, đều không có ý nghĩa.
- **Theo stratum:**
  - 4B **thắng** description A: +0,0156 [0,0053; 0,0290];
  - 4B **hòa** description B: +0,0006 [−0,0036; 0,0065];
  - 4B **thua** title B (−0,0087) và attribute B (−0,0095), Tiki attribute B (−0,0315).
- **Vì sao description không thua:** khi được train trên B, 4B đạt desc 0,909 — bằng 8B. Nghĩa là **4B đủ capacity để học description**; khác biệt 8B nằm ở zero-shot (A desc BLEU 57,6 vs 40,0) và độ chính xác từ ngắn.
- **Vì sao BLEU lệch mà COMET không lệch:** 4B A sinh nhiều trường hơn, đôi khi 8B co về `其他` hoặc dừng sớm; BLEU phạt các trường thừa/khác trật tự, COMET-da thưởng nội dung đúng nghĩa. Ví dụ: cùng một description, 8B cho `其他` (COMET 0,34) còn 4B điền đủ `//; 食品类型：袋泡茶; 内容物(规格)：160; ...` (COMET 0,88). Ngược lại 4B có lỗi lặp (`雾面雾面…`) ở một số câu, kéo BLEU xuống.
- **Kết luận:** 4B là lựa chọn **Pareto tốt** nếu ưu tiên chi phí; chỉ kém 8B ~0,007 COMET ở B, trong khi desc tương đương.

### 3.3. EcomGPT-7B — chuyên ngành nhưng không đa ngữ đủ, B phá model

- **Điểm:** A 0,8214/0,0102; B 0,7661/−0,1455.
- **A mạnh attribute** (0,852) và title (0,808) — nhờ pretraining thương mại điện tử; nhưng description zero-shot chỉ 0,788.
- **B sụp description:** 0,557 (so với A 0,788 dù A không hề học description). Đây là **nghịch lý**: học description lại làm description tệ hơn.
- **Bằng chứng nguyên nhân:**
  - mất số tăng từ 26,0% lên **41,4%**; digit recall desc giảm 0,80 → 0,36;
  - 21/300 (7%) Tiki attribute rỗng, dù attribute không có trong train B;
  - B desc median 320 ký tự, len/ref 0,72, 44% câu <0,6 độ dài reference;
  - chỉ ~1,5–1,9k mẫu description unique, 3–4 mẫu/step, 4,9–13,7% target bị cắt.
- **Cơ chế:** EcomGPT là model chuyên ngành, không phải model dịch đa ngữ mạnh. Khi B dồn token vào description dài với target bị cắt, model học **template/field ngắn** thay vì dịch đủ câu. Các lỗi điển hình: nguồn là mô tả chăn ga nhưng hyp là `牛肉`/`袋装`; nguồn là thông số giày nhưng hyp `iPad 5/6 (9.7')`.
- **Kết luận:** EcomGPT chỉ dùng được cho title/attribute; không dùng cho description, đặc biệt với B.

### 3.4. Moses — phrase-based, mạnh bề mặt, yếu ngữ nghĩa toàn cục

- **Điểm:** A 0,7947/−0,2401; B 0,8103/−0,1466.
- **B tốt hơn A** trên 1688 (+0,0156) và Tiki QE (+0,0935), chủ yếu nhờ description: 0,694 → 0,836 (1688), −0,797 → −0,415 (Tiki).
- **BLEU B = 67,82**, gần Qwen3-4B B (68,77) và chỉ kém 8B B (70,95) — nhưng COMET-QE vẫn âm.
- **Vì sao:** SMT khớp cụm từ cục bộ rất tốt với description dài (nhiều cụm lặp, spec template), giữ số tốt (digit recall 0,95), ít lặp 4-gram (0,037). Nhưng:
  - không có ngữ cảnh toàn câu → sai trật tự/giới từ, QE phạt;
  - 24,0% (A) và 14,1% (B) câu còn ký tự Việt có dấu (tên riêng/thương hiệu không dịch);
  - 12,6–12,8% mất số;
  - eflomal thay GIZA++ là hạn chế pipeline.
- **Kết luận:** Moses B là baseline mạnh về BLEU cho description, nhưng không đạt chất lượng ngữ nghĩa/fluency; chỉ phù hợp làm mốc so sánh.

### 3.5. OpenNMT — thất bại hệ thống, không phải lỗi một lần chạy

- **Điểm:** A 0,3581/−1,3508; B 0,3424/−1,3909; BLEU 0,45/2,05.
- **Bằng chứng mode collapse:**
  - rep4 > 0,3 ở **60,2%** (A) và **53,6%** (B) câu;
  - 96,9%/100% câu mất toàn bộ số;
  - chỉ 71,5%/60,1% hypothesis unique; top-1 chiếm 6,0%/4,0%;
  - A sinh `包包包…`/`件套件套…`; B sinh template `材质 : 否 ; 品牌 : 否 ; 货号 : 否 …`.
- **Nguyên nhân:**
  1. **Under-capacity:** 4 layer, 256 hidden, 32k vocab cho 4–16M token; ppl cuối 823 (A) / 561 (B), acc 16%/21%.
  2. **Data quá ít cho from-scratch:** 4,5M token A và 15,8M token B; trong khi LLM đã có pretraining.
  3. **Exposure bias/mode collapse:** decode beam 5 với `replace_unk`, model học xác suất cao cho vài token phổ biến (`件套`, `包`, `材质:否`), rồi lặp.
  4. **Target char-level + cắt 512 token** ở eval làm giảm tín hiệu với description dài.
  5. A 30k step (~48 epoch) vẫn collapse → không phải chỉ thiếu step; kiến trúc/dữ liệu không đủ.
- **Kết luận:** OpenNMT from-scratch không cạnh tranh được trên bài toán này; không nên dùng làm baseline chính, chỉ giữ để minh họa “NMT truyền thống vs LLM”.

---

## 4. Vì sao recipe B thắng ở description nhưng thua ở attribute

### 4.1. Description

- B dạy trực tiếp description, nên:
  - Qwen8: +0,053 (1688), Tiki desc gần như hòa (+0,215 vs +0,220);
  - Qwen4: +0,038 (1688), Tiki +0,0156;
  - Moses: +0,142 (1688), +0,382 (Tiki);
  - EcomGPT: **−0,231** — ngoại lệ vì truncation + ít mẫu + model yếu.
- Với Qwen/Moses, B là recipe đúng cho description. Với EcomGPT, B là recipe sai vì model không đủ năng lực học từ target dài bị cắt.

### 4.2. Attribute

- B **không có attribute** trong train, nên mọi model đều giảm attribute:
  - Qwen8 −0,028; Qwen4 −0,033; EcomGPT −0,032; Moses −0,026; OpenNMT −0,077.
- Riêng Tiki attribute, Qwen8 vẫn tăng +0,031 (do transfer từ title/desc), nhưng Qwen4 giảm nhẹ và EcomGPT B rỗng 7%.
- Vì attribute ngắn và giống title, model vẫn làm được ở mức khá; đây là lý do B không sụp toàn tập.

### 4.3. Title

- Gần như không đổi giữa A/B ở mọi model (Δ ≤ 0,01), vì cả hai recipe đều có title.

---

## 5. BLEU/COMET và lựa chọn metric

- **BLEU thiên vị trùng n-gram:** Moses B đạt 67,82 BLEU nhưng COMET-QE −0,147; Qwen4 A chỉ 41,66 BLEU nhưng COMET 0,8552.
- **COMET-da thiên về nghĩa:** 4B A điền đủ trường, 8B đôi khi co về `其他`; COMET cho 4B A +0,0156 so với 8B A.
- **COMET-QE Tiki** là metric duy nhất đánh giá không reference; nó phạt mạnh Moses/OpenNMT vì fluency/word order, và chỉ Qwen dương.
- **Khuyến nghị:** dùng COMET-da + COMET-QE làm metric chính; BLEU/chrF chỉ để chẩn đoán bề mặt và so sánh với literature.

---

## 6. Khuyến nghị

1. **Baseline chính:** Qwen3-8B B cho description (0,908; QE 0,215), Qwen3-8B A cho attribute (0,877). Nếu chỉ chọn một cell: Qwen3-8B A hoặc B đều 0,855–0,856; tùy stratum ưu tiên.
2. **Nếu ưu tiên chi phí:** Qwen3-4B B (0,8480; QE 0,057) chỉ kém 8B B 0,0072 COMET, trong khi desc tương đương; là lựa chọn Pareto tốt.
3. **Không dùng** EcomGPT B cho description, Moses cho câu dài cần ngữ nghĩa, OpenNMT cho bất kỳ stratum nào.
4. **Recipe tiếp theo:** train trên **A ∪ B** hoặc route theo `pair_type` (A cho attribute, B cho description) để tránh đánh đổi; nếu giữ B, tăng budget lên 12–16M token để đạt ~1 epoch và giảm truncation.
5. **Nếu cần NMT from-scratch:** tăng model (6 layer/512 hidden), vocab 16k, target word-level/BPE, và ít nhất 50–100M token; nhưng ROI thấp so với LLM.

---

## 7. File dữ liệu

- `reports/MATRIX_5X2_RESULTS.md` — bảng điểm + paired bootstrap.
- `reports/MODEL_DIAGNOSTICS.md` — bảng chẩn đoán đầy đủ + ví dụ lỗi.
- `reports/DATA_COVERAGE.csv` — số mẫu/step, unique rows, truncation.
- `reports/FINETUNE_CONFIGS_5MODELS.md` — cấu hình 5 model.
