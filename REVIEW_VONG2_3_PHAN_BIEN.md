# BIÊN BẢN PHẢN BIỆN — VÒNG 2 · BA PHẢN BIỆN ĐẦU TIÊN

**Dự án:** NMT thương mại điện tử Trung–Việt (nhánh Quang: CADO/A–B–C + ma trận 5×2 + COMET; nhánh Nhật Anh: dữ liệu do LLM sinh + Judge DeepSeek + fine-tune Qwen3-4B + 5 mô hình + demo).
**Hội đồng:** R1 — Phương pháp luận; R2 — MLOps (độc lập).
**Ngày:** 17/11/2026.
**Bối cảnh vòng này:** nhóm đã upload bộ phản hồi cho vòng 1 tại `tiki-4model-eval/reviews/2026-11-16/` (`REVIEW_RESPONSE.md`, `LEAKAGE_AUDIT.md`, `OPENNMT_FAIRNESS.md`, `QUOTA_LENGTH_AUDIT.md`, `MATRIX_5X2_EVAL_F_V2.md`, `CLAIMS_REVISION.md`, `RUN_HASH_MANIFEST.*`). Vòng 2 không lặp lại vòng 1: hội đồng **kiểm tra chính các bản sửa đó**, và chỉ ra chỗ nào vẫn chưa đủ để bảo vệ trước hội đồng.

---

## 0. BẰNG CHỨNG ĐÃ TẢI VỀ `hf_dataset/` VÀ KIỂM CHỨNG ĐỘC LẬP

### 0.1. Tải dữ liệu từ Hugging Face (Python + `huggingface_hub`)

Script: `download_hf_evidence_v2.py` (hỗ trợ `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` → token cache `hf login` → anonymous). Manifest: `hf_dataset/DOWNLOAD_MANIFEST_V2.json`.

| Nguồn | Revision | Remote | Đã tải | Nội dung |
|---|---|---|---|---|
| `ntquang0410/zh-vie_ecom` | `1f950116f4351e43699102fd3fe55d545366515e` | 173 file / 781,0 MB | **131 file / 452,2 MB** | `*.md`, `*.json`, `*.txt`, parquet `data/LLM`, `data/cleaned`, `data/{1688,tiki}/cleaned` + 1 shard raw phục vụ truy nguồn gốc |
| `zh-vie_ecom` @ lịch sử | `126a564f88c72ac7feb2d4cc79e7183348f5aad9` | 187 file | **22 file / 39,8 MB** | Toàn bộ trace Judge DeepSeek đã bị gỡ khỏi bản hiện tại: `validation_report.json`, `quality_signals_summary.json`, `*_provenance_validation.json`, `pending_judge_items.parquet`, `title_review_signals.parquet`, `seven_items_assistant_review.*`… |
| `ntquang0410/tiki-4model-eval` | `a7df35dad97970ced84f2765c9c6b78f64d48ef0` | 374 file / 1.249 MB | **246 file / 17,5 MB** | docs, reports, configs, code, **`reviews/2026-11-16/`** (bản trả lời phản biện vòng 1), `data/weight` runs/scores/eval metadata, BT summaries; **không tải** corpora lớn (units/full/train/hyps) |

Tổng: **509,6 MB**. Ghi chú MLOps:
- **`HF_TOKEN` vẫn không tồn tại** trong môi trường (`env` trống, `hf auth whoami` → "Not logged in", không token cache). Cả hai dataset là public nên tải anonymous thành công; script ghi rõ `authenticated: false`. README_EVIDENCE.md:10 nói "đã thiết lập biến môi trường HF_TOKEN" — **sai so với thực tế**, cần sửa tài liệu (và nếu sau này dataset private thì quy trình tải sẽ gãy).
- README_EVIDENCE.md vẫn mô tả bằng chứng cũ (`session_export.json`, `github_code/`) — hai thứ này **không còn ở workspace**; bằng chứng thực tế nằm ở `ntquang_branch/`, `nhatanh_branch/`, `context/` và `claude_session/`.
- Dataset `zh-vie_ecom` **không có thư mục `eval/metrics`**. Toàn bộ kết quả/sửa lỗi của vòng 1 nằm trong `tiki-4model-eval/reviews/2026-11-16/`; nếu không tải `tiki-4model-eval`, reviewer sẽ không thấy được chúng.

### 0.2. Kiểm chứng độc lập (không chỉ đọc báo cáo)

| Kiểm tra | Kết quả |
|---|---|
| `eval_F` v1 vs `units_A/B` (local matrix data, chuẩn hoá NFKC+lower) | A: source trùng **76**, pair khít **14**, `product_id` **0**; B: source **11**, pair **0**, `product_id` **0** — khớp `LEAKAGE_AUDIT` |
| `eval_F_v2` vs `units_A/B` | A/B: `product_id` 0, source **0**, pair **0** — v2 sạch theo 3 khoá; `eval_F_v2_meta.json`: 858 dòng, leak union v1 = **642/1.500 (42,8%)**, A 584 / B 511, near-dup 44 |
| `eval_F_v2` / common | 858 / **831** dòng; theo stratum: 1688 = 90 title + 96 attribute + 45 description; Tiki = 291 + 224 + 112 |
| Ma trận v2 | Qwen3-4B A **0,8584**; Qwen3-8B B 0,8572; Qwen3-8B A 0,8567; Qwen3-4B B 0,8471; EcomGPT A 0,8250; Moses B 0,8037; Moses A 0,7862; EcomGPT B 0,7634; OpenNMT A 0,3488; OpenNMT B 0,3387 (COMET-da/QE, common 831) |
| Sensitivity v1→v2 | 18 cell: max |Δ| = **0,0178** (Qwen3-8B B Tiki); 1688 max |Δ| = 0,0092 |
| NLLB-200 control | **Không có kết quả**: chỉ có builder `ntquang_branch/kaggle/matrix_nllb/build.py`; mọi tài liệu vẫn ghi "đang chạy/PLANNED"; không file output/score nào trong `tiki-4model-eval` hay nhánh |
| Quota exposure | `review_audit.json` khớp `QUOTA_LENGTH_AUDIT`: B có epoch nhóm **0,128×–2,318×**, description 11,8–13,5% dòng unique, title lặp 1,4–2,3×, 87% description không được học; token Qwen thật **6,56M/5,93M**; tiki_desc target >960 không cap = **30,4%** |
| Run hash | chỉ **3 run JSON** matrix được backfill (theo `LEAKAGE_AUDIT` §5); các cell còn lại dựa vào manifest — tái lập chưa đầy đủ 10/10 |

---

## 1. PHẢN BIỆN 1 — “DATA LEAKAGE **FIXED**” CHƯA ĐÚNG: ĐÓ LÀ LỌC HẬU NGHIỆM, KHÔNG PHẢI TEST ĐỘC LẬP; VÀ NHÁNH LLM LẶP LẠI ĐÚNG LỖI ĐỊNH NGHĨA CŨ

**Mức độ: BLOCKER (giữ nguyên).** Nhóm ghi `FIXED` trong `REVIEW_RESPONSE.md` §0, trong khi chính `LEAKAGE_AUDIT.md` §5 ghi *“v2 dùng để kiểm tra độ bền, không thay thế một eval lớn hơn. Vẫn cần eval_F_v3 được thiết kế trước khi sinh hypothesis”*. **Đây là hai tuyên bố mâu thuẫn**: một bản sửa chưa hoàn tất không thể gắn nhãn `FIXED`.

### 1.1. Phần hội đồng chấp nhận

- Việc phát hiện và định lượng leak là đúng và cần thiết: `product_id` disjoint không đủ; v1 leak **642/1.500 (42,8%)**, chủ yếu qua nhóm near-dup (`nhom_trung` 1688, title cluster Tiki, MinHash LSH). A leak 584 (38,9%), B leak 511 (34,1%); 1688/title leak tới **210/300** mỗi bộ.
- `eval_F_v2` sạch theo 5 khoá (hội đồng kiểm lại: source/pair **0/0** với train+dev), re-score 10 cell, sensitivity đầy đủ, xếp hạng **họ mô hình** không đổi; chênh lớn nhất 0,018.
- Quy tắc trích số mới (`CLAIMS_REVISION.md`) và việc tự hạ mức tuyên bố là thái độ đúng.

### 1.2. Vì sao vẫn BLOCKER

**(a) `eval_F_v2` được tạo sau khi đã sinh hypothesis — không phải benchmark độc lập.** Việc lọc dựa trên dữ liệu (không dựa điểm) nên không phải chọn theo kết quả, nhưng nó **không thay thế được một split thiết kế từ đầu**: 42,8% dòng bị loại là các sản phẩm phổ biến/near-dup, tức **đúng phần đại diện nhất của phân phối 1688**; phần còn lại là sản phẩm hiếm/diễn đạt khác. Vì vậy `eval_F_v2` đo một phân phối khác, không đo hiệu năng trên traffic thật. Với 1688, chỉ còn **231 dòng (90 title/96 attr/45 desc)**, tức **27,8%** của eval; Tiki chiếm **72,2%**. Mọi trung bình “tổng” trên v2 bị dịch sang Tiki (COMET-QE, không reference) — trong khi 1688 mới là nơi có reference người. Kết luận “xếp hạng không đổi” đúng ở mức họ mô hình, nhưng **top cell đổi** (v1: Qwen3-8B A 0,8560; v2: Qwen3-4B A 0,8584), và **macro-average đảo thứ hạng** (Qwen3-8B B dẫn 1688 macro 0,8656; Qwen3-4B A dẫn Tiki macro 0,0974). Nói “không đổi” mà không nói rõ mức so sánh là thiếu.
**(b) Test-set reuse/selection bias chưa được xử lý.** `eval_F = eval_A + 150 description/nguồn`; `eval_A/eval_AB` đã được dùng để chọn recipe A-natural/B-balanced ở weight survey; `NEXT_STEPS.md` §2 tự ghi *“Selection trên eval đã dùng nhiều lần. eval_A/eval_B lấy từ test và đã dùng để chọn config. Cần holdout mới (từ dev)”*. Lọc nội dung **không xoá** được việc test đã được dùng để chọn cấu hình. `eval_F_v2` vẫn lấy từ cùng pool test đó → **không phải test “một lần”**; muốn có test độc lập phải tách từ dev **trước** khi chọn recipe, điều chưa làm.
**(c) Nhánh LLM lặp lại đúng lỗi định nghĩa cũ.** `A7` của nhánh Nhật Anh kết luận **“KHÔNG RÒ RỈ”** với 0 vi phạm ở ngưỡng Jaccard title 0,5 / mô tả 0,7; nhưng cùng tài liệu ghi: *“Tỉ lệ dòng mô tả của test đã có nguyên văn trong pool: **85,0%**”* (thuộc tính khuôn mẫu). Tức là: **product-level Jaccard không bắt được line-level template leak** — chính là lỗi “product_id-only” phiên bản mới. Họ có chấm thêm “dòng mới” để giảm nhẹ, nhưng bảng headline vẫn dùng chrF trên toàn bộ description → **điểm description bị thổi phồng** bởi 85% dòng đã thấy trong train; kết luận “description B tốt hơn”/“chrF mô tả 83–91” không đứng vững nếu không tách template vs new-line.
**(d) Vòng lặp tham chiếu cùng họ mô hình (self-preference / self-training).** (i) Trong nhánh Quang, target phía Tiki của cả A/B là **BT của Qwen3-4B** (`FINETUNE_CONFIGS_5MODELS.md`); Qwen3-4B/8B được đánh giá trên chính target cùng họ sinh ra — lợi thế hệ thống. (ii) Trong nhánh Nhật Anh, tham chiếu title/description là bản dịch của **Gemma-4/Qwen3-14B**; Qwen3-4B/8B được fine-tune trên dữ liệu một phần do Qwen3-14B sinh và chấm bằng tham chiếu do Gemma/Qwen sinh. Đây là **label circularity**, không phải “leak” theo nghĩa split, nhưng cùng bản chất: test không độc lập khỏi quá trình sinh nhãn. Tài liệu nhánh Nhật Anh tự nhận “Qwen3-14B (tạo dữ liệu) cùng họ với Qwen3-4B (được fine-tune)” nhưng chưa đưa vào phân tích.
**(e) Tái lập chưa đầy đủ.** Chỉ 3/10 run JSON matrix có hash; 7 cell còn lại dựa manifest; `verify_review.py` là script audit offline, **không phải cổng chặn trong kernel huấn luyện**. Một assertion fail cứng chưa được chứng minh là chạy trong mọi lần train.

### 1.3. Yêu cầu bắt buộc (để gỡ BLOCKER)

1. **`eval_F_v3` thiết kế trước inference**: giữ quy mô ~1.500, **chia theo nhóm near-dup ngay từ split** (không lọc hậu nghiệm), cân bằng 1688/Tiki như thiết kế gốc; freeze hash; mọi bảng số chính thức dùng v3 (hoặc v2 + tuyên bố rõ là sensitivity, không phải benchmark).
2. **Holdout chọn recipe từ dev** (hoặc từ 1 fold ngoài) trước khi chọn A/B/C, để eval cuối chưa từng dùng; nếu không, ghi rõ “final numbers are selection-contaminated”.
3. **Nhánh LLM**: thêm kiểm tra **line-level** (exact source line, template-line rate) giữa test và pool; mọi bảng description báo cáo **template-line vs new-line** tách riêng; nếu $>X\%$ test line có trong pool thì không được gọi test là “held-out” cho description.
4. **Công bố circularity**: nêu rõ tham chiếu sinh bởi Gemma/Qwen3-14B; tách báo cáo cho từng họ mô hình; nếu có thể, bổ sung **test người** nhỏ (100–300 mẫu) làm tham chiếu độc lập.
5. Gắn hash (dataset/sample/eval/config/seed) vào **cả 10 cell**; thêm assertion leak vào preflight kernel; nếu assertion fail thì fail cứng.

**Điều kiện rút phản biện:** có eval_v3 (hoặc holdout) pre-registered + báo cáo template/new-line + hash đủ 10 cell; hoặc tuyên bố lại toàn bộ kết luận chỉ ở mức “sensitivity analysis”.

---

## 2. PHẢN BIỆN 2 — OpenNMT FROM-SCRATCH: BẢN SỬA “RELABEL + FLOPs + NLLB ĐANG CHẠY” LÀ ĐÚNG HƯỚNG NHƯNG CHƯA ĐỦ; CHÍNH DỰ ÁN ĐÃ CÓ BẰNG CHỨNG BASELINE NMT CÓ THỂ NGANG LLM KHI CẤU HÌNH ĐÚNG

**Mức độ: MAJOR (giữ nguyên).**

### 2.1. Phần hội đồng chấp nhận

- Đã sửa nhãn `OpenNMT-B` thành **B-natural** trong mọi bảng; loại dòng B−A của OpenNMT khỏi bảng hiệu ứng dữ liệu — đúng.
- Bảng compute trung thực: OpenNMT ~**37M tham số** (4 lớp/256 hidden/32k vocab) vs Qwen3-8B 8,2B / EcomGPT 7B; token-pass A **245,8M** / B **188,3M** vs LLM 4,0M → **61,4× / 47,1×**; FLOPs fine-tune OpenNMT A **5,46e16** **nhỏ hơn** Qwen3-8B **1,97e17** → không thể nói “NMT tốn compute hơn”.
- Kết luận được phép đã được viết đúng: *“một cấu hình from-scratch 37M/4 layer/256 hidden sụp mode collapse”*, không kết luận cho paradigm NMT.

### 2.2. Vì sao vẫn MAJOR

**(a) Control pretrained NMT vẫn chưa có kết quả.** `NLLB-200-distilled-600M` mới ở trạng thái “đang chạy”; trong toàn bộ bằng chứng tải về **không có file output/score NLLB**. Không có nó, mọi so sánh “LLM vs NMT” vẫn là so sánh **có pretraining vs không pretraining**, chưa phải kiến trúc/paradigm. NLLB-600M lại là một lựa chọn đơn lẻ, nhỏ hơn Qwen 13×; cần NLLB-1.3B/3.3B hoặc mBART để có đường scaling.
**(b) Chính dự án đã có bằng chứng mâu thuẫn: OpenNMT nhánh Nhật Anh gần như ngang Qwen3-4B.** Trong `D3` (pair, SentencePiece 16k, Noam 2e-3, 20 epoch, early-stop, trung bình 5 checkpoint), OpenNMT from-scratch có **“Chính (câu TB)” +0,39 [−1,09; +1,90] so với Qwen3-4B** (pair), và **−5,57 [−6,42; −4,85]** (cum), tức **không có bằng chứng thua ở pair**. Trong khi đó OpenNMT nhánh Quang (char-level 32k, beam 512, `replace_unk`, 30k step A / 22,990 B) sụp hoàn toàn (COMET 0,349). **Hai baseline cùng paradigm, hai kết quả trái ngược → kết luận “NMT from-scratch yếu” là kết luận về cấu hình, không phải paradigm.** Báo cáo phải hợp nhất hai kết quả này và trình bày như bằng chứng về độ nhạy cấu hình.
**(c) Các trục confound vẫn còn nguyên, chỉ được liệt kê chứ chưa được kiểm soát:** capacity 37M vs 7–8B; pretraining (không vs hàng nghìn tỷ token); token-pass 61×/47×; epoch 48,3 vs 0,885 (và B 10 vs 0,253); tokenizer 32k char-level vs BPE 150k+; `max_length` 512 vs 2.048; decode beam+`replace_unk` vs greedy; **dữ liệu**: OpenNMT học toàn bộ train (line-level pair), LLM học quota balanced (block-level) — khác biểu diễn và khác tập. “Cùng 4M token” chỉ đúng trong nội bộ LLM, không áp cho NMT; `OPENNMT_B = B-natural` đã bị loại khỏi bảng dữ liệu, nhưng OpenNMT vẫn nằm trong bảng so sánh model ở cột B.
**(d) Chỉ 1 seed, không learning curve, không tối ưu hyperparameter trên dev; NLLB dự kiến dùng LoRA r16 — thêm một biến khác.** Không có learning curve thì không thể phân biệt “under-fit do cấu hình” với “paradigm yếu”.
**(e) Kết luận trong báo cáo cũ vẫn có thể bị trích dẫn.** `CLAIMS_REVISION.md` đã sửa câu chữ, nhưng phải rà toàn bộ `MATRIX_4X2_RESULTS.md`, `MODEL_CAUSE_ANALYSIS.md`, slide/báo cáo cũ — bất kỳ câu “NMT from-scratch thất bại / NMT không cạnh tranh” nào còn sót đều không được phép.

### 2.3. Yêu cầu bắt buộc

1. **Hoàn tất NLLB control** (≥1 size, lý tưởng 2 size: 600M + 1.3B/3.3B, cùng 4M quota, cùng max_new, cùng eval protocol); báo cáo kèm chrF/COMET và CI.
2. **Recipe NMT from-scratch chuẩn** (6 lớp/512 hidden, BPE 16k, 50–100M token nếu có, early-stop theo dev, checkpoint averaging, beam + length penalty, bỏ `replace_unk`) + **≥3 seed**, báo cáo learning curve và kết quả theo length-bucket.
3. **Hợp nhất hai baseline OpenNMT** (Quang và Nhật Anh) vào một bảng có ghi rõ cấu hình/dữ liệu/tokenizer; dùng kết quả D3 làm bằng chứng “cấu hình đúng có thể ngang LLM ở pair” thay vì che nó.
4. **Hạ tuyên bố**: chỉ được viết “cấu hình from-scratch 37M/4 layer/256 hidden, char-level 32k, beam 512 + replace_unk, 1 seed — sụp”; không dùng “NMT paradigm”, “LLM > NMT”.
5. Bảng compute phải công bố **cả pretraining budget** (hoặc tuyên bố không thể ước tính) để người đọc thấy rõ đây không phải so sánh compute-matched.

**Điều kiện rút phản biện:** có NLLB/mBART control hoàn chỉnh + recipe chuẩn + ≥3 seed + hợp nhất hai baseline; khi đó mới được bàn “paradigm”.

---

## 3. PHẢN BIỆN 3 — QUOTA SAMPLING: TRONG NỘI NHÓM ĐÃ ĐÚNG, NHƯNG “KHẮC PHỤC LENGTH BIAS” VẪN CHƯA THÀNH SỰ THẬT Ở CẤP THÍ NGHIỆM; VÀ KHẢO SÁT QUOTA CỦA NHÁNH LLM CÒN CONFOUND

**Mức độ: MAJOR (giữ nguyên).**

### 3.1. Phần hội đồng chấp nhận

- Sampler mới đã sửa đúng lỗi toán học `p_i ∝ t_i` (bias bình phương độ dài) **trong nội bộ nhóm**: token share thực 25/25/25/25; median/P90 mẫu ≈ quần thể (478,1/472,5; 46,5/46,5; 631,0/623,8; 34,4/34,4).
- Bản audit `QUOTA_LENGTH_AUDIT.md` minh bạch và đúng các con số hội đồng nêu: epoch nhóm **0,128×–2,318×**, coverage 11,8–90,2%, draws/row 0,127–2,322, 87% description không học, `epochs_seen 0,2529` che spread 18×; tokenizer Qwen thật **6,56M/5,93M** thay vì “4,0M”; tiki_desc target >960 không cap **30,4%**.
- Việc chuyển eval sang `eval_F_v2_common` 831 dòng và báo cáo macro/bucket là cải thiện.

### 3.2. Vì sao vẫn MAJOR — câu trả lời cho README vẫn là “CHƯA”

**(a) Quota đã sửa “cách chọn dòng trong nhóm”, nhưng chưa sửa “cái model thực sự học”.** Chính audit nói rõ: `balanced token ≠ balanced learning`. Trong cell B: 1688_desc epoch **0,128**, tiki_desc **0,146**, trong khi 1688_title **1,397**, tiki_title **2,318**; mỗi optimizer step chỉ có **4,05 / 3,20** mẫu description so với 42,9 / 54,1 mẫu title; ~87% description không bao giờ được học. Nếu mục tiêu là “khắc phục length/field bias” thì đây là **bias mới ở cấp exposure**, không phải nghiệm pháp đã hoàn tất.
**(b) “4M token” không phải token của model; đuôi dài bị cắt.** `tokens_est = |src|/3.6 + |tgt|/1.6`, cap 960; thực tế Qwen tokens **6,56M/5,93M**. Target description bị cắt: 4,9% (1688) / 13,7% (Tiki) theo cap; **30,4%** tiki_desc nếu không cap. Quota dựa trên `eff_tokens` cap 960 nên **khối lượng đuôi dài bị đếm thiếu**, đúng loại bias mà quota được kỳ vọng loại bỏ.
**(c) Thiết kế A/B vẫn confound field × quota.** A = `{title, attribute}` + natural; B = `{title, description}` + balanced → đổi 2 biến cùng lúc; **2×2 control (A-balanced, B-natural cho từng model) chưa chạy**; OpenNMT-B là B-natural bị dán nhãn balanced; chỉ **1 seed** training (42); paired bootstrap 2.000 chỉ resample sản phẩm eval, không phủ variance khởi tạo → CI hẹp hơn thực tế.
**(d) Eval sau khi “fix” vẫn bị selection bias và quá nhỏ.** `eval_F_v2_common` 1688 chỉ **231 dòng**, trong đó description **45 dòng**; all 1688-description conclusions (ví dụ “B giúp description +0,0614” của Qwen3-8B) dựa trên 45 dòng. `eval_F` lại nằm trong pool đã dùng để chọn recipe (xem phản biện 1). Vì vậy “quota đã khắc phục length bias” không thể chứng minh bằng một eval đã bị lọc + tái sử dụng.
**(e) Khảo sát quota của nhánh LLM có confound riêng và chưa chứng minh “natural/11%” là tối ưu.** Nhánh Nhật Anh dùng quota screening (3 cấu hình × 2 bộ × 2 seed, 300 mẫu/nhóm, không CI) và chốt natural; nhưng:
- metric “Chính” áp đặt trọng số **10/90**; **w hòa vốn** = 0,39 (cụm balanced), 0,55 (cụm title_heavy), 0,50 (pair attr_heavy), 0,58 (pair balanced) — **kết luận đổi nếu hội đồng coi title nặng hơn ~39–58%**, chứ không phải một kết luận khách quan;
- chỉ thử **3 mức** title (1:1, ~11%, 75%); chưa thử dải 11–50%, nên 11% chỉ chứng minh “tốt hơn 50%”, **chưa chứng minh tối ưu**;
- chính tài liệu ghi *“screening không tách được % title khỏi số lần mô hình thấy title (natural ~0,31 lượt/epoch, balanced ~1,4 lượt)”* — tức **quota thay đổi cả tần suất xuất hiện lẫn tỉ trọng**, đúng loại confound length/exposure;
- **bản cuối dùng `w_title` trong loss (0,17247 / 0,16915), không dùng quota sampling** — nên kết quả cuối không chứng minh “quota sampling đã khắc phục length bias”; nó chỉ chứng minh “một cách điều khiển khác cho kết quả tương tự trên screening”;
- năm mô hình chỉ đo trên **eval fold 0**, chưa đo test; test khóa 2.558 mới dùng cho adapter B8, không dùng cho so sánh 5 mô hình; reference là bản dịch máy, title reference lỗi **5–14%**, judge pair phủ **~14,5%** đơn vị tranh chấp.

### 3.3. Yêu cầu bắt buộc

1. **Trả lời đúng câu hỏi**: báo cáo chỉ được viết “đã sửa bias chọn dòng trong nội bộ nhóm; **chưa** sửa end-to-end (exposure/truncation/eval)” — không dùng “đã khắc phục length bias”.
2. **2×2 đầy đủ** (A-natural, A-balanced, B-natural, B-balanced) cho ít nhất Qwen3-4B/8B + OpenNMT/Moses; **≥3 seed** cho các cell chính; báo cáo CI theo seed.
3. **Token accounting thật** (Qwen tokenizer) và **báo cáo truncation theo stratum**; báo cáo per-group exposure trong mọi run JSON.
4. **Eval cố định pre-registered** (v3) + macro/bucket + CI; không trộn v1/v2/v3 trong cùng bảng.
5. **Nhánh LLM**: nếu muốn claim “natural/11% tốt nhất” phải chạy thêm mức 25–35%, báo cáo đường cong, CI theo seed và tách “% title” khỏi “số lần thấy title” (ví dụ cùng tỉ trọng token nhưng khác số epoch title); nếu không, chỉ được viết “trong 3 mức đã thử, natural tốt nhất dưới metric 10/90”.
6. **Tách rõ quota sampling vs loss weighting**: hai cơ chế khác nhau; kết luận về cái nào phải dựa trên thí nghiệm của cơ chế đó.

**Điều kiện rút phản biện:** có 2×2 + ≥3 seed + token accounting thật + eval pre-registered; hoặc hạ toàn bộ tuyên bố về mức “đã sửa trong nội bộ nhóm; chưa kết luận end-to-end”.

---

## 4. BẢNG TRẠNG THÁI: NHÓM TỰ ĐÁNH GIÁ vs HỘI ĐỒNG

| Phản biện | Nhóm tự ghi (`REVIEW_RESPONSE`) | Hội đồng vòng 2 | Lý do chính |
|---|---|---|---|
| 1 — Data leakage | **FIXED** | **PARTIAL — BLOCKER** | v2 là post-hoc; test reuse; 1688 còn 231 dòng; nhánh LLM còn line-level leak 85%; circularity tham chiếu; hash chưa đủ 10/10 |
| 2 — OpenNMT fairness | **PARTIAL + đang chạy** | **PARTIAL — MAJOR** | NLLB chưa có kết quả; D3 cho thấy OpenNMT có thể ngang LLM khi cấu hình đúng; confound pretraining/data/tokenizer/seed chưa kiểm soát |
| 3 — Quota/length bias | **PARTIAL + đang chạy** | **PARTIAL — MAJOR** | Đã sửa trong nhóm; end-to-end chưa; 2×2 + seed chưa; khảo sát quota nhánh LLM còn confound và chưa chứng minh tối ưu; quota ≠ loss weighting |

---

## 5. KẾT LUẬN CỦA HỘI ĐỒNG

- Bộ phản hồi vòng 1 là **bước tiến thực chất** về minh bạch và công cụ audit; hội đồng ghi nhận.
- Nhưng **chưa có phản biện nào được đóng**: (1) vẫn BLOCKER vì chưa có test độc lập thực sự; (2) và (3) vẫn MAJOR vì control/thí nghiệm quyết định chưa chạy xong.
- **Không được bảo vệ** nếu báo cáo vẫn dùng: `eval_F` gốc; `eval_F_v2` như “benchmark chính” mà không nêu post-hoc; câu “NMT from-scratch thất bại/không cạnh tranh”; câu “quota đã khắc phục length bias”; số liệu 5 mô hình nhánh LLM như “test” (mới là eval fold 0); kết quả demo 13 trang như “triển khai production”.
- **Được phép bảo vệ** nếu báo cáo trình bày đúng trạng thái: nêu rõ sensitivity, scope, confound, và các kết luận đã hạ mức theo `CLAIMS_REVISION.md` — kèm kế hoạch v3/2×2/NLLB/human eval.

---

## PHỤ LỤC A — LỆNH TÁI LẬP TẢI BẰNG CHỨNG

```bash
# kiểm tra quyền + revision
python3 download_hf_evidence_v2.py --check

# tải đầy đủ bộ bằng chứng vòng 2 (~510 MB)
python3 download_hf_evidence_v2.py

# nếu có token (khuyến nghị, tránh rate limit):
HF_TOKEN=hf_xxx python3 download_hf_evidence_v2.py
```

Manifest: `hf_dataset/DOWNLOAD_MANIFEST_V2.json` (revision, file count, size, sha256 cho file ≤32 MB).
Cây bằng chứng chính:
- `hf_dataset/` — metadata/results `zh-vie_ecom` hiện hành.
- `hf_dataset/_history/126a564f88/` — trace Judge DeepSeek chi tiết.
- `hf_dataset/tiki-4model-eval/` — docs/reports/configs/**`reviews/2026-11-16/`**/runs/scores/metadata.

## PHỤ LỤC B — VÌ SAO KHÔNG CÓ “EVAL/METRICS” TRÊN HF

README_EVIDENCE.md giả định có `eval/metrics`; thực tế `zh-vie_ecom` không có thư mục này. Kết quả nằm trong `tiki-4model-eval` (bản cập nhật 16/11) — nếu chỉ tải `zh-vie_ecom` sẽ **không thấy** bộ phản hồi và các bảng re-score. Đây là lỗ hổng tái lập cần sửa: publish `eval_F_v2`, scores, `review_audit.json`, `RUN_HASH_MANIFEST` cùng dataset hoặc tạo dataset `reviews/` riêng.

## PHỤ LỤC C — FILE/TRÍCH DẪN CHÍNH

| Nội dung | Đường dẫn |
|---|---|
| Bản trả lời phản biện vòng 1 | `hf_dataset/tiki-4model-eval/reviews/2026-11-16/` |
| Leakage audit + eval v2 | `LEAKAGE_AUDIT.md`, `data/cleaned/eval_F_v2*.parquet`, `eval_F_v2_meta.json` |
| Ma trận v2 + sensitivity | `MATRIX_5X2_EVAL_F_V2.md`, `EVAL_F_V2_SENSITIVITY.csv` |
| Quota audit | `QUOTA_LENGTH_AUDIT.md`, `review_audit.json` khóa `quota`, `RUN_HASH_MANIFEST.json` |
| OpenNMT fairness | `OPENNMT_FAIRNESS.md`; builder NLLB `ntquang_branch/kaggle/matrix_nllb/` |
| Nhánh LLM — split/leak | `nhatanh_branch/docs/ket_qua_kaggle/A7_*.md` (0 vi phạm Jaccard; 85% dòng test có trong pool) |
| Nhánh LLM — OpenNMT | `nhatanh_branch/docs/ket_qua_kaggle/D3_*.md` (pair +0,39 [−1,09; +1,90] so với Qwen3-4B) |
| Nhánh LLM — quota | `nhatanh_branch/docs/QUOTA_KET_LUAN.md`, `TRONG_SO_TITLE_MO_TA_DA_CHOT.md` |
| Nhánh LLM — giới hạn | `nhatanh_branch/docs/pipeline/10_demo_hien_trang_va_gioi_han.md` (title ref lỗi 5–14%; judge pair 14,5%; 5 mô hình mới eval fold 0) |
| Nhánh Quang — test reuse | `ntquang_branch/reports/NEXT_STEPS.md` §2 (“eval_A/B đã dùng để chọn config… cần holdout mới”) |
| Tuyên bố phải sửa | `hf_dataset/tiki-4model-eval/reviews/2026-11-16/CLAIMS_REVISION.md`; `nhatanh_branch/docs/BANG_KET_QUA_CUOI.md` (bản cũ đã bị thay bởi `QUOTA_KET_LUAN.md`) |
