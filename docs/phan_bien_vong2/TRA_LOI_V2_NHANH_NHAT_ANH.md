# TRẢ LỜI PHẢN BIỆN VÒNG 2: NHÁNH NHẬT ANH

Ngày 17/11/2026, cập nhật 19/11/2026. Phần P0/P1 (17/11): OpenCode, theo giao việc `PROMPT_OPENCODE_P0_P1.md`, chấm lại trên **eval fold 0**, CPU, không train, không sinh bản dịch, không đọc test khóa. Phần GPU (18-19/11, mục 8): Claude chạy Kaggle L4x4 và Colab T4 (2 seed OpenNMT, tỉ trọng title 25% và 35%, zero-shot), cũng chỉ trên eval fold 0, không đọc test khóa. Ba phân tích bổ sung trên đầu ra GPU (P2, 19/11/2026): `KE_HOACH_PHAN_TICH_P2.md` (khóa trước khi chạy) và `KET_QUA_P2.md`. Kế hoạch phân tích khóa trước khi xem số: `KE_HOACH_PHAN_TICH_P1.md`, sha256 `3743d3eeabcf30ab52a7d2f12cb6ec2aaa3ae7f9634e42c7be82100da81e1ef0`. Số thô: `data/processed/ft_run/phan_bien_v2/*.json`. Nhật ký từng sửa đổi: `NHAT_KY_OPENCODE.md`.

## Bảng tóm tắt

| Mục biên bản | Hội đồng nói (tóm tắt) | Nhánh mình chấp nhận gì | Đã sửa ở file nào | Số mới | Còn mở |
|---|---|---|---|---|---|
| 1.2(c), 1.3.3 | Test held-out theo sản phẩm nhưng phần lớn dòng mô tả đã có trong pool; bảng chính chưa tách dòng khuôn/dòng mới | Đúng. Test chỉ held-out ở cấp sản phẩm, không held-out từng dòng; điểm mô tả toàn phần nghiêng về tái tạo khuôn | A7, `pipeline/08`, `pipeline/09` mục 6, `TONG_HOP` mục 6.1, `KE_HOACH_VA_KET_QUA_NAM_MO_HINH` mục 5 | 85,91% dòng mô tả pair zh→vi đã thấy; OpenNMT trên dòng mới −8,29 (pair zh→vi), −10,07 (cụm zh→vi) [ĐO] | Test, dòng mô tả mới (Bước 0, từ tổng hợp B8): fine-tune hơn zero-shot khoảng 30 điểm (cụm zh→vi +30,1; pair zh→vi +30,6; pair vi→zh +29,7) so với +44 đến +60 trên dòng khuôn; mục 3 | Chưa có khoảng tin cậy cho phần tách trên test (cần chấm lại bản dịch B8 bằng tham chiếu test, chờ duyệt) |
| 1.2(d)(ii), 1.3.4 | Tham chiếu do Gemma/Qwen sinh, Judge chọn; Qwen3 cùng họ một mô hình sinh tham chiếu; chưa kiểm vòng lặp | Đúng một phần. Đã kiểm trên title; mô tả fold 0 không có nhóm Qwen chủ đạo để kiểm | `TONG_HOP` mục 6.3, `pipeline/10` mục 4.1 | Title: 55 Qwen / 245 Gemma; kép so EcomGPT −7,68, so OpenNMT +5,17, so Moses +3,11 [ĐO], không nhất quán theo họ. **Zero-shot cụm (P2)**: Qwen3-4B zero-shot điểm cao hơn 6,4 ở nhóm tham chiếu Qwen trong khi mọi mô hình khác thấp hơn; kép +10,20 / +16,17 / +14,41 so EcomGPT / OpenNMT / Moses, khoảng không chứa 0; tinh chỉnh xóa phần lớn lợi thế (kép −13,72) [ĐO], mục 8.5 | Người chấm 100–300 mẫu; nhóm Qwen chỉ 55 sản phẩm; chỉ title |
| 1.2(b) | Fold 0 là tập chọn, không phải test; số 5 mô hình chưa held-out | Đúng. Đã ghi rõ ở mọi bảng 5 mô hình | `pipeline/09`, `TONG_HOP`, `KE_HOACH_VA_KET_QUA_NAM_MO_HINH`, D6 | 5 mô hình: eval fold 0, r16, chưa đo test; test duy nhất B8 [ĐO] | Test lần hai có đăng ký trước |
| 2.2(b), 2.3.3–5 | Câu "OpenNMT hòa 4B" quá rộng; thiếu bảng compute, đường học, độ dài, hợp nhất baseline | Đúng. Hòa chỉ ở một ô chrF câu TB; khác biệt theo loại dòng và hướng | D3, `pipeline/09`, `TONG_HOP` 6.1, `KE_HOACH_VA_KET_QUA_NAM_MO_HINH`, `KET_QUA_P1` 3.2/3.4/3.5/3.6 | D3 pair: +0,39 [−1,09; +1,90] câu TB, corpus 74,38 so 79,98, macro −8,33; cụm −5,57 [−6,42; −4,85] [ĐO]. **3 seed OpenNMT** (chrF corpus chính): pair 74,73 ± 0,70, cụm 80,61 ± 0,11, kém Qwen3-4B 5,3 (pair) và 6,2 (cụm) [ĐO], mục 8.1 | "Hòa câu TB" đã tính lại cho 3 seed: pair chính +0,30 [−1,07; +1,71], chỉ đúng ở thước đó (macro −8,65 [−9,66; −7,64]), mục 8.1 và 8.4. Chưa chạy recipe 6 lớp/512; NLLB/mBART là việc nhánh Quang |
| 3.2(e), 3.3.5–6 | `w_title` dựa trên quota sampling có confound tần suất; câu "tối ưu" quá mức; thiếu CI và seed | Đúng. Bằng chứng chính là thí nghiệm trọng số loss; quota chỉ kiểm chéo | `pipeline/09` mục 3, `TONG_HOP` mục 4, `TRONG_SO_TITLE_MO_TA_DA_CHOT`, `QUOTA_KET_LUAN` | 3 mức 50/11/75%, 3 fold; w hòa vốn 0,39 (cụm), 0,50 (pair) [ĐO]; CI chỉ phủ chọn mẫu eval. **Mức 25% và 35% (cụm, fold 0, 1 seed)**: chrF macro 75,954 / 75,933 so đối chứng 11% 75,898, theo thước 10/90 zh→vi 89,93 / 89,66 so 90,15 [ĐO]; **CI (P2)**: 25% − 11% −0,19 [−0,41; +0,03], 35% − 11% −0,39 [−0,68; −0,09] (title +1,19, mô tả −0,56), mục 8.2 | Chỉ bản cụm, 1 seed (CI chỉ phủ chọn mẫu eval); chưa tách "% title" khỏi "số lần thấy title"; pair chưa chạy |
| Mục 5 | 5 mô hình là eval fold 0; demo không phải production | Đúng | `pipeline/10` mục 1 và 4.7, `TONG_HOP` mục 0 và 9.2, `demo/CHAY_DEMO.md` | Demo: 13 trang, đọc tay 8 trang, không tham chiếu, một người dùng [ĐO] | 8B công thức cuối chưa chạy |

Không nhận thay nhánh Quang các mục chỉ thuộc về họ: eval_F, NLLB, ma trận 2×2 A/B, hash 10 cell, README_EVIDENCE.

## 1. Rò rỉ cấp dòng và bảng dòng khuôn/dòng mới (1.2(c), 1.3.3)

Đúng. A7 kiểm ở **cấp sản phẩm** theo Jaccard, 0 vi phạm, nhưng ở **cấp dòng** thì 85,0% dòng mô tả của test đã có nguyên văn trong pool. Vì vậy test là held-out theo sản phẩm, không phải held-out cho từng dòng mô tả.

P1 tách lại trên eval fold 0 bằng đúng định nghĩa `cham_diem.xay_tap_da_thay`: cặp `(category, dòng nguồn đã chuẩn hoá)` có trong train fold 1–4.

| Bộ | Hướng | Tỉ lệ dòng mô tả đã thấy | Số dòng | Số sản phẩm |
|---|---|---:|---:|---:|
| pair | zh→vi | **85,91%** | 298 | 285 |
| pair | vi→zh | 75,67% | 300 | 280 |
| cum | zh→vi | **86,06%** | 7.128 | 296,5 |
| cum | vi→zh | 73,92% | 2.093,5 | 298,5 |

Với pair, mỗi mục mô tả là một dòng thuộc tính, nên số dòng bằng số mục; với cum, mỗi mục là cả khối và số dòng là số mảnh sau khi tách `"; "`. Số sản phẩm cum là trung bình 2 seed. Kiểm ref với nguồn: 0 lệch ở pair, 1/300 mục ở cum zh→vi do dấu `";"` nằm trong giá trị. Ở cum, EcomGPT-7B bị loại 35/298 sản phẩm (zh→vi) và 9/300 (vi→zh) vì lệch số dòng, nên điểm của nó không cùng tập với các mô hình khác.

Trên **dòng mô tả mới**, khoảng cách tới Qwen3-4B lớn hơn nhiều so với trên dòng khuôn. (Các số "chênh [95%]" dưới đây là chênh của chrF tính **theo từng sản phẩm** rồi lấy trung bình, bootstrap theo sản phẩm; cột "Mô tả mới" ở các bảng 5 mô hình là chrF corpus gộp mọi dòng mới, nên độ lớn khác: ví dụ pair zh→vi OpenNMT 44,88 so 4B 63,41, tức −18,5 theo corpus và −8,29 theo sản phẩm; hai thước cùng chiều.) Pair zh→vi: OpenNMT −8,29 [−15,49; −0,88], EcomGPT −12,70 [−17,70; −8,09], Moses học đủ −2,50 [−8,11; +3,02], Qwen3-8B −0,43 [−3,98; +2,86]. Pair vi→zh: OpenNMT −17,76 [−25,85; −9,90], EcomGPT −9,69, Moses −2,78, 8B +0,38. Cum zh→vi: OpenNMT −10,07 [−11,68; −8,30], EcomGPT −8,17, Moses −3,68, 8B **+1,83 [+0,85; +2,89]**.

Chiều ngược lại, trên **dòng khuôn** pair zh→vi, OpenNMT lại hơn 4B +2,12 [+0,78; +3,50] và Moses +2,26. Vì vậy câu đúng là: điểm mô tả toàn phần bị chi phối bởi dòng khuôn; phần dịch câu mới cho thấy khoảng cách 8–22 điểm, không phải 0–5 điểm. Số sản phẩm có dòng mới nhỏ (42 pair zh→vi, 71 pair vi→zh) nên CI rộng; đã ghi trong `KET_QUA_P1.md` mục 6.

Đã sửa: A7 thêm câu cấp dòng ngay sau dòng đọc từ JSON; `pipeline/08` bỏ câu "KHÔNG RÒ RỈ" đứng một mình; ba bảng 5 mô hình thêm cột "Mô tả mới (P1)". Test đã được tách dòng mới ở mức điểm ước (mục 3, Bước 0 của B8); chưa có khoảng tin cậy cho phần này.

## 2. Vòng lặp tham chiếu (1.2(d)(ii), 1.3.4)

Tham chiếu title và mô tả do Gemma 4 hoặc Qwen sinh (Qwen3-14B; một phần title có thể là Qwen3.5-9B **[CHƯA KIỂM]**), Judge DeepSeek chọn bản; dữ liệu huấn luyện lấy cùng nguồn. Qwen3-4B/8B cùng họ với một mô hình sinh tham chiếu.

**Title, eval fold 0 zh→vi:** 55 sản phẩm nhóm Qwen, 245 nhóm Gemma. Chênh lệch kép (lợi thế của mô hình Qwen ở nhóm Qwen trừ nhóm Gemma):

| Cặp | Kép [95%] |
|---|---:|
| Qwen3-4B − EcomGPT-7B | −7,68 [−10,36; −5,18] |
| Qwen3-4B − OpenNMT | +5,17 [+2,11; +8,19] |
| Qwen3-4B − Moses học đủ | +3,11 [+0,43; +5,99] |
| Qwen3-8B − EcomGPT-7B | −7,72 [−10,50; −5,07] |
| Qwen3-8B − OpenNMT | +5,12 [+2,22; +8,13] |
| Qwen3-8B − Moses học đủ | +3,07 [+0,47; +5,77] |

(Hai đầu khoảng là bản chạy với thứ tự mã sản phẩm xác định, xem ghi chú cuối mục 2.)

Kết quả **không nhất quán theo họ baseline**: so EcomGPT (BLOOM) thì lợi thế nhỏ hơn ở nhóm Qwen, so OpenNMT/Moses thì lớn hơn. Không thể nói "Qwen được lợi vì tham chiếu cùng họ" như quy luật chung.

**Zero-shot Qwen3-4B:** lúc viết P1 (17/11) chưa tính được vì không có bản dịch zero-shot đầy đủ trên eval fold 0. Ngày 18/11 đã sinh bản zero-shot **bản cụm** trên eval fold 0 (mục 8.3), nên có điểm số và đã tính **chênh lệch kép của title** theo nhóm tham chiếu Qwen/Gemma (mục 8.5, `KET_QUA_P2.md` mục C): zero-shot Qwen3-4B điểm **cao hơn 6,4 điểm** ở nhóm tham chiếu Qwen (55 sản phẩm) so nhóm Gemma (245), trong khi mọi mô hình khác điểm **thấp hơn** ở nhóm Qwen; chênh lệch kép so với EcomGPT, OpenNMT, Moses là +10,20 / +16,17 / +14,41, khoảng 95% không chứa 0. Tức **có lợi thế sẵn có của họ Qwen ở title có tham chiếu do Qwen sinh, không do fine-tune**; fine-tune không khuếch đại mà xóa phần lớn lợi thế đó (mục 8.5). Bản **pair** zero-shot không dùng được (mục 8.3). Kết luận ở đoạn trên ("không nhất quán theo họ baseline") là kết luận của P1 trước khi có zero-shot; đọc cùng mục 8.5.

**Mô tả, eval fold 0 zh→vi:** 285 sản phẩm pair đều Gemma chủ đạo (cum 297 Gemma, 1 không xác định), **không có sản phẩm Qwen chủ đạo**, nên chênh lệch kép theo kế hoạch không tính được. Đây là kết quả, không phải lỗi. Phân tích thêm sau khi xem số, nhóm "có mảnh Qwen" (170) so "không có" (115), cả 6 khoảng 95% đều chứa 0.

Giới hạn: Judge chọn Qwen khi bản Gemma kém nên hai nhóm khác độ khó; chênh lệch kép chỉ khử phần chung. Còn mở: tham chiếu độc lập do người chấm 100–300 mẫu.

Ghi chú về hai đầu khoảng tin cậy ở mục này: P1 rút bootstrap theo thứ tự bám của `set` Python (không xác định giữa các lần chạy). Chạy lại với thứ tự xác định (`p7_kiem_p1_xac_dinh.py`, `phan_bien_v2/kiem_xac_dinh_vong_tham_chieu.json`) trên cả 36 khoảng của P1: điểm ước giữ nguyên (lệch 5e-15), hai đầu dịch chuyển tối đa 0,40, **không khoảng nào đổi dấu**. Các giá trị ổn định để trích cho title: pair, Qwen3-4B trừ EcomGPT −7,68 [−10,36; −5,18], trừ OpenNMT +5,17 [+2,11; +8,19], trừ Moses +3,11 [+0,43; +5,99]; cụm, trừ EcomGPT −3,52 [−5,87; −1,17], trừ OpenNMT +2,45 [−0,80; +5,56], trừ Moses +0,69 [−1,82; +3,45]. Bảng phía trên đã dùng các giá trị xác định này; `KET_QUA_P1.md` giữ các hai đầu của lần chạy P1 gốc, lệch tối đa 0,4.

## 3. Fold 0 là tập chọn, không phải test (1.2(b))

Đúng. Năm mô hình đo trên eval fold 0, train = fold 1–4, cấu hình sàng lọc r16, 4M token, chưa đo trên test. Fold 0 đã dùng để chọn cấu hình (B6, tỉ trọng title, 5 mô hình, adapter demo), nên số trên fold 0 bị ảnh hưởng bởi việc chọn; tỉ trọng title có thêm kiểm độc lập ở fold 1 và 2.

Số test duy nhất là B8 (07/11, adapter fold 0 so với zero-shot): chênh +30,14 đến +34,14 tùy nhánh × hướng. Chênh này bị phóng đại một phần vì tham chiếu là bản dịch máy cùng kiểu với dữ liệu huấn luyện. **Đã tách dòng khuôn/dòng mới trên test** (Bước 0, `KET_QUA_B8_DONG_MOI.md`, đọc từ tổng hợp của đúng lần đo B8, không chấm lại): 86 đến 87% dòng mô tả zh→vi (73% vi→zh) đã có trong train; trên dòng mô tả **mới** fine-tune vẫn hơn zero-shot khoảng 30 điểm chrF corpus (cụm zh→vi +30,1; pair zh→vi +30,6; pair vi→zh +29,7; cụm vi→zh không so được vì zero-shot lệch dòng ở 920/1.073 mục) so với +44 đến +60 trên dòng khuôn; điểm mô tả toàn phần (92 đến 94) bị nâng bởi dòng khuôn, chất lượng dòng mới khoảng 76. Chưa có khoảng tin cậy. Không dùng B8 để xếp hạng năm mô hình.

Đã ghi lại ở `pipeline/09` mục 4 và 6, `TONG_HOP` mục 5 và 6, `KE_HOACH_VA_KET_QUA_NAM_MO_HINH` mục 5 và 7, D6.

## 4. D3, bảng compute, đường học, độ dài, hợp nhất OpenNMT (2.2(b), 2.3.3–5)

**D3.** Với cấu hình Transformer nhỏ học từ đầu, OpenNMT hòa Qwen3-4B ở pair theo chrF câu trung bình trên eval fold 0 (+0,39 [−1,09; +1,90]), nơi phần lớn dòng đã có trong train. Nhưng chrF corpus chính là 74,38 so 79,98; macro câu là −8,33 [−9,40; −7,26]; cụm là −5,57 [−6,42; −4,85]. Trên dòng mô tả mới, OpenNMT kém −8,29 (pair zh→vi) và −10,07 (cụm zh→vi); trên dòng khuôn pair zh→vi lại hơn +2,12. Kết quả cho thấy độ nhạy theo cấu hình và theo loại dòng, không kết luận về paradigm NMT. **Cập nhật 19/11:** chạy thêm seed 43 và 44 (cùng cấu hình D3), chrF corpus chính pair 74,38 / 74,28 / 75,54 (trung bình 74,73, độ lệch chuẩn 0,70), cụm 80,61 / 80,50 / 80,72 (80,61 ± 0,11); khoảng cách tới Qwen3-4B (79,98 pair, 86,81 cụm, 2 seed) là 5,3 và 6,2 điểm, lớn hơn nhiều dao động giữa seed. Câu "hòa chrF câu TB" ở pair đã tính lại cho 3 seed (mục 8.1): chênh "chính" +0,39 / +0,12 / +0,37, trung bình 3 seed +0,30 [−1,07; +1,71], khoảng nào cũng chứa 0, nên câu "hòa" được giữ nhưng **chỉ cho thước chrF câu trung bình chính (0,1 title + 0,9 mô tả, zh→vi)**; ở macro OpenNMT kém 8,3 đến 9,0 điểm ở mọi seed, và ở cụm kém 5,4 đến 5,6 điểm (chính). Chi tiết mục 8.1.

**Compute.** GPU-giờ đọc từ report: Qwen3-4B 0,19; Qwen3-8B 0,37; EcomGPT-7B 0,33; OpenNMT 1,08; Moses 2,93. Pretraining: Qwen3 khoảng 36.000B token, BLOOM khoảng 366B token, đều **[CHƯA KIỂM]** theo báo cáo công bố; OpenNMT và Moses bằng 0. Đây không phải so sánh compute-matched.

**Đường học OpenNMT.** zh→vi: ppl dev 19,05 → 10,82, acc 65,73 → 79,58, best step 5.685, dừng sớm ở 10.233. vi→zh: ppl 204,15 → 32,66, acc 30,70 → 66,39, chạy đủ 5.620 bước; log không ghi best step. Hình trong `docs/phan_bien_vong2/hinh/opennmt_zhvi.png` và `opennmt_vizh.png`.

**Độ dài.** Mô tả pair zh→vi, nhóm dài nhất Q4: Qwen3-4B 72,2; OpenNMT 58,5; Moses 71,2; EcomGPT 53,8. Ở Q1, OpenNMT lại hơn 4B +2,0. Đây là bằng chứng cho "cấu hình NMT nhỏ yếu ở câu dài", không phải "NMT nói chung yếu".

**Hợp nhất OpenNMT.** Hai baseline cùng 37M, 4 lớp/256 hidden, nhưng khác tokenizer (SentencePiece 16k so char-level 32k), beam (5 so 512 + `replace_unk`), dữ liệu (dòng của bản pair so A/B block), tập đo (eval fold 0 so eval_F) và thước (chrF so COMET-da/QE 0,349). Không so điểm trực tiếp, chỉ so cấu hình. Cột nhánh Quang lấy từ biên bản, chưa đối chiếu cấu hình gốc.

## 5. Trọng số loss so với quota sampling (3.2(e), 3.3.5–6)

Đúng là khảo sát quota có confound "tỉ trọng so với số lần thấy": natural thấy khoảng 0,31 lượt title/epoch, balanced khoảng 1,4. Vì vậy **bằng chứng chính** cho `w_title` của bản cuối là thí nghiệm **trọng số loss** (3 mức 1:1 khoảng 50%, khoảng 11%, 75%; 3 fold; có khoảng tin cậy; luật khóa trước). Trong cơ chế này mọi title được thấy đúng 2 epoch như nhau, chỉ đổi hệ số nhân loss từng mẫu; dữ liệu không bị lấy mẫu lại theo tỉ lệ. Đã kiểm trong `scripts/ft_train/kaggle_train.py` trước khi viết câu này.

Khảo sát quota chỉ là **kiểm chéo**, rank 16, một lượt, 2 seed, 300 mẫu/nhóm, không có khoảng tin cậy chính thức. Câu được phép: "Trong 3 mức đã thử, khoảng 11% title cho kết quả cao nhất dưới thước 10/90 (zh→vi). Kết luận đổi nếu trọng số title của thước đo từ khoảng 0,39 (cụm) và 0,50 (pair) trở lên. Chưa thử 11–50%; mức 25% và 35% chưa chạy." Câu này nay đổi thành (đã có khoảng tin cậy ghép cặp, mục 8.2 và `KET_QUA_P2.md` mục B): "Ở bản cụm (1 seed, fold 0), dưới thước 10/90 mức 11% không kém 25% (−0,19 [−0,41; +0,03]) và hơn 35% một chút (35% − 11% = −0,39 [−0,68; −0,09]; title zh→vi +1,19, mô tả zh→vi −0,56: đánh đổi); dưới thước macro ba mức không phân biệt được; khoảng tin cậy chỉ phủ dao động chọn mẫu, chưa có dao động khởi tạo; chưa chạy bản pair." Đã bỏ chữ "tối ưu" khỏi các câu kết luận.

Khoảng tin cậy của các bảng chỉ phủ dao động do chọn mẫu eval, không phủ dao động khởi tạo. Số seed: sàng lọc 2 seed (42, 123); công thức chính 1 seed.

## 6. Mục 5 biên bản: 5 mô hình và demo

Năm mô hình là eval fold 0, không phải test; đã ghi ở mọi bảng. Demo là thử nghiệm quy mô nhỏ, không phải hệ thống đưa vào dùng thật: 13 trang 1688 thật, đọc tay 8 trang, không có tham chiếu người, một người dùng, máy chủ trên Kaggle T4 hoặc laptop. Chưa có số về độ ổn định, tải, nhiều người dùng hay trang ngoài 1688. Bản nén Q5_K_M đạt cổng đặt trước trên laptop; Q4_K_M bị loại. 8B theo công thức cuối chưa chạy được vì thiếu GPU hơn 24 GB một card.

## 7. Việc chờ Nhật Anh quyết (chỉ liệt kê, không làm)

1. Phần tách dòng mới trên test đã có điểm ước từ tổng hợp B8 (`KET_QUA_B8_DONG_MOI.md`, không cần dùng test lần hai). Còn mở: có chấm lại bản dịch B8 bằng tham chiếu test (Bước 1 của `KE_HOACH_B8_DONG_MOI_TEST.md`) để có khoảng tin cậy hay không; và có dùng test lần hai có đăng ký trước cho 5 mô hình hay không.
2. Có mời người song ngữ chấm 100–300 mẫu làm tham chiếu độc lập hay không.
3. Đã có số cho mức title 25% và 35% (chỉ bản cụm) và 2 seed OpenNMT (mục 8). Còn mở, chờ quyết: chạy w_title bản pair và thêm seed (tốn GPU); chạy recipe OpenNMT 6 lớp/512 (tốn GPU); đối chứng NLLB/mBART là việc nhánh Quang.

## 8. Kết quả phiên GPU bổ sung (18-19/11/2026)

Tất cả trên **eval fold 0** (train = fold 1–4), chưa đo test, không đọc test khóa. Số thô và hash: `data/processed/ft_run/phan_bien_v2/onmt_3seed.json`, `kaggle_tlc_v2_tiep/` (`SHA256_ket_qua_phien2.txt`), `kaggle_tlc_song_song/bao_cao_zero_shot.json`.

### 8.1. OpenNMT ba seed (Colab T4, cấu hình D3)

| | Seed 42 | Seed 43 | Seed 44 | Trung bình ± độ lệch chuẩn | Qwen3-4B (2 seed) |
|---|---:|---:|---:|---:|---:|
| Pair, chrF corpus chính | 74,38 | 74,28 | 75,54 | 74,73 ± 0,70 | 79,98 |
| Pair, macro | 54,89 | 54,02 | 54,99 | 54,63 ± 0,54 | 68,52 |
| Cụm, chrF corpus chính | 80,61 | 80,50 | 80,72 | 80,61 ± 0,11 | 86,81 |
| Cụm, macro | 58,72 | 58,12 | 58,31 | 58,38 ± 0,31 | không đưa vào bảng này |

[ĐO] Seed 42 chấm lại khớp D3 từng số. Khoảng cách tới Qwen3-4B (5,3 pair, 6,2 cụm theo chrF chính) lớn hơn nhiều độ lệch chuẩn giữa seed của OpenNMT, nên câu "OpenNMT kém Qwen3-4B theo chrF corpus" không phụ thuộc seed. "Hòa theo chrF câu TB" ở pair đã tính lại cho 3 seed (`KET_QUA_P2.md` mục A): chênh chính +0,39 / +0,12 / +0,37, trung bình 3 seed +0,30 [−1,07; +1,71], chứa 0 ở mọi seed, nhưng macro kém 8,3 đến 9,0 điểm và cụm kém 5,4 đến 5,6 điểm (chính), khoảng không chứa 0. Giới hạn: Qwen3-4B chỉ 2 seed; cấu hình vẫn là Transformer nhỏ (4 lớp/256), chưa phải recipe chuẩn 6 lớp/512 mà biên bản yêu cầu.

### 8.2. Tỉ trọng title 25% và 35% (Kaggle L4x4, Qwen3-4B LoRA r64, bản cụm, fold 0, seed 42)

Ba việc cùng cấu hình (2 epoch, 1.837 bước, ngân sách token 4.096, gradient checkpointing); đối chứng 11% chạy cùng môi trường để làm thước đo trôi môi trường. 1.000 mẫu eval (250 mỗi nhóm hướng × loại).

| Việc | w_title | chrF macro (corpus) | chrF câu TB macro | Thước 10/90 zh→vi |
|---|---:|---:|---:|---:|
| 25% | 0,33668 | 75,954 | 76,859 | 89,93 |
| 35% | 0,54521 | 75,933 | 76,953 | 89,66 |
| 11% (đối chứng) | 0,12029 | 75,898 | 76,685 | 90,15 |

Theo nhóm (chrF corpus): title zh→vi 59,05 / 60,32 / 59,32; title vi→zh 61,17 / 60,48 / 60,54; mô tả zh→vi 93,36 / 92,92 / 93,58; mô tả vi→zh 90,24 / 90,02 / 90,16 (thứ tự 25% / 35% / 11%). Loss kiểm epoch 2 gần như bằng nhau (title zh→vi 0,649 / 0,648 / 0,653).

[ĐO] Tăng tỉ trọng title lên 25% hoặc 35% **không cho điểm cao hơn 11%** theo thước chính. Khoảng tin cậy bootstrap (P2, `KET_QUA_P2.md` mục B, chỉ phủ dao động chọn mẫu eval): 25% − 11% = −0,19 [−0,41; +0,03] (chứa 0); **35% − 11% = −0,39 [−0,68; −0,09] (không chứa 0)**: ở 35% title zh→vi tăng +1,19 [+0,22; +2,25] nhưng mô tả zh→vi giảm −0,56 [−0,89; −0,25], tức đánh đổi title lấy mô tả, và dưới thước 10/90 thì không lời. Dưới thước macro (title và mô tả ngang nhau) cả ba mức không phân biệt được. Title zh→vi ở 25% lại không tăng, nên chưa nói được đường cong đơn điệu. Giới hạn: một seed, 250 mẫu mỗi nhóm; chỉ bản cụm; chưa tách "% title" khỏi "số lần thấy title"; không hiệu chỉnh đa so sánh; môi trường Kaggle khác các lần chạy trước (torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2), nên không đặt chung đường cong với các mức 50% và 75% đã đo ở lần chạy cũ. Đây không phải bằng chứng cho "11% tối ưu".

### 8.3. Zero-shot Qwen3-4B

Bản cụm, eval fold 0, 1.200 mục: chrF corpus zh→vi 47,05 (title 38,10, mô tả 56,01), vi→zh 41,15, macro 44,10 [ĐO]. Theo thước 10/90 zh→vi, zero-shot cụm là 54,22 so 86,81 của Qwen3-4B tinh chỉnh, chênh khoảng 32,6, cùng cỡ chênh +30,14 đến +34,14 của B8 trên test (chưa đối chiếu chặt chẽ cùng tập mẫu). Chênh lệch kép của title bằng zero-shot: mục 8.5. Bản **pair không dùng được**: 511 khối lệch số dòng, 45% rỗng sau ghép, tỉ lệ độ dài 0,11, nên điểm 25,34 phản ánh việc mô hình nền không giữ định dạng nhiều dòng, không phải chất lượng dịch.

### 8.4. Phân tích bổ sung trên đầu ra GPU (P2, 19/11/2026)

Kế hoạch khóa trước khi chạy (`KE_HOACH_PHAN_TICH_P2.md`, sha256 `79422bf0e407bb4e38087fa4963847eb2d36c01350e744e35f4a5582a72ad421`); kết quả đầy đủ và kiểm tái lập ở `KET_QUA_P2.md`. Cả ba phân tích đều tái lập đúng số đã công bố trước khi báo số mới. Tóm tắt: (A) "hòa 4B" ở pair còn đúng cho thước chrF câu trung bình chính trên cả 3 seed, chỉ thước đó; (B) tỉ trọng title 35% thấp hơn 11% 0,39 điểm ở thước chính (title +1,19, mô tả −0,56), 25% không phân biệt được; (C) zero-shot cho thấy lợi thế sẵn có theo họ ở title (mục 8.5).

### 8.5. Chênh lệch kép của title bằng zero-shot (cụm, eval fold 0, 55 sản phẩm tham chiếu Qwen / 245 Gemma)

| | Nhóm Qwen | Nhóm Gemma | Qwen − Gemma |
|---|---:|---:|---:|
| Qwen3-4B zero-shot | 43,32 | 36,92 | **+6,40** |
| Qwen3-4B tinh chỉnh | 48,24 | 55,57 | −7,33 |
| Moses (học đủ) | 48,34 | 56,36 | −8,02 |
| OpenNMT | 39,26 | 49,03 | −9,77 |
| EcomGPT-7B | 43,02 | 46,83 | −3,81 |

Kép loại 1 (zero-shot trừ mốc, nhóm Qwen trừ nhóm Gemma): so EcomGPT +10,20 [+5,90; +14,37], so OpenNMT +16,17 [+11,91; +20,39], so Moses +14,41 [+10,05; +18,62] [ĐO]. Kép loại 2 (tinh chỉnh trừ zero-shot): −13,72 [−17,55; −9,55] [ĐO]. Đọc: tham chiếu do Qwen sinh **có lợi cho mô hình Qwen nền** ở title (ủng hộ lo ngại của hội đồng 1.2(d)(ii)); tinh chỉnh trên dữ liệu chủ yếu kiểu Gemma xóa phần lớn lợi thế này thay vì khuếch đại; ở bản pair, kép của 4B tinh chỉnh so OpenNMT và Moses vẫn dương (+5,17, +3,11), ở cụm thì chứa 0, nên tín hiệu cùng họ sau fine-tune không ổn định và chưa bác bỏ được. Giới hạn: nhóm Qwen chỉ 55 sản phẩm (18% title); Judge chọn Qwen khi bản Gemma kém nên hai nhóm khác độ khó (kép chỉ khử độ khó chung); chỉ title (mô tả không có sản phẩm Qwen chủ đạo); đã kiểm độ dài và định dạng bản zero-shot, không khác nhau giữa hai nhóm; còn thiếu tham chiếu người.

### 8.6. Ghi chú vận hành (để người đọc biết số này đến từ đâu)

Lần chạy Kaggle đầu (18/11) lỗi hết ở cuối epoch 1 do lỗi mã của người chạy, mất khoảng 5,7 giờ và không có điểm nào; đã sửa, thêm lưu checkpoint và kiểm thử trên máy và trên L4x4 thật, rồi chạy lại. Phiên 2 nối tiếp từ checkpoint phiên 1 qua tài khoản Kaggle khác, có kiểm mã băm checkpoint, pool và môi trường trước khi chạy. Bản zero-shot cụm lấy từ lần chạy đầu (không bị ảnh hưởng bởi lỗi).

Truy vết nguồn số (thư mục dưới `data/processed/ft_run/`, không đưa vào git):

| Việc | Chạy ở đâu | Thư mục output | Ghi chú kiểm |
|---|---|---|---|
| Zero-shot Qwen3-4B (cụm, eval fold 0) | Kaggle L4x4, `trinhnhatanh/nlp-tlc-song-song` (lần đầu, 3 việc huấn luyện lỗi) | `kaggle_tlc_song_song/zero_shot/` | pair lỗi định dạng nên không dùng |
| w_title 25%/35% và đối chứng 11%, phiên 1 | Kaggle L4x4, `trinhnhatanh/nlp-tlc-v2-that` | `kaggle_tlc_v2_that/` | dừng ở bước 679/681/679 trên 1.837; mã băm checkpoint trong `SHA256_checkpoint_phien1.txt` |
| w_title, phiên 2 (nối tiếp) | Kaggle L4x4, tài khoản 2, `honghuynanh/nlp-tlc-v2-tiep` | `kaggle_tlc_v2_tiep/` | 1.837/1.837 bước, rc 0 cả ba; mã băm kết quả trong `SHA256_ket_qua_phien2.txt` |
| OpenNMT seed 43, 44 | Colab T4 (mã `scripts/colab/onmt_colab.py`, `onmt_seed_vm.py`) | `colab_onmt/s43/`, `colab_onmt/s44/` | chấm bằng cùng bộ chấm như D3 |

Điểm trong mục 8.2 và `KET_QUA_P2.md` mục B được chấm lại trên máy từ `sinh_ep2.jsonl` (kiểm khớp báo cáo Kaggle đến 1e-6) bằng `scripts/phan_bien_v2/p5_w_title_ci.py`; mục A bằng `p4_hoa_4b_3seed.py`; mục C bằng `p6_kep_zero_shot.py`; kiểm thứ tự xác định bằng `p7_kiem_p1_xac_dinh.py`. Một seed cho mỗi mức w_title.
