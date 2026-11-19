# KẾT QUẢ P1: PHÂN TÍCH LẠI TRÊN EVAL FOLD 0

Ngày 17/11/2026. Người làm: OpenCode. Chỉ dùng eval fold 0 (train = fold 1–4). Không đọc test, không huấn luyện, không sinh bản dịch mới. Kế hoạch khóa trước khi xem số: `KE_HOACH_PHAN_TICH_P1.md`, sha256 `3743d3eeabcf30ab52a7d2f12cb6ec2aaa3ae7f9634e42c7be82100da81e1ef0`. Số thô: `data/processed/ft_run/phan_bien_v2/*.json`; sha256 của 52 tệp đầu vào đã đọc: `sha256_dau_vao.json` (không có test).

## 0. Tái lập 5 mô hình (3.1)

Chạy lại toàn bộ phép chấm của `tong_hop_5mh.py` trên cả `pair` và `cum`, ghi ra thư mục mới. So từng ô `tb_chinh`, `tb_macro`, `chrf_corpus` với `tong_hop_5mh{,_cum}/ket_qua.json`.

| Bộ | Số ô so | Lệch lớn nhất | Kết luận |
|---|---:|---:|---|
| pair | 48 | 0,00000 | khớp tuyệt đối |
| cum | 48 | 0,00000 | khớp tuyệt đối |

[ĐO] Tái lập đúng từng ô, nên các bảng dưới dùng cùng pipeline chấm với bảng gốc.

## 1. Dòng khuôn và dòng mới (3.2)

Định nghĩa "dòng đã thấy": cặp `(category, dòng nguồn đã chuẩn hoá)` có trong train fold 1–4, đúng `cham_diem.xay_tap_da_thay`. Chỉ tính sản phẩm khớp số dòng (`can_dong`).

### 1.1. Tỉ lệ dòng mô tả đã thấy

| Bộ | Hướng | Theo dòng | Theo sản phẩm | Số dòng | Số sản phẩm |
|---|---|---:|---:|---:|---:|
| pair | zh→vi | **85,91%** | 86,49% | 298 | 285 |
| pair | vi→zh | 75,67% | 76,25% | 300 | 280 |
| cum | zh→vi | **86,06%** | 83,84% | 7.128 | 296,5 |
| cum | vi→zh | 73,92% | 75,36% | 2.093,5 | 298,5 |

[ĐO] Với pair, mỗi mục mô tả là một dòng thuộc tính, nên số dòng bằng số mục và số sản phẩm là số mã sản phẩm duy nhất (285 và 280). Với cum, mỗi mục là cả khối, số dòng là số mảnh sau khi tách `"; "`. Số dòng và số sản phẩm của cum là trung bình 2 seed vì số sản phẩm `can_dong` thay đổi theo bản dịch. Con số 85,91% của pair zh→vi khớp với 85,0% của A7 (test) và thay cho con số 87% ước lượng trước đây trong doc 09; với cum, tỉ lệ tương đương 86,06%. Kiểm thêm ref với nguồn: 0 lệch ở pair, 1/300 mục ở cum zh→vi do dấu `";"` nằm trong giá trị; `can_dong` hiện hành chỉ kiểm hyp=ref, nên 1 mục cum zh→vi không đạt đủ điều kiện ref=nguồn.

Số sản phẩm bị loại vì `can_dong=False`: pair 0 ở mọi mô hình; cum zh→vi, EcomGPT-7B bị loại 35/298 và vi→zh 9/300, các mô hình còn lại 0 đến 1,5; vì vậy điểm EcomGPT ở cum chỉ tính trên 263 sản phẩm (zh→vi) và 291 sản phẩm (vi→zh), không cùng tập với các mô hình khác. Đây là khác biệt phải đọc kèm khi so EcomGPT ở cum.

### 1.2. chrF trên dòng khuôn và dòng mới, pair

| Mô hình | Khuôn zh→vi | Mới zh→vi | Chênh mới so 4B [95%] | Khuôn vi→zh | Mới vi→zh | Chênh mới so 4B [95%] |
|---|---:|---:|---|---:|---:|---|
| Qwen3-4B | 96,80 | 63,41 | mốc | 97,32 | 78,42 | mốc |
| Qwen3-8B | 97,00 | 54,38 | −0,43 [−3,98; +2,86] | 97,75 | 62,34 | +0,38 [−3,70; +3,78] |
| EcomGPT-7B | 94,37 | 40,19 | −12,70 [−17,70; −8,09] | 94,46 | 57,24 | −9,69 [−14,03; −5,48] |
| OpenNMT | 98,79 | 44,88 | **−8,29 [−15,49; −0,88]** | 91,67 | 42,81 | **−17,76 [−25,85; −9,90]** |
| Moses (học đủ) | 98,85 | 61,53 | −2,50 [−8,11; +3,02] | 98,71 | 69,74 | −2,78 [−7,24; +1,83] |
| Moses (khớp 4M) | 97,43 | 60,18 | −4,76 [−10,34; +0,89] | 97,31 | 66,48 | −7,10 [−12,27; −2,11] |
| 4B công thức cuối | 99,45 | 65,05 | +4,44 [+0,51; +8,87] | 99,05 | 84,96 | +8,08 [+4,20; +12,64] |

Chú thích thước đo (thêm 19/11): cột "Chênh mới so 4B [95%]" là chênh của chrF tính **theo từng sản phẩm** (chrF corpus trên các dòng mới của sản phẩm đó) rồi lấy trung bình, bootstrap theo sản phẩm; cột "Mới" là chrF corpus gộp mọi dòng mới. Hai thước cùng chiều nhưng khác độ lớn (OpenNMT pair zh→vi: −18,5 theo corpus gộp, −8,29 theo sản phẩm).

Đọc: trên dòng khuôn, OpenNMT **hơn** 4B (+2,12 zh→vi; −5,07 vi→zh), nhưng trên dòng mới lại kém −8,29 và −17,76. Moses học đủ cũng từ +2,26 trên khuôn xuống −2,50 trên dòng mới (zh→vi). Khoảng cách tăng đúng giả thuyết. Số sản phẩm có dòng mới nhỏ (42 zh→vi, 71 vi→zh) nên CI rộng.

### 1.3. chrF trên dòng khuôn và dòng mới, cum

| Mô hình | Khuôn zh→vi | Mới zh→vi | Chênh mới so 4B [95%] | Khuôn vi→zh | Mới vi→zh | Chênh mới so 4B [95%] |
|---|---:|---:|---|---:|---:|---|
| Qwen3-4B | 97,53 | 72,70 | mốc | 98,27 | 72,00 | mốc |
| Qwen3-8B | 97,97 | 74,75 | **+1,83 [+0,85; +2,89]** | 98,30 | 75,79 | **+2,54 [+1,07; +4,10]** |
| EcomGPT-7B | 96,13 | 61,22 | −8,17 [−9,55; −6,76] | 97,13 | 68,89 | −4,55 [−6,87; −2,32] |
| OpenNMT | 96,10 | 56,98 | **−10,07 [−11,68; −8,30]** | 91,23 | 47,56 | **−22,06 [−25,48; −18,59]** |
| Moses (học đủ) | 96,87 | 67,71 | −3,68 [−5,05; −2,22] | 98,97 | 63,95 | −6,77 [−9,30; −4,33] |
| Moses (khớp 4M) | 95,51 | 61,34 | −8,60 [−10,05; −7,11] | 97,91 | 58,32 | −11,47 [−13,93; −8,84] |
| 4B công thức cuối | 99,07 | 79,82 | +5,47 [+4,34; +6,72] | 99,51 | 78,21 | +5,43 [+3,55; +7,53] |

Đọc: cùng chiều, khoảng cách NMT so với 4B lớn hơn nhiều trên dòng mới. Ở cum, Qwen3-8B **hơn** 4B trên dòng mới ở cả hai hướng, khác pair (hòa).

**Kết luận 3.2:** giả thuyết ghi trước **đúng**. Điểm mô tả toàn phần của OpenNMT/Moses bị thổi phồng vì phần lớn dòng là khuôn; trên dòng mới, khoảng cách tới Qwen3-4B là 8–22 điểm, không phải 0–5 điểm. Đây là câu trả lời trực tiếp cho 1.2(c) và 1.3 mục 3.

## 2. Vòng lặp tham chiếu (3.3)

Nhóm theo nguồn tham chiếu: title theo `judge_chon_title`; mô tả theo tỉ phần mảnh Qwen trong `judge_chon_description` (Qwen ≥ 2/3; Gemma ≤ 1/3).

- Title trong eval zh→vi: **55 sản phẩm Qwen, 245 Gemma** (giống nhau ở pair và cum vì cùng title).
- Mô tả trong eval zh→vi: **285 sản phẩm, tất cả đều Gemma chủ đạo** ở pair (cum: 297 Gemma, 1 không xác định). **Không có sản phẩm Qwen chủ đạo**, nên chênh lệch kép theo kế hoạch cho mô tả **không tính được**. Đây là kết quả, không phải lỗi.

### 2.1. Chênh lệch kép trên title zh→vi

| Cặp | Nhóm Qwen: n, chênh | Nhóm Gemma: n, chênh | Kép [95%] |
|---|---|---|---|
| Qwen3-4B − EcomGPT-7B | 55, +5,15 | 245, +12,83 | **−7,68 [−10,22; −5,00]** |
| Qwen3-4B − OpenNMT | 55, +5,92 | 245, +0,75 | **+5,17 [+2,02; +8,29]** |
| Qwen3-4B − Moses học đủ | 55, −2,14 | 245, −5,25 | **+3,11 [+0,24; +5,75]** |
| Qwen3-8B − EcomGPT-7B | 55, +7,05 | 245, +14,77 | **−7,72 [−10,33; −5,03]** |
| Qwen3-8B − OpenNMT | 55, +7,81 | 245, +2,69 | **+5,12 [+2,18; +8,25]** |
| Qwen3-8B − Moses học đủ | 55, −0,24 | 245, −3,31 | **+3,07 [+0,19; +5,96]** |

Ghi chú (thêm 19/11): hai đầu khoảng ở các bảng mục 2 là của một lần chạy P1; bootstrap của P1 rút theo thứ tự bám của `set` Python nên hai đầu lệch tối đa 0,40 giữa các lần chạy (điểm ước không đổi, không khoảng nào đổi dấu; `phan_bien_v2/kiem_xac_dinh_vong_tham_chieu.json`, `scripts/phan_bien_v2/p7_kiem_p1_xac_dinh.py`). Giá trị xác định để trích: `docs/phan_bien_vong2/TRA_LOI_V2_NHANH_NHAT_ANH.md` mục 2.

Cum title, cùng cách tính:

| Cặp | Kép [95%] |
|---|---:|
| Qwen3-4B − EcomGPT-7B | −3,52 [−5,87; −1,12] |
| Qwen3-4B − OpenNMT | +2,45 [−0,83; +5,59] |
| Qwen3-4B − Moses học đủ | +0,69 [−2,00; +3,31] |
| Qwen3-8B − EcomGPT-7B | −4,38 [−7,19; −1,56] |
| Qwen3-8B − OpenNMT | +1,59 [−1,43; +4,70] |
| Qwen3-8B − Moses học đủ | −0,17 [−2,81; +2,54] |

[ĐO] Kết quả **không nhất quán theo họ**: so với EcomGPT (BLOOM), lợi thế của Qwen **nhỏ hơn** ở nhóm tham chiếu Qwen (kép âm); so với OpenNMT/Moses, lợi thế **lớn hơn** (kép dương). Vì vậy không thể nói "Qwen được lợi vì tham chiếu cùng họ" như một quy luật chung. Cần đọc kèm độ khó nhóm: Judge chọn Qwen khi Gemma kém, nên nhóm Qwen là nhóm khó hơn cho mọi mô hình.

### 2.2. Phân tích thêm sau khi xem số (không thay kế hoạch): mô tả, nhóm "có mảnh Qwen" so "không có"

| Cặp | Có Qwen: n, chênh | Không Qwen: n, chênh | Kép [95%] |
|---|---|---|---|
| 4B − EcomGPT | 170, +3,50 | 115, +4,73 | −1,23 [−3,79; +1,21] |
| 4B − OpenNMT | 170, −0,99 | 115, −0,06 | −0,92 [−4,19; +2,52] |
| 4B − Moses | 170, −1,39 | 115, −1,72 | +0,33 [−2,42; +3,09] |
| 8B − EcomGPT | 170, +3,86 | 115, +4,97 | −1,10 [−3,78; +1,42] |
| 8B − OpenNMT | 170, −0,63 | 115, +0,17 | −0,80 [−3,86; +2,17] |
| 8B − Moses | 170, −1,03 | 115, −1,49 | +0,46 [−2,08; +2,95] |

[ĐO] Cả 6 khoảng đều chứa 0: **không thấy lợi thế cùng họ** trên mô tả ở mức nhóm "có mảnh Qwen". Giới hạn: nhóm này không phải "Qwen chủ đạo"; Judge chỉ chọn Qwen cho một phần mảnh. Ở cum, nhóm "có mảnh Qwen" chỉ có 7 sản phẩm, không đủ để kết luận; bảng trên chỉ trình bày pair (170 so 115).

### 2.3. Zero-shot Qwen3-4B: không tính được

**[CHƯA KIỂM]** Bản dịch zero-shot Qwen3-4B đầy đủ trên eval fold 0 không có trên máy. Các tệp zero-shot hiện có là `kaggle_cheo_that/cheo/sinh_zero_shot_{pair,cum}.jsonl`, 600 mục mỗi tệp, **toàn bộ là mục mô tả dạng khối** (id `..._mo_ta`), không có id title của eval; mã sản phẩm trùng 139/1.082 nhưng khác đơn vị đo. Vì vậy không thể lập nhóm Qwen/Gemma cho title và không thể tính chênh lệch kép zero-shot theo kế hoạch. Không dùng tập con này để thay thế vì sẽ đổi đơn vị và mất tính so sánh.

**Kết luận 3.3:** với title, có tín hiệu lệch theo nhóm nhưng **ngược chiều nhau tuỳ baseline**, nên không kết luận self-preference; zero-shot 4B chưa kiểm được vì thiếu bản dịch đầy đủ trên eval fold 0; với mô tả, eval fold 0 **không có nhóm Qwen chủ đạo** để kiểm, và nhóm "có mảnh Qwen" cho kết quả hòa. Cần tham chiếu người (100–300 mẫu) mới trả lời dứt điểm.

## 3. Theo độ dài (3.4)

Tứ phân vị độ dài ký tự nguồn, tính riêng từng bộ × hướng × loại trên eval. Bảng dưới là mô tả zh→vi (Q1 ngắn nhất, Q4 dài nhất).

**Pair, mô tả zh→vi**

| Mô hình | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Qwen3-4B | 95,8 | 98,3 | 95,1 | **72,2** |
| Qwen3-8B | 96,8 | 98,0 | 94,9 | 65,3 |
| EcomGPT-7B | 91,5 | 95,4 | 93,4 | 53,8 |
| OpenNMT | 97,8 | 99,6 | 98,3 | **58,5** |
| Moses học đủ | 98,4 | 98,3 | 97,1 | 71,2 |
| 4B công thức cuối | 99,5 | 100,0 | 97,4 | 73,9 |

**Cum, mô tả zh→vi**

| Mô hình | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Qwen3-4B | 94,5 | 90,3 | 91,6 | 88,2 |
| Qwen3-8B | 94,8 | 91,7 | 92,5 | 89,1 |
| EcomGPT-7B | 91,6 | 84,7 | 83,9 | 77,3 |
| OpenNMT | 91,8 | 84,8 | 85,5 | 80,7 |
| Moses học đủ | 93,8 | 88,8 | 89,2 | 86,0 |
| 4B công thức cuối | 96,5 | 93,7 | 94,8 | 91,3 |

**Pair, title zh→vi và mô tả vi→zh**

| Mô hình | Title Q1 | Title Q2 | Title Q3 | Title Q4 | Mô tả Q1 | Mô tả Q2 | Mô tả Q3 | Mô tả Q4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 49,5 | – | 50,4 | 50,5 | 98,3 | 96,8 | 95,1 | 79,8 |
| Qwen3-8B | 51,1 | – | 53,1 | 52,8 | 97,7 | 98,5 | 95,5 | 64,6 |
| EcomGPT-7B | 37,5 | – | 40,9 | 37,3 | 96,2 | 93,8 | 91,0 | 59,6 |
| OpenNMT | 46,8 | – | 49,7 | 49,7 | 91,9 | 92,6 | 86,5 | 46,5 |
| Moses học đủ | 54,1 | – | 56,4 | 54,9 | 98,9 | 94,6 | 97,2 | 72,4 |
| 4B công thức cuối | 61,7 | – | 62,6 | 63,7 | 100,0 | 98,5 | 99,6 | 85,5 |

**Cum, title zh→vi và mô tả vi→zh**

| Mô hình | Title Q1 | Title Q2 | Title Q3 | Title Q4 | Mô tả Q1 | Mô tả Q2 | Mô tả Q3 | Mô tả Q4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 53,5 | – | 55,6 | 53,7 | 97,1 | 93,9 | 94,4 | 77,4 |
| Qwen3-8B | 56,4 | – | 56,7 | 56,4 | 98,0 | 94,9 | 94,4 | 81,9 |
| EcomGPT-7B | 44,6 | – | 47,2 | 46,8 | 97,0 | 93,9 | 91,1 | 70,7 |
| OpenNMT | 45,3 | – | 48,6 | 48,9 | 84,6 | 83,2 | 83,5 | 63,0 |
| Moses học đủ | 54,1 | – | 56,4 | 54,9 | 95,5 | 93,0 | 91,8 | 74,1 |
| 4B công thức cuối | 62,4 | – | 62,7 | 62,3 | 98,9 | 96,8 | 95,2 | 84,0 |

Đọc: với mô tả zh→vi, mọi mô hình đều giảm ở nhóm dài, nhưng cấu hình OpenNMT rơi mạnh hơn ở pair Q4 (58,5 so 72,2 của 4B, tức −13,7 điểm) trong khi ở Q1 lại hơn 4B (+2,0). Đây là bằng chứng định lượng cho "cấu hình NMT nhỏ kém ở câu dài", không phải kết luận về cả họ NMT. Với **title zh→vi**, không có xu hướng giảm theo độ dài: 4B khoảng 49,5 đến 50,5, 8B khoảng 51,1 đến 53,1; độ dài không phải trục chính của title. Với **mô tả vi→zh**, Q4 giảm ở mọi mô hình; riêng pair, 8B Q4 64,6 thấp hơn 4B 79,8. Với **title vi→zh**, OpenNMT phẳng ở 27,2 đến 31,8 qua cả bốn nhóm, nên vấn đề của cấu hình này ở hướng đó không phải độ dài mà là toàn dải. Ô Q2 của title zh→vi để trống vì tứ phân vị 25% là 30 và 50% là 30,5, không có độ dài nguyên nào nằm trong khoảng (30; 30,5]; đây là đặc điểm của thang độ dài rời rạc, không phải thiếu số.

## 4. Đường học OpenNMT (3.5)

Nguồn: `kaggle_onmt_that/onmt/{zh-vi,vi-zh}_train/train.log`. Hình: `docs/phan_bien_vong2/hinh/opennmt_zhvi.png`, `opennmt_vizh.png`.

| Hướng | Số điểm train | Số điểm dev | PPL dev đầu → cuối | Acc dev đầu → cuối | Dừng sớm |
|---|---:|---:|---|---|---|
| zh→vi | 51 | 9 | 19,05 → 10,82 (thấp nhất 10,62 ở bước 7.959) | 65,73 → 79,58 | bước 5.685, dừng ở 10.233 |
| vi→zh | 28 | 20 | 204,15 → 32,66 | 30,70 → 66,39 | không; chạy đủ 5.620 bước/20 epoch |

[ĐO] zh→vi hội tụ rồi dừng sớm; vi→zh vẫn còn giảm chậm ở cuối, phù hợp với việc vi→zh yếu hơn (ít dữ liệu hơn, 61.401 dòng so 269.895). Log không ghi "best step" cho vi→zh; ghi đúng như vậy, không suy ra. Checkpoint dùng để dịch là trung bình 5 checkpoint (`avg.pt`; `trung_binh_checkpoint=True` trong `onmt_report.json`).

## 5. Compute và hợp nhất OpenNMT (3.6)

### 5.1. Compute 5 mô hình

| Mô hình | Tham số | Pretraining | Fine-tune | Lượt | Bước | GPU-giờ |
|---|---|---|---:|---:|---:|---:|
| Qwen3-4B | 4,0B (LoRA 132.120.576) | ~36.000B token **[CHƯA KIỂM]**, báo cáo Qwen3 | 4,0M token (sàng lọc, 2 seed) | 1 lượt 4M token (2 seed) | 241 | 0,19 |
| Qwen3-8B | 8,2B | ~36.000B token **[CHƯA KIỂM]** | 4,0M token | 1 lượt 4M token (2 seed) | 241 | 0,37 |
| EcomGPT-7B | 7B (BLOOMZ-7b1-mt) | ~366B token **[CHƯA KIỂM]**, báo cáo BLOOM | 3,7M token | 1 lượt 3,7M token (2 seed) | 191 | 0,33 |
| OpenNMT | 37M **[theo biên bản, chưa đối chiếu report]** | 0 | 4,71M token SP; 269.895 + 61.401 dòng | ≤20 epoch; zh→vi dừng sớm, vi→zh đủ 20 | 22.740 + 5.620 | 1,08 |
| Moses | không có tham số học sâu | 0 | 269.895 + 61.401 dòng; 2 biến thể | MERT 10 lượt | — | 2,93 |

[ĐO] Số token/bước/giây lấy từ report của từng phiên. Pretraining là số công bố, chưa kiểm ngoại tuyến. Đây **không phải so sánh compute-matched**: LLM có pretraining hàng chục nghìn tỷ token, NMT/Moses = 0.

### 5.2. Hợp nhất hai baseline OpenNMT

| | Nhánh Nhật Anh (D3) | Nhánh Quang (theo biên bản) |
|---|---|---|
| Tham số | 37M | 37M |
| Lớp / hidden | 4 / 256 | 4 / 256 |
| Tokenizer | SentencePiece unigram 16k mỗi ngôn ngữ | char-level 32k |
| Huấn luyện | Noam 2e-3, ≤20 epoch, dừng sớm, trung bình 5 checkpoint | 30k step A / 22.990 B |
| Giải mã | beam 5 | beam 512 + `replace_unk` |
| Dữ liệu | dòng của bản pair | A/B (block) |
| Seed | 1 | không ghi |
| Tập đo | eval fold 0 | eval_F |
| Thước | chrF | COMET-da/QE 0,349 |

[ĐO] Hai tập đo và hai thước đo khác nhau, **không so điểm trực tiếp**; bảng này chỉ so cấu hình. Cùng 37M/4 lớp/256 hidden nhưng khác tokenizer, beam và dữ liệu, đủ để thấy mức kém của cấu hình NMT nhỏ học từ đầu là kết luận về **cấu hình**, không phải về paradigm NMT.

## 6. Giới hạn

- Eval fold 0 là tập đã dùng để chọn cấu hình (B6, tỉ trọng title, 5 mô hình, adapter demo), nên số trên fold 0 bị ảnh hưởng bởi việc chọn. Tỉ trọng title có kiểm độc lập ở fold 1–2; 5 mô hình thì chưa.
- Pair chỉ có 42 (zh→vi) và 71 (vi→zh) sản phẩm có ít nhất một dòng mới, nên CI của chênh trên dòng mới rộng.
- Mô tả eval fold 0 không có sản phẩm Qwen chủ đạo, nên câu hỏi self-preference cho mô tả chưa trả lời được; zero-shot Qwen3-4B không có bản dịch đầy đủ trên eval fold 0.
- Bootstrap chỉ phủ dao động chọn mẫu eval, không phủ dao động khởi tạo; LLM lấy trung bình 2 seed, NMT/Moses 1 seed. Đơn vị bootstrap là mã sản phẩm, không phải dòng.
- 1/300 mục cum zh→vi có số mảnh ref khác nguồn do dấu `";"` trong giá trị; `can_dong` hiện hành chỉ kiểm hyp=ref. Ở cum, EcomGPT-7B bị loại 35/298 sản phẩm (zh→vi) và 9/300 (vi→zh) vì lệch số dòng, nên điểm của nó không cùng tập với các mô hình khác.
- Chưa có tham chiếu người; mọi điểm là độ giống bản dịch máy.
