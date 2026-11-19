# KẾT QUẢ P2: BA PHÂN TÍCH BỔ SUNG TRÊN ĐẦU RA GPU

Ngày 19/11/2026. Người làm: Claude. Chỉ eval fold 0 (train = fold 1–4), CPU, không train, không sinh bản dịch mới, không đọc test khóa. Kế hoạch khóa trước khi chạy: `KE_HOACH_PHAN_TICH_P2.md`, sha256 `79422bf0e407bb4e38087fa4963847eb2d36c01350e744e35f4a5582a72ad421`. Script: `scripts/phan_bien_v2/p4_hoa_4b_3seed.py`, `p5_w_title_ci.py`, `p6_kep_zero_shot.py`. Số thô: `data/processed/ft_run/phan_bien_v2/hoa_4b_3seed.json`, `w_title_ci.json`, `kep_zero_shot_cum.json`.

## 0. Kiểm tái lập (làm trước, đều đạt)

| Việc | Kiểm | Kết quả |
|---|---|---|
| A | OpenNMT seed 42 một mình so với 4B phải trùng `tong_hop_5mh{,_cum}/ket_qua.json` (điểm ước và hai đầu khoảng, chính và macro, pair và cụm) | lệch lớn nhất 0,00000 |
| B | Dựng lại mẫu bằng `kaggle_train.chon_mau_sinh` trên `pool_cum.jsonl` và chấm lại; `chrf_macro_huong` và 4 `chrf_corpus` theo nhóm phải trùng báo cáo Kaggle | lệch lớn nhất 0 ở cả 3 việc (75,954 / 75,933 / 75,898) |
| C | Chênh lệch kép của Qwen3-4B so với 3 mốc phải trùng `vong_tham_chieu_cum.json` của P1 (title) | điểm ước lệch 2e-15; hai đầu khoảng lệch tối đa 0,14 |

Về dòng cuối: P1 sắp thứ tự mã sản phẩm theo thứ tự bám của `set` Python, không xác định giữa các lần chạy, nên **hai đầu khoảng bootstrap của P1 có thể dao động khoảng ±0,1 giữa các lần chạy; điểm ước thì không đổi**. P2 sắp xếp mã sản phẩm nên xác định. Điểm ước của P1 không đổi. **Đã chạy lại `vong_tham_chieu` của P1 với thứ tự xác định** (`p7_kiem_p1_xac_dinh.py`, ghi `kiem_xac_dinh_vong_tham_chieu.json`, không ghi đè tệp P1): trên 36 khoảng, điểm ước lệch 5e-15, hai đầu dịch chuyển tối đa 0,40, **không khoảng nào đổi dấu** (chứa 0 hay không chứa 0 giữ nguyên). Khoảng đứng sát 0 nhất là chênh lệch kép title pair, Qwen3-4B trừ Moses: P1 ghi [+0,24; +5,75], xác định [+0,43; +5,99], vẫn không chứa 0. Khi trích, dùng giá trị xác định.

## A. "OpenNMT hòa Qwen3-4B" tính lại cho 3 seed

chrF **câu trung bình**, chênh OpenNMT trừ Qwen3-4B (4B lấy trung bình 2 seed), bootstrap theo sản phẩm 2.000 lần, cùng ma trận rút. Dấu `*` nghĩa là khoảng 95% **không** chứa 0.

| | Pair, chính (0,1 title + 0,9 mô tả, zh→vi) | Pair, macro | Cụm, chính | Cụm, macro |
|---|---|---|---|---|
| Seed 42 (D3) | +0,39 [−1,09; +1,90] | −8,33 [−9,40; −7,26] * | −5,57 [−6,42; −4,85] * | −12,42 [−13,29; −11,57] * |
| Seed 43 | +0,12 [−1,29; +1,53] | −9,03 [−10,10; −7,94] * | −5,56 [−6,50; −4,75] * | −12,97 [−13,85; −12,13] * |
| Seed 44 | +0,37 [−1,03; +1,83] | −8,57 [−9,64; −7,54] * | −5,44 [−6,29; −4,69] * | −12,84 [−13,67; −12,04] * |
| **Trung bình 3 seed theo mẫu** | **+0,30 [−1,07; +1,71]** | **−8,65 [−9,66; −7,64] \*** | **−5,52 [−6,39; −4,77] \*** | **−12,75 [−13,56; −11,95] \*** |

[ĐO] Theo luật khóa trước: khoảng của chênh "chính" pair (trung bình 3 seed) **chứa 0**, nên câu "hòa" ở pair được giữ, nhưng **chỉ đúng cho thước này** (chrF câu trung bình, chính 0,1/0,9, zh→vi): cả 3 seed đều cho khoảng chứa 0, điểm ước từ +0,12 đến +0,39. Không có kết luận phụ thuộc seed. Ở **macro** (có vi→zh và title), mỗi seed và trung bình đều kém từ 8,3 đến 9,0 điểm, khoảng không chứa 0. Ở **cụm**, OpenNMT kém ở cả hai thước, khoảng không chứa 0 ở mọi seed. Nối với chrF corpus ở mục 8.1 của `TRA_LOI_V2`: OpenNMT kém 5,3 (pair) và 6,2 (cụm) điểm; "hòa" chỉ còn ở một thước câu TB trên pair, không phải ở chrF corpus. Giới hạn: Qwen3-4B 2 seed; OpenNMT vẫn là cấu hình 4 lớp/256, chưa phải recipe chuẩn.

## B. Khoảng tin cậy cho tỉ trọng title 25% và 35% so với 11% (cụm, fold 0, 1 seed)

chrF **câu trung bình**, bootstrap ghép cặp theo mã sản phẩm (2.000 lần, `default_rng(42)`, cùng ma trận rút). `*`: khoảng không chứa 0. Điểm câu trung bình, chính / macro: 25% = 90,741 / 76,859; 35% = 90,547 / 76,953; 11% = 90,931 / 76,685.

| Cặp | Chính (0,1/0,9 zh→vi) | Macro | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả |
|---|---|---|---|---|---|---|
| 25% − 11% | −0,19 [−0,41; +0,03] | +0,17 [−0,33; +0,70] | −0,18 [−1,14; +0,79] | −0,19 [−0,43; +0,03] | +1,19 [−0,62; +2,98] | −0,13 [−0,41; +0,12] |
| 35% − 11% | **−0,39 [−0,68; −0,09] \*** | +0,27 [−0,24; +0,78] | **+1,19 [+0,22; +2,25] \*** | **−0,56 [−0,89; −0,25] \*** | +0,64 [−1,15; +2,43] | −0,18 [−0,42; +0,05] |
| 35% − 25% | −0,19 [−0,44; +0,03] | +0,10 [−0,26; +0,45] | **+1,36 [+0,45; +2,32] \*** | **−0,37 [−0,63; −0,13] \*** | −0,55 [−1,55; +0,42] | −0,05 [−0,28; +0,20] |

[ĐO] Phát hiện chính, và **sửa một câu em đã viết trước đó** ("chênh trong nhiễu"): ở 35%, title zh→vi **tăng** +1,19 so với 11% nhưng mô tả zh→vi **giảm** −0,56, cả hai khoảng không chứa 0; vì thước chính cho mô tả 0,9 nên tổng "chính" **thấp hơn 0,39** điểm so với 11%, khoảng không chứa 0. Đây là đánh đổi title/mô tả đúng như trọng số dự đoán: tăng tỉ trọng title đổi một phần chất lượng mô tả lấy chất lượng title, và dưới thước 10/90 thì không lời. Macro (title và mô tả có trọng số ngang nhau) không phân biệt được ở cả ba cặp. Ở 25% không có khoảng nào đứng ngoài 0. Giới hạn: một seed (khoảng chỉ phủ dao động chọn mẫu eval, không phủ dao động khởi tạo); không hiệu chỉnh đa so sánh (3 cặp × 6 chỉ số); title ở 35% tăng nhưng ở 25% không tăng, nên chưa nói được đường cong đơn điệu; chỉ bản cụm. Kết luận được phép: "dưới thước 10/90, 11% không kém 25% và hơn 35% một chút (0,4 điểm); dưới thước ngang title/mô tả thì ba mức không phân biệt được; tăng tỉ trọng title cải thiện title zh→vi nhưng đổi bằng mô tả". Không dùng chữ "tối ưu".

## C. Chênh lệch kép của title bằng zero-shot Qwen3-4B (cụm, title zh→vi)

Nhóm theo `judge_chon_title`: 55 sản phẩm tham chiếu Qwen, 245 tham chiếu Gemma. Điểm trung bình chrF corpus của title theo nhóm:

| Mô hình | Nhóm Qwen | Nhóm Gemma | Qwen − Gemma |
|---|---:|---:|---:|
| **Qwen3-4B zero-shot** | **43,32** | **36,92** | **+6,40** |
| Qwen3-4B tinh chỉnh | 48,24 | 55,57 | −7,33 |
| Moses (học đủ) | 48,34 | 56,36 | −8,02 |
| OpenNMT | 39,26 | 49,03 | −9,77 |
| EcomGPT-7B | 43,02 | 46,83 | −3,81 |

Chênh lệch kép loại 1 (thiên vị sẵn có, zero-shot trừ mốc, nhóm Qwen trừ nhóm Gemma):

| Mốc | Nhóm Qwen (n=55) | Nhóm Gemma (n=245) | Kép [95%] |
|---|---:|---:|---:|
| EcomGPT-7B | +0,30 | −9,91 | **+10,20 [+5,90; +14,37] \*** |
| OpenNMT | +4,06 | −12,11 | **+16,17 [+11,91; +20,39] \*** |
| Moses (học đủ) | −5,03 | −19,43 | **+14,41 [+10,05; +18,62] \*** |

Chênh lệch kép loại 2 (phần do fine-tune, tinh chỉnh trừ zero-shot): nhóm Qwen +4,92, nhóm Gemma +18,65, kép **−13,72 [−17,55; −9,55] \***.

[ĐO] Mọi mô hình khác (kể cả bản tinh chỉnh) điểm **thấp hơn** ở nhóm tham chiếu Qwen (nhóm khó hơn, như P1 đã ghi), riêng **zero-shot Qwen3-4B điểm cao hơn 6,4 điểm** ở nhóm tham chiếu Qwen. Theo luật khóa trước, kép loại 1 dương và khoảng không chứa 0 với cả 3 mốc: **zero-shot Qwen3-4B có lợi thế sẵn có ở các title có tham chiếu do Qwen sinh, không do fine-tune**; đây là bằng chứng ủng hộ lo ngại self-preference của biên bản (1.2(d)(ii)) cho title, mạnh hơn P1 (P1 chưa có zero-shot nên chỉ kết luận "không nhất quán"). Loại 2 âm: fine-tune nâng nhóm Gemma nhiều hơn nhóm Qwen (+18,65 so +4,92), nên **fine-tune không khuếch đại lợi thế này, mà xóa phần lớn nó** (sau tinh chỉnh, 4B có cùng kiểu với Moses và OpenNMT: nhóm Qwen khó hơn); điều này nhất quán với việc dữ liệu huấn luyện chủ yếu mang kiểu Gemma (245/300 sản phẩm). Tuy vậy, ở bản pair của P1, kép của 4B tinh chỉnh so OpenNMT (+5,17) và Moses (+3,11) vẫn dương và không chứa 0; ở cụm thì chứa 0. Vậy sau fine-tune, tín hiệu cùng họ ở title **không ổn định giữa hai bản dữ liệu**, không thể bác bỏ.

Kiểm nhanh định dạng (để loại một giải thích dễ): độ dài bản dịch zero-shot theo ký tự trung bình 131,7 (nhóm Qwen) và 130,8 (nhóm Gemma), tỉ lệ độ dài với tham chiếu 1,17 và 1,16; không có xuống dòng ở cả hai nhóm; chữ "translation/dịch" lọt vào 1,8% và 0,8% bản dịch. Nên khác biệt giữa hai nhóm không do lỗi định dạng của zero-shot.

Giới hạn: Judge chọn Qwen khi bản Gemma kém, nên hai nhóm khác độ khó (chênh lệch kép chỉ khử phần độ khó chung cho mọi mô hình); nhóm Qwen chỉ 55 sản phẩm (18% title); chỉ bản cụm (pair zero-shot không dùng được); mô tả không phân tích được (eval fold 0 không có sản phẩm Qwen chủ đạo); điểm là độ giống bản dịch máy chứ chưa có tham chiếu người. Việc còn mở: tham chiếu độc lập do người (100–300 mẫu) vẫn là cách duy nhất trả lời dứt điểm.
