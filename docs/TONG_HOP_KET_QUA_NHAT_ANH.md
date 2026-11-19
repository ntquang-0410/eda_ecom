# Tổng hợp kết quả hướng Nhật Anh (tới 16/11/2026)

Viết cho Quang và nhóm để nắm tình hình mà không cần mở máy của Nhật Anh. Mọi số trong file này chép từ file kết quả đã tải về từ Kaggle (thư mục `data/processed/ft_run/`, **không có trong git** vì `.gitignore` loại `data/processed/`). File nào muốn xem tận nơi thì hỏi Nhật Anh.

Nhãn: **[ĐO]** đo trên dữ liệu thật; **[ƯỚC]** ước tính; **[CHỜ]** chưa chốt. Số lẻ trong bảng dùng dấu chấm.

**Lưu ý chung cho mọi điểm số:** tham chiếu là **bản dịch máy** (Gemma/Qwen đã qua Judge), chưa có người chấm hay LLM chấm so cặp. chrF ở đây đo độ giống tham chiếu đó, không phải chất lượng tuyệt đối. Số của hướng này và của hướng Quang **không so thẳng được** (khác tập đo, khác thước đo), xem mục 13.

---

## 0. Đọc nhanh

1. **Dữ liệu:** 25.575 sản phẩm (1688: 14.837, Tiki: 10.738) sau lọc luật; bản dịch chọn giữa Gemma và Qwen bằng Judge DeepSeek; hai dạng **cụm** và **pair**; HF `ntquang0410/zh-vie_ecom`, commit `82aa9c9`. Chia tập theo sản phẩm: pool 23.017 (5 fold) + **test khóa 2.558**.
2. **Mô hình chính:** Qwen3-4B + LoRA rank 64, alpha 128, lr 2e-4, 2 epoch, `max_len` 2.048, tất cả lớp tuyến tính.
3. **Trọng số đóng góp title : mô tả (đã chốt):** title chiếm **10,74%** (cụm) / **10,86%** (pair) loss theo token đích; `w_title` bản cuối **0,17247** (cụm) / **0,16915** (pair).
4. **Test khóa, đo MỘT lần (07/11):** fine-tune thắng zero-shot ở cả 4 nhánh × hướng, chênh chrF +30 đến +34 (cận dưới 95% xa 0). chrF fine-tune: cụm vi→zh 75,63, zh→vi 77,36; pair vi→zh 75,41, zh→vi 77,65.
5. **Năm mô hình (fold 0, tập dùng để chọn cấu hình, cấu hình sàng lọc rank 16, 4M token, chưa đo test):**
   - Cụm: Qwen3-8B hơn 4B **+1,03 [+0,71; +1,36]**.
   - Pair: 8B so với 4B **+0,44 [−0,24; +1,14]**, tức hòa. Moses học đủ dữ liệu hơn 4B +1,86, nhưng khi khớp ngân sách 4M token thì hòa (+0,05).
   - EcomGPT-7B kém rõ ở cả hai dạng; OpenNMT kém rõ ở dạng cụm. Trên **dòng mô tả mới** (P1): OpenNMT −8,29 (pair zh→vi) và −10,07 (cụm zh→vi); EcomGPT −12,70 và −8,17.
6. **Bản cuối đã huấn luyện:** `cum_cuoi`, `pair_cuoi` trên toàn pool, đã loại 6.386 title bị gắn cờ. **Chưa đo trên test** (không còn dữ liệu giữ lại để đo).
7. **Demo:** extension Chrome + máy chủ dịch. Trên laptop (RTX 3050 4 GB) bản nén **Q5_K_M + LoRA đạt cổng**, Q4_K_M bị loại. Thử 13 trang 1688 thật: 13/13 đạt kiểm cấu trúc, đọc tay 8 trang, không có tham chiếu.
8. **Chưa làm:** 8B theo công thức cuối (mất GPU RTX Pro 6000), đo test cho năm mô hình, B9 (LLM chấm so cặp), người chấm.

---

## 1. Chỗ chứa

| Thứ | Ở đâu |
|---|---|
| Dữ liệu sau Judge | HF `ntquang0410/zh-vie_ecom`, thư mục `data/LLM/bilingual_judged`, commit `82aa9c9` (hash ở mục 2.5) |
| Mã | `scripts/ft_clean/`, `scripts/judge/`, `scripts/ft_train/`, `demo/` (repo này) |
| Kết quả, bản dịch sinh ra, log | máy Nhật Anh `data/processed/ft_run/` (không vào git) |
| Nhật ký đầy đủ từng lần chạy 4B | `data/processed/ft_run/NHAT_KY_CHAY.md` (sinh tự động từ file kết quả) |
| Kết quả chi tiết từng phiên Kaggle | `docs/ket_qua_kaggle/INDEX.md` — mỗi phiên một file, số lấy từ JSON gốc |
| Tài liệu đã có trong git | `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md`, `docs/QUOTA_KET_LUAN.md`, `docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md`, `docs/CAC_DIEM_CAN_CHOT.md`, `docs/archive/BANG_KET_QUA_CUOI_09-11_DA_THAY.md` (bản 09/11, đã bị `QUOTA_KET_LUAN.md` thay) |
| Adapter (trọng số LoRA) | Kaggle tài khoản `trinhnhatanh`, notebook kho `ft-qwen3-4b-*-luu` (riêng tư), xem mục 10 |

---

## 2. Dữ liệu

### 2.1 Nguồn và hướng gốc
- 1688: gốc tiếng Trung, bản Việt do máy dịch, nên hướng gốc là zh→vi.
- Tiki: gốc tiếng Việt, bản Trung do máy dịch, nên hướng gốc là vi→zh.

### 2.2 Phễu làm sạch bằng luật (`scripts/ft_clean/`) [ĐO]
Nguồn là `bilingual_official_context.parquet` (bản cụm, 25.956 sản phẩm, HF commit `4a23574`). Mỗi dòng chỉ bị loại theo lý do đầu tiên nó dính.

| Bước | Luật | Loại |
|---|---|---|
| B1a | title_vi 1688 rò lời nhắc / ảo giác | 46 |
| B1b | số đoạn thuộc tính hai bản khác nhau | 49 (1688 38, Tiki 11) |
| B1c | còn chữ Hán trong bản Việt 1688 | 7 |
| B2 | trần 200 dòng mỗi khuôn thuộc tính | 279 (Tiki) |
| | **Còn lại** | **25.575** (1688 14.837, Tiki 10.738) |

Căn hàng chắc (số đoạn khớp và số trong từng đoạn khớp): 22.706/25.575 = 88,8%.

### 2.3 Chọn bản dịch bằng Judge (`scripts/judge/`)
- Judge là DeepSeek. Mỗi mục được hỏi mù, hai lượt có hoán đổi vị trí, theo rubric v2, gom nhiều mục mỗi lời gọi. Thiết kế có tầng 2 (thêm ngữ cảnh cho mục không nhất quán), nhưng theo file đánh dấu, lần chạy đầy đủ không dùng tầng 2.
- Phân công: Nhật Anh thiết kế giao thức, viết mã, chạy thí điểm trên máy và soạn hướng dẫn chạy; **Huy chạy lần Judge đầy đủ** và đưa kết quả lên HF. Chi tiết: [`ket_qua_kaggle/A6`](ket_qua_kaggle/A6_judge_deepseek_chon_ban_dich_gemma_hay_qwen_cum.md).
- Chỉ thay bản hiện có khi hai lượt cùng chọn một mô hình; mục chưa chấm hoặc không quyết thì giữ hiện trạng.
- Không đưa vào so cặp những mục mà một ứng viên rỗng.
- Bộ pair: Judge chấm 5.000/34.534 đơn vị dùng nhiều nhất, phủ khoảng 84% mảnh. Bộ cụm: 4.439/4.446 mục.

Nguồn của bản cuối theo cột `judge_chon_*` [ĐO, đếm trên file HF]:

| | Bộ cụm | Bộ pair |
|---|---|---|
| Title: gemma / qwen | 19.261 / 6.314 | 19.261 / 6.314 |
| Mảnh mô tả: gemma / qwen / quy_tac / chua_ro | 217.818 / 493 / 140.080 / 83.451 | 316.057 / 12.756 / 237.909 / 305.312 |

`quy_tac` là mảnh khóa bằng từ điển hoặc luật; `chua_ro` là mảnh chưa chấm hoặc không quyết, nên giữ hiện trạng.

### 2.4 Hai định dạng
- **Cụm:** thuộc tính nối bằng "; " trong một khối.
- **Pair:** mỗi dòng một "tên: giá trị".

Hai bộ cùng 25.575 sản phẩm và cùng title.

### 2.5 Chia tập (`chia_tap_v2`, seed 20261104) [ĐO]
- **Test khóa 2.558 sản phẩm** (10%; zh→vi 1.485, vi→zh 1.073).
- Pool 23.017 sản phẩm chia 5 fold (4.484 / 5.084 / 4.483 / 4.483 / 4.483).
- Chia theo nhóm gần-trùng (Jaccard: title 0,5, mô tả 0,7): 18.448 nhóm, nhóm lớn nhất 969 sản phẩm.
- Tập test chỉ dùng để đo, chưa từng dùng để dò hay chọn cấu hình. Dataset Kaggle chỉ chứa mẫu pool.
- Hash để kiểm:

| File | sha256 |
|---|---|
| `bo_cum_chon_deepseek.parquet` | `e0a625b2…a20def` |
| `bo_pair_chon_deepseek.parquet` | `d4535c62…7230d` |
| `split.parquet` | `d66a39c7…ee4fa3` |
| danh sách test | `0d7c7c15…e09101` |

### 2.6 Lọc title bị gắn cờ (`loc_title.py`)
- 6.386/23.017 sản phẩm pool (27,7%) bị gắn cờ title: vi→zh 48,2%, zh→vi 12,9%.
- Bản cuối **loại mẫu title** của các sản phẩm này và **giữ mẫu mô tả**.

---

## 3. Fine-tune Qwen3-4B (máy RTX Pro 6000 qua Kaggle)

### 3.1 Cấu hình cố định
- Nền `Qwen/Qwen3-4B` (revision `1cfa9a72`), bf16, tắt suy luận.
- LoRA trên tất cả lớp tuyến tính, alpha = 2 × rank, dropout 0,05.
- AdamW, clip gradient 1,0; warmup 3% rồi giảm cosine.
- `max_len` 2.048; 4.096 token mỗi lô nhỏ; seed 42.
- Đánh giá: sinh greedy trên 1.000 mẫu cố định của fold kiểm; chrF++ cho zh→vi, chrF ký tự cho vi→zh; bootstrap theo sản phẩm.
- Tổng giờ GPU các phiên chính: 32,02 giờ.

### 3.2 Dò siêu tham số (B6, fold 0, bản cụm) [ĐO]

| Cấu hình | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung | Title | Mô tả |
|---|---|---|---|---|---|
| lr 5e-5, r16 | 0.4090 | – | – | – | – |
| lr 1e-4, r16 | 0.3900 | – | – | – | – |
| lr 2e-4, r16 | 0.3770 | – | – | – | – |
| lr 5e-5, r64 | 0.3808 | – | – | – | – |
| lr 1e-4, r64 | 0.3680 | 0.3587 | 75.51 | 60.21 | 90.81 |
| **lr 2e-4, r64 (chọn)** | 0.3703 | **0.3512** | **75.76** | 60.22 | 91.30 |

- lr 2e-4 so với 1e-4 (cùng r64): chrF theo câu chênh +0,33 [−0,12; +0,81], tức hòa; chọn 2e-4 vì loss thấp hơn.
- LoRA chỉ trên attention (q, k, v, o) kém LoRA tất cả lớp: **−2,20 [−2,75; −1,65]**, nên giữ tất cả lớp tuyến tính.

### 3.3 Tỉ trọng title trong loss: kiểm trên 3 fold [ĐO]
Mốc cũ là 1 : 1 theo mẫu, khiến title chiếm khoảng 50% loss dù chỉ khoảng 11% token đích. Mức mới khoảng 11% làm bằng cách nhân `w_title` vào loss của mẫu title.

| Bản | Fold | chrF chung (corpus), mới − mốc | Dòng mới mô tả zh→vi, mới − mốc [95%] | Title hai hướng (corpus) |
|---|---|---|---|---|
| cụm | 0 | +0.38 | +1.43 [+0.54; +2.32] | +0.14 [−0.99; +1.34] |
| cụm | 1 | +0.40 | +1.95 [+1.13; +2.82] | +0.33 [−0.64; +1.30] |
| cụm | 2 | +1.15 | +2.69 [+1.76; +3.69] | +0.33 [−0.46; +1.16] |
| pair | 0 | −0.08 (theo câu) | +1.36 [+0.57; +2.19] | **−0.96 [−1.81; −0.15]** |
| pair | 1 | +0.28 (theo câu) | +2.10 [+1.23; +2.94] | +0.01 [−0.96; +1.00] |
| pair | 2 | +0.47 (theo câu) | +1.58 [+0.72; +2.44] | +0.34 [−0.65; +1.25] |

- Cụm đạt luật khóa trước: cả 3 fold dương, trung bình 0,64 lớn hơn độ lệch chuẩn 0,44.
- Pair đạt luật 3 fold riêng của pair. Lợi chủ yếu ở dòng mô tả mới hướng zh→vi; title fold 0 của pair tụt thật.
- Mức 75% title: thua ở pair (−0,58 [−0,99; −0,19]), hòa ở cụm, nên bỏ.
- Kiểm lr 1e-4 ở mức 11% (cụm): +0,03 [−0,43; +0,47], không đạt, giữ lr 2e-4.

### 3.4 Bản cuối (Kaggle bản 43, 07/11) [ĐO]
- Huấn luyện trên **toàn pool 23.017 sản phẩm**, 2 epoch, loại 6.386 title bị cờ; `w_title` tính lại để giữ tỉ trọng.
- `cum_cuoi`: 39.734 mẫu, 15,65M token, 1.990 bước, loss ep2 0,1599.
- `pair_cuoi`: 39.724 mẫu, 14,83M token, 2.008 bước, loss ep2 0,1655.
- Không còn fold giữ lại nên **không có điểm chất lượng**. Chỉ sinh thử 100 mẫu để bắt lỗi hỏng, và bản cuối **không đo trên test**.

---

## 4. Bộ trọng số đóng góp title : mô tả (đã chốt)

| | Cụm | Pair |
|---|---|---|
| % title trong loss (theo token đích) | **10,74%** | **10,86%** |
| % mô tả | 89,26% | 89,14% |
| `w_title` fold 0 | 0,12029 | 0,11818 |
| `w_title` bản cuối (sau khi loại title bị cờ) | **0,17247** | **0,16915** |

"Trọng số" ở đây là hệ số trong loss, không phải trọng số mô hình.

**Bằng chứng chính là thí nghiệm trọng số loss**, không phải khảo sát quota: 3 mức (1:1 khoảng 50%, khoảng 11%, 75%), 3 fold, có khoảng tin cậy, luật khóa trước. Trong cơ chế này mọi title được thấy đúng 2 epoch như nhau, chỉ đổi hệ số nhân loss, nên không dính confound "tỉ trọng so với số lần thấy". Khảo sát quota dưới đây chỉ là **kiểm chéo**, có confound tần suất (natural khoảng 0,31 lượt/epoch, balanced khoảng 1,4), rank 16, 300 mẫu/nhóm, không có khoảng tin cậy chính thức.

**Kiểm chéo bằng khảo sát lấy mẫu theo hạn ngạch** (quota; rank 16, một lượt, 2 seed, eval 1.200 mẫu; điểm "Chính" = 0,10 × title + 0,90 × mô tả hướng zh→vi) [ĐO, `docs/QUOTA_KET_LUAN.md`]:

| Bộ | Cấu hình | % title | Chính | Macro 50/50 | Title zh→vi | Mô tả zh→vi |
|---|---|---|---|---|---|---|
| cụm | **natural (chốt)** | 10.8 | **86.81** | 71.87 | 54.19 | 90.44 |
| cụm | balanced | 50 | 85.04 | 73.11 | 57.98 | 88.04 |
| cụm | title_heavy | 80 | 81.84 | 72.43 | 59.12 | 84.36 |
| pair | **natural (chốt)** | 10.3 | **79.98** | 68.52 | 49.98 | 83.31 |
| pair | balanced | 50 | 75.44 | 68.60 | 54.00 | 77.82 |
| pair | attr_heavy | 15 | 78.86 | 68.19 | 51.38 | 81.92 |

- Tăng tỉ trọng title thì title tăng và mô tả giảm, cùng chiều ở cả hai seed.
- **w hòa vốn** là trọng số title tối thiểu để cấu hình kia thắng natural: cụm balanced 0,39; cụm title_heavy 0,55; pair balanced 0,58; pair attr_heavy 0,50. Nếu nhóm coi title nặng hơn khoảng 40% thì kết luận ở cụm đổi.
- Bảng trên là vòng sàng lọc (mức 11% và 50%). **Cập nhật 18/11:** đã chạy thêm 25% và 35% (bản cụm, fold 0, 1 seed, công thức cuối): 25% − 11% = −0,19 [−0,41; +0,03] (không phân biệt được); 35% − 11% = −0,39 [−0,68; −0,09] (title zh→vi +1,19, mô tả zh→vi −0,56, đánh đổi); macro không phân biệt được ba mức (`docs/phan_bien_vong2/KET_QUA_P2.md` mục B). Không dùng chữ "tối ưu". Các mức 12–24%, 26–34% chưa thử; chưa chạy bản pair. Khoảng tin cậy chỉ phủ dao động chọn mẫu eval, không phủ dao động khởi tạo; khảo sát quota là kiểm chéo, không phải bằng chứng chính cho `w_title`.

---

## 5. Đo test khóa MỘT lần (Kaggle bản 41, 07/11) [ĐO]

- Adapter fold 0 (`cum_f0_p11`, `pair_f0_p11`) so với zero-shot trên 5.116 mục / 2.558 sản phẩm mỗi nhánh.
- Luật khóa trước khi đo: thắng khi cận dưới khoảng 95% (bootstrap theo sản phẩm) của chênh lớn hơn 0.

| Nhánh | Hướng | Zero-shot | Fine-tune | Chênh [95%] |
|---|---|---|---|---|
| cụm | vi→zh | 41.49 | 75.63 | +34.14 [+33.30; +34.95] |
| cụm | zh→vi | 47.22 | 77.36 | +30.14 [+29.66; +30.63] |
| pair | vi→zh | 43.92 | 75.41 | +31.50 [+30.73; +32.24] |
| pair | zh→vi | 46.34 | 77.65 | +31.31 [+30.81; +31.83] |

- Kiểm riêng hướng zh→vi của cụm: tỉ lệ sai chữ Việt giảm từ 0,308 (zero-shot) xuống 0,058 (mốc tham chiếu 0,049); rò lời dẫn từ 0,326 xuống 0.
- Chênh lớn một phần vì tham chiếu là bản dịch máy cùng kiểu với dữ liệu huấn luyện. Đã tách dòng khuôn/dòng mới trên test (19/11, đọc từ tổng hợp B8, `docs/phan_bien_vong2/KET_QUA_B8_DONG_MOI.md`): trên dòng mô tả mới fine-tune hơn zero-shot khoảng 30 điểm chrF corpus, chất lượng dòng mới khoảng 76; chưa có khoảng tin cậy.

---

## 6. So sánh năm mô hình (eval fold 0, không đụng test) [ĐO]

**Cách đo:**
- Huấn luyện trên fold 1–4, đo trên 1.200 mẫu cố định của fold 0.
- Ba LLM dùng cấu hình **sàng lọc**: LoRA r16, lr 2e-4, lấy mẫu natural 4M token một lượt, trung bình 2 seed. Cấu hình này không phải công thức cuối.
- Hai mô hình cổ điển học từ đầu trên dòng của bản pair: OpenNMT là Transformer nhỏ, Moses là phrase-based với Hán tách từng chữ.
- "Chính" = 0,10 × title + 0,90 × mô tả hướng zh→vi; "Macro" = trung bình đều hai hướng.

### 6.1 chrF (corpus) và chênh chrF câu trung bình so với 4B [95%, bootstrap theo sản phẩm]

**Pair (1.198 mẫu)**

| Mô hình | Title | Mô tả | Chính | Macro | Chênh chính (câu) so với 4B | Mô tả mới (P1) |
|---|---|---|---|---|---|---|
| Qwen3-4B | 49.98 | 83.31 | 79.98 | 68.52 | mốc | 63.41 |
| Qwen3-8B | 52.03 | 79.86 | 77.08 | 66.61 | +0.44 [−0.24; +1.14] | 54.38 |
| EcomGPT-7B | 38.45 | 71.45 | 68.15 | 56.78 | −4.81 [−5.95; −3.79] | 40.19 |
| OpenNMT (học đủ) | 48.27 | 77.28 | 74.38 | 54.89 | +0.39 [−1.09; +1.90] | 44.88 |
| Moses (học đủ 270k dòng) | 54.93 | 83.65 | 80.78 | 68.08 | +1.86 [+0.67; +3.04] | 61.53 |
| Moses (khớp 4M token) | 51.97 | 82.34 | 79.30 | 65.36 | +0.05 [−1.14; +1.23] | 60.18 |

**Cụm (1.200 mẫu)**

| Mô hình | Title | Mô tả | Chính | Macro | Chênh chính (câu) so với 4B | Mô tả mới (P1) |
|---|---|---|---|---|---|---|
| Qwen3-4B | 54.19 | 90.44 | 86.81 | 71.87 | mốc | 72.70 |
| Qwen3-8B | 56.52 | 91.34 | 87.86 | 73.92 | **+1.03 [+0.71; +1.36]** | 74.75 |
| EcomGPT-7B | 45.85 | 82.49 | 78.83 | 64.44 | −6.42 [−7.36; −5.55] | 61.22 |
| OpenNMT (học đủ) | 47.01 | 84.34 | 80.61 | 58.72 | −5.57 [−6.42; −4.85] | 56.98 |
| Moses (học đủ) | 54.93 | 88.48 | 85.12 | 69.77 | −1.51 [−2.06; −1.02] | 67.71 |
| Moses (khớp 4M token) | 51.97 | 85.71 | 82.33 | 66.48 | −4.17 [−4.79; −3.60] | 61.34 |

Zero-shot macro (pair / cụm): 4B 41,97 / 43,95; 8B 45,19 / 51,65; EcomGPT 13,69 / 12,78.

**Cách đọc hai bảng:**
- **Đây là eval fold 0**, tập đã dùng để chọn cấu hình (B6, tỉ trọng title, 5 mô hình, adapter demo), train = fold 1–4; **chưa đo test** cho 5 mô hình. Số trên fold 0 bị ảnh hưởng bởi việc chọn. Tỉ trọng title có thêm kiểm độc lập ở fold 1 và 2.
- Ở pair, chrF corpus và chrF câu trung bình xếp 8B với 4B khác nhau. Corpus của 8B thấp hơn chủ yếu vì vài dòng dài, ví dụ danh sách thành phần mà tham chiếu giữ nguyên tên Latin còn 8B dịch sang chữ Hán.
- **85,91% dòng mô tả của eval pair zh→vi đã có nguyên văn trong train** (P1 3.2, [ĐO]; A7 ghi 85,0% trên test). Vì vậy điểm mô tả toàn phần nghiêng về tái tạo khuôn mẫu, và cột "Mô tả mới (P1)" mới là phần dịch câu chưa từng thấy. Trên dòng mới, OpenNMT kém Qwen3-4B −8,29 [−15,49; −0,88] (pair zh→vi) và −10,07 [−11,68; −8,30] (cụm zh→vi); Moses học đủ −2,50 và −3,68. Chiều ngược lại, trên dòng khuôn OpenNMT lại hơn 4B +2,12 (pair zh→vi).
- Với cấu hình Transformer nhỏ học từ đầu (D3), OpenNMT hòa Qwen3-4B ở pair **chỉ theo chrF câu trung bình chính (0,1/0,9, zh→vi)** trên eval fold 0, nơi phần lớn dòng đã có trong train (+0,30 [−1,07; +1,71], trung bình 3 seed 42/43/44, cả ba seed chứa 0); ở macro kém −8,65, ở cụm kém −5,52, ở chrF corpus kém 5,3 (pair) và 6,2 (cụm) điểm, và kém rõ ở vi→zh (`docs/phan_bien_vong2/KET_QUA_P2.md` mục A). Kết quả cho thấy độ nhạy theo cấu hình và theo loại dòng, không kết luận về paradigm NMT.
- OpenNMT kém nhất ở vi→zh (title 29,6) vì chỉ có khoảng 61 nghìn dòng train theo hướng này; cùng hướng, khoảng cách trên dòng mô tả mới còn lớn hơn (−17,76 ở pair, P1).

### 6.2 BLEU / SacreBLEU / TER theo cách đo của bài VLSP
- Chữ ký sacrebleu 2.6.0: zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`.
- BLEU kiểu mteval-v13a ra trùng SacreBLEU nên không lập bảng riêng.

| Mô hình | Cụm SacreBLEU chính | Cụm TER chính | Pair SacreBLEU chính | Pair TER chính |
|---|---|---|---|---|
| Qwen3-4B | 81.19 | 16.98 | 71.01 | 24.70 |
| Qwen3-8B | 82.31 | 15.48 | 68.27 | 26.42 |
| EcomGPT-7B | 69.34 | 36.07 | 54.46 | 49.03 |
| OpenNMT (học đủ) | 74.62 | 25.74 | 67.95 | 29.08 |
| Moses (học đủ) | 75.50 | 25.11 | 65.69 | 31.13 |
| Moses (khớp 4M) | 72.49 | 28.24 | 65.23 | 32.49 |

- Cụm: 8B hơn 4B cả ở SacreBLEU (+1,11 [+0,67; +1,58]) và TER (−1,51 [−2,04; −0,97]).
- Pair: các chênh giữa 4B, 8B, OpenNMT và Moses đều có khoảng chứa 0 ở điểm chính.
- BLEU trên title và dòng thuộc tính rất ngắn thì bất ổn. Các số này không so với số trong bài VLSP vì khác dữ liệu và miền.

### 6.3 Vòng lặp tham chiếu (bổ sung 17/11, P1 3.3) [ĐO]

Tham chiếu title và mô tả do **Gemma 4 hoặc Qwen** sinh (Qwen3-14B; một phần title có thể là Qwen3.5-9B **[CHƯA KIỂM]**), Judge DeepSeek chọn bản; dữ liệu huấn luyện lấy cùng nguồn đó. Vì vậy Qwen3-4B/8B **cùng họ** với một mô hình sinh tham chiếu, còn EcomGPT (BLOOM) thì không.

- **Title, eval fold 0 zh→vi:** 300 sản phẩm, trong đó 55 có tham chiếu Qwen và 245 có tham chiếu Gemma. Chênh lệch kép (lợi thế của mô hình Qwen ở nhóm Qwen trừ nhóm Gemma): so EcomGPT −7,68 [−10,36; −5,18]; so OpenNMT +5,17 [+2,11; +8,19]; so Moses học đủ +3,11 [+0,43; +5,99] (thứ tự mã sản phẩm xác định). **Không nhất quán theo họ baseline** với mô hình đã fine-tune, nên không kết luận self-preference từ phần này. Bổ sung 18/11 với **zero-shot** Qwen3-4B (bản cụm): lợi thế sẵn có ở nhóm tham chiếu Qwen (kép +10,20 / +16,17 / +14,41 so EcomGPT / OpenNMT / Moses, khoảng 95% không chứa 0); fine-tune xóa phần lớn lợi thế đó. Chi tiết `docs/phan_bien_vong2/KET_QUA_P2.md` mục C.
- **Mô tả, eval fold 0 zh→vi:** 285 sản phẩm pair đều Gemma chủ đạo (cụm: 297 Gemma, 1 không xác định), **không có nhóm Qwen chủ đạo** để tính chênh lệch kép theo kế hoạch. Phân tích thêm sau khi xem số, nhóm "có mảnh Qwen" (170) so "không có" (115), cả 6 khoảng 95% đều chứa 0.
- Giới hạn: Judge chọn Qwen khi bản Gemma kém, nên hai nhóm khác độ khó; chênh lệch kép chỉ khử phần độ khó chung cho mọi mô hình. Còn mở: tham chiếu độc lập do người chấm 100–300 mẫu.

---

## 7. Thử chéo dạng pair và dạng cụm (600 sản phẩm fold 0, mô tả zh→vi) [ĐO]

| So sánh | Chênh chrF câu TB [95%] |
|---|---|
| pair + adapter pair so với cụm + adapter cụm | +0.04 [−0.16; +0.23], hòa |
| adapter cụm nhận đầu vào dạng pair (sai dạng) | −1.01 [−1.39; −0.67] |
| adapter pair nhận đầu vào dạng cụm (sai dạng) | −0.47 [−0.68; −0.28] |
| adapter so với zero-shot | +37.7 đến +37.9 |

Hai dạng cho chất lượng ngang nhau khi dùng đúng adapter của dạng đó. Chọn pair hay cụm cho extension là vì tiện, không vì điểm.

---

## 8. Công thức cuối đo trên fold 0 (T4 fp16, 14/11) [ĐO]

- Adapter `cum_f0_p11`, `pair_f0_p11`: công thức cuối, học fold 1–4.
- Chấm trên cùng eval với năm mô hình. Lời dẫn và cách dịch khác vòng sàng lọc (pair dịch cả khối rồi tách dòng), nên dòng này **không xếp hạng chung** với năm dòng ở mục 6.

| Bộ | chrF chính (corpus) | Chênh chrF câu so với 4B sàng lọc | SacreBLEU chính | TER chính |
|---|---|---|---|---|
| Cụm | 90.38 | +3.36 [+2.96; +3.78] | 85.73 | 12.04 |
| Pair | 83.28 | +3.95 [+2.98; +4.99] | 75.00 | 18.89 |

Trên T4 fp16, pair dịch 1.165 lượt, không có khối nào lệch dòng và không có bản dịch rỗng.

---

## 9. Demo (extension Chrome + máy chủ dịch)

### 9.1 Cổng nén để chạy trên laptop
- Laptop RTX 3050 4 GB, adapter `pair_f0_p11`, 150 sản phẩm fold 0, hướng zh→vi, so với bản fp16 trên Kaggle.
- Ngưỡng đặt trước khi đo: chrF tụt ≤ 1,0 điểm; lệch dòng ≤ 2%; không có bản rỗng.

| Bản | chrF câu TB (macro zh→vi) | Tụt so với fp16 | Khối lệch dòng | Rỗng | Title (giây, trung vị) | Khối ~22 dòng (giây, trung vị) | Kết luận |
|---|---|---|---|---|---|---|---|
| fp16 Kaggle (mốc) | 79.76 | – | 0/72 | 0% | – | – | – |
| Q4_K_M + LoRA | 76.95 | −2.81 | 1/72 | 0.6% | 0.99 | 12.09 | **loại** |
| **Q5_K_M + LoRA** | 79.62 | −0.14 | 0/72 | 0% | 1.09 | 14.08 | **đạt** |

### 9.2 Thử trên trang 1688 thật (15/11)
- 13 trang lưu sẵn, chạy qua máy chủ Kaggle T4 fp16 mở bằng đường hầm.
- 13/13 trang đạt kiểm cấu trúc. Số dòng thuộc tính đọc từ DOM khớp JSON nhúng trong trang.
- Đọc tay 8 trang mới, không có tham chiếu: khoảng 95% dòng đúng và tự nhiên. Lỗi thấy được:
  - Tự thêm chữ "(Thượng Hải)".
  - 秋葵 (đậu bắp) dịch thành "khoai môn".
  - 树脂 dịch thành "Nhựa polyester".
  - Chữ "外贸" rụng khỏi title.
  - Tên thương hiệu bị phiên âm sang chữ Latin.
- Bộ đệm đã dịch sẵn 13 trang, có chế độ chỉ dùng bộ đệm (không cần GPU). Đã chọn 5 trang để demo. Chưa thử đầy đủ trong Chrome thật.
- **Đây là thử nghiệm quy mô nhỏ**: 13 trang 1688 thật, đọc tay 8 trang, không có tham chiếu người, một người dùng, máy chủ trên Kaggle T4 hoặc laptop. Không có số về độ ổn định, tải, nhiều người dùng hay trang ngoài 1688.

---

## 10. Trọng số (adapter LoRA) đã lưu

| Adapter | Học trên | Dùng cho | Nơi lưu (Kaggle `trinhnhatanh`) | sha256 `adapter_model.safetensors` |
|---|---|---|---|---|
| `cum_cuoi` | toàn pool | bản cuối dạng cụm | `ft-qwen3-4b-cuoi-luu` | `fc8481ad…5b131` |
| `pair_cuoi` | toàn pool | bản cuối dạng pair, demo | `ft-qwen3-4b-cuoi-luu` | `7fd7f157…2319b` |
| `cum_f0_p11` (tài khoản 1) | fold 1–4 | đo test (mục 5), C0 | `ft-qwen3-4b-tl-luu` | `ce352c70…` (đầy đủ trong `MANIFEST_TL.json`) |
| `pair_f0_p11` (tài khoản 1) | fold 1–4 | đo test (mục 5), C0 | `ft-qwen3-4b-tl-luu` | `4d06b578…` |
| `cum_f0_p11` (huấn luyện lại ở tài khoản 2, cùng cấu hình) | fold 1–4 | thử chéo (mục 7), công thức cuối trên fold 0 (mục 8) | output notebook `honghuynanh/nlp-group3` | `ff28993c…` |
| `pair_f0_p11` (huấn luyện lại ở tài khoản 2, cùng cấu hình) | fold 1–4 | thử chéo (mục 7), mục 8, cổng nén demo (mục 9.1) | output notebook `honghuynanh/nlp-group3-pair` | `56967f4a…` |
| các lần thử fold khác | | | `ft-qwen3-4b-{b6,b78,b8f1,b8f2,tl2,tl3}-luu` | trong manifest của từng kho |

- Mỗi adapter nặng 528.550.256 byte (rank 64), nền `Qwen/Qwen3-4B` revision `1cfa9a72`.
- Bản sao trên máy Nhật Anh ở `data/processed/ft_run/ban_cuoi/cuoi/{cum_cuoi,pair_cuoi}/ep2/`, hash khớp manifest.
- Bản nén cho demo nằm ở `data/processed/ft_run/demo_gguf/`.
- Adapter 8B và EcomGPT của vòng sàng lọc nằm ở đầu ra notebook tài khoản 2 (`honghuynanh/nlp-8b`, `nlp-ecom`). Chúng không phải bản cuối.

---

## 11. Đã chốt và chưa chốt

**Đã chốt:**
- lr 2e-4, rank 64, 2 epoch, LoRA tất cả lớp tuyến tính.
- Tỉ trọng title 10,74% (cụm) và 10,86% (pair).
- Loại 6.386 title bị cờ khỏi bản cuối.
- Đo test đúng một lần.
- Bộ thước đo cho năm mô hình: chrF cùng BLEU, SacreBLEU, TER, tách title và mô tả, cả hai hướng.

**[CHỜ]:**
- Chạy 8B theo công thức cuối cho bản cụm (cần GPU lớn hơn 24 GB một card).
- Extension dùng dạng pair hay cụm (hiện demo dùng `pair_cuoi`).
- Đo test cho mô hình được chọn.
- B9: LLM chấm so cặp.
- Sửa title thật cho bản dữ liệu v2.
- Commit mã.
- Danh sách đầy đủ ở `docs/CAC_DIEM_CAN_CHOT.md`.

---

## 12. Giới hạn phải ghi khi trích số

- Tham chiếu là bản dịch máy. Tham chiếu title có lỗi lõi khoảng 5–14%, nên chrF title chỉ khoảng 50–62.
- Khoảng tin cậy chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại. Vòng sàng lọc có 2 seed, các lần chạy 4B chỉ 1 seed.
- Năm mô hình chỉ đo trên eval fold 0, chưa đo trên test.
- Hai mô hình cổ điển chỉ chạy một lần và học trên dòng của bản pair. EcomGPT chạy ít bước hơn vì tokenizer BLOOM cho ít token hơn.
- Fold 0 cũng là nơi phát hiện tín hiệu dòng mới, nên chỉ fold 1 và 2 là kiểm độc lập.
- Mã `kaggle_train.py` bản trước 05/11 23:06 (dùng cho B6 và bản 25) không còn bản lưu riêng.

---

## 13. Đặt cạnh hướng của Quang

| | Hướng Nhật Anh | Hướng Quang (`main`) |
|---|---|---|
| Dữ liệu | bản dịch Gemma/Qwen chọn bằng Judge DeepSeek; dạng cụm và pair | làm sạch CADO và lọc bitext; bộ A/B/C, Tiki dịch ngược bằng Qwen3 |
| Tập đo | eval fold 0 (1.200 mẫu) và test khóa 2.558 sản phẩm | eval_F 1.500 dòng |
| Thước chính | chrF++ (kèm BLEU, SacreBLEU, TER) | COMET-da (1688) và COMET-QE (Tiki), kèm BLEU và chrF |

Điểm hai bên **không so thẳng được**. Muốn chọn theo điểm thì cách công bằng là cho mô hình cuối của mỗi bên dịch cùng một tập rồi chấm cùng một thước đo.

---

## 14. Tình trạng tải đầu ra Kaggle

- Mọi phiên huấn luyện, đo và demo có kết quả đều đã tải về `data/processed/ft_run/`, mỗi phiên một thư mục `kaggle_*`.
- Đầu ra mỗi phiên huấn luyện 4B được lưu sang kho `*-luu` trước khi đẩy phiên kế, kèm manifest sha256.
- Hai lần chạy khói 8B bản cuối (13/11) bị cấp Tesla T4: lần 1 hết bộ nhớ khi nạp mô hình, lần 2 dừng ở chốt chặn VRAM; không có kết quả (phiên F3).
- Notebook `notebook59a0de1490` (tài khoản 1, gắn cuộc thi Gemma 4) chưa chạy gì.

---

## 15. Mục lục chi tiết từng phiên

Mỗi phiên có một file riêng trong `docs/ket_qua_kaggle/`, sinh bởi `scripts/ft_train/xuat_ket_qua_kaggle.py` từ file kết quả gốc (số không gõ tay). Mỗi file có: notebook, tài khoản, bản Kaggle, thời gian, GPU, mục đích, bảng số, kết luận, ghi chú/giới hạn, sha256 file gốc. Các lần chạy khói/thử ngắn gom ở cuối [`INDEX.md`](ket_qua_kaggle/INDEX.md).

| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |
|---|---|---|---|---|
| A1 | [Bộ song ngữ 1: thuộc tính 1688 dịch theo đơn vị (Gemma, Qwen3-14B, CometKiwi)](ket_qua_kaggle/A1_bo_song_ngu_1_thuoc_tinh_1688_dich_theo_don_vi_g.md) | 1 (`trinhnhatanh`) | – | 15–22/10/2026 |
| A2 | [Bộ song ngữ 2: thuộc tính Tiki (vi→zh) dịch theo đơn vị](ket_qua_kaggle/A2_bo_song_ngu_2_thuoc_tinh_tiki_vi_zh_dich_theo_do.md) | 1 (`trinhnhatanh`) | – | 23/10/2026 |
| A3 | [Bộ song ngữ 3: rà soát theo cụm có ngữ cảnh (Tiki)](ket_qua_kaggle/A3_bo_song_ngu_3_ra_soat_theo_cum_co_ngu_canh_tiki.md) | 1 (`trinhnhatanh`) | – | 23–24/10/2026 |
| A4 | [Bộ song ngữ 4: rà soát theo cụm có ngữ cảnh (1688)](ket_qua_kaggle/A4_bo_song_ngu_4_ra_soat_theo_cum_co_ngu_canh_1688.md) | 1 (`trinhnhatanh`) | – | 24–26/10/2026 |
| A5 | [Lọc luật cho fine-tune (trên máy)](ket_qua_kaggle/A5_loc_luat_cho_fine_tune_tren_may.md) | – (chạy trên máy) | – | 30/10/2026 |
| A6 | [Judge DeepSeek chọn bản dịch Gemma hay Qwen (cụm và pair)](ket_qua_kaggle/A6_judge_deepseek_chon_ban_dich_gemma_hay_qwen_cum.md) | – (gọi API DeepSeek, không chạy trên Kaggle; thí điểm: Nhật Anh trên máy; lần chạy đầy đủ: Huy) | – | thí điểm 29–31/10/2026; bản phát hành tải về 03/11/2026 |
| A7 | [Chia tập theo sản phẩm, kiểm rò rỉ, gắn cờ title (trên máy)](ket_qua_kaggle/A7_chia_tap_theo_san_pham_kiem_ro_ri_gan_co_title_t.md) | – (chạy trên máy) | – | 04–05/11/2026 |
| B0 | [Dựng mẫu huấn luyện và đóng gói Dataset Kaggle (trên máy)](ket_qua_kaggle/B0_dung_mau_huan_luyen_va_dong_goi_dataset_kaggle_t.md) | – (dựng trên máy; Dataset ở tài khoản 1 và 2) | – | 04–12/11/2026 |
| B1 | [B6: dò siêu tham số Qwen3-4B (lr × rank)](ket_qua_kaggle/B1_b6_do_sieu_tham_so_qwen3_4b_lr_rank.md) | 1 (`trinhnhatanh`) | 20, 21, 22 (nối nhau) | 05/11/2026 |
| B2 | [B7 + B8 phần đầu: LoRA attention, pair fold 0, cụm fold 1 (dở)](ket_qua_kaggle/B2_b7_b8_phan_dau_lora_attention_pair_fold_0_cum_fo.md) | 1 (`trinhnhatanh`) | 25 (bản 26 nối tiếp bị hủy) | 05/11/2026 |
| B3 | [Thử tỉ trọng title fold 0 (~11% và 75%, cụm và pair)](ket_qua_kaggle/B3_thu_ti_trong_title_fold_0_11_va_75_cum_va_pair.md) | 1 (`trinhnhatanh`) | 29 | 05–06/11/2026 |
| B4 | [B8 fold 1, 2 (mốc 1 : 1) + cụm ~11% fold 1](ket_qua_kaggle/B4_b8_fold_1_2_moc_1_1_cum_11_fold_1.md) | 1 (`trinhnhatanh`) | 32 | 06/11/2026 |
| B5 | [Cụm ~11% fold 2 + luật 3 fold bản cụm](ket_qua_kaggle/B5_cum_11_fold_2_luat_3_fold_ban_cum.md) | 1 (`trinhnhatanh`) | 34 | 06/11/2026 |
| B6 | [Kiểm lr 1e-4 (cụm) + pair fold 1](ket_qua_kaggle/B6_kiem_lr_1e_4_cum_pair_fold_1.md) | 1 (`trinhnhatanh`) | 36 | 06–07/11/2026 |
| B7 | [Pair fold 2 + luật 3 fold bản pair](ket_qua_kaggle/B7_pair_fold_2_luat_3_fold_ban_pair.md) | 1 (`trinhnhatanh`) | 38 (bản 37 là khói) | 07/11/2026 |
| B8 | [Đo test khóa MỘT lần](ket_qua_kaggle/B8_do_test_khoa_mot_lan.md) | 1 (`trinhnhatanh`) | 41 (bản 39, 40 là khói) | 07/11/2026 |
| B9 | [Bản cuối trên toàn pool (cum_cuoi, pair_cuoi)](ket_qua_kaggle/B9_ban_cuoi_tren_toan_pool_cum_cuoi_pair_cuoi.md) | 1 (`trinhnhatanh`) | 43 (bản 42 là khói) | 07/11/2026 |
| C1 | [Khảo sát quota vòng 1 (rank 16, max_len 960, 1.200 bước)](ket_qua_kaggle/C1_khao_sat_quota_vong_1_rank_16_max_len_960_1_200.md) | 1 (`trinhnhatanh`) | 47 (pair), 48 (cụm) | 09/11/2026 |
| C2 | [C0: adapter chính thức chấm trên eval của vòng sàng lọc](ket_qua_kaggle/C2_c0_adapter_chinh_thuc_cham_tren_eval_cua_vong_sa.md) | 1 (`trinhnhatanh`) | 50 (cụm), 51 (pair) | 09/11/2026 |
| C3 | [Khảo sát quota vòng 2 (ngân sách token cố định, max_len 2.048)](ket_qua_kaggle/C3_khao_sat_quota_vong_2_ngan_sach_token_co_dinh_ma.md) | 1 (`trinhnhatanh`) | 53 | 10/11/2026 |
| C4 | [Khảo sát quota vòng 3: 3 cấu hình × 2 bộ × 2 seed (bảng đã chốt)](ket_qua_kaggle/C4_khao_sat_quota_vong_3_3_cau_hinh_2_bo_2_seed_ban.md) | 1 (`trinhnhatanh`) | 55 | 10/11/2026 |
| C5 | [Tái lập ở tài khoản 2 (check1)](ket_qua_kaggle/C5_tai_lap_o_tai_khoan_2_check1.md) | 2 (`honghuynanh`) | – | 11/11/2026 |
| C6 | [Qwen3-4B natural với lr 1e-4](ket_qua_kaggle/C6_qwen3_4b_natural_voi_lr_1e_4.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| D0 | [Danh sách mẫu natural dùng chung cho các LLM (chạy khô, CPU)](ket_qua_kaggle/D0_danh_sach_mau_natural_dung_chung_cho_cac_llm_cha.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| D1 | [Qwen3-8B, vòng sàng lọc (cụm + pair, 2 seed)](ket_qua_kaggle/D1_qwen3_8b_vong_sang_loc_cum_pair_2_seed.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| D2 | [EcomGPT-7B, vòng sàng lọc (cụm + pair, 2 seed)](ket_qua_kaggle/D2_ecomgpt_7b_vong_sang_loc_cum_pair_2_seed.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| D3 | [OpenNMT Transformer nhỏ, học từ đầu (pair + dịch bản cụm)](ket_qua_kaggle/D3_opennmt_transformer_nho_hoc_tu_dau_pair_dich_ban.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| D4 | [Moses vòng 1: chọn cách tách chữ Hán (từng chữ hay jieba)](ket_qua_kaggle/D4_moses_vong_1_chon_cach_tach_chu_han_tung_chu_hay.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| D5 | [Moses vòng 2: học đủ dữ liệu và khớp 4M token, kèm eval bản cụm](ket_qua_kaggle/D5_moses_vong_2_hoc_du_du_lieu_va_khop_4m_token_kem.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| D6 | [Tổng hợp năm mô hình (trên máy, không GPU)](ket_qua_kaggle/D6_tong_hop_nam_mo_hinh_tren_may_khong_gpu.md) | – (Qwen3-4B lấy từ C4, tài khoản 1; các mô hình khác tài khoản 2) | – | 12–14/11/2026 |
| E1 | [Huấn luyện lại adapter fold 0 ở tài khoản 2 (cum_f0_p11, pair_f0_p11)](ket_qua_kaggle/E1_huan_luyen_lai_adapter_fold_0_o_tai_khoan_2_cum.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| E2 | [Thử chéo dạng pair và dạng cụm](ket_qua_kaggle/E2_thu_cheo_dang_pair_va_dang_cum.md) | 2 (`honghuynanh`) | – | 12/11/2026 |
| E3 | [Công thức chính trên eval năm mô hình ("dòng thứ sáu", T4 fp16)](ket_qua_kaggle/E3_cong_thuc_chinh_tren_eval_nam_mo_hinh_dong_thu_s.md) | 2 (`honghuynanh`) | – | 14/11/2026 |
| F1 | [Cổng nén để chạy demo trên laptop (không phải Kaggle)](ket_qua_kaggle/F1_cong_nen_de_chay_demo_tren_laptop_khong_phai_kag.md) | – | – | 14/11/2026 |
| F2 | [Máy chủ dịch demo trên Kaggle + thử 13 trang 1688 thật](ket_qua_kaggle/F2_may_chu_dich_demo_tren_kaggle_thu_13_trang_1688.md) | 1 (`trinhnhatanh`) | – | 15/11/2026 |
| F3 | [Qwen3-8B theo công thức chính: không chạy được (bị cấp T4)](ket_qua_kaggle/F3_qwen3_8b_theo_cong_thuc_chinh_khong_chay_duoc_bi.md) | 2 (`honghuynanh`) | – | 13/11/2026 |

Nhật ký đầy đủ các phiên fine-tune 4B (bản 20–43, bản chụp 07/11): [`PHU_LUC_NHAT_KY_CHAY.md`](PHU_LUC_NHAT_KY_CHAY.md).
