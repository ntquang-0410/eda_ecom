# PLAN v0.6.1: Fine-tune Qwen3-4B (LoRA) trên hai bộ sau Judge (cụm và pair): kiểm tra bộ dữ liệu và chọn siêu tham số

Trạng thái: **v0.6.1, 05/11/2026. Xong B0–B6. B7 xong; B8 dừng giữa chừng do Nhật Anh hủy phiên (`cum_f1` còn checkpoint ở bước 1.102, `cum_f2` chưa chạy). Mục tiêu (d): bước 1, 2, 3, 6 xong ngày 06/11, cả hai bản không có ứng viên, giữ mức hiện tại ~50% (kết quả ở cuối mục 0, chỉ trên fold 0), bỏ qua bước 4, 5. Việc tiếp: hai mục tiêu còn lại (nối B8, sửa bộ đo, đo test), chờ lệnh của Nhật Anh. Xem log trực tiếp: `kaggle kernels logs -f trinhnhatanh/nlp-group3`.** Việc tốn GPU dài chỉ chạy khi Nhật Anh ra lệnh. Mọi thay đổi ghi vào "Nhật ký thay đổi" cuối file.
Nhãn: **[ĐO]** = đã đo trên dữ liệu thật; **[ƯỚC]** = ước tính/giả định; **[THEO TÀI LIỆU]** = lấy từ ghi chú dự án, không suy ra được từ dữ liệu; **[MẶC ĐỊNH]** = em tự đặt, anh phản đối thì đổi; **[CHỜ]** = chờ Nhật Anh.

---

## 0. Plan khảo sát tỉ lệ title : mô tả (mục tiêu d, v0.6)

### Bối cảnh
Nhóm muốn biết khi huấn luyện nên cho title và mô tả tỉ lệ bao nhiêu thì bản dịch tốt nhất, không học lệch, **riêng cho bản cụm và bản pair**, tỉ lệ có thể là số lẻ, khảo sát bằng K-fold trên baseline **Qwen3-4B + LoRA**, với bộ đã chốt (lr 2e-4, rank 64, 2 epoch). Tỉ lệ là trọng số trong loss: mình đặt, rồi đo bằng huấn luyện. Khảo sát dữ liệu chỉ cho ứng viên.

K-fold đủ 3 mức × 2 bản × 3 fold ~26 giờ, vượt quỹ ~19,9 giờ [ƯỚC], nên thử 2 mức trên fold 0 để chọn ứng viên rồi mới K-fold.

**Ưu tiên (Nhật Anh chốt):** giữ cả ba mục tiêu, nhưng tỉ lệ (d) làm trước; có tỉ lệ rồi mới tới (b) mỗi bộ có ổn không và (c) bộ siêu tham số, vì đo ổn định và đo test phải dùng cấu hình cuối. Dò tỉ lệ cần sẵn lr, rank nên dùng bộ B6 làm nền, không làm lại B6. Giữ ~50% thì B6 nguyên giá trị; đổi tỉ lệ thì kiểm lại lr (Nhật Anh chọn).

**Số đã có [ĐO], % title:** trọng số loss hiện tại cụm 49,65, pair 50,30 (đo trong phiên bản 29; số ghi trước đó là 49,56 và 50,26); theo token đích cụm 10,74, pair 10,86; theo lượng lỗi cụm 55,10, pair 56,45. Mức hiện tại đã có fold 0 của cả hai bản; cụm fold 1 dừng ở bước 1.102/1.800.

### Tiêu chí, khóa trước khi xem kết quả [MẶC ĐỊNH]
- **Không học lệch:** title, mô tả, và chrF hướng zh→vi đều không tụt quá 1 điểm so với mức hiện tại. Mô tả xem thêm dòng mới.
- **Qua cổng fold 0 (bước 3):** chrF chung cao hơn mức hiện tại, khoảng tin cậy bootstrap 95% của chênh nằm hẳn trên 0, và không học lệch. Dùng cùng luật "chứa 0 là hòa" như B6, vì một lần qua cổng tốn tới ~5,7 giờ K-fold.
- **Kết luận đổi tỉ lệ (bước 6), đạt đủ:** (1) chrF chung cao hơn ở cả 3 fold; (2) chênh trung bình lớn hơn độ lệch chuẩn của chênh giữa các fold; (3) không học lệch, tính theo trung bình 3 fold. Không đạt thì giữ mức hiện tại: đây cũng là kết quả hợp lệ.
- Thước đo chính là chrF trên bản dịch, không dùng loss để chọn: B6 đã đo lệch loss 2,1% mà chrF vẫn hòa. Tập test không tham gia chọn tỉ lệ.

### 6 bước

| Bước | Việc | Khi nào chạy | GPU [ƯỚC] |
|---|---|---|---|
| **1. Chuẩn bị** | Lưu đầu ra bản 25; viết, kiểm mã; chạy khói | Luôn | ~0,25 giờ |
| **2. Thử 2 mức trên fold 0** | Mỗi bản thử ~11% và ~75% title, đủ 2 epoch, chấm 1.000 mẫu, so theo cặp với mức hiện tại đã có. Làm cụm trước, pair sau | Luôn, cả hai bản đều đủ 2 mức (Nhật Anh chọn phương án B) | ~2,8 giờ cụm + ~2,7 giờ pair |
| **3. Chọn ứng viên** | Áp cổng fold 0, riêng từng bản | Luôn | 0 |
| **4. Kiểm lại lr** | Thử lr 1e-4, đủ 2 epoch, ở tỉ lệ ứng viên, so với lr 2e-4 cùng tỉ lệ | Chỉ khi có ứng viên | 0–1,4 giờ |
| **5. K-fold** | Ứng viên cụm và mức hiện tại trên fold 1, 2; pair chạy K-fold riêng chỉ khi mức thắng khác cụm | Chỉ khi có ứng viên | 0–5,0 giờ cụm; pair +~5,6 giờ khi cần |
| **6. Chốt tỉ lệ** | Áp tiêu chí, ghi tỉ lệ từng bản kèm độ dao động | Luôn | 0 |

**Riêng tìm tỉ lệ:** ~5,7 giờ nếu không có ứng viên, ~12,1 giờ nếu có ứng viên cụm. **Cộng hai mục tiêu còn lại** (B8 cụm 2,1 giờ nếu chưa chạy ở bước 5, đo test ~1,5 giờ): **~9,3 đến ~13,6 giờ**, quỹ ~19,9 giờ. Pair phải K-fold riêng thì thêm ~5,6 giờ, thành ~19,3 giờ, chỉ dư ~0,6 giờ, không đủ cho chốt chặn C1–C3, nên báo và xin quyết định trước khi chạy. Cuối mỗi bước dừng báo Nhật Anh, có lệnh mới chạy bước sau.

### Bước 1. Chuẩn bị (không tốn GPU dài)
1. **Lưu đầu ra bản 25** sang notebook riêng `ft-qwen3-4b-b78-luu`, theo mẫu `B6_LUU`, không GPU, gồm cả `moi_nhat` và `opt.pt` của `cum_f1`. Kiểm trạng thái trong bản lưu đúng bước 1.102, qua đó kiểm luôn việc Kaggle gắn đầu ra bản 25 chứ không phải bản 26 bị hủy.
2. **Sửa `kaggle_train.py`, phần em tự làm:**
   - `trong_so_loai`, hàm thuần: tính tỉ trọng title theo đúng cách chia lô, và tìm `w_title` để đạt % mục tiêu bằng chia đôi, riêng từng bản. Đầu ra là tỉ lệ số lẻ.
   - Loss có trọng số trong `chay_b78`: token nhân trọng số của mẫu, chia cho tổng trọng số token; mỗi micro-batch theo tổng trọng số mẫu. Trọng số 1:1 phải về đúng công thức cũ. Đánh giá vẫn không trọng số. Việc không đặt trọng số giữ nguyên đường mã cũ.
   - Danh sách việc lấy từ cấu hình, mỗi việc có % title, lr, có đo loss cả fold hay không.
   - `chon_trang_thai`: khi gắn cả bản lưu và đầu ra chính notebook, lấy trạng thái tiến xa hơn. Hiện `sorted(...)[0]` sẽ chọn nhầm bản lưu cũ.
3. **Sửa `dong_goi_kaggle.py`:** thêm `luu_b78`, lệnh chạy theo từng bước, dùng lại `b78_tu_dong`, `chay_b5`, `tai_ket_qua`.
4. **Sonnet làm, em đọc lại và tự chạy test:** `tong_hop_ti_le.py` + test: so theo cặp từng fold bằng `bootstrap_chenh` từ bản dịch đã lưu, cổng fold 0, không học lệch, kết luận K-fold, bảng dòng mới, nội suy khi được duyệt.
5. **Chạy khói hai phần** như B7 + B8: dừng giữa một việc, phiên mới nối đúng. Phiên có một phép thử: loss có trọng số 1:1 trên lô thật trộn title và mô tả phải khớp `out.loss` của mô hình.

### Bước 2. Thử 2 mức trên fold 0
- **Vì sao đủ 2 epoch, không dò 1 epoch:** B6 đã đo thứ hạng đảo: epoch 1 lr 1e-4 thắng (0,3680 so với 0,3703), epoch 2 lr 2e-4 thắng (0,3587 so với 0,3512). Bản pair cũng chưa có adapter epoch 1 để chấm mốc. Chạy đủ 2 epoch so được thẳng với chrF đã có của `cum_f0`, `pair_f0`, và lần chạy thắng dùng lại làm fold 0 của K-fold, không phải chạy lại.
- **Vì sao hai mức này:** ~11% có ý nghĩa rõ (theo token, đo được riêng từng bản: cụm 10,74, pair 10,86), ~75% đủ xa về phía title. Mức gần 50% không phân biệt được với mức hiện tại.
- Cùng seed, cùng lô, cùng 1.000 mẫu với mốc nên so theo cặp được. Tỉ lệ trong báo cáo là số lẻ thật, ghi cả % và `w_title`. Không đo loss cả fold ở bước này, chỉ đo cho ứng viên ở bước 5, để bớt giờ.
- Làm cụm trước, pair sau, cả hai đều đủ 2 mức để mỗi bản có đường so sánh riêng (Nhật Anh chọn phương án B, không bỏ pair).

### Bước 3. Chọn ứng viên
- Một mức qua cổng thì lấy mức đó; cả hai qua thì lấy mức chênh lớn hơn; không mức nào qua thì giữ mức hiện tại và sang mục tiêu còn lại.
- **Điểm nội suy số lẻ là tùy chọn, xin duyệt riêng (+~1,4 giờ mỗi bản):** chỉ khi có mức qua cổng, nội suy bậc hai 3 điểm chrF trên logit(%) để tìm điểm tốt hơn nằm giữa, chạy xác nhận trên fold 0. Không ngoại suy ra ngoài khoảng đã thử. Sai số chrF ~±0,5 nên số lẻ này có độ dao động, không phải số chính xác. Trường hợp cao nhất không còn chỗ trong quỹ cho điểm này.

### Bước 4. Kiểm lại lr
- Chỉ thử **lr 1e-4**, đủ 2 epoch, ở tỉ lệ ứng viên, cùng fold 0. Bỏ lr 5e-5 vì ở B6 nó kém hẳn hai mức kia. Không dò lại rank, số epoch.
- Chỉ đổi lr khi chrF thắng, khoảng tin cậy nằm hẳn trên 0. Nếu đổi, lần chạy này chính là fold 0 mới của ứng viên, không phải chạy lại.
- Cả hai bản có ứng viên thì chỉ kiểm trên bản cụm, bản pair dùng lr đó như B6 và B7; nếu lr đổi thì pair phải chạy lại fold 0 (+~1,3 giờ), báo trước khi chạy. Chỉ pair có ứng viên thì kiểm trên bản pair.

### Bước 5. K-fold K=3
- **Cụm:** ứng viên trên fold 1, 2 (fold 0 có từ bước 2 hoặc 4) và mốc `cum_f1`, `cum_f2` (nối từ bước 1.102, chính là B8).
- **Pair (phương án B):** cùng mức thắng với cụm thì không K-fold riêng, tỉ lệ pair dùng kết quả fold 0 của nó cộng K-fold của cụm, ghi rõ giới hạn này. Mức thắng khác cụm, hoặc chỉ pair có ứng viên, thì chạy ứng viên và mốc pair trên fold 1, 2 (~5,6 giờ), báo trước khi chạy. Không có ứng viên thì giữ ~50%, không K-fold. C3 bật thì chạy mốc pair như B8.
- **Mỗi lần chạy:** 2 epoch, loss cả fold theo hướng, loại, category, chrF 1.000 mẫu cố định, tách dòng đã thấy và dòng mới, cùng mẫu giữa các mức. Mỗi việc ~1,4–1,5 giờ [ĐO từ phiên 1].
- **Dừng sớm:** ứng viên thua ở fold 1 thì không chạy fold 2 của ứng viên đó, vì tiêu chí (1) đã không thể đạt.
- **Lưu ý:** fold 0 dùng để chọn ứng viên nên số ở đó hơi lạc quan; fold 1, 2 mới là bằng chứng độc lập.
- Mỗi phiên tối đa 2,5 giờ, tự lưu, nối tiếp, tải số liệu và bản dịch về máy như B6.

### Bước 6. Chốt tỉ lệ
Áp 3 tiêu chí, riêng từng bản. Kết quả: tỉ lệ title : mô tả của bản cụm và bản pair, kèm độ dao động và bảng K-fold. Giữ mức hiện tại cũng là kết quả hợp lệ. Xong bước này mới sang hai mục tiêu còn lại.

### Sau 6 bước: hai mục tiêu còn lại
1. **Không có ứng viên thì chạy nốt B8 cụm** ở mức hiện tại (2,1 giờ). **Tính C1–C3** cho cấu hình cuối bằng `tong_hop_b78.py`, thêm hậu tố tên việc nếu đổi tỉ lệ.
2. **Sửa trước khi đo test, chấm lại trên máy không tốn GPU:** mốc tham chiếu cho `giu_so`, `giu_latin`, `sai_chu_viet`; xử lý lời dẫn zero-shot chép "产品标题（仅上下文"; đổi trần `max_new_tokens` của `sinh` từ độ dài tham chiếu sang độ dài nguồn, vì trên test và extension không được dựa vào tham chiếu.
3. **Lưu adapter cuối** sang notebook riêng trước lần đẩy kế tiếp.
4. **Đo test một lần (B10):** Dataset test riêng (D17), zero-shot và mô hình fold 0 ở tỉ lệ đã chốt của hai bản, ~1,5 giờ. Không đổi tỉ lệ sau khi thấy test.
5. **B9** Gemini ~8 lần gửi, Claude hiệu chuẩn 50 dòng. **Báo cáo:** tỉ lệ từng bản, bảng K-fold, giới hạn. Hướng dẫn cho mô hình khác: tỉ lệ là điểm xuất phát.

### Cố ý không làm, để khỏi nặng tay
Không thêm mức thử ngoài 2 mức, trừ điểm nội suy khi được duyệt; không chia tỉ lệ theo hướng (ghi số theo hướng để xem sau); không dò lại rank, số epoch, chỉ kiểm 1 mức lr; không đo loss cả fold ở bước 2; gác thử 3 epoch; K-fold ứng viên không quá K=3.

### Giới hạn cần ghi trong báo cáo
Tham chiếu title có lỗi lõi ~5–14% nên tăng trọng số title có thể học lỗi nhiều hơn và điểm title bị nhiễu. Tỉ lệ đo trên title chưa sửa; khi có title đã sửa (D15), so lại nhanh trên fold 0 trước mô hình cuối, ngoài ngân sách này. Không dò lại rank, số epoch ở tỉ lệ mới. K=3 chỉ cho 3 số, nhìn được chiều hướng chứ khó khẳng định chắc. "Tốt nhất" dựa trên cách tính title và mô tả ngang nhau. Tỉ lệ là của Qwen3-4B trên dữ liệu này.

### Kiểm chứng
- `test_kaggle_train.py`: tỉ trọng tính tay đúng; tìm `w_title` đạt % mục tiêu; trọng số 1:1 về công thức cũ; chọn đúng trạng thái tiến xa hơn. `test_tong_hop_ti_le.py`: biên các cổng, khoảng tin cậy chứa 0 là không qua, học lệch, thiếu dữ liệu không tính là đạt, nội suy có cực trị và đơn điệu.
- Trên máy: chấm lại bản dịch mức hiện tại phải ra đúng 75,7613 (cụm) và 75,2587 (pair).
- Khói: nối phiên đúng số bước; `cum_f1` nối từ 1.102; phép thử trọng số 1:1 khớp; `khop_b6` true; không có file test trong input.
- Thật: đầu mỗi phiên xem log trực tiếp có "chay tiep tu" và số bước đúng; so tỉ trọng title thực tế với % mục tiêu.

### Kết quả bước 2, 3, 6 (06/11) [ĐO], fold 0, 1.000 mẫu, rà bằng số, không phải đánh giá của người
Phiên 20.421 giây (5,67 giờ), bản 29, không lỗi, không bất thường. Chênh so với mức ~50% tính theo câu, khoảng 95%; các cột còn lại là chrF corpus.

| Bản | Mức title | chrF | Chênh [95%] | Title | Mô tả | zh→vi | Mô tả dòng mới zh→vi | Sinh lặp | Cổng |
|---|---|---|---|---|---|---|---|---|---|
| Cụm | ~50% | 75,76 | mốc | 60,22 | 91,30 | 75,90 | 77,17 | 1 | mốc |
| Cụm | 10,74% (w 0,120) | 76,14 | +0,11 [-0,39; +0,61] | 60,36 | 91,92 | 76,43 | 79,05 | 1 | không qua |
| Cụm | 75% (w 3,039) | 75,66 | -0,06 [-0,51; +0,42] | 60,67 | 90,65 | 75,65 | 74,54 | 3 | không qua |
| Pair | ~50% | 75,26 | mốc | 60,35 | 90,17 | 76,17 | 76,36 | 0 | mốc |
| Pair | 10,86% (w 0,118) | 75,16 | -0,08 [-0,54; +0,36] | 59,39 | 90,94 | 76,00 | 78,68 | 1 | không qua |
| Pair | 75% (w 2,959) | 74,60 | -0,58 [-0,99; -0,19] | 59,75 | 89,46 | 75,49 | 74,07 | 1 | không qua |

**Kết luận theo tiêu chí đã khóa:** giữ ~50% cho cả hai bản, không vào K-fold ứng viên. **Giới hạn:** chỉ fold 0; clip gradient cắt 262/263 và 263/263 bước toàn title ở mức 75% nên tác dụng thật thấp hơn danh nghĩa; mức ~11% gần như không bị clip (12/1.837). **Tín hiệu để xem sau, chưa kiểm định, không dùng để đổi kết luận:** mức ~11% nâng mô tả dòng mới zh→vi ~1,9–2,3 điểm, title gần như không đổi ở cụm (+0,14) nhưng giảm ở pair (-0,96). Quỹ còn ~14 giờ [ƯỚC]. Kết quả ở `ft_run/kaggle_tl/phien_1_20261105_231941/tong_hop/`.

**Kiểm tín hiệu dòng mới, ngoài 6 bước, Nhật Anh duyệt 06/11, 0 GPU [ĐO]:** `kiem_dong_moi.py`, mức ~11% trừ 1 : 1, dòng mới mô tả zh→vi, theo sản phẩm, khoảng 95%, luật khóa trước khi xem số: cụm +1,43 [+0,54; +2,32], pair +1,36 [+0,57; +2,19], cả hai qua. Độ nhạy không phân biệt hoa thường và theo cụm chung dòng: cụm +1,44 [+0,71; +2,12], pair +0,95 [+0,22; +1,69]. Mô tả zh→vi toàn bộ: cụm +0,69 [+0,39; +1,00], pair +0,82 [+0,52; +1,16]. Title zh→vi: cụm +0,24 [-0,86; +1,35], pair -1,19 [-2,20; -0,25]. Fold 0 là nơi phát hiện tín hiệu nên chưa đổi 1 : 1, chỉ đáng kiểm trên fold khác. **Đề xuất, chờ lệnh:** lưu đầu ra bản 29 trước khi đẩy; chạy `cum_f1` (mốc, nối từ bước 1.102), `cum_f1_p11`, `cum_f2` (~3,5 giờ [ƯỚC]). Luật đề xuất cho fold 1: dòng mới mô tả zh→vi khoảng 95% nằm hẳn trên 0, title không tụt quá 1 điểm; không đạt thì giữ 1 : 1. Đạt thì mới chạy `cum_f2_p11` (+~1,35 giờ) và áp tiêu chí 3 fold.

**Kết quả fold 1 (06/11, bản 32, 3,63 giờ) [ĐO]:** ĐẠT luật khóa: dòng mới mô tả zh→vi +1,95 [+1,13; +2,82], title hai hướng +0,33 [-0,64; +1,30]. chrF chung corpus +0,38 (fold 0), +0,40 (fold 1); theo câu +0,11 [-0,39; +0,61], +0,40 [-0,05; +0,85]; không học lệch ở cả hai fold. B8 mức 1 : 1: 75,76, 74,74, 74,16, trung bình 74,89 ± 0,81; C1–C3 không bật. Quỹ còn ~10,1 giờ [ƯỚC]. Kết quả ở `ft_run/kaggle_b8f1/phien_1_20261106_102741/`. Chưa đổi 1 : 1, chờ lệnh `cum_f2_p11`; đạt 3 fold thì kiểm lr (Nhật Anh đã chọn) và quyết bản pair. Trước lần đẩy kế phải lưu đầu ra bản 32.

**Kết quả fold 2 (06/11, bản 34, 1,49 giờ) [ĐO]:** ĐẠT đủ luật khóa (Prompt 796): dòng mới mô tả zh→vi fold 2 +2,69 [+1,76; +3,69], title hai hướng +0,33 [-0,46; +1,16]. Tiêu chí 3 fold: chrF chung +0,38, +0,40, +1,15, trung bình 0,64 > độ lệch chuẩn 0,44; không học lệch. **Bản cụm đổi sang tỉ trọng title 10,74% (w_title ~0,12), kiểm lr xong mới là bộ cuối.** Giới hạn: fold 2 tăng chung chủ yếu do vi→zh (+1,88; zh→vi +0,43); bootstrap không tính dao động do huấn luyện lại; mới so 3 mức. Pair vẫn 1 : 1, chưa kiểm trên fold khác. Quỹ còn ~8,5 giờ [ƯỚC]. Kết quả ở `ft_run/kaggle_b8f2/phien_1_20261106_165058/`. Đầu ra bản 32 đã lưu ở `ft-qwen3-4b-b8f1-luu`; trước lần đẩy kế phải lưu đầu ra bản 34. Chờ lệnh kiểm lr và quyết pair.

## 1. Bối cảnh và mục tiêu

Hai bộ song ngữ thương mại điện tử sau Judge (DeepSeek chọn giữa Gemma 4 và Qwen3-14B) trên HF `ntquang0410/zh-vie_ecom`, thư mục `data/LLM/bilingual_judged` (commit `82aa9c9`): `bo_cum_chon_deepseek.parquet` (cụm), `bo_pair_chon_deepseek.parquet` (pair), kèm hai file `*_danh_dau`. Mỗi file 25.575 sản phẩm (1688 14.837 zh→vi, Tiki 10.738 vi→zh). Đã tải về `data/processed/bo_chon_hf/`.

**Mục tiêu (Nhật Anh chốt 04/11):**
- (b) Giữ cả hai bộ, chỉ cần biết **mỗi bộ có ổn không** (không chọn một bộ, không so hơn thua).
- (c) Có **bộ siêu tham số** trên baseline **Qwen3-4B + LoRA** để chuẩn bị fine-tune các mô hình nhóm đang khảo sát (chưa chốt danh sách).
- **(d) MỤC TIÊU CUỐI CÙNG (Nhật Anh chốt 05/11, tối):** khi huấn luyện, nên cho **title và mô tả tỉ lệ bao nhiêu** thì bản dịch tốt nhất, để mô hình không học lệch. Tỉ lệ ở đây là trọng số trong hàm loss, do người huấn luyện đặt, chọn bằng thí nghiệm. Hiện [ĐO] ~49,6% title, ~50,4% mô tả, trong khi title chỉ 10,7% token đích. "Học lệch" [MẶC ĐỊNH, chờ duyệt] = một loại tăng điểm nhưng loại kia tụt quá 1 điểm chrF. Thí nghiệm chờ Nhật Anh duyệt.
  - **Khảo sát riêng từng bản, không GPU [ĐO, 05/11] (% title):** trọng số loss hiện tại cụm 49,56, pair 50,26; theo token đích cụm 10,74, pair 10,86; theo tổng lượng lỗi fold 0 sau 2 epoch (loss × token) cụm 55,10, pair 56,45. Theo hướng, lượng lỗi: zh→vi cụm 52,78, pair 54,52; vi→zh cụm 61,70, pair 61,73. Khảo sát chỉ cho ứng viên, tỉ lệ tốt nhất phải đo bằng huấn luyện rồi chấm bản dịch.
  - **Cách làm: xem mục 0.**

**Ứng dụng đích:** extension trình duyệt: mở trang mua sắm Trung Quốc, bấm là dịch sang tiếng Việt (zh→vi). Vẫn huấn luyện **cả hai hướng** để mô hình học nhiều hơn. Dữ liệu chỉ gồm **title và thuộc tính sản phẩm** (không có đánh giá khách, chữ giao diện), extension chỉ nên dịch hai loại đó.

**Tài nguyên:**
- Kaggle: notebook `nlp-group3` (tài khoản `trinhnhatanh`), GPU **RTX PRO 6000** có do gắn cuộc thi ARC Prize 2026; phiên hiện **tắt internet**; Kaggle Models không có Qwen3-4B nên đóng gói thành Kaggle Dataset.
- Người chấm: không có người HSK 5. **Claude Pro** (không API; Weekly 27% ngày 05/11; ~1 điểm mỗi 120 mục) + **Gemini Pro bản web** (gửi CSV thủ công).
- Thời gian GPU có hạn (Nhật Anh 05/11): không chạy thứ không phục vụ trực tiếp mục tiêu (b), (c).

**Ràng buộc từ điều lệ nhóm** (`CLAUDE.md` §5.7): chia theo `product_id`, cùng `title_zh` cùng một phía; tập test cần người hoặc rà soát; không đẩy HF khi chưa có lệnh; ghi mọi prompt vào `WALKTHROUGH.md`. Kết luận rà bằng LLM phải ghi đúng là "rà soát bằng LLM", không ghi "người rà".

## 2. Quyết định đã chốt (Nhật Anh, 04/11/2026)

| # | Quyết định |
|---|---|
| D1 | Kết quả cuối: (b) cả hai bộ có ổn không + (c) bộ siêu tham số. **Không chọn một bộ, bỏ so cặp "bản nào hơn".** |
| D2 | LoRA; danh sách mô hình sau chưa chốt; siêu tham số Qwen3-4B làm điểm xuất phát |
| D3 | Huấn luyện và đánh giá **cả hai hướng**; đánh giá chính nhấn zh→vi (demo) |
| D4 | Không có người HSK 5; chấm bằng LLM (xem D12) |
| D5 | Ngưỡng "ổn" của dữ liệu: **lỗi nặng ≤ 5%**, tính riêng từng hướng, từng loại (title, mô tả) |
| D6 | Không chờ Huy chạy Judge rộng hơn cho pair; ghi rõ độ phủ judge pair 14,5% |
| D7 | Không làm thí nghiệm title đo giá trị Judge |
| D8 | Chạy trên notebook Kaggle `nlp-group3`, RTX PRO 6000; đóng gói mô hình, thư viện, dữ liệu thành Kaggle Dataset |
| D9 | Chưa có tập test; tạo ở B1 |
| D10 | K-fold vẫn dùng; dữ liệu đầy, bỏ mẫu con |
| D11 | Bỏ DeepSeek làm người chấm (đã chọn nhãn, tránh thiên vị vòng tròn) |
| **D12** | **Phương án 2 (kết hợp):** **Gemini Pro web chấm phần lớn B2 và B9; Claude chỉ hiệu chuẩn** trên mẫu chung nhỏ (vì chỉ có Claude Pro, Weekly 17%) |
| D13 | Gửi Gemini bằng CSV lô nhỏ, theo quy trình mục 6 |
| D14 | (05/11) Cổng title dùng chỉ số **nặng lõi** (sai_nghia, so_don_vi, thuong_hieu_ma, sot_chu_nguon); nặng do thiếu chỉ báo cáo kèm. Mô tả vẫn tính theo mảnh. |
| D15 | (05/11) **Hướng A cho title:** cho qua có điều kiện để đi tiếp. Huấn luyện cả title và mô tả; báo cáo ghi rõ mô tả ổn, title chưa đạt, lỗi lõi khoảng 5–14%. Trước mô hình cuối cho extension mới dò và sửa title (`loc_title.py`, so Gemma với Qwen, từ điển lỗi lặp, Gemini Pro chấm phần đã sàng). |
| D16 | (05/11) **B6 dò loại dần trên 1 fold, ~7–8 giờ; K-fold giữ ở B8.** Thay cho 18 lần chạy (6 cấu hình × K=3, ~33 giờ). Điều chỉnh D10 ở riêng bước B6. |
| D17 | (05/11) Gói dữ liệu huấn luyện trên Kaggle **chỉ có mẫu pool**, không có tập test; test lên Dataset riêng khi cần đánh giá. |
| D19 | (05/11) B6 tối đa 2 epoch. **Không bỏ K-fold (B8 K=3), không bỏ cấu hình LoRA attention** (Nhật Anh chốt); giữ đủ lưới 6 cấu hình; đo test đủ 5.116 mẫu. Ước ~12–17 giờ. |
| D21 | (05/11) **Mục tiêu (d): tỉ lệ title : mô tả riêng từng bản, khảo sát bằng K=3 trên Qwen3-4B + LoRA, thử 2 mức trên fold 0 trước để chọn ứng viên** (mục 0). Nhật Anh chốt: giữ cả ba mục tiêu, ưu tiên tỉ lệ trước; đổi tỉ lệ thì kiểm lại lr. Đổi dò thô 1 epoch thành đủ 2 epoch vì B6 đã đo thứ hạng đảo giữa epoch 1 và 2. Nhật Anh chọn phương án B cho pair (đủ 2 mức, K-fold riêng chỉ khi mức thắng khác cụm) và giữ bước kiểm lại lr. |
| D20 | (05/11) **Chốt từ B6: lr2e-4, rank 64, 2 epoch, LoRA mọi lớp.** Chấm thêm chrF LoRA attention so với lr1e-4_r16 ở epoch 1, gộp vào phiên B7 (Nhật Anh duyệt). Adapter B6 lưu ở notebook riêng `ft-qwen3-4b-b6-luu`. Chốt chặn không tự bật thêm việc, `tong_hop_b78.py` tính rồi người quyết. Sửa bộ đo giu_so, giu_latin, sai_chu_viet trước khi đo test, bắt đầu bằng mốc tham chiếu. |
| D18 | (05/11) **Cắt phần thừa sau B6, có chốt chặn tự kiểm (mục 4b):** B7 1 cấu hình; B8 K=3, dùng lại fold 0, bỏ seed phụ, bản pair chỉ chạy thêm khi lệch; B10 không huấn luyện lại, đo test bằng mô hình fold 0; B9 ~400 dòng; bỏ COMET vòng này. Điều chỉnh D10 ở B8. |

## 3. Hiện trạng dữ liệu

Hash 4 file khớp `manifest.json` **[ĐO]**. Không có lỗi chặn.

| Hạng mục | Kết quả [ĐO] |
|---|---|
| Cấu trúc | 25.575 sp duy nhất, 0 null/rỗng; cùng thứ tự/category; **title giống 100%**, chỉ mô tả khác nhau |
| Quan hệ hai bộ | Pair có trước (bộ 1 = 1688, bộ 2 = Tiki; dịch từng tên/giá trị riêng, không ngữ cảnh); cụm làm sau (bộ 3 = Tiki, bộ 4 = 1688) là **hiệu đính bản nháp pair** có ngữ cảnh, khóa mục từ điển/quy tắc/thương hiệu/số đo/mã **[THEO TÀI LIỆU, CLAUDE.md 23-26/10]**. Dữ liệu chỉ cho thấy hai bộ **không** là bản sao cơ học của nhau |
| Độ giống nhau | Token mô tả trùng 99,1%; 43,5% sp có mô tả dịch giống hệt; 70,9% sp trùng ≥99%. Trên 81,5% sp đo được từng mảnh: **4,8% mảnh khác** (tên 0,7%, giá trị 4,2%), phần lớn đồng nghĩa/chuẩn hóa đơn vị/từ vay (500GR→500g; miliampe→mAh; Android Box) |
| Judge tác động | Judge chỉ xem các "cuộc đấu" (Gemma ≠ Qwen, cả hai không rỗng): cụm 17.170/441.842 mảnh (3,9%), pair 195.311/872.034 (22,4%, 31.510 chưa chấm). Đổi mô tả: cụm 615 (0,14%), pair 9.054 (1,0%); đổi title 7.604 (29,7%). Judge chọn **một trong hai**, không chứng nhận đúng/sai tuyệt đối |
| Căn dòng | pair 100%; cụm lệch 53 sp vì `; ` nằm trong giá trị (14 sp lệch ≥3) |
| Nhất quán | tên nguồn >1 cách dịch: pair 0/3.044, cụm 40/3.044 (1,3%); giá trị: pair 459/56.193 (0,82%, chưa rõ nguyên nhân), cụm 676 (1,20%) |
| Rò rỉ | chia ngẫu nhiên: 2,8% title kiểm trùng nguồn với train; chia theo nhóm: 0%. Dòng mô tả kiểm trùng nguyên dòng với train: 86,7% (ngẫu nhiên), 85,5% (theo nhóm) |
| Dòng "mới" (chia nhóm) | 15,3% số dòng nhưng 33,8% ký tự dịch (Tiki 24,2%/49,4%; 1688 13,4%/32,6%) |
| Nhóm gần-trùng | 1.903 nhóm ≥2 sp phủ 7.322 sp (Tiki 4.983 = 46%, 1688 2.339 = 16%), lớn nhất 300 |
| Dư thừa | pair: 436.017 dòng, 75.433 duy nhất theo (category, dòng) |
| Nhãn "bạc" (mô tả) | cụm: Gemma 49,3% / quy tắc 31,7% / chưa rõ 18,9% / Qwen 0,1%; pair: 36,2% / 27,3% / 35,0% / 1,5%; title: Gemma 19.261 / Qwen 6.314 |
| Độ phủ judge | pair chấm 5.000/34.534 đơn vị; cụm 4.439/4.446. README HF: chưa người kiểm chứng toàn bộ, không coi là gold |
| Hướng/khối lượng | 1688 = 58% sp, ~82% token mô tả; Tiki 42% sp |
| Token (Qwen3 + 40 khung) | mô tả TB ~565 token/mẫu, p99 ~1.881; >2.048: 181 (cụm)/173 (pair); >4.096: 7; ~17,0M token/epoch (cụm), ~16,3M (pair), gồm title + mô tả |
| Bản trước Judge | `data/processed/ft_clean/ft_buoc4_giu.parquet` (title khác sau Judge đúng 7.604). Không dùng (D7), ghi để tra cứu |
| Nhỏ | 13 title Tiki giữ nguyên (tên model); 4 title 1688 dài >6× nguồn (≥2 lặp cụm); 64/381 nhóm cùng title nguồn có 2 bản dịch; "mất số" 4,3% title chủ yếu dương tính giả (25款→2025). Dữ liệu cá nhân: không thấy |

Phía **đích của cả hai hướng đều là bản dịch máy**, nên dữ liệu không bảo đảm "tự nhiên". Đảo chiều Tiki (dịch ngược) là ý tưởng chưa đo, **ngoài phạm vi**.

## 4. Quy trình B0–B10

**Thứ tự và mốc:** B0 → B1 → B1b → B2, B3, B4, B5 (đã xong) → **B6 → B7 (mốc 1: có bộ siêu tham số, mục tiêu c, ~9 giờ)** → **B8 → đo test B10 → B9 (mốc 2: biết mỗi bộ có ổn không, mục tiêu b)**. Xong mốc 1 Nhật Anh có thể dừng xem kết quả trước khi đi tiếp. Cổng B2 đi theo D14, D15.

### B0. Đóng băng dữ liệu, đóng gói mô hình và thư viện
- **Đã làm (05/11):** ghi revision HF `82aa9c9` + SHA256 4 file vào `data/processed/ft_run/DU_LIEU_DONG_BANG.json`. Mô hình và thư viện đóng gói bằng hai notebook có Internet chạy trên Kaggle (`ft-qwen3-4b-mo-hinh`, `ft-qwen3-4b-thu-vien`: peft, sacrebleu và phụ thuộc thiếu), đầu ra gắn thẳng vào `nlp-group3` (`dong_goi_kaggle.py`).
- **Mô hình:** [MẶC ĐỊNH] `Qwen/Qwen3-4B`, tắt suy luận (`enable_thinking=False`).
- **Qua khi:** hash khớp; notebook mở được các Dataset **khi tắt internet**.

### B1. Chia tập dùng chung
- **Làm:** gom nhóm gần-trùng (cách làm thật ở mục "ĐÃ LÀM" bên dưới, thay MinHash bằng phép nối chính xác). Khóa **10% làm test**, phân tầng hướng × category. Phần còn lại ("pool") chia **5 fold** theo nhóm, phân tầng hướng × category. **Một phép chia cho cả hai bộ.**
- **Ra:** `data/processed/ft_run/split.parquet` (product_id, nhom_id, fold5, la_test) + báo cáo tỉ lệ.
- **Qua khi:** không nhóm nào nằm hai phía; tỉ lệ hướng × category sát gốc.
- **ĐÃ LÀM (04/11, khóa) [ĐO]:** thay MinHash bằng phép nối tập **chính xác** (lọc tiền tố AllPairs, kiểm khớp vét cạn) vì mã cũ bỏ xô ≥60 sp và dùng `hash()` ngẫu nhiên theo phiên. **6 góc nhìn, xét cả hai phía ngôn ngữ cho mọi sp** (chặn rò chéo hướng): title zh/vi (4-gram, **ngưỡng 0,5**, chặt hơn kế hoạch), mô tả zh/vi bản pair và bản cụm (tập thuộc tính, **ngưỡng 0,7**; 0,6 làm mô tả khuôn mẫu nối thành nhóm 2.992 sp). **v2 (05/11, bản khóa hiện hành):** thêm luật **cùng thương hiệu cụ thể + title chứa trong nhau** (hệ số chứa 4-gram ≥0,7, title zh hoặc vi), vá ca rò bị bỏ sót khi kiểm mẫu B3: sp test "Trà kỷ tử" (Dược Liệu Tuệ Tâm) là tin đăng anh em của 3 sp pool cùng hiệu (100/200/500gr), Jaccard thấp vì title ngắn. Khóa lại trước khi bất cứ thứ gì dùng tập test. Kết quả v2: 18.448 nhóm, 2.157 nhóm ≥2 sp, lớn nhất 969; **test 2.558 sp (zh→vi 1.485, vi→zh 1.073)**, pool 23.017 (fold 4.484/5.084/4.483/4.483/4.483). Test 33,9% sp thuộc nhóm ≥2 (pool 36,6%). **Kiểm độc lập `kiem_ro_ri.py`** (vét cạn mọi cặp test×pool và fold×fold khác bằng chỉ mục ngược, mã riêng): 0 vi phạm cả 6 góc nhìn, 0 title trùng chính xác; lần chạy khô đầu đã bắt được 1 sp test + 19 cặp fold giống ≥0,7 qua mô tả bản cụm → thêm 2 góc nhìn cụm vào gom nhóm. Cặp sát ngưỡng nhất (đọc tay) là sp khác nhau cùng loại/shop. Dòng mô tả test đã có nguyên dòng trong pool 84,7% (khuôn mẫu thuộc tính, không phải rò rỉ sp). Khóa: `DU_LIEU_DONG_BANG.json` (hash dữ liệu HF + hash split + hash danh sách test), `chung.doc_split()` kiểm hash, `chia_tap.py` từ chối ghi đè; danh sách test ở `ft_run/test_khoa/ds_test.txt`.

### B1b. Đóng gói dữ liệu và chia sẻ mã test
- **Làm (D17):** đóng gói Dataset riêng tư `ft-qwen3-4b-pool` **chỉ gồm mẫu pool**, kiểm từng dòng không có mã test trước khi đẩy (`dong_goi_kaggle.py du_lieu`). **Đã xong 05/11.** Không đưa 4 file parquet và `split.parquet` lên như bản cũ vì chúng chứa tập test. Gửi danh sách mã sản phẩm test cho nhóm khảo sát mô hình để không ai huấn luyện trên đó [CHỜ: Nhật Anh chọn thời điểm gửi].

### B2. Kiểm dữ liệu (cổng 5%)
- **Mục đích:** biết dữ liệu **đúng/sai tuyệt đối** trước khi tốn giờ GPU; Judge chỉ chọn một trong hai ứng viên và mới xem 3,9% (cụm) / 22,4% (pair) số mảnh.
- **Mẫu (lấy từ tập test, phân tầng category, sản phẩm không trùng mẫu B9):**
  - M1: 300 sp zh→vi + 200 sp vi→zh; mỗi sp chấm **title** (chung hai bộ) và **mô tả bản cụm** → ~1.000 mục. **[ĐO 05/11] Nguồn mô tả bản cụm khác bản pair ở 18,3% sp (4.674, gần hết 1688: cụm có thêm thuộc tính như 颜色/尺寸)** → mục mô tả M1 dùng **nguồn của chính bản cụm**, mảnh = thuộc tính nguồn cụm (tách "; "); M2 chỉ lấy sp có nguồn cụm trùng khớp nội dung nguồn pair. Ước lượng pair chỉ phủ thuộc tính chung (ghi trong báo cáo).
  - M2: 300 **mảnh khác nhau** giữa cụm và pair (150/150 hai hướng; chỉ lấy từ sp căn hàng được, 81,5% sp, nên hơi lệch về sp cấu trúc đơn giản; ghi trong báo cáo), chấm **từng bản** (không chọn hơn) → 600 lượt.
  - **Đơn vị tính lỗi:** title = 1 title; mô tả = **1 mảnh** (1 dòng nguồn của bản pair, được đánh số). Tỉ lệ mô tả = Σ mảnh lỗi nặng ÷ Σ mảnh đã chấm. (Nếu tính "cả mô tả có ≥1 lỗi nặng" thì mô tả ~12 mảnh/sp sẽ vượt 5% oan; số đó chỉ báo thêm làm thông tin phụ.)
  - **Bản pair:** `ty_le_pair = ty_le_cum(M1) + 0,048 × (ty_le_M2_pair − ty_le_M2_cum)`, vì chỉ 4,8% mảnh khác nhau giữa hai bản.
- **Người chấm:** **Gemini Pro web chấm toàn bộ M1 + M2** (CSV, mục 6). **Claude hiệu chuẩn trên mẫu chung** M1c: 100 sp (60 zh→vi + 40 vi→zh, lấy ngẫu nhiên từ M1; 200 mục) + M2c: 100 mảnh (200 lượt) ≈ 400 lượt. Claude không thấy kết quả của Gemini khi chấm.
- **Cách chấm (mù, cùng rubric cho cả hai):** chỉ thấy nguồn và bản dịch, không biết bộ nào/mô hình nào. Mức: **Đạt / Lỗi nhẹ / Lỗi nặng**; loại lỗi: sai nghĩa; sai số/đơn vị; sai thương hiệu/tên model; thiếu hoặc thừa; sai thuật ngữ; chữ nguồn sót; văn phong. Lỗi nặng = sai nghĩa, mất/sai số-đơn vị, sai thương hiệu-tên model. **Dùng bốn điểm trung lập của Judge** (viết thương hiệu Latin hay chữ đích; từ ngoại đã ngoại trong nguồn; từ đồng nghĩa ngang vị thế kể cả dạng ngắn/đầy đủ; định dạng số/đơn vị) với **một ngoại lệ**: đổi thương hiệu/model/tên sản phẩm thành tên KHÁC là lỗi nặng (lấy từ `QUY_TAC_CHINH_SACH` trong `scripts/judge/so_cap_rubric.py`; lưu rubric thành `scripts/ft_train/rubric_kiem.md`, ghi hash để hai bên dùng đúng một bản).
- **Cổng D5 và hiệu chỉnh:** tỉ lệ lỗi nặng của Gemini trên mẫu đầy, **riêng title và mô tả**, từng hướng, từng bộ. Hiệu chỉnh theo Claude, **gộp chung mọi nhóm trên mẫu chung M1c+M2c** (mẫu từng nhóm quá nhỏ, 40–60 mục, gây nhiễu): `bo_sot = max(0, (số mục Claude lỗi nặng mà Gemini không) − (số mục Gemini lỗi nặng mà Claude không)) ÷ số mục chung`; `ty_le_hieu_chinh = ty_le_Gemini_mau_day + bo_sot`, cộng cho mọi nhóm. Cổng qua khi `ty_le_hieu_chinh ≤ 5%` ở mọi nhóm. Báo kèm Wilson 95% và Cohen kappa/tỉ lệ bất đồng Gemini–Claude về "lỗi nặng"; [MẶC ĐỊNH] kappa < 0,4 hoặc Claude thấy lỗi nặng mà Gemini cho qua trong >30% ca → coi Gemini không đáng tin, Claude mở rộng mẫu hoặc báo Nhật Anh.
- **Vùng xám:** `ty_le_hieu_chinh` 5–6,5% → Nhật Anh quyết (mở rộng mẫu hay dừng). Mẫu vi→zh nhỏ hơn (n=200, Wilson ~±3 điểm) nên luôn báo CI.
- **Nếu không đạt:** dừng, xem lỗi tập trung ở loại thuộc tính/hướng nào, loại hoặc sửa phần đó, báo Nhật Anh; **không** huấn luyện trên dữ liệu chưa qua cổng.
- **Tham chiếu test:** mục Đạt giữ nguyên; mục lỗi nặng dùng bản sửa (Gemini/Claude, ghi nguồn và cờ). Báo cáo điểm trên **toàn test** (tham chiếu bạc) và trên **tập đã kiểm** (tham chiếu tin cậy hơn).
- **Thử trước:** gửi Gemini 1 lô 20 dòng và cho Claude chấm thử 20 mục để đo % Weekly thực tế, kiểm quy trình CSV, rồi mới chạy cả mẫu.
- **Công thủ công [ƯỚC]:** ~1.600 dòng ÷ ~50 dòng/lô ≈ 32 lần gửi.
- **ĐÃ THỬ (05/11) [ĐO], rà soát bằng LLM:** Opus chấm 180/400 mục hiệu chuẩn: title 16/45 Nặng (zh→vi 10/26, vi→zh 6/19), mô tả 1/785 mảnh, M2 0/88. Sonnet phiên mới chấm mù 45 title đó: 6/45, đều nằm trong 16 của Opus (kappa "nặng" 0,44; "nặng lõi" 0,76); 6 mô tả, khớp Opus. Gemini Pro lô thử 20 mục: khớp Opus 18/20, title 4/10, mô tả 1/157 (cùng mảnh); chạy bằng Flash trước đó cho K00008, K00012 là Nhẹ. Weekly 23% → 25% sau 180 mục Opus + 51 mục Sonnet (thanh chỉ hiện số nguyên). **Nhật Anh duyệt (05/11):** giữ nguyên rubric, báo cáo tách **hai tầng** (nặng lõi: sai_nghia, so_don_vi, thuong_hieu_ma, sot_chu_nguon; nặng do thiếu: chỉ thieu_thua), chấm **200 title mới** (100/100, 4 lô Gemini Pro), Opus chấm mù 40 title trong đó; `scripts/ft_train/kiem_title.py`. Không gửi 53 lô trộn khi chưa cần; quyết cổng title sau khi có kết quả. **Đã chấm đủ 400 mục hiệu chuẩn (Opus, 05/11):** title 26/100 Nặng (zh→vi 15/60, vi→zh 11/40), nặng lõi 14/100 (zh→vi 6/60, vi→zh 8/40); mô tả zh→vi 3/1.466 mảnh, vi→zh 0/282; M2 4/200 (cụm 2/100, pair 2/100). 45 title đầu 35,6%, 55 title sau 18,2%: có thể do ngẫu nhiên hoặc Opus nhẹ tay dần, chưa phân biệt được. **Gemini Pro (chế độ suy luận) 200 title mới (05/11):** nặng 50/200 (zh→vi 41/100, vi→zh 9/100), nặng lõi 10/200 = 5% (Wilson 2,7–9,0%; zh→vi 6/100, vi→zh 4/100); không dùng mức Nhẹ lần nào (150 Đạt, 50 Nặng); 9 ca Nặng trái chữ rubric (8 chỉ thiếu từ kênh bán/quảng cáo, 1 khác định dạng số). Trên 59 title chung với Opus: kappa nặng 0,41; kappa nặng lõi 0,16 (cùng ca nhưng khác mã lỗi là phổ biến); Opus lõi 4/59, Gemini lõi 5/59. Weekly 23% → 27% sau ~490 mục [ĐO, số nguyên]. Cổng title chưa quyết, chờ Nhật Anh. **Áp công thức cổng:** bo_sot lõi = max(0, 3−4)/59 = 0, nên zh→vi 6% (vùng xám), vi→zh 4% (qua); nhưng kappa lõi 0,16 < 0,4 và Gemini cho qua 3/4 ca lõi của Opus, nên theo quy tắc mục B2 Gemini chưa đủ tin ở tầng lõi. **Bộ lọc tự động `loc_title.py` [ĐO]:** gắn cờ 6.360/25.575 title (vi→zh 5.021, zh→vi 1.339); trên 277 title đã chấm: bắt 14/26 title có lỗi lõi, 94 cờ thì 72 không ai chấm nặng. Độ chính xác thấp, chỉ dùng để sàng, không dùng để sửa tự động.

### B3. Dựng mẫu huấn luyện
- **Làm:** mỗi sp sinh 2 mẫu (title, mô tả); chỉ dẫn nêu **hướng** và loại văn bản; [MẶC ĐỊNH] **đưa title nguồn vào chỉ dẫn mô tả cho cả hai nhánh** (~30 token/mẫu); cụm giữ `; `, pair giữ xuống dòng; chat template Qwen3 `enable_thinking=False`; loss chỉ trên phần dịch; `max_len` 2.048; mẫu quá dài (~0,7%) cắt ở ranh giới thuộc tính nguồn, mẫu nào không căn được thì loại khỏi huấn luyện và báo số lượng. [MẶC ĐỊNH] pair **không khử trùng dòng lặp**; báo cáo tách "dòng đã thấy / dòng mới".
- **Ra:** `samples_{cum,pair}.jsonl` kèm fold, hướng; `scripts/ft_train/dung_mau.py` + test.
- **Qua khi:** tổng token khớp số đo (~17,0M cụm, ~16,3M pair/epoch); đọc tay 5 mẫu/bộ/hướng.
- **ĐÃ LÀM (05/11, trên split v2) [ĐO]:** `dung_mau.py` (Sonnet viết, em soát + kiểm độc lập). Pool cụm 46.120 mẫu / 16,22M token; pair 46.110 / 15,39M (token thật theo khuôn chat + chỉ dẫn; cách đo cũ khớp 17,0M/16,3M trong 0,1%; khuôn + chỉ dẫn + title ngữ cảnh làm tăng ~6–7%). Chia phần 127 sp (cụm)/113 (pair); loại 58/55 mô tả (gần hết 1688 có thuộc tính 颜色/规格 liệt kê hàng chục biến thể >2.048 token). Test 5.116 mẫu mỗi nhánh, ghi riêng `test_khoa/`. Kiểm độc lập: 0 sp test trong pool, fold khớp, 0 title test xuất hiện trong văn bản pool.

### B4. Bộ chấm tự động và zero-shot
- **Làm:** `scripts/ft_train/cham_diem.py`: chrF++ (đích zh dùng chrF ký tự); COMET, CometKiwi có sẵn trong mã nhưng **không chạy vòng này** (D18); kiểm riêng: giữ số/đơn vị (có chuẩn hóa), giữ thương hiệu/tên model, sai chữ viết, đúng số dòng, tỉ lệ sinh lặp, tỉ lệ độ dài; mô tả tách "dòng đã thấy / dòng mới"; **trung bình theo hướng** (không theo token). **Cố định giải mã:** greedy, `max_new_tokens` có trần, cùng cách cho mọi lần chạy.
- **Zero-shot:** Qwen3-4B nguyên bản, cùng chỉ dẫn, không ví dụ mẫu; chỉ dẫn nghiêm "chỉ trả bản dịch"; đo và ghi **tỉ lệ đầu ra có lời dẫn/giải thích**, cắt lời dẫn theo quy tắc ghi rõ để zero-shot không bị thiệt oan.
- **Qua khi:** thước đo chấm đúng ~20 ví dụ kiểm tay và cặp cố tình sai (đổi số, đổi thương hiệu); chốt ngưỡng sinh lặp sau khi xem đầu ra zero-shot [CHỜ duyệt].

### B5. Chạy thử và đo tốc độ trên Kaggle
- **Làm:** 100 bước; loss giảm; lưu/nạp lại checkpoint; đo token/giây, bộ nhớ; **kiểm khi tắt internet, PyTorch/CUDA trên Blackwell, vLLM (nếu dùng để sinh)**; kiểm giới hạn phiên; kiểm chọn RTX PRO 6000 bằng API hay phải bấm Commit trên web.
- **Qua khi:** có token/giây; cập nhật bảng ngân sách (mục 8). Nếu < ~10.000 token/giây báo Nhật Anh trước B6.
- **ĐÃ LÀM (05/11) [ĐO]:** B0, B1b xong: notebook `ft-qwen3-4b-thu-vien`, `ft-qwen3-4b-mo-hinh` (Qwen/Qwen3-4B, revision 1cfa9a72), Dataset riêng tư `ft-qwen3-4b-pool` **chỉ có mẫu pool** (đổi so với B1b cũ, tập test không lên Kaggle cho tới khi cần đánh giá, lúc đó dùng Dataset riêng). `nlp-group3` đã ghi đè (Nhật Anh cho phép, sao lưu `notebooks/_sao_luu/nlp-group3_20261105_truoc_ghi_de/`). B5 đạt: Python 3.12, torch 2.10+cu128, transformers 5.0, peft 0.18.1, RTX PRO 6000 95 GB, chạy khi tắt Internet; 100 bước loss kiểm 1,045 → 0,335; lưu/nạp lại khớp. Tốc độ: 3.750–3.820 token/giây khi bật gradient checkpointing dù lô to hay nhỏ; 5.770–5.870 khi tắt (lô 4096 token 52 GB; 6144 token 74 GB; 8192 token tràn). Cấu hình chốt: micro_bs 32, ngân sách 4096 token, không checkpoint. **Dưới 10.000 token/giây: báo Nhật Anh, ~1,19 tỷ token ≈ 56 giờ.** Giới hạn phiên Kaggle chưa kiểm. **Nguyên nhân chậm [ĐO 05/11]:** GPU đã chạy hết sức (94–99%, xung ~2.400/2.430 MHz, ~450/600 W); trong một bước: forward 38%, backward 61%, optimizer 1%, tạo tensor 0,1%, đệm 0,9%; suy luận mô hình gốc 23.530 token/giây, huấn luyện 5.613–5.873 (backward qua 36 lớp + LoRA). Biến thể: LoRA chỉ q/k/v/o 7.397 (nhanh hơn 32%, cần kiểm chất lượng); LoRA bf16 7.129 (không dùng vì cập nhật nhỏ dễ mất ở bf16); không dropout 5.874; lm_head chỉ ở vị trí dịch 5.762; rank 64 5.353. Giờ theo bước ở 5.873 token/giây: B6 ~33, B8 ~17, B7 ~3,5, B10 ~3.

### B6. Dò siêu tham số (bản cụm, dữ liệu đầy, dò loại dần trên 1 fold, D16)
- **Điều kiện:** cổng B2 theo D15 (title cho qua có điều kiện). **Mã:** `kaggle_train.py` `chay_b6`; chạy khói `dong_goi_kaggle.py b6_thu`, chạy thật `dong_goi_kaggle.py b6`.
- **Cấu hình:** lưới 6 cấu hình lr {5e-5, 1e-4, 2e-4} × rank {16, 64} với LoRA mọi lớp tuyến tính, **cộng 1 cấu hình lr 1e-4, rank 16, LoRA chỉ q/k/v/o** để biết có dùng được kiểu nhanh hơn 26–32% không. Cố định [MẶC ĐỊNH]: alpha = 2 × rank, dropout 0,05, batch hiệu dụng 32 (micro_bs 32, ngân sách 4.096 token, không checkpoint), warmup 3%, cosine **tính cho cả 2 epoch**, weight decay 0, bf16, seed 42. Train fold 1–4, kiểm fold 0.
- **Vòng 1 (~4,2 giờ):** mọi cấu hình chạy **1 epoch**, đo loss trên 3.000 mẫu kiểm cố định (750 mỗi hướng × loại). Điểm = trung bình theo hướng, title và mô tả trọng số bằng nhau.
- **Vòng 2 (~1,2 giờ):** 2 cấu hình đầu chạy nốt tới epoch 2 (D19); cấu hình thứ 3 được vào nếu loss lệch ≤1% so với đầu bảng (trần 3, thêm ~0,6 giờ). Loss không hữu hạn thì loại.
- **Vòng 3 (~0,5–1,5 giờ, chưa đo ở lô lớn):** mỗi cấu hình vòng 2, ở epoch loss tốt nhất, **sinh và chấm chrF++** trên 1.000 mẫu cố định (250 mỗi hướng × loại, sản phẩm nguyên). So theo cặp bằng bootstrap 95%.
- **Chọn:** đầu bảng chrF, loại cấu hình sinh lặp vượt ngưỡng; khoảng tin cậy chứa 0 là hòa, chọn cái rẻ hơn. Kết quả mâu thuẫn (loss và chrF ngược nhau, hoặc hòa mà khác nhau rõ về sinh lặp) thì mới chạy thêm fold 1 cho 2 cấu hình (+2,5 giờ) và báo Nhật Anh. LoRA chỉ attention chỉ được dùng tiếp nếu chrF không kém cấu hình all-linear cùng lr, rank.
- **Chạy tiếp:** ghi `b6/b6_trang_thai.json` sau mỗi mốc; dừng có trật tự ở `gio_toi_da` 2,5 giờ để đầu ra phiên được lưu; phiên sau gắn đầu ra chính notebook qua `kernel_sources` là chạy tiếp (đã kiểm chạy thật ở phiên 2, nối đúng từ bước 790). Vòng tự động ngắt thì chờ phiên xong rồi chạy `b6_tu_dong tiep`. Không sửa `kaggle_train.py` khi B6 chưa xong, vì mỗi phiên đẩy lại mã từ đĩa.
- **Đã đo (05/11) [ĐO]:** phiên 1 dài 9.003 giây, xong 3 cấu hình r16 và 790/929 bước của lr5e-5_r64. Mỗi cấu hình r16 ~38 phút, 5.836–5.839 token/giây; lr5e-5_r64 5.546 token/giây. Batch thật trung bình ~40 mẫu (37.123 mẫu/929 bước), cao hơn mức danh nghĩa 32 vì gom micro-batch tới khi đủ, giống nhau ở mọi cấu hình. **Vòng 1 xong (phiên 2, 15:06) [ĐO]:** điểm loss lr1e-4_r64 0,36798; lr2e-4_r64 0,37032; lr2e-4_r16 0,37701; lr5e-5_r64 0,38085; lr1e-4_r16 0,38999; lr5e-5_r16 0,40898; lr1e-4_r16_attn 0,44354 (7.781 token/giây). Chọn vòng 2: lr1e-4_r64, lr2e-4_r64. LoRA attention không vào vòng 2 nên chưa có chrF. **B6 XONG (05/11) [ĐO]:** thời gian chạy trên Kaggle 21.821 giây (~6,06 giờ, 3 phiên). Epoch 2: lr2e-4_r64 loss 0,35120, chrF 75,76, sinh lặp 1/1.000, 0,327 giây/mẫu; lr1e-4_r64 loss 0,35874, chrF 75,51, sinh lặp 3/1.000, 0,372 giây/mẫu. Chênh chrF +0,33, bootstrap 95% [-0,12; +0,81], chứa 0 nên là hòa theo luật; loss và sinh lặp cùng nghiêng về lr2e-4_r64, không mâu thuẫn nên không cần chạy fold 1. Đề xuất chốt lr2e-4_r64, 2 epoch, chờ Nhật Anh duyệt. lr2e-4 nằm ở biên trên của lưới lr.
- **Vì sao bản cụm:** judge 99,9% mục nên ít nhiễu nhất; hai bộ giống 99% nên cấu hình tốt cho cụm gần như chắc dùng được cho pair (kiểm ở B7).

### B7. Kiểm siêu tham số trên bản pair
- **Mã (05/11):** `kaggle_train.py` `chay_b78`, gộp B7 và B8 thành 5 việc: attn, cum_f0, pair_f0, cum_f1, cum_f2; `dong_goi_kaggle.py b78_tu_dong moi|tiep`; tổng hợp `tong_hop_b78.py`. **Khói đạt [ĐO]:** phần 1 dừng ở pair_f0 bước 3, phần 2 nối đúng, 16/16 bước, không làm lại attn và cum_f0, xóa moi_nhat sau khi xong việc. Chưa kiểm ở khói: phép so chấm lại với điểm B6 (`khop_b6`), đã kiểm riêng trên máy, khớp. Ước chạy thật ~4,7 giờ [ƯỚC]. Xong phải lưu adapter sang notebook riêng trước khi đẩy lại nlp-group3.
- **Phiên 1 (05/11, 18:13–20:46) [ĐO]:** LoRA attention epoch 1 chrF 71,54, lr1e-4_r16 epoch 1 chrF 73,89, chênh -2,20, bootstrap 95% [-2,75; -1,65], kém rõ, sinh lặp 4 so với 2/1.000; theo luật B6 không dùng tiếp. cum_f0 chấm lại khớp B6, loss cả fold 0,3497. pair_f0: loss 0,3798 → 0,3587, loss cả fold 0,3549, chrF 75,26, sinh lặp 0/1.000.
- **Làm (D18):** chạy **cấu hình tốt nhất** của B6, đúng số epoch đã chọn, trên pair, fold 0 (~1,2 giờ); sinh và chấm chrF trên cùng 1.000 mẫu (cùng sản phẩm với B6). Không so hơn thua hai bộ (D1): chỉ để biết cấu hình dùng được cho pair và có mô hình pair để đo test.
- **Chốt chặn C1:** chrF pair thấp hơn cụm quá 2 điểm, hoặc sinh lặp cao hơn rõ: chạy thêm cấu hình thứ hai của B6 trên pair (+1,2–1,7 giờ); nếu cấu hình thứ hai tốt hơn rõ thì báo Nhật Anh trước khi dò riêng pair.

### B8. K-fold K=3 đo độ ổn định (D18)
- **Làm:** bản **cụm** chạy thêm fold 1, 2 bằng cấu hình đã chốt, số epoch đã chọn (~2,5 giờ); fold 0 dùng lại kết quả B6. Mỗi fold: loss trên **cả fold** theo hướng, loại, category (không cần sinh) và chrF trên 1.000 mẫu cố định. Báo cáo trung bình ± độ lệch.
- **Bản pair:** fold 0 đã có từ B7. Chỉ chạy thêm fold 1, 2 khi chốt chặn C3 bật (+2,5–3,7 giờ).
- **Mục đích:** đo độ lệch giữa fold và phát hiện túi lỗi theo category, hướng bằng loss. **Không** để chọn bộ hơn (D1).
- **Chốt chặn C2:** chrF giữa 3 fold lệch quá 2 điểm [MẶC ĐỊNH, chốt lại khi thấy số]: chạy thêm fold 3, 4 của cụm. Fold 1, 2 kém fold 0 rõ: nghi B6 chọn may rủi, chạy cấu hình thứ hai trên fold 1 để đối chứng và báo Nhật Anh.
- **Chốt chặn C3:** chrF hoặc loss của pair ở fold 0 lệch cụm quá 2 điểm chrF / 5% loss, hoặc lệch theo một hướng: chạy pair fold 1, 2.
- **Lưu ý:** fold 0 dùng chọn siêu tham số ở B6 (D16) cũng nằm trong B8 nên số ở fold đó **hơi lạc quan**; fold 1, 2 độc lập với việc chọn cấu hình, nên B8 đồng thời kiểm cấu hình đã chọn có giữ được trên fold khác không. Điểm chính lấy ở test khóa (B10).

### B9. Chấm đầu ra mô hình (Gemini chính, Claude hiệu chuẩn)
- **Mẫu (D18):** lấy từ `mau_b9.parquet` (test, **không trùng mẫu B2**): 50 dòng mỗi nhóm hướng × loại × bộ (8 nhóm) = **~400 dòng** so với zero-shot. **Chốt chặn C4:** nhóm nào sau khi bỏ dòng ngang còn dưới 30 dòng, hoặc p gần ngưỡng (0,01–0,1), thì mở rộng riêng nhóm đó lên 150 dòng.
- **Cách chấm:** so cặp mù, thứ tự A/B đảo ngẫu nhiên từng dòng (ghi vị trí để kiểm thiên vị vị trí); mỗi dòng: A hơn / B hơn / ngang / cả hai lỗi nặng. **Gemini chấm toàn bộ** (CSV, mục 6); **Claude hiệu chuẩn trên 50 dòng chung** (ngẫu nhiên). Kết luận bằng **kiểm định dấu** (bỏ dòng ngang), p<0,05, từng hướng, từng bộ, từng loại. Báo cáo đồng thuận Gemini–Claude; nếu Claude cho kết luận ngược hướng thì báo Nhật Anh.
- **Công thủ công [ƯỚC]:** ~400 dòng ÷ ~50 dòng/lô = ~8 lần gửi.
- **Điều kiện:** đã có đầu ra zero-shot và mô hình fold 0 của hai bộ (sau bước đo test ở B10).

### B10. Đo test, báo cáo, bàn giao
- **Làm (D18):** **không huấn luyện lại.** Dùng mô hình fold 0 của cụm (B6) và pair (B7), và zero-shot, đo **một lần** trên test khóa, qua Dataset test riêng (D17, chỉ gắn vào phiên đánh giá). Điểm có thể hơi thấp hơn thực tế vì mô hình chỉ học 80% pool; test vẫn hoàn toàn chưa thấy. Huấn luyện toàn pool để dành cho **mô hình cuối, sau khi sửa title** (D15). **Chốt chặn C5:** hướng nào fine-tune không thắng zero-shot trên test thì báo ngay, không kết luận "ổn". Báo cáo `BAO_CAO_FT_QWEN3_4B.md` ([MẶC ĐỊNH] trong repo EDA, không đẩy HF khi chưa có lệnh): điểm test từng hướng/bộ (toàn test và tập đã kiểm), kết quả B2/B8/B9, bộ siêu tham số, giới hạn (nhãn bạc, rà bằng LLM, pair judge 14,5%). **Hướng dẫn cho mô hình khác:** giữ định dạng, `max_len`, dải rank, số epoch; quét lại 3 mức lr.

## 4b. Các chốt chặn cho phần đã cắt (D18)

| Đã cắt | Rủi ro | Chốt chặn tự kiểm | Nếu bật thì tốn thêm |
|---|---|---|---|
| B7 chỉ 1 cấu hình | Cấu hình tốt cho cụm chưa chắc tốt cho pair | C1: pair kém cụm quá 2 điểm chrF | ~1,2–1,7 giờ |
| B8 K=3 thay K=5, không seed phụ | Ước lượng ổn định kém chính xác; B6 chọn may rủi | C2: lệch giữa fold quá 2 điểm chrF | ~2,5–3,7 giờ |
| Pair không chạy fold 1, 2 | Pair bất ổn mà không thấy | C3: pair fold 0 lệch cụm | ~2,5–3,7 giờ |
| B9 ~400 dòng | Không đủ kết luận nhóm nào đó | C4: mở rộng riêng nhóm thiếu | Vài lần gửi |
| B10 không huấn luyện lại | Điểm test hơi thấp hơn thực tế; chưa có mô hình cuối | C5 + ghi rõ giới hạn; mô hình cuối làm sau khi sửa title | Khi huấn luyện cuối |
| Bỏ COMET | Mất một tín hiệu phụ | chrF, kiểm số/tên hãng, B9 vẫn có | Không |

## 5. Tiêu chí "ổn" (khóa trước B6)

| Đối tượng | Tiêu chí |
|---|---|
| Dữ liệu (mỗi bộ, mỗi hướng, title và mô tả riêng) | lỗi nặng hiệu chỉnh ≤ **5%** (B2; Wilson 95% kèm theo). Đơn vị: title = 1 title; mô tả = **1 mảnh (thuộc tính)**, không phải cả mô tả |
| Mô hình (mỗi bộ, mỗi hướng) | thắng zero-shot trong so cặp mù, kiểm định dấu p<0,05; sinh lặp dưới ngưỡng chốt ở B4; không hướng nào kém zero-shot |
| Ổn định | độ lệch giữa 3 fold (D18) nhỏ hơn chênh lệch fine-tune so zero-shot; chốt chặn C2 ở B8 |
| Không đòi | hai bộ hơn thua nhau (D1) |

Kết quả "ổn nhưng hai bộ như nhau" là hợp lệ.

## 6. Quy trình chấm bằng Gemini web và Claude hiệu chuẩn

- **CSV gửi Gemini:** UTF-8; cột `id, huong, loai(title|mo_ta|manh), nguon, ban_dich` (B2) hoặc `id, huong, loai, nguon, ban_A, ban_B` (B9, đã đảo ngẫu nhiên). **Bỏ mọi cột nhãn** (Gemma/Qwen/judge/bộ). Xuống dòng trong mô tả thay bằng ký hiệu cố định (ví dụ ` ⏎ `) để khỏi vỡ CSV. **Mô tả: nguồn được đánh số theo dòng nguồn của bản pair** (`[1] tên: giá trị ⏎ [2] ...`, đây là "mảnh" chuẩn, đã căn 100%); bản dịch đưa nguyên khối, Gemini/Claude tự đối chiếu và báo số thứ tự mảnh lỗi (bản cụm có `; ` mơ hồ nên không đánh số bản dịch). Lô ≤ ~50 dòng (mô tả dài thì ít hơn). Cùng một đoạn rubric (từ `rubric_kiem.md`) ở đầu mỗi cuộc trò chuyện.
- **Yêu cầu đầu ra:** CSV/JSON đúng từng dòng, không tóm tắt. Title/mảnh khác nhau: `id, muc, loai_loi, ghi_chu`. Mô tả: `id, so_manh_nang, ds_manh_nang (số thứ tự), loai_loi, ghi_chu`; tổng số mảnh lấy từ dữ liệu, không để mô hình tự đếm. Nhật Anh lưu kết quả vào `data/processed/ft_run/kiem_gemini/` (giữ cả văn bản thô).
- **Kiểm tra:** script (`scripts/ft_train/kiem_ket_qua.py`) so số dòng và `id` trả về với lô gửi; lô thiếu/lệch thì gửi lại; không dùng dòng Gemini bịa.
- **Hạn chế đã biết:** bản web khó tái lập; giới hạn kích thước/đầu ra hiện hành chưa kiểm chứng; Gemini cùng họ Google với Gemma nên có thể ưu ái văn phong Gemma hoặc bỏ sót lỗi của Gemma → chỉ tin sau khi đồng thuận với Claude (mục B2).
- **Claude:** chấm lô 20–25 mục mỗi lượt, ghi `data/processed/ft_run/kiem_claude/*.jsonl`; chỉ ghi mã Đạt/Nhẹ/Nặng + loại lỗi, chỉ viết bản sửa khi lỗi nặng; chạy trong **phiên mới** với bản tóm tắt ngắn; chia nhiều ngày hoặc nhiều cửa sổ 5 giờ.
- **Usage Claude Pro** [ƯỚC, không quy ra % tuần được]: ~0,2M token nội dung (B2 ~0,125M + B9 ~0,06M); thực tế cao hơn do đọc lại ngữ cảnh; **đo bằng thử 20 mục, nhìn thanh Weekly**, ngoại suy rồi mới chạy tiếp.

## 7. Rủi ro đã biết

| Rủi ro | Xử lý |
|---|---|
| B6 loại nhầm cấu hình lr nhỏ vì mới 1 epoch (D16) | Giữ thêm cấu hình lệch ≤1% vào vòng 2 |
| B6 so cấu hình chỉ trên 1 fold (D16) | Fold gần 9.000 mẫu, so theo cặp có bootstrap; kết quả mâu thuẫn thì chạy fold 2; B8 kiểm lại trên fold 1, 2 (C2) |
| Mô hình học cả lỗi title (D15) | Ghi rõ trong báo cáo; chưa dùng làm mô hình cuối cho extension khi chưa sửa title |
| Nhãn bạc; Claude/Gemini cũng là LLM | Ghi "rà soát bằng LLM"; mù nguồn gốc; cổng 5% có CI; đồng thuận hai bên |
| Gemini cùng họ Gemma, có thể bỏ sót lỗi | Claude hiệu chuẩn, hiệu chỉnh tỉ lệ, kappa; nếu bất đồng cao thì Claude mở rộng mẫu |
| Usage Claude Pro (Weekly 27% ngày 05/11) | Chỉ còn hiệu chuẩn B9 50 dòng; ~1 điểm mỗi 120 mục |
| Gemini web: bỏ dòng, tóm tắt, khó tái lập, giới hạn chưa biết | Lô nhỏ, kiểm id/dòng bằng script, lưu văn bản thô |
| Công thủ công gửi Gemini (B9 ~8 lần) | Mở rộng riêng nhóm thiếu theo C4 |
| Nhiều dòng "ngang" ở B9 làm cỡ mẫu hiệu dụng của kiểm định dấu nhỏ (title dễ ngang) | Báo cỡ mẫu sau khi bỏ dòng ngang; nếu >60% ngang thì mở rộng mẫu hoặc chỉ kết luận trên mô tả |
| Hiệu chỉnh Claude trên mẫu chung nhỏ nhiễu | Gộp chung mọi nhóm; báo kèm CI; vùng xám 5–6,5% Nhật Anh quyết |
| Phiên Kaggle tắt internet | Đóng gói Dataset, kiểm ở B5 |
| PyTorch/vLLM không chạy trên Blackwell | Kiểm ở B5; sinh bằng `generate` của transformers |
| Mất phiên Kaggle | Checkpoint theo bước; mỗi lần chạy ghi file riêng; chạy lại không lặp việc đã xong |
| Điểm mô tả bị thổi phồng bởi dòng khuôn mẫu | Báo cáo tách "dòng đã thấy / dòng mới" |
| Cụm so với pair lẫn độ phủ judge (99,9% so với 14,5%) | Nêu rõ trong báo cáo |
| Chọn cấu hình bị 1688 lấn át | Trung bình theo hướng |
| Zero-shot bị thiệt vì lời dẫn thừa | Chỉ dẫn nghiêm + cắt lời dẫn có ghi rõ (B4) |
| Quyền dùng RTX PRO 6000 gắn cuộc thi có thể có hạn | Kiểm ở B5; checkpoint đủ để chuyển GPU khác |
| Điều khoản Gemini API về dùng đầu ra Gemma để huấn luyện (`CLAUDE.md` dòng 113) | Câu hỏi cũ chưa đóng, chưa kiểm điều khoản; việc dùng Gemini để chấm không dính |
| Cụm: `; ` mơ hồ | Giữ định dạng gốc; chấm bằng văn bản chuẩn hóa; báo lỗi căn dòng riêng |

## 8. Ngân sách

**GPU** (dữ liệu đầy; [ĐO] 13,06M token mỗi epoch trên 4 fold bản cụm, 5.873 token/giây với LoRA mọi lớp, 7.397 với LoRA chỉ attention):

| Bước | Lượt chạy | Token | Giờ ở 5.873 token/giây [ƯỚC] |
|---|---|---|---|
| B6 vòng 1: 7 cấu hình × 1 epoch | 7 epoch | ~91M | ~4,2 |
| B6 vòng 2: 2 cấu hình × 1 epoch còn lại | 2 epoch | ~26M | ~1,2 |
| B7 pair: 1 cấu hình × 2 epoch | 2 epoch | ~25M | ~1,2 |
| B8 cụm: fold 1, 2 × 2 epoch | 4 epoch | ~52M | ~2,5 |
| B10 không huấn luyện | 0 | 0 | 0 |
| **Tổng huấn luyện** | 15 epoch | **~194M** | **~9,1** |

Thêm sinh bản dịch [ƯỚC, chưa đo ở lô lớn]: B6 vòng 3 ~0,5–1,5 giờ; B7, B8 ~1–3 giờ; đo test 3 mô hình ~2–4 giờ. **Tổng ước ~12–17 giờ; mốc 1 (B6 + B7) ~8–9 giờ; quỹ Kaggle 30 giờ, dùng riêng, còn ~29 giờ (05/11).** Chốt chặn C1–C3 bật hết thì thêm tối đa ~10 giờ. Mỗi phiên thêm ~3 phút chuẩn bị. LoRA chỉ attention đạt chất lượng thì phần sau B6 giảm ~20%. Nguyên nhân chậm xem mục B5: GPU đã chạy hết sức, tốn chủ yếu ở backward; ước tính cũ 10.000–30.000 token/giây là giả định quá lạc quan.

**Claude Pro:** B2 đã xong, B9 hiệu chuẩn 50 dòng [ƯỚC ~0,5 điểm Weekly]. **Gemini web:** B9 ~8 lần gửi CSV [ƯỚC].

## 9. Việc tạo trong repo (khi có lệnh bắt đầu)

Trong `D:\nam4_ky1\e-commerce - EDA` (mẫu theo `scripts/judge/`, `scripts/ft_clean/`; test chạy trực tiếp bằng python; Python của venv `e-commerce - crawl\ecom_crawler\.venv`):
`scripts/ft_train/`: `chia_tap.py` (B1), `dong_goi_kaggle.py` (B0/B1b), `lay_mau_kiem.py` (B2/B9: dựng mẫu và CSV lô, đảo A/B), `kiem_ket_qua.py` (kiểm id/dòng trả về, tính tỉ lệ lỗi, Wilson CI, kappa, hiệu chỉnh, kiểm định dấu), `rubric_kiem.md`, `dung_mau.py` (B3), `cham_diem.py` (B4/B8), `kaggle_train.py` hoặc notebook (B5–B10), `README.md`; test từng file. Đầu ra `data/processed/ft_run/` (gitignore). Ghi mục mới vào `CLAUDE.md`, `WALKTHROUGH.md`.

## 10. Kiểm chứng

- B1: không nhóm nào ở hai phía; tỉ lệ phân tầng; test không giao fold nào.
- B2: thử 20 dòng; tính tỉ lệ/CI/kappa bằng script, không bằng tay; mẫu M1/M2/M1c/M2c không chồng B9.
- B3: đếm token khớp số đo; đọc tay 5 mẫu/bộ/hướng.
- B4: 20 ví dụ kiểm tay; cặp cố tình sai bị phát hiện; đo tỉ lệ lời dẫn của zero-shot.
- B5: loss giảm; nạp lại checkpoint cho cùng loss; có token/giây; chạy được khi tắt internet.
- B6–B10: mỗi lần chạy ghi cấu hình, seed, fold, điểm; chạy lại một lần xác nhận lặp lại trong sai số.
- Rubric: cùng hash cho Gemini và Claude.

## 11. Còn mở

1. Thời điểm gửi danh sách mã test cho nhóm [Nhật Anh].
2. Danh sách mô hình nhóm khảo sát (ảnh hưởng mức chuyển siêu tham số).
3. Ngưỡng sinh lặp, ngưỡng ổn định giữa fold, ngưỡng kappa: chốt sau B4/B8 và sau thử 20 dòng.
4. Phiên bản Qwen3-4B: [MẶC ĐỊNH] `Qwen/Qwen3-4B`.
5. Nếu sau này có người HSK: dùng để thay hoặc đối chiếu rà bằng LLM.
6. Xử lý vùng xám 5–6,5% ở B2 do Nhật Anh quyết khi xảy ra.

## 12. Nhật ký thay đổi

| Ngày | Thay đổi |
|---|---|
| 03/11/2026 | v0.1: soạn từ cuộc bàn Prompt 698–705 (9 bước). |
| 04/11/2026 | v0.2: theo Prompt 706–717. Bỏ so cặp cụm với pair (D1), bỏ mẫu con (D10), bỏ thí nghiệm title (D7), bỏ DeepSeek làm người chấm (D11); thêm cả hai hướng (D3); thêm Claude rà vai A (cổng 5%) và vai B; đóng gói Kaggle (D8); thêm B7; đổi sang B0–B10; viết lại ngân sách theo dữ liệu đầy; thêm số đo mới. |
| 04/11/2026 | **v0.3:** theo Prompt 718–724. **Phương án 2 (D12, D13): Gemini Pro web chấm chính B2/B9, Claude chỉ hiệu chuẩn** (~0,2M token); thêm mục 6 (quy trình CSV, kiểm kết quả, hiệu chỉnh, kappa); sửa thứ tự B0/B1 (thêm B1b); mẫu B9 tách khỏi B2; rubric dùng bốn điểm trung lập + ngoại lệ của Judge; B2 chạy song song B3–B5, B6 chỉ sau cổng; gom cổng riêng title/mô tả; vùng xám 5–6,5% do Nhật Anh quyết; thêm lưu ý lạc quan nhẹ của 3 fold, giới hạn mảnh khác nhau (81,5% sp), zero-shot cắt lời dẫn, báo điểm trên toàn test và tập đã kiểm, cắt mẫu dài ở ranh giới thuộc tính; số lần gửi Gemini thủ công ~70. |
| 07/11/2026 | **v0.7:** theo Prompt 806–808. Pair duyệt 10,86% title; cả hai bản ~11%, lr 2e-4. Sửa bộ đo (mốc tham chiếu, cắt lời dẫn `dong_ngu_canh`, trần sinh theo nguồn). Đo test MỘT lần (bản 41, Dataset riêng tư): fine-tune adapter fold 0 thắng zero-shot ở cả 4 hướng×nhánh (chrF +30 đến +34, khoảng 95% theo sản phẩm xa 0; cụm 75,6/77,4, pair 75,4/77,7 so zero-shot 41–47), không có cảnh báo; nhưng chênh bị phóng đại vì tham chiếu là bản dịch máy, B9 chưa làm. Bản cuối trên toàn pool (bản 43, 3,2 giờ): loại 6.386/23.017 title bị cờ (cách 1, Nhật Anh chọn), giữ mô tả; adapter `cum_cuoi`, `pair_cuoi` ở `ft-qwen3-4b-cuoi-luu` và `ft_run/ban_cuoi/`, hash khớp. Bản cuối không đo lại trên test. Còn lại: B9, báo cáo `BAO_CAO_FT_QWEN3_4B.md`, sửa title thật (cách 2) nếu muốn bản v2. Quỹ GPU phiên chính 32,0 giờ (hai tuần quota). |
| 07/11/2026 | **v0.6.6:** theo Prompt 804–805. Pair fold 2 (bản 38, 2,77 giờ) xong; luật pair 3 fold (khóa trước, `kiem_3fold_pair.py`) ĐẠT cả bốn điều: dòng mới mô tả zh→vi +1,36 / +2,10 / +1,58, chrF chung -0,10 / +0,13 / +0,97 (tb +0,33), title hai hướng tb -0,21, title zh→vi tb -0,13. Giới hạn: chỉ lợi ở zh→vi, title fold 0 tụt thật, mốc fold 2 thấp. Chưa đổi, chờ Nhật Anh duyệt pair sang 10,86%. Đầu ra bản 38 lưu ở `ft-qwen3-4b-tl3-luu`. Quỹ GPU phiên chính 26,27 giờ; quota reset 7h 07/11. |
| 07/11/2026 | **v0.6.5:** theo Prompt 800. Bản 36 (`tl2`, 4,16 giờ) xong. Luật lr không đạt (chrF chung theo câu +0,03 [-0,43; +0,47]), giữ lr 2e-4: bản cụm chốt 10,74% title, lr 2e-4. Cổng pair fold 1 đạt (dòng mới mô tả zh→vi +2,10 [+1,23; +2,94], title +0,01); chrF chung pair gần như đứng yên (fold 0 -0,10, fold 1 +0,13). Chờ lệnh pair fold 2 (~2,7 giờ) và anh duyệt cách áp tiêu chí 3 fold cho pair. Đầu ra bản 36 lưu ở `ft-qwen3-4b-tl2-luu`. |
| 06/11/2026 | **v0.6.4:** theo Prompt 795–796. Thêm điều kiện dòng mới và title trên fold 2; bản cụm đạt đủ, đổi sang tỉ trọng title 10,74%, chờ kiểm lr và quyết pair. |
| 06/11/2026 | **v0.6.3:** theo Prompt 788–794. Kiểm tín hiệu mô tả dòng mới ngoài 6 bước; mức ~11% bản cụm đạt luật khóa trên fold 1; B8 mức 1 : 1 xong. Chưa đổi tỉ lệ, chờ fold 2. |
| 06/11/2026 | **v0.6.2:** kết quả bước 2, 3, 6: không có ứng viên ở cả hai bản, giữ ~50% (fold 0). Bỏ qua bước 4, 5. |
| 05/11/2026 | **v0.6.1:** theo Prompt 774. Viết lại mục 0 thành 6 bước. Bỏ dò thô 1 epoch bằng loss, thử đủ 2 epoch và so bằng chrF; siết cổng fold 0 theo bootstrap; thêm zh→vi vào luật không học lệch; kiểm lr 1 mức, đủ 2 epoch; dừng sớm ở fold 1. Theo Prompt 777–779: chọn phương án B cho pair, giữ kiểm lr. Tổng ~9,3–13,6 giờ, pair K-fold riêng +~5,6 giờ. |
| 05/11/2026 | **v0.6:** theo Prompt 763–773. Mục tiêu (d), mục 0, D21. B8 dừng do Nhật Anh hủy phiên. |
| 05/11/2026 | **v0.5.2:** theo Prompt 755–757. B6 xong, D20, mã và khói B7 + B8 đạt, chờ lệnh chạy thật. |
| 05/11/2026 | **v0.5.1:** theo Prompt 746–751. D19: B6 tối đa 2 epoch, giữ K-fold và LoRA attention, đủ lưới, test đủ mẫu. Ngân sách ~9,1 giờ huấn luyện, ~12–17 giờ cả sinh. |
| 05/11/2026 | **v0.5:** theo Prompt 744–745. D18: cắt phần thừa sau B6 (B7 1 cấu hình; B8 K=3, dùng lại fold 0, bỏ seed phụ, pair chạy thêm khi lệch; B10 không huấn luyện lại; B9 ~400 dòng; bỏ COMET) kèm chốt chặn C1–C5 (mục 4b). Giờ huấn luyện ~29 → ~10,5–12. Dọn các mô tả cũ ở B0, B1, thứ tự bước, Weekly. |
| 05/11/2026 | **v0.4:** theo Prompt 725–743. D14 (cổng title dùng nặng lõi), D15 (hướng A), D16 (B6 dò loại dần trên 1 fold, K-fold ở B8), D17 (Kaggle chỉ có pool). Viết lại B6 và ngân sách theo số đo thật. |
| 04/11/2026 | **v0.3.1** (soát lại sau v0.3): **đơn vị tính lỗi mô tả = 1 mảnh**, không phải cả mô tả (tránh vượt 5% oan); công thức lỗi pair = lỗi cụm + 0,048 × (chênh trên M2); hiệu chỉnh Claude gộp chung mọi nhóm (tránh nhiễu từ mẫu 40–60 mục); CSV mô tả đánh số mảnh theo dòng nguồn pair; thêm rủi ro cỡ mẫu hiệu dụng của kiểm định dấu. |
