# NHẬT KÝ OPENCODE: PHẢN BIỆN VÒNG 2 (NHÁNH NHẬT ANH)

Ghi theo yêu cầu `docs/phan_bien_vong2/PROMPT_OPENCODE_P0_P1.md`. Không sửa `WALKTHROUGH.md` (phần của Claude). Mọi việc dưới đây chạy trên CPU, không train, không sinh bản dịch, không gọi API, không đọc `test_khoa/`, `split.parquet`, `ds_test.txt`.

## 17/11/2026

### A. P1, chấm lại trên eval fold 0

- **Khóa kế hoạch trước khi xem số:** `docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P1.md`, sha256 `3743d3eeabcf30ab52a7d2f12cb6ec2aaa3ae7f9634e42c7be82100da81e1ef0`. Ghi hash vào nhật ký trước khi chạy bất kỳ phép chấm mới nào.
- **3.1 Tái lập 5 mô hình** (`scripts/phan_bien_v2/p1_tai_lap.py`, `tai_lap_{pair,cum}.json`): khớp **0,00000** ở cả 48 ô pair và 48 ô cum (tb_chinh, tb_macro, chrF corpus từng nhóm). Không dừng theo luật > 0,01.
- **3.2 Tách dòng khuôn/mới** (`p1_phan_tich.py`, `tach_dong_{pair,cum}.json`): tỉ lệ dòng mô tả đã thấy pair 85,91% (zh→vi) / 75,67% (vi→zh); cum 86,06% / 73,92%. Giả thuyết "khoảng cách NMT lớn hơn trên dòng mới" **đúng**. OpenNMT trên dòng mới kém Qwen3-4B −8,29 [−15,49; −0,88] (pair zh→vi) và −17,76 [−25,85; −9,90] (pair vi→zh); cum −10,07 [−11,68; −8,30] / −22,06. Trên dòng khuôn, pair zh→vi OpenNMT lại hơn +2,12. Đã sửa đơn vị đếm: gom theo mã sản phẩm trước khi lấy trung bình và bootstrap, nên số sản phẩm pair là 285/280, không phải số dòng. EcomGPT ở cum bị loại 35/298 (zh→vi) và 9/300 (vi→zh) vì lệch số dòng; đã ghi trong báo cáo.
- **3.3 Vòng lặp tham chiếu** (`vong_tham_chieu_{pair,cum}.json`): title eval zh→vi có 55 sản phẩm Qwen / 245 Gemma. Chênh lệch kép: so EcomGPT −7,68 [−10,22; −5,00], so OpenNMT +5,17 [+2,02; +8,29], so Moses +3,11 [+0,24; +5,75], **không nhất quán theo họ**. Zero-shot Qwen3-4B **không tính được**: không có bản dịch zero-shot đầy đủ trên eval fold 0, các tệp hiện có chỉ 600 mục mô tả dạng khối của thí nghiệm chéo. Mô tả eval fold 0 **không có sản phẩm Qwen chủ đạo** (pair 285/285 Gemma; cum 297 Gemma, 1 không xác định), nên chênh lệch kép theo kế hoạch không tính được. Phân tích thêm sau khi xem số, "có mảnh Qwen" (170) so "không" (115): cả 6 khoảng 95% chứa 0.
- **3.4 Theo độ dài** (`do_dai_{pair,cum}.json`): mô tả pair Q4, Qwen3-4B 72,2; OpenNMT 58,5; Moses 71,2; EcomGPT 53,8. Cum Q4: 4B 88,2; OpenNMT 80,7.
- **3.5 Đường học OpenNMT** (`p1_opennmt_hoc.py`, `opennmt_hoc.json`, 2 PNG trong `docs/phan_bien_vong2/hinh/`): zh→vi ppl dev 19,05 → 10,82, acc 65,73 → 79,58, best step 5.685, dừng 10.233; vi→zh ppl 204,15 → 32,66, acc 30,70 → 66,39, chạy đủ 5.620 bước, log không ghi best step.
- **3.6 Compute và hợp nhất** (`p1_compute.py`, `compute.json`): GPU-giờ 4B 0,19; 8B 0,37; EcomGPT 0,33; OpenNMT 1,08; Moses 2,93. Pretraining Qwen ~36.000B token, BLOOM ~366B token, đều **[CHƯA KIỂM]**; OpenNMT/Moses 0.
- **Hash đầu vào:** `p1_sha256.py` ghi sha256 52 tệp đã đọc vào `sha256_dau_vao.json`; không có tệp test.
- Viết `docs/phan_bien_vong2/KET_QUA_P1.md` với bảng 3.2–3.6, nhãn [ĐO]/[ƯỚC]/[CHƯA KIỂM], mục giới hạn.
- Test: `scripts/phan_bien_v2/test_p1_phan_tich.py`, 6 test pass.

### B. P0, sửa tuyên bố trong tài liệu

Mỗi mục ghi file, câu cũ → câu mới, lý do, mục biên bản.

1. `scripts/ft_train/xuat_ket_qua_kaggle.py`, A7, mục 3:
   - Giữ dòng đọc từ JSON "Kết luận trong file: KHÔNG RÒ RỈ".
   - Thêm ngay sau: "Ở **cấp sản phẩm**, không có sản phẩm gần-trùng giữa test và pool theo ngưỡng Jaccard. Nhưng ở **cấp dòng**, 85.0% dòng mô tả của test đã có nguyên văn trong pool. Vì vậy test là held-out theo sản phẩm, **không** phải held-out cho từng dòng mô tả; điểm mô tả toàn phần chủ yếu đo khả năng tái tạo thuộc tính khuôn mẫu."
   - `ket_luan` A7: "Không rò rỉ theo ngưỡng đã đặt" → "Không rò rỉ ở **cấp sản phẩm** theo ngưỡng Jaccard đã đặt", thêm câu cấp dòng. Lý do: 1.2(c) và 1.3 mục 3. Chạy lại bộ sinh, `git diff --stat docs/ket_qua_kaggle` chỉ đổi A7, D3, INDEX. Định dạng trong câu thêm đổi từ 85.0% thành 85,0% cho thống nhất.
2. `docs/pipeline/08_chia_tap_va_dung_mau.md:29`: "KHÔNG RÒ RỈ, 0 vi phạm." → "0 vi phạm **ở cấp sản phẩm**... đây là held-out theo sản phẩm, **không** phải held-out cho từng dòng mô tả." Mục 1.2(c).
3. Bảng 5 mô hình, thêm cột "Mô tả mới (P1)":
   - `docs/pipeline/09_fine_tune_va_danh_gia.md` mục 6, pair và cụm.
   - `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` mục 6.1, pair và cụm.
   - `docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md` mục 5, pair và cụm.
   - Số: pair 4B 63,41; 8B 54,38; EcomGPT 40,19; OpenNMT 44,88; Moses đủ 61,53; Moses 4M 60,18. Cụm 4B 72,70; 8B 74,75; EcomGPT 61,22; OpenNMT 56,98; Moses đủ 67,71; Moses 4M 61,34. Lý do: 1.3 mục 3.
4. `scripts/ft_train/xuat_ket_qua_kaggle.py`, D3, `ket_luan`:
   - Cũ: "Pair: hòa với 4B theo chrF câu; cụm: kém rõ. Yếu nhất ở vi→zh..."
   - Mới: nêu đủ bốn số pair (chrF câu TB +0,39 [−1,09; +1,90]; chrF corpus chính 74,38 so 79,98; macro câu −8,33 [−9,40; −7,26]), cụm −5,57 [−6,42; −4,85], thêm dòng mới P1 và câu "không kết luận về paradigm NMT". Lý do: 2.2(b), 2.3 mục 3–4.
5. `docs/pipeline/09` mục 6 và `docs/TONG_HOP` mục 6.1: bỏ câu "NMT yếu", thay bằng câu D3 được phép, kèm số dòng mới và số dòng khuôn. `docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md:55` cũng sửa "OpenNMT yếu ở vi→zh" thành "với cấu hình Transformer nhỏ học từ đầu, OpenNMT kém nhất ở vi→zh" + số dòng mới.
6. Tỉ trọng title:
   - `docs/pipeline/09` mục 3, `docs/TONG_HOP` mục 4, `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md` mục 2, `docs/QUOTA_KET_LUAN.md` mục 5: nói rõ **bằng chứng chính là thí nghiệm trọng số loss** (3 mức, 3 fold, CI, luật khóa trước; mọi title thấy 2 epoch, chỉ đổi hệ số nhân loss); quota chỉ là **kiểm chéo**, có confound tần suất.
   - Đã kiểm trong `scripts/ft_train/kaggle_train.py`: `w_title` là hệ số nhân loss từng mẫu, dữ liệu không bị lấy mẫu lại theo tỉ lệ.
   - Hạ câu "tốt nhất" ở QUOTA thành "cao nhất **trong 3 bảng đã thử** trên metric chính 10/90"; bỏ chữ "tối ưu" ở pipeline/09:67 và TRONG_SO.
   - Thêm câu CI chỉ phủ dao động chọn mẫu eval, không phủ khởi tạo; sàng lọc 2 seed, công thức chính 1 seed.
7. Fold 0 là tập chọn:
   - Thêm ghi chú vào `docs/pipeline/09` mục 6, `docs/TONG_HOP` mục 6, `docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md` mục 5 và mục 7, `docs/pipeline/10` mục 4.7, `docs/ket_qua_kaggle/D6` (đầu bảng và ghi chú).
   - Câu: eval fold 0 (tập đã dùng để chọn cấu hình, train = fold 1–4), cấu hình sàng lọc r16, chưa đo test; số test duy nhất là B8, chênh khoảng 30 điểm bị phóng đại một phần và **chưa tách dòng mới trên test**.
8. Vòng lặp tham chiếu:
   - Thêm `docs/TONG_HOP` mục 6.3 và `docs/pipeline/10` mục 4.1: tham chiếu do Gemma 4/Qwen sinh, Judge chọn, dữ liệu cùng nguồn; dẫn số P1 3.3; nêu việc còn mở là người chấm 100–300 mẫu.
9. Demo:
   - `docs/pipeline/10` mục 1, `docs/TONG_HOP` mục 9.2, `demo/CHAY_DEMO.md` mục phạm vi: thêm "thử nghiệm quy mô nhỏ: 13 trang 1688 thật, đọc tay 8 trang, không có tham chiếu người, một người dùng, máy chủ trên Kaggle T4 hoặc laptop". Bỏ chữ "triển khai" khỏi câu mô tả.
10. `git mv docs/BANG_KET_QUA_CUOI.md docs/archive/BANG_KET_QUA_CUOI_09-11_DA_THAY.md`; giữ banner, sửa link tương đối thành `../QUOTA_KET_LUAN.md` và `../TONG_HOP_KET_QUA_NHAT_ANH.md`; sửa link trong `docs/TONG_HOP:36`, `docs/ket_qua_kaggle/C1`, comment và chuỗi ghi chú trong `scripts/ft_train/xuat_ket_qua_kaggle.py`. Lý do: 4.8.
11. Rà grep:
    - `docs/CAC_DIEM_CAN_CHOT.md:27`: câu "mô tả ~83–91" thêm số dòng mới 63,41 (pair) và 72,70 (cụm).
    - `docs/ket_qua_kaggle/C4` và generator: "natural cao nhất ... cả hai seed" → "theo trung bình 2 seed; ở pair attr_heavy seed 123 nhỉnh +0,38 nhưng trung bình vẫn thấp hơn natural". Đây là sửa câu chữ theo bảng sẵn có, không đổi số đo.
    - `docs/pipeline/08`: câu "Moses mạnh ở bản pair" thêm giới hạn dòng mới.
    - Các chỗ còn chữ "tối ưu" là nghĩa optimizer ("Tối ưu: AdamW", "trạng thái tối ưu") hoặc tên đề tài, không phải tuyên bố kết quả. Không sửa.
12. Slide `docs/bao_cao_thay/Bao_cao_bo_song_ngu_chinh_thuc.pptx`: đọc chữ bằng `zipfile` + XML (`.venv` không có `python-pptx`). Nội dung 7 slide nói về bộ dữ liệu 26.038 sản phẩm, 2 mô hình dịch, phễu lọc, cân bằng nhóm hàng; slide 6 đã ghi "Bản dịch do máy, mới kiểm theo mẫu, chưa có người rà toàn bộ". Không thấy câu về 5 mô hình, fold 0, production hay "vượt trội". **Không sửa file.** Câu nên cân nhắc khi thuyết trình: tiêu đề dùng "bộ chính thức" là tên bộ dữ liệu, không phải tuyên bố chất lượng; nên nói rõ tham chiếu vẫn là bản dịch máy.

### C. Chưa làm / còn mở

- Chưa có người chấm 100–300 mẫu; chưa dùng test khóa lần hai để tách dòng mới trên test. Đây là việc chờ Nhật Anh quyết.
- Chưa chạy lại toàn bộ `xuat_ket_qua_kaggle.py` sau khi sửa chuỗi C1/C4/D6; các file sinh tương ứng đã được sửa tay khớp nội dung, generator cũng đã sửa để lần sinh sau không lùi. Lần chạy kiểm 4.1 trước đó chỉ đổi A7, D3, INDEX.
- Không sửa `WALKTHROUGH.md`, không sửa tài liệu nhánh Quang, không sửa `.pptx`.
