# KẾ HOẠCH: TÁCH DÒNG KHUÔN / DÒNG MỚI TRÊN KẾT QUẢ B8 (TEST KHÓA)

Ngày soạn 19/11/2026. Người soạn: Claude. **TRẠNG THÁI: CHỜ NHẬT ANH DUYỆT. Chưa mở, chưa đọc, chưa liệt kê bất kỳ tệp test nào để soạn kế hoạch này**; mọi chi tiết dưới đây lấy từ mã (`kaggle_train.chay_do_test`, `cham_diem.cham`) và tài liệu (`docs/ket_qua_kaggle/B8_do_test_khoa_mot_lan.md`, `docs/pipeline/09` mục 4, biên bản `REVIEW_VONG2_3_PHAN_BIEN.md` mục 1.2(c), 1.3.3).

## 1. Mục tiêu

Phản biện 1 (BLOCKER) mục 1.2(c) và yêu cầu 1.3.3: tài liệu ghi 85,0% dòng mô tả của test đã có nguyên văn trong pool, nhưng bảng B8 (adapter fold 0 hơn zero-shot +30 đến +34 điểm chrF) gộp cả dòng khuôn lẫn dòng mới. Hội đồng đòi bảng mô tả **tách dòng khuôn và dòng mới**. P1 đã làm việc đó trên eval fold 0 nhưng chưa làm trên **test**, nên câu trả lời hiện còn ghi "chưa tách dòng mới trên test".

Câu hỏi cần trả lời bằng số: trên test, fine-tune còn hơn zero-shot **ở các dòng mô tả mới** không, và hơn bao nhiêu so với dòng khuôn? Điều này quyết định có được giữ câu "fine-tune thắng zero-shot trên test" hay chỉ được nói ở mức dòng khuôn.

Không phải mục tiêu: đo lại chất lượng, so 5 mô hình trên test, đo bản cuối (B9), hay chọn lại bất kỳ cấu hình nào. Test **không** được dùng để lựa chọn gì.

## 2. Hai bước, có cổng duyệt giữa chừng

### Bước 0 (không chạm tệp test; chỉ đọc tổng hợp đã sinh cùng lần đo B8)

Hàm chấm của B8 (`D.cham(..., tap_da_thay=tap)`, `kaggle_train.py` dòng 1133 và 1140) **đã lưu** `dong_da_thay_moi` (chrF corpus trên dòng đã thấy và dòng mới, theo hướng, cùng số dòng) vào `do_test/do_test_trang_thai.json` (18.212 byte, sha256 `c93d6e1a3d4bdeed…`, theo B8). Nếu trường này có mặt thì **số tách khuôn/mới trên test đã nằm sẵn trong kết quả của đúng lần đo duy nhất**, không cần chấm lại gì. Việc làm: đọc riêng trường `ket_qua.<nhánh>/<zero_shot|ft>.dong_da_thay_moi` và số dòng, đối chiếu tổng với con số 85,0% của A7 và với bảng B8, ghi thành bảng. Không đọc `sinh_*.jsonl`, không đọc tham chiếu.
- Cần duyệt: được mở **một tệp tổng hợp** `do_test_trang_thai.json` (chỉ số tổng hợp, không có nguồn/bản dịch từng mục; chưa kiểm chứng điều này, sẽ dừng nếu thấy nội dung từng mục).
- Thời gian ước: 10–20 phút. Chạy trên máy cục bộ, CPU, không mạng.
- Kết quả: đủ để sửa câu "chưa tách dòng mới trên test" thành số thật (điểm chrF corpus, không có khoảng tin cậy).

### Bước 1 (chỉ làm nếu Bước 0 thiếu trường, hoặc cần khoảng tin cậy; **chạm tham chiếu test**)

Chấm lại **các bản dịch đã sinh sẵn** (không sinh mới, không GPU) để có điểm theo từng sản phẩm, rồi bootstrap theo sản phẩm.
- Đầu vào (tất cả cố định, ghi sha256 **trước** khi chạy): bản dịch `sinh_{cum,pair}_{ft,zero_shot}.jsonl` (4 tệp, theo B8: 5.415.370 / 6.023.822 / 5.318.098 / 6.092.578 byte); điểm từng mục `sinh_diem_*.jsonl` (chỉ để đối chiếu); **tham chiếu test** (`goi_do_khoa/do_khoa_{cum,pair}.jsonl*`, nguồn + bản dịch tham chiếu + nhóm hàng của 2.558 sản phẩm mỗi nhánh); `pool_{cum,pair}.jsonl` để dựng tập "đã thấy" (cặp `(category, dòng nguồn chuẩn hóa)` có trong fold 1–4, đúng `cham_diem.xay_tap_da_thay`, `fold_adapter=0`).
- Việc làm: chạy `cham_diem.cham` trên bản dịch đã sinh; **cổng tái lập bắt buộc**: tám con số theo loại và bốn con số tổng của bảng B8 (ví dụ cụm zh→vi 47,22 / 77,36; pair zh→vi 46,34 / 77,65) phải trùng đến 0,005; lệch thì dừng, không báo số tách. Sau đó tách mô tả theo dòng khuôn / dòng mới và tính chênh fine-tune trừ zero-shot trên từng phần, bootstrap theo sản phẩm 2.000 lần, `default_rng(42)`, cùng ma trận rút (như P1/P2).
- Cần duyệt: được mở **tham chiếu test** để chấm lại **đúng các bản dịch B8 đã có**. Không có lần sinh mới, không đổi bản dịch, không đổi adapter.
- Thời gian ước: viết script và test khoảng 1–1,5 giờ; chấm lại 4 tệp × khoảng 5.100 mục ước 20–40 phút CPU (P1 chấm 1.200 mục mỗi mô hình mất khoảng 1 phút); bootstrap vài phút. Tổng khoảng 2–3 giờ làm việc, tính toán dưới 1 giờ.

## 3. Phân tích và luật diễn giải (khóa ở đây, không sửa sau khi xem số)

- Chỉ số chính: chrF corpus của **mô tả** (zh→vi và vi→zh; cụm và pair), cho dòng **mới** và dòng **khuôn**, của zero-shot và fine-tune, và chênh fine-tune − zero-shot. Title không có khái niệm dòng khuôn, không đổi.
- Tỉ lệ dòng đã thấy: theo dòng và theo sản phẩm, đối chiếu 85,0% (A7). Nếu lệch quá 2 điểm phần trăm thì ghi lệch và lý do.
- **Giả thuyết ghi trước:** chênh fine-tune − zero-shot trên dòng khuôn **lớn hơn** trên dòng mới (fine-tune thuộc khuôn), nhưng trên dòng mới vẫn **dương**.
- **Luật theo B8:** thắng khi cận dưới khoảng 95% của chênh > 0.
  - Trên dòng mới cận dưới > 0 ở nhánh × hướng nào thì ghi "vẫn thắng trên dòng mô tả mới" cho nhánh × hướng đó, kèm độ lớn.
  - Cận dưới ≤ 0 thì ghi đúng như vậy và rút câu "fine-tune thắng zero-shot" cho phần mô tả dòng mới của nhánh đó; khi đó headline +30 điểm chỉ còn đúng cho dòng khuôn và title.
  - Nếu chênh trên dòng mới nhỏ hơn đáng kể so với dòng khuôn thì câu mới phải nêu cả hai số cạnh nhau; không dùng headline gộp.
- Không so các mô hình khác trên test. Không đo `cum_cuoi`/`pair_cuoi` trên test.
- Khoảng tin cậy chỉ phủ dao động chọn sản phẩm test, không phủ dao động khởi tạo (adapter một seed).

## 4. Biện pháp kiểm soát (vì test chỉ được đo một lần)

1. Khóa kế hoạch này bằng sha256 trước khi chạy; ghi sha256 từng tệp đầu vào trước khi chạy.
2. Chỉ đọc bản dịch đã có; không sinh, không GPU, không mạng, không Kaggle/Colab.
3. Kết quả ghi vào thư mục mới `data/processed/ft_run/phan_bien_v2/b8_dong_moi/`, không ghi đè tệp B8.
4. Cổng tái lập (Bước 1) trước khi báo bất kỳ số tách nào.
5. Không in nội dung từng mục (nguồn, bản dịch, tham chiếu) ra màn hình hay tài liệu; chỉ số tổng hợp theo nhóm.
6. Mã mới có test với dữ liệu giả; quét bí mật trong phần thay đổi.
7. Kết quả không được dùng để chọn lại cấu hình, adapter, ngưỡng hay tỉ trọng nào.

## 5. Đầu ra

`docs/phan_bien_vong2/KET_QUA_B8_DONG_MOI.md` (bảng khuôn/mới, đọc bảng, giới hạn), số thô `phan_bien_v2/b8_dong_moi/*.json`, cập nhật `TRA_LOI_V2_NHANH_NHAT_ANH.md` (bỏ câu "chưa tách dòng mới trên test") và các dòng "chưa tách dòng mới trên test" trong `pipeline/09` mục 4, `TONG_HOP`, `B8` (qua bộ sinh `xuat_ket_qua_kaggle.py`).

## 6. Rủi ro và giới hạn

- Hội đồng có thể cho rằng mở lại tham chiếu test là "xem test lần hai". Giảm nhẹ: chỉ chấm lại đúng bản dịch đã sinh, kế hoạch khóa, cổng tái lập, không dùng kết quả để chọn gì. Nếu Bước 0 đủ thì **không cần Bước 1**.
- Dòng mới của test vẫn có tham chiếu là bản dịch máy cùng kiểu với dữ liệu huấn luyện; chưa thay được người chấm.
- Bước 0 giả định `dong_da_thay_moi` có trong tệp tổng hợp; chưa kiểm chứng (không mở tệp).

## 7. Cần Nhật Anh quyết

1. Duyệt Bước 0 (mở một tệp tổng hợp)? Có / Không.
2. Nếu Bước 0 không đủ hoặc muốn có khoảng tin cậy, duyệt Bước 1 (chấm lại bản dịch B8 bằng tham chiếu test)? Có / Không.
