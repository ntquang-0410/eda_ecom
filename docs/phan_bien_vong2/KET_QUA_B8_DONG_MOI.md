# KẾT QUẢ BƯỚC 0: DÒNG KHUÔN / DÒNG MỚI TRÊN TEST (B8)

Ngày 19/11/2026. Người làm: Claude, theo `KE_HOACH_B8_DONG_MOI_TEST.md` (sha256 `fa75f69a2b4b4647901cdc4783d935c45c6435e408c58d67b9f322c36cc05565`), Nhật Anh duyệt **Bước 0** ngày 19/11/2026. **Chỉ mở một tệp**: `data/processed/ft_run/kaggle_do_test/phien_1_20261107_080604/do_test/do_test_trang_thai.json` (18.212 byte; sha256 bắt đầu `c93d6e1a3d4bdeed`, trùng giá trị ghi ở `B8_do_test_khoa_mot_lan.md`). Chỉ đọc các trường tổng hợp (`ket_qua.*.dong_da_thay_moi`, `theo_nhom`, `mau_do`); **không** mở bản dịch, điểm từng mục hay tham chiếu test; không chấm lại, không sinh gì. Số dưới đây đã được tính trong **đúng lần đo B8 duy nhất** (07/11) bởi hàm `cham_diem.cham(..., tap_da_thay=...)`; bản này chỉ đọc ra. **Không có khoảng tin cậy** (cần Bước 1, chưa duyệt).

## 1. Tỉ lệ dòng mô tả của test đã có nguyên văn trong train

Định nghĩa "đã thấy": cặp `(category, dòng nguồn chuẩn hóa)` có trong fold 1–4, mà adapter fold 0 đã học (đúng như P1). Chỉ tính mục căn được dòng (`can_dong`).

| Nhánh | Hướng | Dòng đã thấy | Dòng mới | % đã thấy |
|---|---|---:|---:|---:|
| cụm | zh→vi | 31.478 | 5.099 | 86,1% |
| pair | zh→vi | 31.269 | 4.831 | 86,6% |
| cụm | vi→zh | 5.404 | 2.022 | 72,8% |
| pair | vi→zh | 5.418 | 2.036 | 72,7% |

[ĐO] Số dòng lấy từ bản dịch fine-tune (mục căn được gần như đầy đủ). Gần với 85,0% của A7 (A7 đếm theo cách khác); zh→vi cụm 86,1% và pair 86,6%, lệch 1,1 và 1,6 điểm phần trăm so với 85,0%, trong ngưỡng 2 điểm đặt trước. Hướng vi→zh có tỉ lệ thấp hơn (khoảng 73%).

## 2. chrF corpus mô tả trên dòng khuôn và dòng mới, zero-shot so với fine-tune

| Nhánh | Hướng | Dòng khuôn: zero-shot → fine-tune (chênh) | Dòng mới: zero-shot → fine-tune (chênh) |
|---|---|---|---|
| cụm | zh→vi | 54,89 → 99,24 (**+44,35**) | 46,32 → 76,38 (**+30,06**) |
| pair | zh→vi | 55,73 → 99,31 (**+43,58**) | 46,02 → 76,61 (**+30,59**) |
| pair | vi→zh | 35,77 → 96,17 (**+60,40**) | 46,46 → 76,15 (**+29,69**) |
| cụm | vi→zh | **không so được** (xem dưới); fine-tune khuôn 96,49 | **không so được**; fine-tune mới 76,90 |

[ĐO] Hai điều đọc được.
1. **Trên dòng mô tả mới, fine-tune vẫn hơn zero-shot khoảng 30 điểm** ở cả ba ô so được (+30,06; +30,59; +29,69), tức cùng cỡ với headline +30 đến +34 của B8. Giả thuyết ghi trước (chênh trên dòng khuôn lớn hơn trên dòng mới, và trên dòng mới vẫn dương) **đúng về điểm ước**: chênh trên dòng khuôn lớn hơn 13 đến 31 điểm.
2. **Mức tuyệt đối của fine-tune bị thổi phồng bởi dòng khuôn:** trên dòng khuôn fine-tune đạt 99,2 đến 99,3 (zh→vi) và 96,2 đến 96,5 (vi→zh), trên dòng mới chỉ 76,2 đến 76,9. Điểm chrF mô tả toàn phần của B8 (92 đến 94) phản ánh chủ yếu việc tái tạo khuôn; con số "chất lượng dịch dòng mới" là khoảng 76, không phải 92 đến 94. Câu "dịch tốt 77%" không được dùng như độ chính xác.

Zero-shot cũng điểm cao hơn khoảng 9 điểm ở dòng khuôn so với dòng mới (54,9 so 46,3 ở cụm zh→vi), nên một phần của "khuôn" là các dòng thuộc tính ngắn, dễ.

## 3. Giới hạn (đọc trước khi trích số)

- **Không có khoảng tin cậy.** Luật thắng của B8 (cận dưới khoảng 95% > 0) chưa áp dụng được cho dòng mới; số trên là điểm ước chrF corpus. Với hàng nghìn dòng mới và chênh khoảng 30 điểm, khoảng tin cậy nhiều khả năng xa 0, nhưng chưa tính. Cần Bước 1 (chấm lại đúng bản dịch đã sinh bằng tham chiếu test) mới có.
- **Tập mục không giống nhau giữa zero-shot và fine-tune.** Zero-shot hay làm lệch số dòng nên bị loại nhiều mục khỏi phần tách dòng (`so_muc_khong_can_duoc`): cụm zh→vi 70 trên 1.485 mục, cụm vi→zh **920 trên 1.073** (chỉ còn 153 mục, nên **không so được ở ô này**), pair zh→vi **371 trên 1.485**, pair vi→zh 8 trên 1.073; fine-tune chỉ bị loại 0 đến 6 mục mỗi ô. Chênh ở ô pair zh→vi so trên tập mục hơi khác nhau (zero-shot thiếu một phần tư mục), nên con số +30,59 nên đọc là xấp xỉ.
- Điểm chrF corpus theo dòng ở đây khác thước "chrF theo mục, trung bình theo sản phẩm" ở bảng B8 gốc (ví dụ mô tả cụm zh→vi fine-tune ở bảng B8 là 93,78, theo nhóm corpus ở đây là 92,19); không so trực tiếp hai thước.
- Tham chiếu vẫn là bản dịch máy cùng kiểu với dữ liệu huấn luyện; chưa có người chấm. Một seed. Chỉ mô tả (title không có dòng khuôn).
- Mẫu test: 2.558 sản phẩm mỗi nhánh, 5.116 mục, `so_phan_khac_0` = 0, nguồn đo `test`, đã xong, không có bất thường (theo trường trạng thái).

## 4. Câu được phép viết

"Trên test, 86 đến 87% dòng mô tả zh→vi (73% vi→zh) đã có nguyên văn trong dữ liệu huấn luyện của adapter. Trên các dòng mô tả **mới**, adapter fold 0 vẫn hơn zero-shot khoảng 30 điểm chrF corpus ở các ô so được (cụm zh→vi +30,1; pair zh→vi +30,6; pair vi→zh +29,7), và chênh trên dòng khuôn lớn hơn (+44 đến +60). Điểm mô tả toàn phần (92 đến 94) bị nâng bởi dòng khuôn; chất lượng trên dòng mới khoảng 76. Chưa có khoảng tin cậy cho phần tách này; tham chiếu là bản dịch máy."
