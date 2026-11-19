# Bộ trọng số đóng góp Title : Mô tả đã chốt (bản cụm và bản pair)

Ghi ngày 11/11/2026 theo lệnh của Nhật Anh ("lưu lại kẻo quên"). Nhật Anh xác nhận lại ngày 12/11/2026 khi xem bảng v55 ở mục 2: chốt "natural" (title khoảng 11%) cho cả hai bộ. Nhãn: **[ĐO]** đã đo; **[ƯỚC]** ước tính; **[MẶC ĐỊNH]** em tự đặt.

## 1. Cái đã chốt

| | Bản cụm | Bản pair |
|---|---|---|
| % title trong loss (theo token đích) | **10,74%** | **10,86%** |
| % mô tả | 89,26% | 89,14% |
| `w_title` của bản cuối (mô tả nhân 1) | **0,17247** | **0,16915** |
| Cơ chế | trọng số loss mỗi mẫu title (mọi title được thấy 2 epoch) | như cụm |

- "Trọng số" ở đây là hệ số trong loss, không phải trọng số mô hình. Cách cũ 1:1 làm title chiếm khoảng 50% loss (cụm 49,65; pair 50,30) dù chỉ khoảng 11% token đích.
- `w_title` bản cuối cao hơn fold 0 (khoảng 0,12 và 0,118) vì đã loại 6.386/23.017 title bị cờ `loc_title.py`.
- Cấu hình nền: Qwen3-4B + LoRA rank 64, alpha 128, lr 2e-4, 2 epoch (xem `BAN_GIAO_AI_MOI.md` mục 6). Adapter chính thức và hash: mục 6.5 cùng file.
- Hai file chính thức trong đời thực: `data/processed/ft_run/ban_cuoi/cuoi/{cum_cuoi,pair_cuoi}/ep2/`.

### 1.1. Bảng 4 nhóm của cấu hình natural (screening v55, token đích, cap 2.048) [ĐO]

| Bộ | 1688_title | 1688_description / attribute | tiki_title | tiki_description / attribute | Title tổng |
|---|---:|---:|---:|---:|---:|
| Cụm | 7,13 | 75,83 | 3,67 | 13,38 | **10,79** |
| Pair (line) | 6,83 | 77,49 | 3,51 | 12,17 | **10,34** |

Nguồn: `data/processed/ft_run/kaggle_quota_step1/quota_report.json`; tỉ lệ thực hiện lệch ≤0,0003; 0 mẫu bị bỏ (dài nhất 2.047 cụm / 1.907 pair). Đây là bảng **tần suất lấy mẫu theo token** (dùng để dựng mẫu quota), **không phải** `w_title` của bản cuối ở mục 1; pair ở đây là dạng **line** (title + 1 dòng attribute), khác bản **block** đang giao.

## 2. Kiểm chéo bằng screening quota (Kaggle v55, 10/11/2026) [ĐO]

**Bằng chứng chính không phải mục này.** Bằng chứng chính cho `w_title` của bản cuối là thí nghiệm **trọng số loss** (B/TL: 3 mức 1:1 khoảng 50%, khoảng 11%, 75%; 3 fold; có khoảng tin cậy; luật khóa trước). Trong cơ chế đó mọi title được thấy đúng 2 epoch như nhau, chỉ đổi hệ số nhân loss từng mẫu; dữ liệu **không** bị lấy mẫu lại theo tỉ lệ (đã kiểm trong `scripts/ft_train/kaggle_train.py`). Khảo sát quota dưới đây chỉ là **kiểm chéo**, có confound tần suất (natural khoảng 0,31 lượt/epoch, balanced khoảng 1,4), rank 16, một lượt, 300 mẫu/nhóm, không có khoảng tin cậy chính thức; không dùng làm căn cứ chính.

Cấu hình screening: rank 16, một lượt, ngân sách 1,726M token đích (cụm), eval 1.200 mẫu (300 mỗi nhóm hướng x loại), 2 seed (42, 123), chrF corpus hướng zh→vi, điểm "Chính" = 0,10 x Title + 0,90 x Mô tả (hằng `W_TITLE_ZHVI` trong `scripts/ft_train/kaggle_quota.py`). Nguồn: `data/processed/ft_run/kaggle_quota_step1/quota_report.json`.

**Cụm**

| Cấu hình | % title | Title | Mô tả | Chính (10/90) | Macro 50/50 |
|---|---|---|---|---|---|
| natural (chốt) | 10,79 | 54,19 | 90,44 | 86,81 | 71,87 |
| balanced | 50 | 57,98 | 88,04 | 85,04 | 73,11 |
| title_heavy | 80 | 59,12 | 84,36 | 81,84 | 72,43 |

**Pair**

| Cấu hình | % title | Title | Mô tả | Chính (10/90) | Macro 50/50 |
|---|---|---|---|---|---|
| natural (chốt) | 10,34 | 49,98 | 83,31 | 79,98 | 68,52 |
| attr_heavy | 15 | 51,38 | 81,92 | 78,86 | 68,19 |
| balanced | 50 | 54,00 | 77,82 | 75,44 | 68,60 |

Zero-shot macro: cụm 43,95; pair 41,97.

**Kết luận:** giữ natural (title khoảng 11%) cho cả hai bộ. Đánh đổi title/mô tả là thật và cùng chiều ở cả hai seed (cụm balanced: title +3,80, mô tả -2,40; pair balanced: title +4,02, mô tả -5,49).

**w hòa vốn** (trọng số title mà cấu hình bắt đầu hơn natural) [ƯỚC, trung bình 2 seed, chưa có khoảng tin cậy]: cụm balanced 0,39; cụm title_heavy 0,55; pair attr_heavy 0,50; pair balanced 0,58. Nghĩa là: nếu nhóm coi title quan trọng từ khoảng 39% trở lên thì ở cụm balanced đáng chọn hơn. Cụm balanced cũng hơn natural ở macro 50/50 (+1,14 và +1,33 theo hai seed).

## 3. Giới hạn bắt buộc ghi kèm
- Mỗi nhóm chỉ 300 mẫu, chưa có khoảng tin cậy; mô tả pair zh→vi lệch 1,4 đến 1,9 điểm giữa hai seed.
- Screening khác bản cuối (rank 16, một lượt, lấy mẫu theo token): chỉ dùng so cấu hình với nhau. Screening không tách được "% title" khỏi "số lần mô hình thấy title" (natural thấy khoảng 0,31 lượt kho title, balanced khoảng 1,4 lượt [ƯỚC]), nên không dùng nó để kết luận 11% làm title kém đi.
- **Cập nhật 18/11:** đã chạy thêm mức 25% và 35% (bản cụm, fold 0, 1 seed, Kaggle L4x4) và tính khoảng tin cậy ghép cặp (`docs/phan_bien_vong2/KET_QUA_P2.md` mục B). Dưới thước chính 10/90: 25% − 11% = −0,19 [−0,41; +0,03] (không phân biệt được); 35% − 11% = −0,39 [−0,68; −0,09] (title zh→vi +1,19, mô tả zh→vi −0,56: đánh đổi). Dưới thước macro (title và mô tả ngang nhau) ba mức không phân biệt được. Vì vậy **không nói "11% tối ưu"**; câu được phép: "dưới thước 10/90, 11% không kém 25% và hơn 35% một chút (0,4 điểm); tăng tỉ trọng title cải thiện title zh→vi nhưng đổi bằng mô tả". Các mức 12–24% và 26–34% chưa thử; chỉ bản cụm, một seed; chưa chạy bản pair.
- (Câu cũ trước 18/11, giữ để truy vết) "Chưa thử mức title nằm giữa 11% và 50%; 11% là 'tốt hơn 50% ở dòng mô tả mới'… mức 25% và 35% chưa chạy." Câu này đã lỗi thời.
- Khoảng tin cậy trong các bảng chỉ phủ dao động do chọn mẫu eval, không phủ dao động khởi tạo. Số seed: bảng sàng lọc 2 seed (42, 123); công thức chính 1 seed.
- Pair attr_heavy seed 42 lệch 1,18 giữa v53 và v55, chưa rõ nguyên nhân (khởi tạo hoặc không tất định GPU).
- Tham chiếu là bản dịch máy, nên chrF title thấp (khoảng 50 đến 59) một phần do tham chiếu title có lỗi lõi 5 đến 14%.
- Bản dịch title mô hình sinh ra không lệch hình thức sang kiểu mô tả (độ dài bằng tham chiếu, không có ";" hay xuống dòng thừa) [ĐO].
- Cột "Chính (10/90)" cũ trong bảng v55 gốc bị sai: trọng số thực là cụm 8,33%, pair 67,48% (tính theo token mẫu eval). Đã sửa bằng hằng số 0,10; bảng ở trên là bảng đã tính lại.
