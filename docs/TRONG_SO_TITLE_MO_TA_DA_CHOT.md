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

## 2. Kiểm chéo bằng screening quota (Kaggle v55, 10/11/2026) [ĐO]

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
- Chưa thử mức title nằm giữa 11% và 50%; 11% là "tốt hơn 50% ở dòng mô tả mới", chưa chứng minh tối ưu.
- Pair attr_heavy seed 42 lệch 1,18 giữa v53 và v55, chưa rõ nguyên nhân (khởi tạo hoặc không tất định GPU).
- Tham chiếu là bản dịch máy, nên chrF title thấp (khoảng 50 đến 59) một phần do tham chiếu title có lỗi lõi 5 đến 14%.
- Bản dịch title mô hình sinh ra không lệch hình thức sang kiểu mô tả (độ dài bằng tham chiếu, không có ";" hay xuống dòng thừa) [ĐO].
- Cột "Chính (10/90)" cũ trong bảng v55 gốc bị sai: trọng số thực là cụm 8,33%, pair 67,48% (tính theo token mẫu eval). Đã sửa bằng hằng số 0,10; bảng ở trên là bảng đã tính lại.
