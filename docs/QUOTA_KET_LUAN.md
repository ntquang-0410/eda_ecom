# QUOTA SURVEY — KẾT LUẬN CHỐT (10/11/2026)

Nguồn: `kaggle_quota_step1/quota_report.json` — 12 run (3 cấu hình × 2 bộ × 2 seed), cap 2.048, 0 mẫu bị bỏ, ngạch thực hiện lệch ≤ 0,0003, loss chuẩn hoá theo token đích.

## 1. Kết quả (trung bình 2 seed)

| Bộ | Cấu hình | Chính 10/90 | Macro 50/50 | Title zh->vi | Mô tả zh->vi |
|---|---|---:|---:|---:|---:|
| cum | **natural** | 86.81 | 71.87 | 54.19 | 90.44 |
| cum | balanced | 85.04 | 73.11 | 57.98 | 88.04 |
| cum | title_heavy | 81.84 | 72.43 | 59.12 | 84.36 |
| pair | **natural** | 79.98 | 68.52 | 49.98 | 83.31 |
| pair | balanced | 75.44 | 68.60 | 54.00 | 77.82 |
| pair | attr_heavy | 78.86 | 68.19 | 51.38 | 81.92 |

## 2. Chênh so natural, từng seed

| Bộ | Cấu hình | Seed | Δ chính | Δ title | Δ mô tả |
|---|---|---:|---:|---:|---:|
| cum | balanced | 42 | -1.79 | +3.67 | -2.40 |
| cum | balanced | 123 | -1.76 | +3.92 | -2.39 |
| cum | title_heavy | 42 | -5.27 | +4.62 | -6.37 |
| cum | title_heavy | 123 | -4.68 | +5.25 | -5.79 |
| pair | balanced | 42 | -6.06 | +3.66 | -7.14 |
| pair | balanced | 123 | -3.02 | +4.37 | -3.84 |
| pair | attr_heavy | 42 | -2.61 | +1.78 | -3.10 |
| pair | attr_heavy | 123 | +0.38 | +1.02 | +0.31 |

## 3. Ngưỡng trọng số title để cấu hình kia thắng natural

Chỉ thắng khi `Δtitle / |Δmô tả| > (1-w)/w`. Suy ra w hòa vốn:

| Bộ | Cấu hình | Δtitle | Δ mô tả | Tỉ lệ | w hòa vốn |
|---|---|---:|---:|---:|---:|
| cum | balanced | +3.80 | -2.40 | 1.58 | **0.39** |
| cum | title_heavy | +4.93 | -6.08 | 0.81 | **0.55** |
| pair | balanced | +4.02 | -5.49 | 0.73 | **0.58** |
| pair | attr_heavy | +1.40 | -1.39 | 1.00 | **0.50** |

Đọc: với w=0,10 (title 10% trọng số), cả 4 cấu hình đều thua natural vì tỉ lệ đánh đổi 1,0-1,6 << 9. Nhưng nếu hội đồng cho title nặng ≥ 0,39 (cụm) hoặc ≥ 0,50 (pair), kết luận đổi. Con số w hòa vốn không phụ thuộc w hiện tại, chỉ phụ thuộc Δtitle và Δmô tả.

## 4. Đã chứng minh / chưa chứng minh

**Đã chứng minh [ĐO]:**

- Pipeline quota chạy đúng: 0 mẫu bị bỏ ở cap 2.048 (dài nhất 2.047), ngạch thực hiện lệch ≤0,0003, loss chuẩn hoá theo token đích, mỗi bước ~8.000 token đích, không cycle.
- Với metric chính (zh->vi, 10/90) và 2 seed: natural > balanced > title_heavy (cụm); natural > attr_heavy > balanced (pair); cả 4 ứng viên đều âm hoặc đổi dấu.
- Đánh đổi title↔mô tả là thật và nhất quán: tăng title luôn làm mô tả giảm; tỉ lệ đánh đổi 0,31-1,58, giảm dần khi title tăng.
- Trên macro 50/50, cụm balanced hơn natural; tức kết luận phụ thuộc trọng số w.

**Chưa chứng minh:**

- Chưa khảo sát mức title 11-50%: tỉ lệ đánh đổi đo được là trung bình trên khoảng rộng, không phải đạo hàm tại 11%, nên không loại trừ được một điểm gấp khúc.
- Chưa tách được nguyên nhân lệch 1,18 điểm của pair attr_heavy giữa v53 và v55 (nghi init/RNG hoặc phi tất định GPU; 5/6 cấu hình còn lại lặp trong 0,20).
- 300 mẫu/nhóm, không có khoảng tin cậy chính thức; rank16, một lượt; tham chiếu là bản dịch máy.
- Không đo lại test.

## 5. Câu viết cho báo cáo

> Khảo sát quota trên 2 bộ (3 cấu hình/bộ, 2 seed, cap 2.048, loss chuẩn hoá theo token đích) cho thấy bảng natural — title khoảng 10,8% (cụm) và 10,3% (pair) tính trên token đích — là tốt nhất trên metric triển khai (zh->vi, title 10% / mô tả 90%). Tăng tỉ lệ title làm title tăng chậm dần nhưng mô tả giảm đều, với tỉ lệ đánh đổi 0,31-1,58, thấp hơn xa ngưỡng hòa vốn 9 của metric này. Trên macro 50/50 thì cụm balanced nhỉnh hơn, cho thấy kết luận phụ thuộc trọng số title; ngưỡng để cấu hình khác thắng natural là w≈0,39 (cụm) và w≈0,50 (pair). Khảo sát chỉ dùng để chọn bảng, chưa phải kết quả test.

> Lưu ý cơ chế: screening điều khiển tần suất lấy mẫu theo token; bản giao dùng trọng số loss `w_title` để đạt cùng mức ~11% token title. Hai cách khác nhau, cùng nhắm một đại lượng.

## 6. Việc tiếp

- Không chạy thêm GPU để đổi bảng: natural là bảng đang dùng.
- Nếu hội đồng yêu cầu đường cong đầy đủ: chạy thêm mức title 25% trên cụm, chấm 1.000 mẫu/nhóm 1688 (~2-3h).
- Viết báo cáo với các mục 4 và 5 ở trên.
