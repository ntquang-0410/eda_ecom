# BẢNG KẾT QUẢ CUỐI — QUOTA SAMPLING (cụm + pair)

Sinh tự động từ report: `version48/quota_report.json` (cụm), `version47/quota_report.json` (pair),
`kaggle_c0/c0_report_v50_cum.json`, `kaggle_c0/c0_report_v51_pair.json`.
Ngày 09/11/2026. Mọi số dưới đây là **[ĐO]**, trừ chỗ ghi [ƯỚC].

## A. BẢNG CHẠY ĐƯỢC (cấu hình đang dùng) — **KẾ THỪA, không do phiên hôm nay sinh ra**

| Bộ | Định nghĩa mẫu | Title | Description / Attribute | Nguồn con số |
|---|---|---:|---:|---|
| **Cụm** | 1 title + 1 description (khối) | **10,74%** | **89,26%** | `p_muc_tieu`/`p_thuc` của `cum_f0_p11` — bản cũ, giai đoạn 06-07/11 |
| **Pair** | 1 title + cả khối mô tả | **10,86%** | **89,14%** | `p_muc_tieu`/`p_thuc` của `pair_f0_p11` — bản cũ, giai đoạn 06-07/11 |

Phân rã 4 ô của bộ cụm (khớp bảng natural 10,72, dùng được để dựng quota):

| 1688_title | 1688_description | tiki_title | tiki_description |
|---:|---:|---:|---:|
| 6,26 | 74,21 | 4,46 | 15,07 |

Pair dạng **line** (title + 1 dòng attribute) thì bảng natural là 6,60 / 74,17 / 4,70 / 14,53 (title 11,30),
nhưng đây là **định nghĩa khác** với adapter `pair_f0_p11` (block), không được trộn.

## A2. NGUỒN GỐC TỪNG SỐ (bắt buộc ghi khi trích dẫn)

| Số | Nguồn | Thời điểm | Do phiên này chạy? |
|---|---|---|---|
| 10,74% / 89,26% (cụm) | `p_muc_tieu` = `p_thuc` = 0,1074 trong `tl_trang_thai.json` của `cum_f0_p11`; `p_khi_w_1` = 49,65% | 06-07/11 | **Không**, kế thừa |
| 10,86% / 89,14% (pair) | `p_muc_tieu` = `p_thuc` = 0,1086 trong `tl_trang_thai.json` của `pair_f0_p11`; `p_khi_w_1` = 50,30% | 06-07/11 | **Không**, kế thừa |
| 76,14 / 75,16 | `cham` trong `tl_trang_thai.json` (1,000 mẫu, 250/nhóm) | 06-07/11 | **Không**, kế thừa |
| 5 cấu hình cụm (74,06 … 70,07) | `version48/quota_report.json` | file ghi 09/11 11:52 | **Có** |
| 5 cấu hình pair line (71,51 … 67,65) | `version47/quota_report.json` | file ghi 09/11 08:01 | **Có** |
| C0 cụm 76,49 / C0 pair 74,55 | `c0_report_v50_cum.json`, `c0_report_v51_pair.json` | 09/11 13:12 và 13:18 | **Có** |
| Bảng 4 ô cụm 6,26-74,21-4,46-15,07 | `s_g` cấu hình natural của `version48/quota_report.json` | 09/11 | **Có** |

Hệ quả: nếu báo cáo chỉ được trích số **do đợt này chạy ra**, thì bảng trọng số hợp lệ là
**cụm natural 10,72 / 89,28** (4 ô như trên), còn **10,74 / 10,86 là số kế thừa**, phải ghi rõ nguồn là bản cũ.

## B. KẾT QUẢ 5 CẤU HÌNH — CỤM (block, 1,200 mẫu, rank16/960/1,200 step)

| Cấu hình | 1688_title đặt | thực | 1688_desc đặt | thực | tiki_title đặt | thực | tiki_desc đặt | thực | Title đặt | Title thực | chrF | Zero-shot | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 6,26 | 9,55 | 74,21 | 62,10 | 4,46 | 6,74 | 15,07 | 21,61 | 10,72 | 16,29 | 74,06 | 43,95 | +30,10 |
| balanced | 25,00 | 28,62 | 25,00 | 15,03 | 25,00 | 29,07 | 25,00 | 27,27 | 50,00 | 57,69 | 72,55 | 43,95 | +28,60 |
| title_heavy | 40,00 | 42,24 | 10,00 | 6,22 | 40,00 | 41,57 | 10,00 | 9,98 | 80,00 | 83,81 | 70,07 | 43,95 | +26,11 |
| desc_heavy | 2,00 | 2,83 | 55,00 | 40,61 | 3,00 | 4,19 | 40,00 | 52,38 | 5,00 | 7,01 | 73,33 | 43,95 | +29,37 |
| tiki_heavy | 10,00 | 10,76 | 15,00 | 9,18 | 35,00 | 37,92 | 40,00 | 42,14 | 45,00 | 48,68 | 71,49 | 43,95 | +27,53 |

## C. KẾT QUẢ 5 CẤU HÌNH — PAIR (line, 1,198 mẫu, rank16/960/1,200 step)

| Cấu hình | 1688_title đặt | thực | 1688_attr đặt | thực | tiki_title đặt | thực | tiki_attr đặt | thực | Title đặt | Title thực | chrF | Zero-shot | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 6,60 | 6,93 | 74,17 | 73,39 | 4,70 | 4,90 | 14,53 | 14,79 | 11,30 | 11,82 | 69,35 | 41,97 | +27,38 |
| balanced | 25,00 | 24,75 | 25,00 | 24,79 | 25,00 | 25,11 | 25,00 | 25,35 | 50,00 | 49,86 | 70,25 | 41,97 | +28,28 |
| title_heavy | 40,00 | 40,50 | 20,00 | 19,59 | 25,00 | 25,27 | 15,00 | 14,64 | 65,00 | 65,77 | 71,51 | 41,97 | +29,55 |
| attr_heavy | 10,00 | 10,69 | 60,00 | 58,98 | 5,00 | 5,13 | 25,00 | 25,20 | 15,00 | 15,82 | 67,65 | 41,97 | +25,68 |
| tiki_heavy | 10,00 | 9,76 | 20,00 | 19,31 | 30,00 | 30,11 | 40,00 | 40,82 | 40,00 | 39,87 | 70,21 | 41,97 | +28,24 |

## D. C0 — adapter official chấm trên đúng bộ eval của screen

| Bộ | Adapter | n eval | chrF | vi->zh | zh->vi | So với bảng tốt nhất của screen |
|---|---|---:|---:|---:|---:|---|
| Cụm | `cum_f0_p11` (title 10,74, rank64/2,048/2 epoch) | 1,200 | **76,49** | 75,46 | 77,52 | thắng natural 74,06 (+2,43) |
| Pair | `pair_f0_p11` (title 10,86, rank64/2,048/2 epoch) | 1,198 | **74,55** | 75,65 | 73,45 | thắng title_heavy 71,51 (+3,04) |

Chi tiết nhóm C0:

| Nhóm | Cụm vi->zh | Cụm zh->vi | Pair vi->zh | Pair zh->vi |
|---|---:|---:|---:|---:|
| title | 60,01 | 61,61 | 61,21 | 61,77 |
| mô tả / attribute | 90,91 | 93,42 | 90,09 | 85,12 |

## E. ĐỌC BẢNG

1. Cụm: title thực 7,01 -> 73,33; 16,29 -> **74,06** (đỉnh); 57,69 -> 72,55; 83,81 -> 70,07. Title 10-16% là vùng tốt.
2. Pair line: tăng title **tăng** điểm tới 65% (65,77 -> 71,51), giảm ở 15% (67,65). Cấu trúc dòng ngắn khác cụm.
3. C0 official thắng cả 10 cấu hình screen trên cùng eval.
4. Chênh lệch C0 vs screen **gộp cả công thức** (rank64/2,048/2 epoch vs rank16/960/1,200 step), không tách riêng được tác dụng của trọng số.

## F. HẠN CHẾ (phải ghi khi báo cáo)

- [ĐO] 1 seed (42), 1,200 mẫu eval, 1 lần train mỗi cấu hình, không CI.
- [ĐO] rank16, max_len 960 -> cụm natural realized title 16,29% (đặt 10,72) do bỏ mẫu dài.
- [ĐO] pair screen là **line**, adapter official là **block**: không so trực tiếp hai bảng.
- [CHƯA KIỂM] chưa có B3 line-level; chưa có confirm official recipe cho bảng quota; không đo lại test.
- [ƯỚC] Nếu chạy confirm official recipe (rank64/2,048/2 epoch) cho 1 bảng: 2-4h; cho 2 bộ: 4-8h.
