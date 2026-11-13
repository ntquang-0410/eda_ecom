# BẢNG QUOTA (CÔNG THỨC QUANG) — KẾT QUẢ PHIÊN 09/11/2026

Nguồn: `version48/quota_report.json` (cụm, ghi 09/11 11:52) và `version47/quota_report.json` (pair, ghi 09/11 08:01).
Đây là `s_g` — tỉ lệ token mục tiêu theo 4 nhóm — do quota sinh ra, **không phải** trọng số loss `w_title` cũ.
Công thức: `S_g = Σ e_i²`, `w_g = s_g / S_g`, `p_i ∝ w_g · e_i`. Cụm = Quang B (block), pair = Quang A (line).

## 1. CỤM (Quang B, block: title + description)

| Cấu hình | 1688_title (đặt) | 1688_description (đặt) | tiki_title (đặt) | tiki_description (đặt) | Title đặt | Title thực | chrF |
|---|---:|---:|---:|---:|---:|---:|---:|
| **natural** | 6,26 | 74,21 | 4,46 | 15,07 | 10,72 | 16,29 | 74,06 |
| balanced | 25,00 | 25,00 | 25,00 | 25,00 | 50,00 | 57,69 | 72,55 |
| title_heavy | 40,00 | 10,00 | 40,00 | 10,00 | 80,00 | 83,81 | 70,07 |
| desc_heavy | 2,00 | 55,00 | 3,00 | 40,00 | 5,00 | 7,01 | 73,33 |
| tiki_heavy | 10,00 | 15,00 | 35,00 | 40,00 | 45,00 | 48,68 | 71,49 |

**Chốt cụm = natural:** 1688_title **6,26** / 1688_description **74,21** / tiki_title **4,46** / tiki_description **15,07** (title 10,72).

## 2. PAIR (Quang A, line: title + từng dòng attribute)

| Cấu hình | 1688_title (đặt) | 1688_attribute (đặt) | tiki_title (đặt) | tiki_attribute (đặt) | Title đặt | Title thực | chrF |
|---|---:|---:|---:|---:|---:|---:|---:|
| natural | 6,60 | 74,17 | 4,70 | 14,53 | 11,30 | 11,82 | 69,35 |
| balanced | 25,00 | 25,00 | 25,00 | 25,00 | 50,00 | 49,86 | 70,25 |
| **title_heavy** | 40,00 | 20,00 | 25,00 | 15,00 | 65,00 | 65,77 | 71,51 |
| attr_heavy | 10,00 | 60,00 | 5,00 | 25,00 | 15,00 | 15,82 | 67,65 |
| tiki_heavy | 10,00 | 20,00 | 30,00 | 40,00 | 40,00 | 39,87 | 70,21 |

**Chốt pair line = title_heavy:** 1688_title **40,00** / 1688_attribute **20,00** / tiki_title **25,00** / tiki_attribute **15,00** (title 65,00).

## 3. Ghi chú bắt buộc

- Hai bảng này là **đầu ra của quota**, dùng trực tiếp để dựng mẫu cho lần train sau (truyền `s_g` vào bộ lấy mẫu), không quy đổi ra `w_title`.
- Cụm và pair **khác định nghĩa mẫu** (block vs line), không so điểm với nhau.
- Số điểm đi kèm là screen rank16/960/1,200 step/1 seed, **chưa phải** cấu hình giao.
- Pair title_heavy 65% là line; adapter official hiện tại là block, nên bảng này **chưa thay được** adapter block mà không có B3 + confirm.
