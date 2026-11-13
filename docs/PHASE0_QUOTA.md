# Pha 0: Dry-run quota Survey A

**Ngày:** 09/11/2026  
**Phạm vi:** Survey A (title + attribute/description) × 2 bộ (cụm, pair)  
**GPU:** 0. Không train, không mở test.

## 1. Cấu hình

- **Cụm = Quang B, block-level**: `1688_title`, `1688_description`, `tiki_title`, `tiki_description`.
- **Pair = Quang A, line-level**: `1688_title`, `1688_attribute`, `tiki_title`, `tiki_attribute`.
- 5 cấu hình mỗi bộ: natural, balanced, title_heavy, desc/attr_heavy, tiki_heavy.
- Natural lấy từ dữ liệu; 4 cấu hình còn lại theo Quang.
- Quota: `T_g = s_g × 4.000.000`; lấy đều trong nhóm đến đủ `T_g`; cap `e_i` ở 960 hoặc 2.048.
- Seed 42; đã chạy cả `flt=0` (không lọc title) và `flt=1` (lọc theo `loc_title/pool.parquet`).

## 2. Dữ liệu

- Mỗi bộ 25.575 sản phẩm, 9 cột; 14.837 1688 + 10.738 Tiki; 0 missing/empty.
- Cụm: mô tả một khối `name: value; ...`; median 16 mục, max 70.
- Pair: 436.017 dòng toàn bộ, 0 lệch zh/vi, 0 dòng rỗng, 0 dòng thiếu `:`; 1688 24,42 dòng/sản phẩm, Tiki 6,87.
- Pool train: 18.533 sản phẩm; pair 332.424 dòng; cụm 37.123 mẫu.
- Lọc title: 18.533 title → **13.366** (bỏ 5.167; 1688 10.680→9.324, Tiki 7.853→4.042).

## 3. Natural title share

| Bộ | Cap | Không lọc | Có lọc |
|---|---:|---:|---:|
| Cụm | 960 | 10,72% | 7,85% |
| Cụm | 2.048 | 10,09% | 7,37% |
| Pair | 960 | 11,30% | 8,29% |
| Pair | 2.048 | 11,20% | 8,21% |

Lọc title làm title share giảm khoảng 2,7-3,0 điểm. Cap 960 làm title share cụm tăng 0,63 điểm so với 2.048 (do cắt description dài).

## 4. Dry-run realized share (cap 2.048, có lọc)

| Bộ | Cấu hình | Title đặt | Title thực (token cặp) | Title thực (token đích) | n_draw | unique | max_rep | compute [ƯỚC] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cụm | natural | 7,37% | 7,37% | 7,80% | 12.360 | 10.270 | 5 | 4,86M |
| Cụm | balanced | 50% | 49,98% | 50,52% | 42.220 | 16.898 | 14 | 5,67M |
| Cụm | title_heavy | 80% | 79,99% | 80,33% | 59.658 | 15.172 | **20** | 5,86M |
| Cụm | desc_heavy | 5% | 5,00% | 4,91% | 14.747 | 10.708 | 7 | 5,24M |
| Cụm | tiki_heavy | 45% | 44,99% | 43,63% | 41.892 | 14.789 | 16 | 5,86M |
| Pair | natural | 8,21% | 8,22% | 8,32% | 139.191 | 113.272 | 6 | 16,06M |
| Pair | balanced | 50% | 50,00% | 50,66% | 110.707 | 73.534 | 14 | 11,87M |
| Pair | title_heavy | 65% | 65,00% | 66,00% | 98.484 | 59.430 | 15 | 10,08M |
| Pair | attr_heavy | 15% | 15,00% | 15,38% | 135.436 | 108.309 | 6 | 15,55M |
| Pair | tiki_heavy | 40% | 40,00% | 39,63% | 119.897 | 73.119 | 16 | 13,23M |

- Group share khớp `s_g` tới khoảng 0,0005; overshoot 0,01-0,04%.
- Pair natural: 1688_attribute lấy 110.960 dòng, unique 90.214, max_rep 6.
- **Cảnh báo**: cum `title_heavy` + lọc title có `max_rep = 20` (1688_title 15, tiki_title 20). Một số title bị lặp 20 lần; đây là hệ quả của 80% quota title trên nhóm title đã lọc còn 13.366 mẫu.

## 5. Chi phí screen 10 run (cap 960, có lọc)

| Bộ | 5 run compute [ƯỚC] |
|---|---:|
| Cụm | 27,6M |
| Pair | 67,2M |
| **Tổng** | **94,8M** |

Ở 5.000-5.500 token/giây: khoảng **4,8-5,3h** compute; cộng overhead 20% khoảng **5,8-6,4h**. Ước tính 7h là conservative.

## 6. Eval sample

- Mỗi bộ 1.200 mẫu dev, 300/nhóm, seed 20261109.
- Cụm: 1.200 block; pair: 1.200 dòng.
- File: `data/interim/quota_phase0/eval_{tap}_sample.csv` và `.npy`.

## 7. File đã tạo

| File | Nội dung |
|---|---|
| `scripts/ft_train/quota_sampling.py` | Dựng quota, dry-run, eval sample |
| `scripts/ft_train/test_quota_sampling.py` | Test công thức; tất cả đạt |
| `data/eda/khao_sat_trong_so/quota_phase0/summary.csv` | Tổng hợp 40 cấu hình × cap × lọc |
| `data/eda/khao_sat_trong_so/quota_phase0/dry_run.csv` | Chi tiết theo nhóm |
| `data/eda/khao_sat_trong_so/quota_phase0/meta.json` | Natural share, group counts |
| `data/interim/quota_phase0/idx_q_*.npy` | Index mẫu quota |
| `data/interim/quota_phase0/train_ref_*.csv` | Bảng train tương ứng index |
| `data/interim/quota_phase0/eval_*` | Mẫu eval cố định |

## 8. Quyết định cần chốt trước Kaggle smoke

1. **Lọc title**: `flt=1` (giống bản cuối, natural title 7,4-8,2%) hay `flt=0` (giống Quang, natural 10,1-11,2%)?
2. Nếu `flt=1`, có giữ `title_heavy` cụm 80% không? Nó lặp tới 20 lần. Có thể giảm còn 65-70% hoặc giữ làm stress test.
3. Cap: screen 960, confirm 2.048 như plan.
4. Smoke trên Kaggle: 100 step cum balanced + pair natural, cap 960, đo token/giây, eval 200 mẫu, khoảng 0,3-0,5h.

## 9. Kết luận

- Công thức quota chạy đúng trên cả 2 bộ; group share khớp, độ dài giữ tự nhiên.
- Dữ liệu 9 cột hỗ trợ đầy đủ; pair line-level 436k dòng khớp 100%.
- Vấn đề dữ liệu lớn nhất là **lọc title làm title group nhỏ**, khiến `title_heavy` lặp nhiều. Cần chốt trước khi chạy.
