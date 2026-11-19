# 08. Chia tập, khóa test, cờ title, dựng mẫu huấn luyện

[← 07](07_judge_deepseek.md) · [Tổng quan](00_TONG_QUAN.md) · [09 →](09_fine_tune_va_danh_gia.md)

Thời gian: 31/10 – 12/11/2026, trên máy. Nguồn:
- [`ket_qua_kaggle/A7`](../ket_qua_kaggle/A7_chia_tap_theo_san_pham_kiem_ro_ri_gan_co_title_t.md), [`B0`](../ket_qua_kaggle/B0_dung_mau_huan_luyen_va_dong_goi_dataset_kaggle_t.md);
- `scripts/ft_train/chia_tap.py`, `kiem_ro_ri.py`, `loc_title.py`, `dung_mau.py`;
- `docs/KE_HOACH_FT_QWEN3_4B_plan_v0.7.md`.

## 1. Kiểm chất lượng tham chiếu trước khi chia (31/10)

- Bản dịch trong bộ đều do máy tạo và chưa ai kiểm. Nhật Anh không mua API Claude, chỉ có gói Pro. Phương án chốt (Prompt 724): **Claude chấm hiệu chuẩn** một mẫu nhỏ trong phiên, Gemini Pro web chấm thêm 200 title.
- Kết quả dùng về sau, đều là rà soát bằng LLM:
  - title có lỗi lõi khoảng 5–14%, hai người chấm bất đồng nhiều;
  - mô tả lỗi nặng dưới 1% mảnh.
- Nhật Anh quyết **không làm tập vàng người kiểm**, để Judge làm phần đánh giá dữ liệu (31/10). Điều này **trái điều lệ nhóm** (test phải do người dịch/kiểm) và phải nêu trong báo cáo.

## 2. Chia tập (`chia_tap_v2`, khóa 05/11, seed 20261104) [ĐO]

- **Đơn vị chia là sản phẩm**, và mọi sản phẩm gần trùng phải nằm cùng một phía.
- **Gom nhóm gần trùng:**
  - nối tập chính xác trên 6 góc nhìn ở cả hai phía ngôn ngữ: title Jaccard ≥ 0,5, mô tả ≥ 0,7;
  - thêm luật "cùng thương hiệu và title chứa trong nhau";
  - nối bắc cầu;
  - kết quả **18.448 nhóm**, nhóm lớn nhất 969 sản phẩm.
- **Test khóa 10% = 2.558 sản phẩm** (zh→vi 1.485, vi→zh 1.073), lấy theo nhóm.
- **Pool 23.017 sản phẩm** (zh→vi 13.352, vi→zh 9.665), chia **5 fold** theo nhóm, phân tầng hướng × ngành. Cỡ fold: 4.484 / 5.084 / 4.483 / 4.483 / 4.483.
- Cụm và pair dùng **chung một cách chia** (cùng sản phẩm, cùng fold).
- **Kiểm rò rỉ độc lập** (`kiem_ro_ri.py`): 0 vi phạm **ở cấp sản phẩm**. Không có sản phẩm gần-trùng giữa test và pool theo ngưỡng Jaccard; đây là held-out theo sản phẩm, **không** phải held-out cho từng dòng mô tả.
  - Riêng ở mức dòng thuộc tính: 85,0% dòng mô tả của test đã có nguyên văn trong pool. Đây là đặc điểm của dữ liệu thuộc tính (nhiều dòng lặp như "Xuất xứ: Trung Quốc"), không phải lỗi chia. Chính đặc điểm này làm Moses mạnh ở bản pair trên toàn bộ mô tả; trên dòng mô tả mới, Moses học đủ gần Qwen3-4B ở pair (−2,50) nhưng kém ở cụm (−3,68) (P1, eval fold 0).

Hash để kiểm:

| File | sha256 |
|---|---|
| `bo_cum_chon_deepseek.parquet` | `e0a625b2…a20def` |
| `bo_pair_chon_deepseek.parquet` | `d4535c62…7230d` |
| `split.parquet` | `d66a39c7…ee4fa3` |
| danh sách test | `0d7c7c15…e09101` |

### Quy tắc khóa test (áp suốt dự án)

- Không mở `data/processed/ft_run/test_khoa/`, `split.parquet`, `ds_test.txt` để xem ví dụ hay để chọn gì.
- Test chỉ đo **một lần** (07/11). Sau đó không đổi tỉ lệ, lr, adapter theo test, và không đo lại bản cuối trên test.
- Dataset Kaggle `ft-qwen3-4b-pool` chỉ chứa pool. Test nằm ở Dataset riêng tư `ft-qwen3-4b-do-khoa`, chỉ gắn vào phiên đo test. Mã có guard chặn đọc nhầm.
- Mọi luật quyết định được **khóa trước khi xem số** (ghi vào WALKTHROUGH rồi mới chạy).

## 3. Gắn cờ title nghi lỗi (`loc_title.py`) [ĐO]

- Toàn bộ 25.575 sản phẩm: 7.127 title có cờ (1688 1.920, Tiki 5.207).
- Trên pool: **6.386/23.017 (27,7%)**. Theo hướng: vi→zh 4.657/9.665 (48,2%), zh→vi 1.729/13.352 (12,9%).
- Độ chính xác thấp: trên 277 title đã chấm, cờ bắt được 18/26 title lỗi lõi, nhưng phần lớn cờ là nhầm (18/104 title có cờ là lỗi thật). Đây là **bộ sàng**, không phải bằng chứng sai.
- Dùng ở **bản cuối**: loại mẫu title của các sản phẩm có cờ, giữ mẫu mô tả (file 09). Phương án sửa title thật bằng Gemini (khoảng 1.729 title zh→vi) **chưa làm**.

## 4. Dựng mẫu huấn luyện (`dung_mau.py`)

Mỗi sản phẩm cho **2 mẫu**: một mẫu title, một mẫu mô tả, theo đúng hướng gốc (1688 zh→vi, Tiki vi→zh). Dùng khuôn chat Qwen3, tắt suy luận (`enable_thinking=False`). Loss chỉ tính trên phần bản dịch.

```text
Title:      Translate this product title from {SRC} to {TGT}. Output only the translation.\n\n{NGUON}
Mô tả pair: Translate this product's attribute list from {SRC} to {TGT}. Each line is one attribute written as "name: value".
            Keep the same number and order of lines, and output only the translation.
            \nProduct title (context only, do not translate): {TITLE}\n\n{NGUON}
Mô tả cụm:  ... The attributes are separated by "; " and each is written as "name: value". Keep the same number and order
            of attributes, separated by "; ", and output only the translation. (+ cùng phần title làm ngữ cảnh)
```

- Title nguồn được đưa vào lời dẫn của mẫu mô tả, làm ngữ cảnh.
- Mô tả vượt `max_len` 2.048 token thì chia thành nhiều phần theo thuộc tính. Thuộc tính đơn lẻ vẫn quá dài thì loại.

| Bộ | Mẫu pool | Token pool | zh→vi / vi→zh | Title / mô tả | Token title / mô tả | Sản phẩm bị chia phần | Mẫu bị loại | Mẫu test (chỉ đếm) |
|---|---|---|---|---|---|---|---|---|
| **cụm** | 46.120 | 16.215.152 | 26.791 / 19.329 | 23.017 / 23.103 | 1.973.652 / 14.241.500 | 127 | 58 | 5.116 |
| **pair** | 46.110 | 15.392.361 | 26.781 / 19.329 | 23.017 / 23.093 | 1.973.652 / 13.418.709 | 113 | 55 | 5.116 |

Số token đếm cả lời dẫn. Tỉ trọng title theo **token đích** (10,74% cụm, 10,86% pair) tính từ chính các mẫu này lúc huấn luyện (file 09).

## 5. Dataset riêng tư trên Kaggle

| Dataset | Tài khoản | Nội dung |
|---|---|---|
| `ft-qwen3-4b-pool` | 1 và 2 (hash manifest trùng) | mẫu pool: cụm 46.120, pair 46.110 |
| `ft-qwen3-4b-quota` | 1, 2 | chỉ số mẫu, bảng tham chiếu, eval cố định 1.200 mẫu fold 0 cho khảo sát quota và năm mô hình |
| `ft-song-ngu` | 2 | dữ liệu từng dòng của bản pair cho OpenNMT và Moses: zh-vi train 269.895 dòng, vi-zh 61.401; bản khớp 4M token; eval; dev |
| `ft-qwen3-4b-do-khoa` | 1 | test khóa (cụm 5.116, pair 5.116), chỉ gắn vào phiên đo test |

Tài khoản 1 = `trinhnhatanh`, tài khoản 2 = `honghuynanh`.
