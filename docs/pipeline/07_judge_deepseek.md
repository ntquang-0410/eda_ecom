# 07. Judge DeepSeek: chọn bản Gemma hay Qwen (bộ `bilingual_judged`)

[← 06](06_loc_luat_ft_clean.md) · [Tổng quan](00_TONG_QUAN.md) · [08 →](08_chia_tap_va_dung_mau.md)

Thời gian: thí điểm 29–31/10/2026 trên máy; lần chạy đầy đủ do Huy chạy; bản phát hành tải về 03/11/2026. Nguồn:
- [`ket_qua_kaggle/A6`](../ket_qua_kaggle/A6_judge_deepseek_chon_ban_dich_gemma_hay_qwen_cum.md);
- `HUONG_DAN_JUDGE.md` (bàn giao cho Huy, 18 quyết định D1–D18);
- `scripts/judge/README.md`;
- `data/processed/bo_chon_hf/README.md`, `manifest.json`;
- WALKTHROUGH Prompt 570–698.

## 1. Bài toán

- Mỗi đơn vị dịch có hai ứng viên: bản Gemma và bản Qwen (file 03–04).
- CometKiwi đã dùng để chọn ở bộ 1–2. Nhật Anh muốn thêm một **LLM làm giám khảo** để chọn ở những chỗ hai bản khác nhau, cho cả title và mô tả, cả bản cụm và bản pair. Lý do: chọn thuần bằng CometKiwi thì đề tài thiếu tính mới (Prompt 610).
- Các phương án đã cân nhắc và bỏ:
  - Claude chấm toàn bộ: gói Pro không đủ hạn mức;
  - Gemini web;
  - NVIDIA free: lỗi tốc độ.
- Chốt: **`deepseek-v4.1-flash`, tắt suy luận**, gọi qua Alibaba Cloud Token Plan.

## 2. Giao thức (chốt sau thí điểm)

- **Mù tên mô hình.** Hai bản được gọi là bản 1 và bản 2.
- **Hai lượt, hoán đổi vị trí.** Chỉ thay bản hiện có khi **cả hai lượt cùng chọn một mô hình**. Lý do: thí điểm cho thấy nếu cố định thứ tự, mô hình đứng trước được lợi khoảng 26 điểm; tỉ lệ chọn vị trí 1 là 62,6–67,1% (Prompt 661).
- Rubric **v2**, siết để judge chọn được trong một lần. Có quy tắc cho trường hợp hai bản tương đương thật (ví dụ 电泵 và 电动泵).
- Mục khai **tương đương**, **không nhất quán**, hoặc **chưa chấm** thì **giữ hiện trạng**: bản đang có trong bộ, mặc định theo pipeline trước đó.
- **Chặn** bản mới nếu nó gây sót chữ Hán (1688) hoặc thêm chữ Việt (Tiki).
- Không đưa vào so cặp mục mà một ứng viên rỗng.
- **Gom** khoảng 15 mục mỗi lời gọi: token mỗi mục giảm từ 1.060 xuống 360; kết quả trùng 223/247 (90,3%) mục bền vững so với chấm từng mục.
- Đơn vị chấm theo từng bản:

| | Title | Mô tả bản cụm | Mô tả bản pair |
|---|---|---|---|
| Đơn vị chấm | nguyên title | từng cặp `tên: giá trị`. Cách lai: cặp duy nhất không ngữ cảnh (4.446 mục), thêm ngữ cảnh sản phẩm khi cần | từng **tên** hoặc **giá trị** riêng. Chấm 5.000 đơn vị tranh chấp dùng nhiều nhất trên 34.534, phủ khoảng 84% mảnh |
| Số mục DeepSeek chấm | mọi title (Nhật Anh chốt "cho judge xem tất cả", Prompt 682); số lời gọi chi tiết nằm ở revision `126a564f` chưa tải | 4.439/4.446. 7 mục bị nhà cung cấp chặn được rà riêng | 5.000/34.534. Phần chưa chấm giữ bản trước theo pipeline |

- Thiết kế có **tầng 2** (thêm ngữ cảnh cho mục không nhất quán), nhưng file đánh dấu cho thấy lần chạy đầy đủ **không dùng tầng 2** (judge tầng 2 = 0).

## 3. Thí điểm (Nhật Anh, trên máy) [ĐO]

So cặp mù hai thứ tự, 100 mục mỗi loại, quy tắc v1:

| Loại | Gemma | Qwen | Hòa | Không nhất quán | Chọn vị trí 1 | Gemma trong số có người thắng |
|---|---|---|---|---|---|---|
| title | 67 | 5 | 0 | 28 | 62,8% | 93,1% |
| cụm | 62 | 9 | 1 | 28 | 62,6% | 87,3% |
| pair | 46 | 5 | 11 | 38 | 67,1% | 90,2% |

- Gemma thắng áp đảo ở cả ba loại.
- Khoảng 28–38% mục không nhất quán giữa hai thứ tự. Vì vậy cần luật "hai lượt cùng chọn".
- Bảng đầy đủ 30 lần thí điểm (mô hình, token, giây): A6 mục 1.

## 4. Lần chạy đầy đủ và bản phát hành

- **Huy chạy** theo `HUONG_DAN_JUDGE.md` và gói dữ liệu Nhật Anh chuẩn bị (Prompt 694–697), rồi đưa lên HF: `ntquang0410/zh-vie_ecom`, `data/LLM/bilingual_judged`, **commit `82aa9c9`**.
- Bản tải về máy: `data/processed/bo_chon_hf/`. Hash kiểm trong `scripts/ft_train/chung.py`.
- Bản trước khi thu gọn (có file truy vết chi tiết): revision `126a564f`. Bản này **chưa tải về**.

| File | Nội dung |
|---|---|
| `bo_cum_chon_deepseek.parquet` | Bản **cụm** cuối, 25.575 sản phẩm, 9 cột |
| `bo_pair_chon_deepseek.parquet` | Bản **pair** cuối, cùng 25.575 sản phẩm, cùng thứ tự |
| `bo_cum_danh_dau.parquet`, `bo_pair_danh_dau.parquet` | Trạng thái quyết định Judge theo sản phẩm |

- 9 cột: `product_id, source, category, title_zh, title_vi, description_zh, description_vi, judge_chon_title, judge_chon_description`.
  - `judge_chon_title` là mô hình tạo bản title cuối (`gemma`/`qwen`). Hai title đã được sửa tay.
  - `judge_chon_description` là **đếm nguồn** nội dung cuối theo mảnh, không phải điểm.

Nguồn nội dung cuối [ĐO, đếm trên file HF]:

| | Bản cụm | Bản pair |
|---|---|---|
| Title gemma / qwen | 19.261 / 6.314 | 19.261 / 6.314 (title giống hệt) |
| Mảnh mô tả gemma / qwen / quy_tac / chua_ro | 217.818 / 493 / 140.080 / 83.451 | 316.057 / 12.756 / 237.909 / 305.312 |

- `quy_tac` là mảnh khóa bằng từ điển hoặc luật.
- `chua_ro` là mảnh chưa chấm hoặc không quyết, nên giữ hiện trạng. Ở bản cụm, `chua_ro` còn gồm mảnh có tên và giá trị đến từ hai mô hình khác nhau.
- Bản cụm và bản pair đếm mảnh khác nhau, **không so trực tiếp** số đếm giữa hai bản.

Judge thực sự thay đổi bao nhiêu (từ file đánh dấu) [ĐO]:

| | Bản cụm | Bản pair |
|---|---|---|
| Mảnh tranh chấp | 17.170 | 195.311 |
| Judge quyết ở tầng 1 | 9.755 | 59.470 |
| Khai tương đương | 7.408 | 104.331 |
| Chưa chấm | – | 31.510 |
| **Mảnh bị đổi bản** | **615** | **9.054** |
| Title bị đổi | 7.604 (chung hai bản) | |

## 5. Đọc kết quả thế nào

- Judge chủ yếu **xác nhận bản đang có**, thường là Gemma. Số mảnh mô tả thực sự đổi bản nhỏ: 615 ở cụm, 9.054 ở pair. Riêng title thay đổi nhiều (7.604).
- Bộ sau Judge **chưa được người kiểm chứng toàn bộ**, không phải tập vàng.
- Ước lượng chất lượng bằng LLM chấm (bàn giao 07/11):
  - title có lỗi lõi khoảng 5–14%. Hai người chấm (Claude, Gemini) bất đồng nhiều.
  - mô tả (một mảnh) lỗi nặng dưới 1%: zh→vi 3/1.466 mảnh, vi→zh 0/282.
- Độ phủ Judge của bản pair thấp (khoảng 14,5% đơn vị tranh chấp được chấm). Pair không khử trùng dòng lặp.
