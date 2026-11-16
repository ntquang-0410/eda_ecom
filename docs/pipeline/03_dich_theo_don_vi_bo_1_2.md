# 03. Dịch lại bằng LLM theo đơn vị: bộ 1 (1688) và bộ 2 (Tiki), nguồn của **bản pair**

[← 02](02_eda_va_lam_sach_mo_ta.md) · [Tổng quan](00_TONG_QUAN.md) · [04 →](04_ra_soat_theo_cum_bo_3_4.md)

Thời gian: 04/10 – 23/10/2026. Nguồn:
- WALKTHROUGH Prompt 99–400;
- [`ket_qua_kaggle/A1`](../ket_qua_kaggle/A1_bo_song_ngu_1_thuoc_tinh_1688_dich_theo_don_vi_g.md), [`A2`](../ket_qua_kaggle/A2_bo_song_ngu_2_thuoc_tinh_tiki_vi_zh_dich_theo_do.md), đọc từ thẻ nguồn gốc `the_nguon_goc_*.json`;
- thẻ dữ liệu `data/release/description_zh_to_vi/README.md`, `description_vi_to_zh/README.md`.

Chỉ mô tả (thuộc tính) đi qua giai đoạn này. Title do Huy dịch riêng (file 05).

## 1. Chọn mô hình dịch (04 – 21/10)

| Ứng viên | Kết quả | Lý do |
|---|---|---|
| Qwen3-8B | bỏ khỏi vai trò dịch | để dành làm mốc (baseline) |
| HY-MT1.5-7B (Tencent) | **bỏ** | giấy phép cấm dùng đầu ra để cải thiện mô hình khác |
| DeepSeek-V4.1-Flash qua NVIDIA / Evolus | bỏ | quá chậm, giới hạn tốc độ (log thử 02/10) |
| Qwen3-30B-A3B, Qwen3-14B | chọn **Qwen3-14B-AWQ** (vLLM, Kaggle T4 ×2) | dịch thử 200 đơn vị, 14B không kém 30B mà nhẹ hơn |
| Gemma 4 31B / 26B-A4B qua Google AI Studio | chọn **`gemma-4-26b-a4b-it`** | 31B bị giới hạn RPM; 26B gọi được với 5 khóa xoay vòng; điều khoản Gemma không cản đề tài |

Trong mã và thẻ, **A = Gemma** (`api_tuy_chon`) và **B = Qwen3-14B** (`qwen3_14b`). Cả hai dịch **toàn bộ** đơn vị. Điểm chấm bằng **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`, không cần tham chiếu) chạy trên Kaggle.

Rủi ro Nhật Anh nêu từ đầu (Prompt 334): Qwen3-14B **cùng họ** với Qwen3-4B được fine-tune về sau, nên khi đọc điểm cần thận trọng.

## 2. Cách dịch "theo đơn vị"

1. **Định tuyến** từng cặp của `desc_v1`. Phần cố định không qua mô hình:
   - có/không;
   - mã thật, ngày;
   - tên hàng (Latin giữ nguyên);
   - giá trị chép nguyên;
   - từ điển đã duyệt: **P6** cho tên thuộc tính, **P7** cho giá trị. Nhật Anh duyệt danh sách và tự sửa vài mục, ví dụ 吊带, 纯色, 有无内胆.
2. Phần còn lại gom thành **đơn vị** là các tên hoặc giá trị *khác nhau*. Mỗi đơn vị dịch một lần, kèm luật kiểm tự động: chữ Hán sót, số, 斤/寸/两/万, độ dài, lặp từ.
3. **Chọn bản** theo luật ngưỡng trên điểm CometKiwi và luật kiểm:
   - ưu tiên A;
   - chỉ đổi sang B khi A trượt luật hoặc kém điểm rõ: `tau` 60, `kappa` 0,5, `delta` 0,25;
   - đơn vị cả hai cùng trượt thì **cắt riêng**.
   - Ngưỡng là tạm, đặt từ phân bố điểm, **không có nhãn đúng/sai**.
4. Ráp lại thành cặp, gắn **độ tin cậy** cao / vừa / thấp theo cách chọn. Ví dụ: hai bản giống hệt → cao; chỉ một bản qua luật → thấp.

Kiểm mẫu (rà soát bằng Claude, Nhật Anh đọc lại; không phải người song ngữ):
- **Bộ 1:** mẫu 100 cặp, tập trung vào cặp lấy từ Qwen và cặp độ tin cậy thấp. Kết quả ổn; sau đó đổi luật 斤 để không còn ca "cân" lọt vào (Prompt 346–350).
- **Bộ 2:** mẫu 200 cặp. Tỉ lệ "cặp hoàn toàn đúng" theo nhãn: cao 76/76, vừa 47/63, thấp 35/60 (Prompt 410).

## 3. Bộ 1: 1688, mô tả zh→vi (15 – 22/10) [ĐO, thẻ `the_nguon_goc_20261022_2126.json`]

- Đầu vào: 405.275 cặp của `desc_v1`.
- Định tuyến cặp: mô hình 338.273; chép nguyên 49.128; tên hàng 13.878; mã thật 2.479; ngày 72.
- Đơn vị cần dịch: **43.687** (1.980 tên + 41.707 giá trị).

| Lượt | Mô hình | Thời lượng |
|---|---|---|
| Gemma (2 phiên API) | gemma-4-26b-a4b-it | 1,01 + 1,41 giờ |
| Qwen | Qwen3-14B-AWQ, vLLM | 4,17 giờ |
| CometKiwi chấm A / B | | 0,24 / 0,26 giờ |

Kết quả chọn theo **đơn vị**: giữ 37.907/43.687.
- Cách chọn:
  - khác chữ cùng nghĩa → A: 14.937;
  - gần giống → A: 10.866;
  - giống hệt: 6.917;
  - chỉ A qua luật: 2.024;
  - chỉ B qua luật: 672;
  - còn lại là từ điển/quy tắc.
- Lý do cắt: khác nhiều mà điểm không đủ 3.277; gần giống nhưng điểm thấp 1.053; một bản qua luật nhưng điểm thấp 951; cả hai trượt 499.

Kết quả theo **cặp**: giữ **394.662**, cắt riêng **10.613**. Độ tin cậy: cao 188.708, vừa 187.755, thấp 18.199.

Đã đẩy HF thư mục `description_zh_to_vi/`. Thẻ dữ liệu viết tiếng Anh, ở `data/release/description_zh_to_vi/README.md`.

## 4. Bộ 2: Tiki, mô tả vi→zh (23/10) [ĐO, A2]

- Đầu vào: `tiki_eda.parquet` (HF `data/cleaned/tiki`), 13.200 sản phẩm, 89.300 cặp (89.325 trước làm sạch).
- Dùng lại pipeline của bộ 1 theo chiều ngược, có file duyệt P6/P7 riêng cho Tiki.
- Kết quả theo đơn vị: giữ 13.439/13.518.
- Kết quả theo cặp: giữ **87.056**, cắt riêng **2.244**. Độ tin cậy: cao 77.018, vừa 9.181, thấp 857.
- Đã đẩy HF `description_vi_to_zh/`.

## 5. Bộ 1–2 đi về đâu

- Bảng cặp đã chọn của bộ 1 và bộ 2, mỗi giá trị dịch **không ngữ cảnh**, là nguồn mô tả của file `*_full_attributes_independent_*`. File này Huy ghép với title thành **bản pair** (file 05).
- Bản dịch theo đơn vị của **cả hai mô hình** (A và B) được giữ lại. Judge sau này so chúng ở các đơn vị hai mô hình dịch khác nhau (34.534 đơn vị tranh chấp, file 07).
- Bộ 1–2 cũng là **bản nháp** cho bước rà soát theo cụm (bộ 3–4, file 04).
