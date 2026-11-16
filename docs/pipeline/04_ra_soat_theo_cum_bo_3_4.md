# 04. Rà soát theo cụm có ngữ cảnh: bộ 3 (Tiki) và bộ 4 (1688), nguồn của **bản cụm**

[← 03](03_dich_theo_don_vi_bo_1_2.md) · [Tổng quan](00_TONG_QUAN.md) · [05 →](05_gop_tiki_vao_1688_official.md)

Thời gian: 23/10 – 26/10/2026. Nguồn:
- WALKTHROUGH Prompt 401–509;
- [`ket_qua_kaggle/A3`](../ket_qua_kaggle/A3_bo_song_ngu_3_ra_soat_theo_cum_co_ngu_canh_tiki.md), [`A4`](../ket_qua_kaggle/A4_bo_song_ngu_4_ra_soat_theo_cum_co_ngu_canh_1688.md);
- thẻ `data/release/description_vi_to_zh_cum/README.md`, `description_zh_to_vi_cum/README.md`.

## 1. Vì sao cần bản cụm

Nhật Anh yêu cầu thêm một bản dịch **cả cụm** (Prompt 401, 419). Mục đích là để dữ liệu fine-tune mang ngữ cảnh sản phẩm: cùng một giá trị có thể dịch khác nhau tùy title và các thuộc tính khác.
- Ví dụ: 粤妆20161271 là mã giấy phép mỹ phẩm Quảng Đông, chỉ biết được khi nhìn cả sản phẩm.

"Cụm" ở đây là **toàn bộ danh sách thuộc tính của một sản phẩm**.

## 2. Cách làm (bản v2, gộp 3 cách thử trên cùng 200 sản phẩm)

1. **Bản nháp** của mỗi cặp lấy từ bộ 1 (1688) hoặc bộ 2 (Tiki), đã dịch theo đơn vị.
2. Phần **khóa** (từ điển P6/P7, có/không, mã thật, chép nguyên, tên hàng) không cho mô hình sửa.
3. Mô hình nhận **title + toàn bộ danh sách thuộc tính + bản nháp**, trả lại danh sách đã rà, giữ nguyên số mục và thứ tự. Việc của mô hình: giữ nháp, sửa, hoặc dịch mới mục còn thiếu.
4. **Luật kiểm cấu trúc:** đủ mục, đúng thứ tự, không sót chữ nguồn, số khớp. Mục trượt luật thì **lùi về nháp**.
5. **Gemma (A) chạy chính. Qwen3-14B (B) chỉ dùng khi cụm của Gemma hỏng hoặc trượt luật** (Prompt 457). Nhật Anh chốt cách này sau khi xem chạy thử 200 mẫu.
6. Gắn độ tin cậy cấp cặp và cấp sản phẩm. Nhãn sản phẩm lấy theo cặp thấp nhất, kèm cột đếm cặp thấp `so_cap_thap`.

Hạ tầng:
- Gemma gọi qua API từ 4 notebook Kaggle chạy song song (`song-ngu-4-cum-gemma`, `-p1..p3`).
- Qwen chạy offline trên RTX Pro 6000 của notebook `nlp-group3` (gắn cuộc thi), cho khoảng 1.733 cụm cần dự phòng (Prompt 502).

## 3. Bộ 3: Tiki, cụm vi→zh (23 – 24/10) [ĐO, A3]

| Thành phần | Nguồn cuối |
|---|---|
| Tên thuộc tính | khóa 85.803; giữ nháp 1.116; Gemma sửa 219; dịch mới 1 |
| Giá trị | khóa 72.850; giữ nháp 12.096; Gemma sửa 2.073; Qwen sửa 25; dịch mới 56; lùi nháp 12 |

- Độ tin cậy cặp: cao 76.820, vừa 8.967, thấp 1.325.
- Độ tin cậy sản phẩm (13.187 sản phẩm có mô tả): cao 6.979, vừa 4.982, thấp 1.226.
- Có các lần **chấm mù** mẫu: `cham_mau_*`, `cham_that_*`, khóa giải mã để riêng.
- HF: `description_vi_to_zh_cum/` (bảng sản phẩm, đã bỏ hai cột JSON thuộc tính cho nhẹ).

## 4. Bộ 4: 1688, cụm zh→vi (24 – 26/10) [ĐO, A4]

Nguồn bản dịch cuối theo cặp:

| Nguồn | Số cặp |
|---|---|
| giữ bản dịch đơn vị (bộ 1) | 206.673 |
| từ điển / quy tắc | 172.338 |
| Gemma sửa trong ngữ cảnh | 15.664 |
| dịch mới (mục nháp thiếu) | 6.999 |
| Qwen sửa | 18 |

- Độ tin cậy cặp: cao 188.362, vừa 188.086, thấp 25.244.
- Độ tin cậy sản phẩm: thấp 12.020, vừa 4.322 (16.342 sản phẩm).
  - Nhãn "thấp" phổ biến vì sản phẩm nhận nhãn của cặp thấp nhất. Về sau quy tắc gộp không dùng nhãn này mà dùng tỉ lệ cặp yếu (file 05).
- Qwen chỉ kịp dịch một phần cụm trong thời gian có GPU (`cum_qwen.chua_dich` lớn). Phần còn lại theo Gemma.
  - Hệ quả: ở bản cụm, số mục mà Judge có hai ứng viên để so ít hơn hẳn bản pair (4.446 mục so với 34.534 đơn vị, file 07).
- HF: `description_zh_to_vi_cum/`.

## 5. Cụm và pair khác nhau đến đâu

- Bản cụm **xuất phát từ** bản dịch đơn vị (bộ 1–2) rồi chỉ sửa ở chỗ mô hình thấy cần trong ngữ cảnh. Phần lớn cặp giống bản pair, khác ở khoảng 4,8% mảnh (bàn giao 07/11).
- Ngày 28/10 Nhật Anh thêm cột `desc_nguon_dich` (nguồn bản dịch mô tả) vào bảng sản phẩm của cả 4 bộ (`data/release/desc_nguon_dich_20261028/`). Huy cần cột này để gộp.
