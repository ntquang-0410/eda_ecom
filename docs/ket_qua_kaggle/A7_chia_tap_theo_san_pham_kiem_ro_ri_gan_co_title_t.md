# A7. Chia tập theo sản phẩm, kiểm rò rỉ, gắn cờ title (trên máy)

[← Mục lục](INDEX.md) · [← A6](A6_judge_deepseek_chon_ban_dich_gemma_hay_qwen_cum.md) · [B0 →](B0_dung_mau_huan_luyen_va_dong_goi_dataset_kaggle_t.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | A7 |
| Giai đoạn | A. Dữ liệu: dịch, lọc, Judge |
| Notebook Kaggle | – (`scripts/ft_train/chia_tap.py`, `kiem_ro_ri.py`, `loc_title.py`) |
| Tài khoản | – (chạy trên máy) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 04–05/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run` |

## Mục đích

Đóng băng bản dữ liệu sau Judge, chia một phép chia dùng chung cho cả hai bộ: gom sản phẩm gần-trùng (title, mô tả) vào cùng nhóm để không rơi vào hai phía, khóa 10% làm test, pool còn lại chia 5 fold phân tầng theo hướng × nhóm hàng; kiểm rò rỉ; gắn cờ title nghi lỗi bằng luật.

## 1. Đóng băng dữ liệu và phép chia (`ft_run/DU_LIEU_DONG_BANG.json`)

| Mục | Giá trị |
|---|---|
| Nguồn | HF `ntquang0410/zh-vie_ecom`, `data/LLM/bilingual_judged`, commit `82aa9c9` |
| Phiên bản phép chia | chia_tap_v2 |
| Seed | 20261104 |
| Tỉ lệ test | 10% |
| Số fold của pool | 5 |
| Test khóa / pool | 2.558 / 23.017 sản phẩm |
| Ngưỡng Jaccard gom nhóm gần-trùng | `{"title_zh": 0.5, "title_vi": 0.5, "mo_ta_zh": 0.7, "mo_ta_vi": 0.7, "mo_ta_zh_cum": 0.7, "mo_ta_vi_cum": 0.7, "chua_cung_hieu": 0.7}` |
| sha256 phép chia / danh sách test | `d66a39c7dbe76bd9…` / `0d7c7c1535d11548…` |
| Quy tắc | Tập test (la_test=True) chỉ dùng để đánh giá. Không mẫu nào của test được đưa vào huấn luyện, dò siêu tham số hay chọn cấu hình. Đọc phép chia qua chung.doc_split(). |

## 2. Kết quả chia (`ft_run/bao_cao_chia_tap.json`)

| Mục | Giá trị |
|---|---|
| Số cặp gần-trùng theo góc nhìn | `{"title_zh": 2076, "title_vi": 4757, "mo_ta_zh": 16915, "mo_ta_vi": 16107, "mo_ta_zh_cum": 16456, "mo_ta_vi_cum": 15556, "chua_cung_hieu": 4716}` |
| Số nhóm / nhóm từ 2 sản phẩm / nhóm lớn nhất | 18.448 / 2.157 / 969 |
| Test theo hướng | `{"zh->vi": 1485, "vi->zh": 1073}` |
| Số sản phẩm mỗi fold | `{"0": 4484, "1": 5084, "2": 4483, "3": 4483, "4": 4483}` |
| Tỉ lệ sản phẩm thuộc nhóm từ 2 | `{"toan_bo": 0.363, "test": 0.3393, "pool": 0.3656}` |

Phân tầng theo hướng × nhóm hàng:

| Hướng | Nhóm hàng | Tổng | Test | Tỉ lệ test | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|---|---|---|
| vi->zh | auto | 1.796 | 152 | 8.5% | 280 | 579 | 265 | 269 | 251 |
| vi->zh | beauty | 2.025 | 192 | 9.5% | 357 | 401 | 353 | 360 | 362 |
| vi->zh | electronics | 2.054 | 268 | 13.1% | 351 | 364 | 359 | 353 | 359 |
| vi->zh | food | 1.708 | 171 | 10.0% | 311 | 307 | 306 | 307 | 306 |
| vi->zh | home | 1.668 | 149 | 8.9% | 247 | 455 | 270 | 269 | 278 |
| vi->zh | mother_baby | 1.487 | 141 | 9.5% | 266 | 312 | 259 | 254 | 255 |
| zh->vi | auto | 954 | 95 | 10.0% | 172 | 171 | 172 | 172 | 172 |
| zh->vi | bags | 2.857 | 286 | 10.0% | 514 | 514 | 514 | 514 | 515 |
| zh->vi | beauty | 1.278 | 128 | 10.0% | 230 | 230 | 230 | 230 | 230 |
| zh->vi | electronics | 1.079 | 108 | 10.0% | 195 | 194 | 194 | 194 | 194 |
| zh->vi | fashion | 1.319 | 132 | 10.0% | 237 | 237 | 238 | 238 | 237 |
| zh->vi | food | 1.376 | 138 | 10.0% | 248 | 247 | 248 | 247 | 248 |
| zh->vi | home | 1.748 | 175 | 10.0% | 315 | 314 | 314 | 315 | 315 |
| zh->vi | mother_baby | 1.927 | 193 | 10.0% | 347 | 346 | 347 | 347 | 347 |
| zh->vi | shoes | 2.299 | 230 | 10.0% | 414 | 413 | 414 | 414 | 414 |

## 3. Kiểm rò rỉ (`ft_run/bao_cao_ro_ri.json`)

Kết luận trong file: **KHÔNG RÒ RỈ**. Title trùng chính xác giữa test và pool: `{"title_zh": 0, "title_vi": 0}`.

| So | Góc nhìn | Ngưỡng | Số truy vấn | Số vi phạm |
|---|---|---|---|---|
| test với pool | title_zh | 0.50 | 2.558 | 0 |
| test với pool | title_vi | 0.50 | 2.558 | 0 |
| test với pool | mo_ta_zh_pair | 0.70 | 2.558 | 0 |
| test với pool | mo_ta_vi_pair | 0.70 | 2.558 | 0 |
| test với pool | mo_ta_zh_cum | 0.70 | 2.558 | 0 |
| test với pool | mo_ta_vi_cum | 0.70 | 2.558 | 0 |
| test với pool | title_zh_cung_hieu | 0.70 | 1.430 | 0 |
| test với pool | title_vi_cung_hieu | 0.70 | 1.436 | 0 |

Fold với các fold khác: 40 phép so, tổng số vi phạm 0.
Tỉ lệ dòng mô tả của test đã có nguyên văn trong pool: 85.0% (thuộc tính khuôn mẫu như "Chất liệu: Cotton"; đây là lý do các phiên B chấm riêng "dòng mới").

## 4. Lọc title bị gắn cờ (`ft_run/loc_title/bao_cao.json`)

Cờ luật trên title của bộ sau Judge (toàn bộ 25.575 sản phẩm): `co_co` = số title dính ít nhất một cờ.

| Hướng | n | Có cờ | Hán trong bản Việt | Việt trong bản Trung | Âm tiết Việt | Mất chữ Latin | Lệch số | Lệch hiệu |
|---|---|---|---|---|---|---|---|---|
| vi->zh | 10.738 | 5.207 | 0 | 316 | 854 | 3.932 | 721 | 788 |
| zh->vi | 14.837 | 1.920 | 0 | 0 | 0 | 228 | 1.132 | 608 |

Đánh giá cờ trên mẫu title đã chấm tay: 277 title đã chấm, 26 title lỗi, cờ bắt được 18; trong 104 title có cờ thì 18 là lỗi thật. Theo từng cờ: `{"han_trong_vi": {"co": 0, "trung_loi": 0}, "viet_trong_zh": {"co": 4, "trung_loi": 2}, "am_tiet_viet": {"co": 12, "trung_loi": 2}, "mat_latin": {"co": 66, "trung_loi": 9}, "so_lech": {"co": 22, "trung_loi": 4}, "hieu_lech": {"co": 23, "trung_loi": 8}}`.

## Kết luận của phiên

- Không rò rỉ theo ngưỡng đã đặt (0 vi phạm test với pool và giữa các fold).
- Cờ title bắt được phần lớn title lỗi trên mẫu chấm tay nhưng cũng gắn cờ nhiều title không lỗi; Nhật Anh chọn loại mẫu title bị cờ ở bản cuối (B9), giữ mẫu mô tả.

## Ghi chú, giới hạn, sự cố

- Tên script ghi theo chức năng; mã nằm trong `scripts/ft_train/`.
- Dòng mô tả của test phần lớn đã có nguyên văn trong pool vì thuộc tính lặp theo khuôn; vì vậy điểm mô tả cao không đồng nghĩa mô hình dịch tốt câu mới, các phiên B có chỉ số "dòng mới" riêng.
- Thư mục `ft_run/test_khoa/` chứa tập test; không mở khi viết tài liệu này.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/DU_LIEU_DONG_BANG.json` | 1.433 | `7d699d8a17ad606e` |
| `ft_run/bao_cao_chia_tap.json` | 4.288 | `344544a258cf26db` |
| `ft_run/bao_cao_ro_ri.json` | 11.604 | `9fcd8bd4117cb561` |
| `ft_run/loc_title/bao_cao.json` | 1.321 | `7363c42758bbae1e` |
