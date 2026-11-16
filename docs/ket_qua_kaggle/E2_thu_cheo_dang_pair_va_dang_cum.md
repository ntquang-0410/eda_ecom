# E2. Thử chéo dạng pair và dạng cụm

[← Mục lục](INDEX.md) · [← E1](E1_huan_luyen_lai_adapter_fold_0_o_tai_khoan_2_cum.md) · [E3 →](E3_cong_thuc_chinh_tren_eval_nam_mo_hinh_dong_thu_s.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | E2 |
| Giai đoạn | E. Thử chéo và công thức chính trên eval năm mô hình |
| Notebook Kaggle | `honghuynanh/nlp-cheo` |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_cheo_that`, `data/processed/ft_run/cheo_ket_qua` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_cheo_that/bao_cao_cheo.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | 0.18.1 | 4.863 giây (1.35 giờ) | có | khong_co_file_test_trong_input |

## Mục đích

Cho mỗi adapter dịch cả hai dạng đầu vào (cụm và pair) trên 600 sản phẩm fold 0, mô tả zh→vi, cộng zero-shot; chấm trên máy với cả hai tham chiếu.

## Mẫu và lượt sinh (`cheo_trang_thai.json`)

| Mục | Giá trị |
|---|---|
| Ứng viên (sản phẩm zh→vi fold 0, mô tả nguyên, nguồn hai nhánh giống hệt) | 1.827 |
| Số sản phẩm chọn | 600 |
| Trong đó thuộc mẫu cố định của các lần huấn luyện | 177 |
| Bỏ vì nguồn hai nhánh khác nhau | 793 |
| Seed chọn mẫu | 43 |

| Điều kiện (adapter / dạng đầu vào) | Số mẫu | Giây mỗi mẫu |
|---|---|---|
| zero_shot/pair | 600 | 1.661 |
| zero_shot/cum | 600 | 1.609 |
| cum_f0_p11/pair | 600 | 1.169 |
| cum_f0_p11/cum | 600 | 1.183 |
| pair_f0_p11/pair | 600 | 1.100 |
| pair_f0_p11/cum | 600 | 1.184 |

## Kết quả chấm trên máy (`cham_cheo.py` → `ft_run/cheo_ket_qua/bang.md`, chép nguyên)


Mẫu: 600 sản phẩm (fold 0, mô tả nguyên, nguồn hai nhánh giống hệt); 177 sản phẩm thuộc mẫu cố định của các lần huấn luyện.

### Điểm từng điều kiện

| Adapter | Dạng đầu vào | chrF câu TB (2 tham chiếu) | chrF corpus vs ref cụm | chrF corpus vs ref pair | Đúng số thuộc tính | Giữ số | Rò lời dẫn | Rỗng | Tỉ lệ dài |
|---|---|---|---|---|---|---|---|---|---|
| cum_f0_p11 | cum | 93.69 | 92.47 | 91.53 | 99.8% | 98.4% | 0.0% | 0.0% | 1.00 |
| cum_f0_p11 | pair | 92.72 | 91.44 | 90.60 | 97.7% | 98.1% | 0.0% | 0.0% | 1.00 |
| pair_f0_p11 | cum | 93.21 | 91.23 | 91.99 | 99.7% | 98.4% | 0.0% | 0.0% | 1.00 |
| pair_f0_p11 | pair | 93.73 | 91.70 | 92.51 | 100.0% | 98.4% | 0.0% | 0.0% | 1.00 |
| zero_shot | cum | 56.01 | 54.83 | 54.79 | 94.3% | 97.2% | 0.0% | 0.0% | 1.12 |
| zero_shot | pair | 55.88 | 54.73 | 54.72 | 10.2% | 96.4% | 0.0% | 0.0% | 1.14 |

### So sánh theo cặp (chênh chrF câu TB, khoảng 95% bootstrap theo sản phẩm)

| So sánh | A | B | Chênh A−B | Khoảng 95% | Kết luận |
|---|---|---|---|---|---|
| Adapter pair: đưa dạng pair hay dạng cụm tốt hơn? | pair_f0_p11/pair | pair_f0_p11/cum | +0.51 | [+0.41; +0.63] | A tốt hơn |
| Adapter cụm: đưa dạng cụm hay dạng pair tốt hơn? | cum_f0_p11/cum | cum_f0_p11/pair | +0.97 | [+0.70; +1.31] | A tốt hơn |
| Cách dùng đúng dạng: pair + adapter pair so với cụm + adapter cụm | pair_f0_p11/pair | cum_f0_p11/cum | +0.04 | [-0.16; +0.23] | chưa phân biệt được |
| Dạng cụm: adapter pair (sai dạng) so với adapter cụm (đúng dạng) | pair_f0_p11/cum | cum_f0_p11/cum | -0.47 | [-0.68; -0.28] | B tốt hơn |
| Dạng pair: adapter cụm (sai dạng) so với adapter pair (đúng dạng) | cum_f0_p11/pair | pair_f0_p11/pair | -1.01 | [-1.39; -0.67] | B tốt hơn |
| Adapter pair so với không adapter (dạng pair) | pair_f0_p11/pair | zero_shot/pair | +37.85 | [+37.02; +38.74] | A tốt hơn |
| Adapter cụm so với không adapter (dạng cụm) | cum_f0_p11/cum | zero_shot/cum | +37.68 | [+36.91; +38.49] | A tốt hơn |

### Đối chiếu với lần huấn luyện (kiểm quy trình)

- cum_f0_p11/cum: n=177, chrF của bản dịch lúc huấn luyện 94.04, sinh lại ở phép thử chéo 94.05, 153/177 bản dịch giống hệt; thiếu ở bản huấn luyện: 0.
- pair_f0_p11/pair: n=177, chrF của bản dịch lúc huấn luyện 94.05, sinh lại ở phép thử chéo 94.00, 155/177 bản dịch giống hệt; thiếu ở bản huấn luyện: 0.

## Kết luận của phiên

- Hai dạng ngang chất lượng khi dùng đúng adapter của dạng đó; dùng sai dạng thì tụt. Chọn pair hay cụm cho extension là vì tiện, không vì điểm.

## Ghi chú, giới hạn, sự cố

- Một fold, chỉ mô tả, không title.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_cheo_that/bao_cao_cheo.json` | 419 | `8cce9ea74396c591` |
| `ft_run/kaggle_cheo_that/cheo/cheo_trang_thai.json` | 1.218 | `f7106cae5e38e577` |
| `ft_run/kaggle_cheo_that/cheo/eval_items.jsonl` | 2.751.215 | `3d1c3844b28457a0` |
| `ft_run/kaggle_cheo_that/cheo/sinh_cum_f0_p11_cum.jsonl` | 819.877 | `fa33a4185f4bf49c` |
| `ft_run/kaggle_cheo_that/cheo/sinh_cum_f0_p11_pair.jsonl` | 823.325 | `4cfa67789987ea22` |
| `ft_run/kaggle_cheo_that/cheo/sinh_pair_f0_p11_cum.jsonl` | 821.808 | `0079d7dee09dc023` |
| `ft_run/kaggle_cheo_that/cheo/sinh_pair_f0_p11_pair.jsonl` | 821.310 | `5625ad5875094b3b` |
| `ft_run/kaggle_cheo_that/cheo/sinh_zero_shot_cum.jsonl` | 893.701 | `64f7a65c17cae860` |
| `ft_run/kaggle_cheo_that/cheo/sinh_zero_shot_pair.jsonl` | 917.109 | `73b2d51083754b16` |
| `ft_run/kaggle_cheo_that/nlp-cheo.log` | 5.746 | `0cabb343be254e14` |
| `ft_run/cheo_ket_qua/bang.md` | 2.713 | `d6304eb3e426dd9d` |
| `ft_run/cheo_ket_qua/ket_qua.json` | 5.051 | `b347edea4843012d` |
