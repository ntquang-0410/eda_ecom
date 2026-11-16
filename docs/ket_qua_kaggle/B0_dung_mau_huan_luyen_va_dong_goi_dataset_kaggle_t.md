# B0. Dựng mẫu huấn luyện và đóng gói Dataset Kaggle (trên máy)

[← Mục lục](INDEX.md) · [← A7](A7_chia_tap_theo_san_pham_kiem_ro_ri_gan_co_title_t.md) · [B1 →](B1_b6_do_sieu_tham_so_qwen3_4b_lr_rank.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B0 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | – (`scripts/ft_train/dung_mau.py`, `dong_goi_kaggle.py`, `dong_goi_quota.py`, `xuat_song_ngu.py`) |
| Tài khoản | – (dựng trên máy; Dataset ở tài khoản 1 và 2) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 04–12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/mau`, `data/processed/ft_run/goi_pool`, `data/processed/ft_run/goi_pool_honghuynanh`, `data/processed/ft_run/goi_quota`, `data/processed/ft_run/goi_song_ngu`, `data/processed/ft_run/goi_do_khoa` |

## Mục đích

Biến mỗi sản phẩm của pool thành mẫu hội thoại (lời dẫn + bản dịch tham chiếu) cho LLM, đếm token, rồi đóng gói thành Dataset riêng tư trên Kaggle cho các phiên huấn luyện; dựng thêm dữ liệu từng dòng cho hai mô hình cổ điển.

## 1. Lời dẫn (chép từ `scripts/ft_train/dung_mau.py`)

Khuôn chat Qwen3, tắt suy luận (`enable_thinking=False`); loss chỉ tính trên phần bản dịch.

```text
Title:   Translate this product title from {SRC} to {TGT}. Output only the translation.\n\n{NGUON}
Mô tả, pair: Translate this product's attribute list from {SRC} to {TGT}. Each line is one attribute written as "name: value". Keep the same number and order of lines, and output only the translation.
             \nProduct title (context only, do not translate): {TITLE}\n\n{NGUON}
Mô tả, cụm:  ... The attributes are separated by "; " and each is written as "name: value". Keep the same number and order of attributes, separated by "; ", and output only the translation. (+ cùng phần title ngữ cảnh)
```

## 2. Mẫu dựng được (`ft_run/mau/bao_cao_dung_mau.json`)

Mỗi sản phẩm cho 2 mẫu: title và mô tả, theo đúng hướng gốc (1688 zh→vi, Tiki vi→zh). Mô tả quá `max_len` thì chia thành nhiều phần theo thuộc tính; thuộc tính đơn lẻ vẫn quá dài thì loại. Tập test chỉ được đếm, không dựng vào gói huấn luyện.

| Bộ | max_len | Mẫu pool | Token pool | Mẫu zh→vi / vi→zh | Mẫu title / mô tả | Token title / mô tả | Sản phẩm bị chia phần | Mẫu bị loại | Lý do loại | Mẫu test (chỉ đếm) |
|---|---|---|---|---|---|---|---|---|---|---|
| cum | 2.048 | 46.120 | 16.215.152 | 26.791 / 19.329 | 23.017 / 23.103 | 1.973.652 / 14.241.500 | 127 | 58 | `{"khong_can_so_thuoc_tinh": 2, "thuoc_tinh_don_le_vuot_max_len": 56}` | 5.116 |
| pair | 2.048 | 46.110 | 15.392.361 | 26.781 / 19.329 | 23.017 / 23.093 | 1.973.652 / 13.418.709 | 113 | 55 | `{"thuoc_tinh_don_le_vuot_max_len": 55}` | 5.116 |

Số token ở đây đếm cả lời dẫn. Tỉ trọng title theo **token đích** (10,74% cụm, 10,86% pair) được tính từ chính các mẫu này lúc huấn luyện, xem cột `% title thực` ở B3–B7.

## 3. Dataset riêng tư đã đóng gói lên Kaggle

| Dataset | Tài khoản | Nội dung (số mẫu / số dòng) | Manifest |
|---|---|---|---|
| `ft-qwen3-4b-pool` | 1 | `{"cum": 46120, "pair": 46110}` | `ft_run/goi_pool/MANIFEST.json` |
| `ft-qwen3-4b-pool` | 2 | `{"cum": 46120, "pair": 46110}` | `ft_run/goi_pool_honghuynanh/MANIFEST.json` |
| `ft-qwen3-4b-quota` | 1, 2 | 17 file (chỉ số mẫu, bảng tham chiếu, mẫu eval) | `ft_run/goi_quota/MANIFEST.json` |
| `ft-song-ngu` | 2 | `{"zh-vi/train": 269895, "vi-zh/train": 61401, "zh-vi/khop4m_s42": 37528, "vi-zh/khop4m_s42": 9006, "zh-vi/khop4m_s123": 37991, "vi-zh/khop4m_s123": 8366, "zh-vi/eval": 598, "vi-zh/eval": 600, "zh-vi/dev": 1000, "vi-zh/dev": 1000}` | `ft_run/goi_song_ngu/MANIFEST.json` |
| `ft-qwen3-4b-do-khoa` | 1 | `{"cum": 5116, "pair": 5116}` | `ft_run/goi_do_khoa/MANIFEST.json` |

- `ft-qwen3-4b-pool`: mẫu pool hai bộ (không có test), dùng cho mọi phiên B, C, D, E.
- `ft-qwen3-4b-quota`: bảng tham chiếu và mẫu eval cố định cho khảo sát quota (C, D). Bảng tỉ trọng natural theo 4 nhóm ở `goi_quota/meta.json`.
- `ft-song-ngu`: dữ liệu từng dòng của bản pair cho OpenNMT và Moses (D3–D5), gồm biến thể học đủ và khớp 4M token.
- `ft-qwen3-4b-do-khoa`: tập test khóa, chỉ gắn vào phiên đo test B8.

## Kết luận của phiên

- Mọi phiên B–E đọc mẫu từ các Dataset này; hash của gói pool khớp giữa tài khoản 1 và 2.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/mau/bao_cao_dung_mau.json` | 21.756 | `0170007a474bbca2` |
| `ft_run/mau/bao_cao_dung_mau.md` | 48.607 | `bbfcf4f570cc7fa2` |
| `ft_run/goi_pool/MANIFEST.json` | 657 | `aca8fa4445bcb556` |
| `ft_run/goi_pool_honghuynanh/MANIFEST.json` | 657 | `aca8fa4445bcb556` |
| `ft_run/goi_pool_honghuynanh/MANIFEST.json` | 657 | `aca8fa4445bcb556` |
| `ft_run/goi_quota/MANIFEST.json` | 640 | `decb82a491c4edfa` |
| `ft_run/goi_quota/meta.json` | 3.107 | `0964fc9dcc4ae16e` |
| `ft_run/goi_song_ngu/MANIFEST.json` | 1.294 | `1467bc4a3b51672a` |
| `ft_run/goi_do_khoa/MANIFEST.json` | 567 | `1a2988c490fedbdd` |
