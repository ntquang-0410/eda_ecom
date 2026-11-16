# B1. B6: dò siêu tham số Qwen3-4B (lr × rank)

[← Mục lục](INDEX.md) · [← B0](B0_dung_mau_huan_luyen_va_dong_goi_dataset_kaggle_t.md) · [B2 →](B2_b7_b8_phan_dau_lora_attention_pair_fold_0_cum_fo.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B1 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 20, 21, 22 (nối nhau) |
| Thời gian | 05/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_b6` |

## Mục đích

Chọn lr và rank LoRA cho bản cụm trên fold 0: vòng 1 chạy 1 epoch cho 7 cấu hình (gồm LoRA chỉ attention), vòng 2 chạy tiếp tới 2 epoch cho 2 cấu hình tốt nhất, sinh 1.000 mẫu và chấm.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| B6 phiên 1 | 20 | 05/11/2026 09:58:22 | 05/11/2026 12:28:25 | 2.50 | 2.50 | không | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |
| B6 phiên 2 | 21 | 05/11/2026 12:33:15 | 05/11/2026 15:03:17 | 2.50 | 2.50 | không | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |
| B6 phiên 3 | 22 | 05/11/2026 15:07:27 | 05/11/2026 16:10:25 | 1.05 | 2.50 | có | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr5e-5_r16 | cum | 0 | 0.00005 | 16 | 33.030.144 | 1 | 929 | 37.123 | 13.061.056 | – | – | – | ep1: 5836 |
| lr1e-4_r16 | cum | 0 | 0.00010 | 16 | 33.030.144 | 1 | 929 | 37.123 | 13.061.056 | – | – | – | ep1: 5838 |
| lr2e-4_r16 | cum | 0 | 0.00020 | 16 | 33.030.144 | 1 | 929 | 37.123 | 13.061.056 | – | – | – | ep1: 5839 |
| lr5e-5_r64 | cum | 0 | 0.00005 | 64 | 132.120.576 | 1 | 929 | 37.123 | 13.061.056 | – | – | – | ep1: 5546 |
| lr1e-4_r64 | cum | 0 | 0.00010 | 64 | 132.120.576 | 2 | 1.837 | 37.123 | 13.061.056 | – | – | – | ep1: 5588 / ep2: 5564 |
| lr2e-4_r64 | cum | 0 | 0.00020 | 64 | 132.120.576 | 2 | 1.837 | 37.123 | 13.061.056 | – | – | – | ep1: 5587 / ep2: 5574 |
| lr1e-4_r16_attn | cum | 0 | 0.00010 | 16 | 11.796.480 | 1 | 929 | 37.123 | 13.061.056 | – | – | – | ep1: 7781 |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lr5e-5_r16 | 0.4090 | – | – | – | – | – | – | – | – / – | – / – | – |
| lr1e-4_r16 | 0.3900 | – | – | – | – | – | – | – | – / – | – / – | – |
| lr2e-4_r16 | 0.3770 | – | – | – | – | – | – | – | – / – | – / – | – |
| lr5e-5_r64 | 0.3808 | – | – | – | – | – | – | – | – / – | – / – | – |
| lr1e-4_r64 | 0.3680 | 0.3587 | 75.51 | 76.29 | 60.21 | 90.81 | – | – | – / – | – / – | 3 |
| lr2e-4_r64 | 0.3703 | 0.3512 | 75.76 | 76.62 | 60.22 | 91.30 | 75.90 | 75.62 | 895 / 77.17 | 456 / 77.79 | 1 |
| lr1e-4_r16_attn | 0.4435 | – | – | – | – | – | – | – | – / – | – / – | – |

## So sánh theo cặp (bootstrap, khoảng 95%)

Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.

| Nhóm | Cặp / chỉ số | Chênh [95%] |
|---|---|---|
| B6 chrF chung (theo câu), epoch 2 | lr2e-4_r64 - lr1e-4_r64 | +0.33 [-0.12; +0.81] |

## Ghi chú tự động từ nhật ký

- lr2e-4_r64 (mốc cum_f0): zh→vi, vi→zh và dòng mới lấy từ lần chấm lại bản dịch B6 ở đợt thử tỉ lệ (`bao_cao_ti_le.json`, mốc); chrF chung chấm lại trùng điểm B6 75,7613 (độ lệch 0).

## Kết luận của phiên

- Chọn **lr 2e-4, rank 64, 2 epoch**: loss kiểm thấp nhất; chênh chrF với lr 1e-4 là hòa (bảng so sánh).
- Ba phiên nối nhau vì mỗi phiên cài trần 2,5 giờ; phiên 1, 2 dừng có trật tự, phiên 3 xong.

## Ghi chú, giới hạn, sự cố

- Mã `kaggle_train.py` dùng cho B6 là bản trước lần sửa 05/11 23:06, không còn bản lưu riêng.
- Báo cáo phân tích thêm: `kaggle_b6/phan_tich_chi_so/BAO_CAO.md`. Lần chạy thử nối phiên `kaggle_b6_noi` là khói, không thuộc B6 (xem INDEX).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/b6_trang_thai.json` | 8.170 | `b243d4d038caf70f` |
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr1e-4_r64/ep2/adapter_config.json` | 1.074 | `0c8e20a007716304` |
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr1e-4_r64/ep2/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr1e-4_r64/sinh_ep2.jsonl` | 946.476 | `85302a373c9319a5` |
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr2e-4_r64/ep2/adapter_config.json` | 1.074 | `0c8e20a007716304` |
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr2e-4_r64/ep2/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr2e-4_r64/sinh_ep2.jsonl` | 932.776 | `935df8a51848de09` |
| `ft_run/kaggle_b6/ket_qua_cuoi/bao_cao_b6.json` | 10.872 | `c3157cfe47db05fe` |
| `ft_run/kaggle_b6/ket_qua_cuoi/nlp-group3.log` | 8.060 | `61a736399fce07b0` |
| `ft_run/kaggle_b6/kho_luu_manifest/b6_luu/MANIFEST.json` | 3.020 | `69d7722b445f14a0` |
| `ft_run/kaggle_b6/kho_luu_manifest/ft-qwen3-4b-b6-luu.log` | 2.833 | `76f9829c0a97186a` |
| `ft_run/kaggle_b6/phan_tich_chi_so/BAO_CAO.md` | 12.465 | `10efdf65f94d9a59` |
| `ft_run/kaggle_b6/phan_tich_chi_so/so_lieu.json` | 9.119 | `73cacec1f610ed24` |
| `ft_run/kaggle_b6/phien_1_20261105_095809/b6/b6_trang_thai.json` | 2.117 | `7b46f73f201eb964` |
| `ft_run/kaggle_b6/phien_1_20261105_095809/bao_cao_b6.json` | 3.481 | `285ce79a430465b8` |
| `ft_run/kaggle_b6/phien_1_20261105_095809/nlp-group3.log` | 9.665 | `aa26375bd640d818` |
| `ft_run/kaggle_b6/phien_2_20261105_123231/b6/b6_trang_thai.json` | 3.185 | `560593affc74516a` |
| `ft_run/kaggle_b6/phien_2_20261105_123231/bao_cao_b6.json` | 4.843 | `605015eeb00e5a3d` |
| `ft_run/kaggle_b6/phien_2_20261105_123231/nlp-group3.log` | 10.493 | `b80d0b354e5667cb` |
| `ft_run/kaggle_b6/phien_3_20261105_150627/b6/b6_trang_thai.json` | 8.170 | `b243d4d038caf70f` |
| `ft_run/kaggle_b6/phien_3_20261105_150627/bao_cao_b6.json` | 10.872 | `c3157cfe47db05fe` |
| `ft_run/kaggle_b6/phien_3_20261105_150627/nlp-group3.log` | 8.060 | `61a736399fce07b0` |
| `ft_run/kaggle_b6/tu_dong.log` | 2.242 | `26f8f0f71e73d819` |

Adapter LoRA (trọng số) có trong output đã tải:

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr1e-4_r64/ep2/adapter_model.safetensors` | 528.550.256 | `195d0bf3bf18adbb` |
| `ft_run/kaggle_b6/ket_qua_cuoi/b6/lr2e-4_r64/ep2/adapter_model.safetensors` | 528.550.256 | `94a3cf3d7d03a5aa` |
