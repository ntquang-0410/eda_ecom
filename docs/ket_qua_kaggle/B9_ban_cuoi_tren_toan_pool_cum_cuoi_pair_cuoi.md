# B9. Bản cuối trên toàn pool (cum_cuoi, pair_cuoi)

[← Mục lục](INDEX.md) · [← B8](B8_do_test_khoa_mot_lan.md) · [C1 →](C1_khao_sat_quota_vong_1_rank_16_max_len_960_1_200.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | B9 |
| Giai đoạn | B. Fine-tune Qwen3-4B (công thức chính) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` → kho `trinhnhatanh/ft-qwen3-4b-cuoi-luu` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 43 (bản 42 là khói) |
| Thời gian | 07/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_cuoi`, `data/processed/ft_run/ban_cuoi` |

## Mục đích

Huấn luyện hai adapter cuối trên toàn pool 23.017 sản phẩm với cấu hình đã chốt; loại mẫu title của 6.386 sản phẩm bị `loc_title.py` gắn cờ, giữ mẫu mô tả.

## Các lần chạy Kaggle trong phiên

| Nhãn | Bản | Bắt đầu (ICT) | Kết thúc (ICT) | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft |
|---|---|---|---|---|---|---|---|---|
| Bản cuối trên toàn pool | 43 | 07/11/2026 10:55:02 | 07/11/2026 14:07:01 | 3.20 | 5.50 | có | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.10.0+cu128 / 5.0.0 / 0.18.1 |

## Cấu hình từng việc huấn luyện

Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.

| Việc | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title thực | % title nếu w=1 | w_title | Token/giây |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_cuoi | cum | – | 0.00020 | 64 | 132.120.576 | 2 | 1.990 | 39.734 | 15.652.689 | 10.74% | 40.77% | 0.17247 | ep1: 5525 / ep2: 5525 |
| pair_cuoi | pair | – | 0.00020 | 64 | 132.120.576 | 2 | 2.008 | 39.724 | 14.829.898 | 10.86% | 41.42% | 0.16915 | ep1: 5564 / ep2: 5567 |

## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)

`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.

| Việc | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_cuoi | 0.2504 | 0.1599 | – | – | – | – | – | – | – / – | – / – | – |
| pair_cuoi | 0.2515 | 0.1655 | – | – | – | – | – | – | – / – | – / – | – |

Số bước bị cắt gradient (clip 1,0) / số bước, theo loại lô (toàn title / trộn / toàn mô tả):

| Việc | Title / trộn / mô tả |
|---|---|
| cum_cuoi | 1/207 / 2/923 / 7/860 |
| pair_cuoi | 3/237 / 8/931 / 6/840 |

## Manifest adapter bản cuối (đã kiểm hash sau khi tải)

| File | Byte | sha256 đầy đủ |
|---|---|---|
| `cum_cuoi/ep2/README.md` | 5.238 | `50795648aeaead7dc05e7363a785bcf3f2dfa48fdd67b6c35238635253bcf30a` |
| `cum_cuoi/ep2/adapter_config.json` | 1.074 | `3c4ac130bf7871e0b487c8835945d415627c56097ce79127824ec1934293926d` |
| `cum_cuoi/ep2/adapter_model.safetensors` | 528.550.256 | `fc8481ad788c3760b131c207cf2485ec93a968c6e593889f7c29a81c698b5131` |
| `cuoi_trang_thai.json` | 18.234 | `7df505c573e775e6fd95ac50098887a7de3078858b3deda223437556491b2c7f` |
| `pair_cuoi/ep2/README.md` | 5.238 | `50795648aeaead7dc05e7363a785bcf3f2dfa48fdd67b6c35238635253bcf30a` |
| `pair_cuoi/ep2/adapter_config.json` | 1.074 | `3c4ac130bf7871e0b487c8835945d415627c56097ce79127824ec1934293926d` |
| `pair_cuoi/ep2/adapter_model.safetensors` | 528.550.256 | `7fd7f157001b1af5de96906e0ae599ad0d40e170d28d892fe17186571a92319b` |

## Kết luận của phiên

- Hai adapter cuối đã lưu và kiểm hash. Demo dùng `pair_cuoi`.

## Ghi chú, giới hạn, sự cố

- Không còn fold giữ lại nên không có điểm chất lượng; loss trong bảng là loss trên mẫu đã học (chỉ để bắt lỗi hỏng).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_cuoi/kho_luu_manifest/MANIFEST_CUOI.json` | 962 | `1255303f77572c80` |
| `ft_run/kaggle_cuoi/phien_1_20261107_105449/bao_cao_cuoi.json` | 150.253 | `935608d6631f6ff8` |
| `ft_run/kaggle_cuoi/phien_1_20261107_105449/cuoi/cuoi_trang_thai.json` | 18.234 | `7df505c573e775e6` |
| `ft_run/kaggle_cuoi/phien_1_20261107_105449/nlp-group3.log` | 15.812 | `f10bf3e81a4c4f6b` |
| `ft_run/kaggle_cuoi/tu_dong.log` | 1.152 | `223943f944d310a3` |
| `ft_run/ban_cuoi/cuoi/cum_cuoi/ep2/adapter_config.json` | 1.074 | `3c4ac130bf7871e0` |
| `ft_run/ban_cuoi/cuoi/cum_cuoi/ep2/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/ban_cuoi/cuoi/cuoi_trang_thai.json` | 18.234 | `7df505c573e775e6` |
| `ft_run/ban_cuoi/cuoi/pair_cuoi/ep2/adapter_config.json` | 1.074 | `3c4ac130bf7871e0` |
| `ft_run/ban_cuoi/cuoi/pair_cuoi/ep2/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/ban_cuoi/ft-qwen3-4b-cuoi-luu.log` | 3.087 | `a333d65d4feea67e` |
| `ft_run/ban_cuoi/MANIFEST_CUOI.json` | 962 | `1255303f77572c80` |

Adapter LoRA (trọng số) có trong output đã tải:

| File | Byte | sha256 |
|---|---|---|
| `ft_run/ban_cuoi/cuoi/cum_cuoi/ep2/adapter_model.safetensors` | 528.550.256 | `fc8481ad788c3760` |
| `ft_run/ban_cuoi/cuoi/pair_cuoi/ep2/adapter_model.safetensors` | 528.550.256 | `7fd7f157001b1af5` |
