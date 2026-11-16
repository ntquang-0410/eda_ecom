# C1. Khảo sát quota vòng 1 (rank 16, max_len 960, 1.200 bước)

[← Mục lục](INDEX.md) · [← B9](B9_ban_cuoi_tren_toan_pool_cum_cuoi_pair_cuoi.md) · [C2 →](C2_c0_adapter_chinh_thuc_cham_tren_eval_cua_vong_sa.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | C1 |
| Giai đoạn | C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 47 (pair), 48 (cụm) |
| Thời gian | 09/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_quota_screen` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_quota_screen/version47/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |
| `ft_run/kaggle_quota_screen/version48/quota_report.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | – | – | – | – |

## Mục đích

Thử 5 bảng tỉ trọng (title/mô tả × 1688/Tiki) cho mỗi bộ bằng cách điều khiển tần suất lấy mẫu, cùng số bước.

## Kết quả từng việc

chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.

**`ft_run/kaggle_quota_screen/version47/quota_report.json`**

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair | natural | 42 | 16 | 960 | 1.200 | 3.410.279 | 130 | 11.30% | 11.82% | – | – | – | 69.35 | 41.97 | 68.44 | 70.26 | 0.0062 | 6681 | – |
| pair | balanced | 42 | 16 | 960 | 1.200 | 3.534.549 | 38 | 50.00% | 49.86% | – | – | – | 70.25 | 41.97 | 68.80 | 71.70 | 0.2236 | 6682 | – |
| pair | title_heavy | 42 | 16 | 960 | 1.200 | 3.438.967 | 30 | 65.00% | 65.77% | – | – | – | 71.51 | 41.97 | 69.85 | 73.18 | 0.1189 | 6689 | – |
| pair | attr_heavy | 42 | 16 | 960 | 1.200 | 3.499.904 | 114 | 15.00% | 15.82% | – | – | – | 67.65 | 41.97 | 68.98 | 66.32 | 0.3052 | 6676 | – |
| pair | tiki_heavy | 42 | 16 | 960 | 1.200 | 3.633.264 | 39 | 40.00% | 39.87% | – | – | – | 70.21 | 41.97 | 67.81 | 72.62 | 0.2658 | 6655 | – |

- Việc `cum/natural` **LỖI**: `ValueError: id_tham_chieu_trung`
- Việc `cum/balanced` **LỖI**: `ValueError: id_tham_chieu_trung`
- Việc `cum/title_heavy` **LỖI**: `ValueError: id_tham_chieu_trung`
- Việc `cum/desc_heavy` **LỖI**: `ValueError: id_tham_chieu_trung`
- Việc `cum/tiki_heavy` **LỖI**: `ValueError: id_tham_chieu_trung`

**`ft_run/kaggle_quota_screen/version48/quota_report.json`**

| Bộ | Cấu hình | Seed | rank | max_len | Bước | Token đích | Mẫu bị bỏ vì dài | % title đặt | % title thực | Title zh→vi | Mô tả zh→vi | Chính (0,1/0,9) | chrF macro | Zero-shot macro | zh→vi | vi→zh | Loss cuối | Token/giây | VRAM đỉnh (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum | natural | 42 | 16 | 960 | 1.200 | 12.408.206 | 1.468 | 10.72% | 16.29% | – | – | – | 74.06 | 43.95 | 74.44 | 73.67 | 0.0291 | 6168 | – |
| cum | balanced | 42 | 16 | 960 | 1.200 | 5.503.317 | 554 | 50.00% | 57.69% | – | – | – | 72.55 | 43.95 | 71.25 | 73.86 | 0.2979 | 6471 | – |
| cum | title_heavy | 42 | 16 | 960 | 1.200 | 4.006.218 | 222 | 80.00% | 83.81% | – | – | – | 70.07 | 43.95 | 67.55 | 72.58 | 0.4442 | 6574 | – |
| cum | desc_heavy | 42 | 16 | 960 | 1.200 | 14.172.871 | 1.178 | 5.00% | 7.01% | – | – | – | 73.33 | 43.95 | 72.96 | 73.69 | 0.0327 | 6258 | – |
| cum | tiki_heavy | 42 | 16 | 960 | 1.200 | 6.298.461 | 363 | 45.00% | 48.68% | – | – | – | 71.49 | 43.95 | 68.96 | 74.01 | 0.3474 | 6476 | – |

- Đây là **vòng sàng lọc** (rank 16, một lượt, khoảng 4M token tính toán), không phải công thức cuối (rank 64, 2 epoch). Điểm chỉ dùng để so các cấu hình hoặc mô hình với nhau.
- Bộ `pair` trong khảo sát quota là **từng dòng** (title hoặc một dòng thuộc tính mỗi mẫu, 331.296 mẫu), khác adapter `pair_*` của công thức cuối (cả khối thuộc tính một lượt).

## Kết luận của phiên

- **Không dùng để chốt**, đã thay bằng C3, C4: `max_len` 960 làm bỏ nhiều mẫu dài (cột `Mẫu bị bỏ vì dài`), nên % title thực lệch xa % đặt, và mỗi cấu hình thấy số token khác nhau.

## Ghi chú, giới hạn, sự cố

- Bản 47 lỗi ở phần cụm (dòng LỖI ở bảng); phần cụm chạy lại ở bản 48.
- Bảng tổng hợp cũ của vòng này: `docs/BANG_KET_QUA_CUOI.md` (đã bị thay).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_screen/nlp-group3.log` | 32.580 | `e51ae56da2e39eeb` |
| `ft_run/kaggle_quota_screen/quota_report.json` | 26.959 | `3175932e09aa6803` |
| `ft_run/kaggle_quota_screen/version47/nlp-group3.log` | 32.580 | `e51ae56da2e39eeb` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_attr_heavy/adapter_config.json` | 1.073 | `05d879095363a289` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_attr_heavy/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_balanced/adapter_config.json` | 1.073 | `05d879095363a289` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_balanced/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_natural/adapter_config.json` | 1.073 | `05d879095363a289` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_natural/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_tiki_heavy/adapter_config.json` | 1.073 | `05d879095363a289` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_tiki_heavy/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_title_heavy/adapter_config.json` | 1.073 | `05d879095363a289` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_title_heavy/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_quota_screen/version47/quota_report.json` | 26.959 | `3175932e09aa6803` |
| `ft_run/kaggle_quota_screen/version48/nlp-group3.log` | 17.553 | `3ab55d4e44c0ea36` |
| `ft_run/kaggle_quota_screen/version48/quota/cum_natural/adapter_config.json` | 1.073 | `a0b9d4e9a16a0894` |
| `ft_run/kaggle_quota_screen/version48/quota/cum_natural/README.md` | 5.238 | `50795648aeaead7d` |
| `ft_run/kaggle_quota_screen/version48/quota_report.json` | 21.714 | `3adef3453f7b2d2e` |

Adapter LoRA (trọng số) có trong output đã tải:

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_quota_screen/version47/quota/pair_attr_heavy/adapter_model.safetensors` | 132.187.888 | `f57922317010a7a3` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_balanced/adapter_model.safetensors` | 132.187.888 | `f36a8a6adc1ada35` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_natural/adapter_model.safetensors` | 132.187.888 | `b492b3a00926f813` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_tiki_heavy/adapter_model.safetensors` | 132.187.888 | `d1f78c4d606e37ed` |
| `ft_run/kaggle_quota_screen/version47/quota/pair_title_heavy/adapter_model.safetensors` | 132.187.888 | `398f97747c8d066c` |
| `ft_run/kaggle_quota_screen/version48/quota/cum_natural/adapter_model.safetensors` | 132.187.888 | `53d9373a8c5c8738` |
