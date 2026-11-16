# E1. Huấn luyện lại adapter fold 0 ở tài khoản 2 (cum_f0_p11, pair_f0_p11)

[← Mục lục](INDEX.md) · [← D6](D6_tong_hop_nam_mo_hinh_tren_may_khong_gpu.md) · [E2 →](E2_thu_cheo_dang_pair_va_dang_cum.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | E1 |
| Giai đoạn | E. Thử chéo và công thức chính trên eval năm mô hình |
| Notebook Kaggle | `honghuynanh/nlp-group3` (cụm), `honghuynanh/nlp-group3-pair` (pair) |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 12/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_cheo_cum`, `data/processed/ft_run/kaggle_cheo_pair` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_cheo_cum/bao_cao_cheo_cum.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | 0.18.1 | 5.314 giây (1.48 giờ) | có | khong_co_file_test_trong_input |
| `ft_run/kaggle_cheo_pair/bao_cao_cheo_pair.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | 2.10.0+cu128 | 5.0.0 | 0.18.1 | 5.084 giây (1.41 giờ) | có | khong_co_file_test_trong_input |

## Mục đích

Cần adapter chưa học fold 0 để đo trên eval fold 0 (adapter bản cuối đã học cả fold 0). Nhật Anh duyệt huấn luyện lại đúng cấu hình B3 (rank 64, 2 epoch, ~11% title) ở tài khoản 2, hai notebook chạy song song.

## Kết quả huấn luyện lại hai adapter fold 0 (tài khoản 2)

| Adapter | Nhánh | Fold kiểm | Cấu hình | lr | Epoch | Bước | Mẫu train | Token train | % title thực | w_title | Loss kiểm ep1 | Loss kiểm ep2 | chrF chung (corpus) | chrF chung (câu) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cum_f0_p11 | cum | 0 | lr2e-4_r64 | 0.00020 | 2 | 1.837 | 37.123 | 13.061.056 | 10.74% | 0.12029 | 0.3663 | 0.3493 | 75.99 | 76.80 |
| pair_f0_p11 | pair | 0 | lr2e-4_r64 | 0.00020 | 2 | 1.843 | 37.114 | 12.395.741 | 10.86% | 0.11818 | 0.3751 | 0.3520 | 75.38 | 76.55 |

Đối chiếu với lần huấn luyện cùng cấu hình ở tài khoản 1 (bản 29, `NHAT_KY_CHAY.json`): `cum_f0_p11` chrF chung 76.14, loss kiểm ep2 0.3487; `pair_f0_p11` chrF chung 75.16, loss kiểm ep2 0.3514.

sha256 adapter tài khoản 2 (ghi trong trạng thái phiên thử chéo E2):

| Adapter | Đường dẫn trên Kaggle | sha256 |
|---|---|---|
| cum_f0_p11 | `/kaggle/input/notebooks/honghuynanh/nlp-group3/cheo_cum/cum_f0_p11/ep2` | `ff28993cc88521324164d4f5954054b2126989839817195da48604e5d3d80db8` |
| pair_f0_p11 | `/kaggle/input/notebooks/honghuynanh/nlp-group3-pair/cheo_pair/pair_f0_p11/ep2` | `56967f4a05e7eae44bed641362b6274ac09805a8c3d10488e6672becdfd3ba92` |

## Kết luận của phiên

- Hai adapter này dùng cho E2 (thử chéo), E3 (công thức chính trên eval năm mô hình) và cổng nén demo (F1).

## Ghi chú, giới hạn, sự cố

- Hai adapter là lần huấn luyện riêng nên hash khác bản tài khoản 1; số của E2, E3, F1 thuộc về adapter tài khoản 2, số của B8 (test) và C2 thuộc về adapter tài khoản 1.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_cheo_cum/bao_cao_cheo_cum.json` | 13.926 | `ca5b51f3ec09af04` |
| `ft_run/kaggle_cheo_cum/cheo_cum/cheo_cum_trang_thai.json` | 10.097 | `b02ab72e26c2b78d` |
| `ft_run/kaggle_cheo_cum/cheo_cum/cum_f0_p11/ep2/adapter_config.json` | 1.096 | `b1e7dbc1301d20b6` |
| `ft_run/kaggle_cheo_cum/cheo_cum/cum_f0_p11/sinh_ep2.jsonl` | 931.964 | `381d2a1bad613f30` |
| `ft_run/kaggle_cheo_cum/nlp-group3.log` | 9.658 | `1a67bd81ded6d50b` |
| `ft_run/kaggle_cheo_pair/bao_cao_cheo_pair.json` | 13.914 | `2dccd3457987d387` |
| `ft_run/kaggle_cheo_pair/cheo_pair/cheo_pair_trang_thai.json` | 10.082 | `134d71f2129d14da` |
| `ft_run/kaggle_cheo_pair/cheo_pair/pair_f0_p11/ep2/adapter_config.json` | 1.096 | `d90dfb9ac1512d89` |
| `ft_run/kaggle_cheo_pair/cheo_pair/pair_f0_p11/sinh_ep2.jsonl` | 920.490 | `a9e08f62476d680a` |
| `ft_run/kaggle_cheo_pair/nlp-group3-pair.log` | 9.688 | `30d5dbc8c299d1c5` |
