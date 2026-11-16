# C2. C0: adapter chính thức chấm trên eval của vòng sàng lọc

[← Mục lục](INDEX.md) · [← C1](C1_khao_sat_quota_vong_1_rank_16_max_len_960_1_200.md) · [C3 →](C3_khao_sat_quota_vong_2_ngan_sach_token_co_dinh_ma.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | C2 |
| Giai đoạn | C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc) |
| Notebook Kaggle | `trinhnhatanh/nlp-group3` |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | 50 (cụm), 51 (pair) |
| Thời gian | 09/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_c0` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_c0/c0_report_v50_cum.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | – | – | – | – | – | – |
| `ft_run/kaggle_c0/c0_report_v51_pair.json` | NVIDIA RTX PRO 6000 Blackwell Server Edition | – | – | – | – | – | – | – |

## Mục đích

Chấm `cum_f0_p11`, `pair_f0_p11` (tài khoản 1, công thức chính: rank 64, 2 epoch) trên đúng eval fold 0 của khảo sát quota để đặt hai cách cạnh nhau.

## Kết quả: adapter chính thức chấm trên đúng eval của vòng sàng lọc

| File | Bộ | Adapter | Kiểu eval | n | chrF macro | zh→vi | vi→zh | zh→vi title | zh→vi mô tả | vi→zh title | vi→zh mô tả | Giây | Đường dẫn adapter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c0_report_v50_cum.json` | cum | cum_f0_p11 | cum_block_1200 | 1.200 | 76.49 | 77.52 | 75.46 | 61.61 | 93.42 | 60.01 | 90.91 | 454.6 | `/kaggle/input/ft-qwen3-4b-tl-luu/tl/cum_f0_p11/ep2` |
| `c0_report_v50_cum.json` | pair | pair_f0_p11 | None | – | – | – | – | – | – | – | – | – | `None` |
| `c0_report_v51_pair.json` | pair | pair_f0_p11 | pair_eval_1198 | 1.198 | 74.55 | 73.45 | 75.65 | 61.77 | 85.12 | 61.21 | 90.09 | 171.3 | `/kaggle/input/ft-qwen3-4b-tl-luu/tl/pair_f0_p11/ep2` |

`c0_report.json` trong cùng thư mục là bản sao của `c0_report_v51_pair.json` (cùng sha256).

## Kết luận của phiên

- Adapter công thức chính cao hơn mọi cấu hình của vòng sàng lọc, nhưng chênh gộp cả công thức (rank, max_len, số epoch), không tách được tác dụng của tỉ trọng.

## Ghi chú, giới hạn, sự cố

- Eval pair ở đây là từng dòng (1.198 mẫu), còn adapter `pair_f0_p11` học cả khối.

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_c0/c0_report.json` | 3.220 | `cc997fe1a308cbee` |
| `ft_run/kaggle_c0/c0_report_v50_cum.json` | 4.180 | `727b425f2c333928` |
| `ft_run/kaggle_c0/c0_report_v51_pair.json` | 3.220 | `cc997fe1a308cbee` |
| `ft_run/kaggle_c0/nlp-group3.log` | 3.545 | `28c6bd6d4a166736` |
