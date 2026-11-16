# F1. Cổng nén để chạy demo trên laptop (không phải Kaggle)

[← Mục lục](INDEX.md) · [← E3](E3_cong_thuc_chinh_tren_eval_nam_mo_hinh_dong_thu_s.md) · [F2 →](F2_may_chu_dich_demo_tren_kaggle_thu_13_trang_1688.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | F1 |
| Giai đoạn | F. Demo và việc chưa chạy được |
| Notebook Kaggle | – (máy Nhật Anh, `demo/do_luong_tu.py`) |
| Tài khoản | – |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 14/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/demo_do_luong_tu` |

## Mục đích

Đo bản nén GGUF (Q4_K_M, Q5_K_M) của Qwen3-4B + LoRA trên RTX 3050 4 GB so với fp16 trên Kaggle, theo ngưỡng đặt trước.

## Cổng nén trên laptop (`demo/do_luong_tu.py`, RTX 3050 4 GB)

So bản nén GGUF + LoRA trên máy với bản fp16 trên Kaggle T4, cùng 150 sản phẩm fold 0, hướng zh→vi, adapter `pair_f0_p11` (tài khoản 2). Ngưỡng đặt trước khi đo: chrF tụt ≤ 1,0; khối lệch dòng ≤ 2%; rỗng ≈ 0; title ≤ 3 giây; khối 20 dòng ≤ 20 giây.

| Bản nén | chrF câu TB máy | chrF câu TB fp16 | Chênh | chrF corpus máy | chrF corpus fp16 | Khối lệch dòng | Rỗng | Trùng khớp fp16 | Token/giây | Giây title (trung vị) | Giây title (p90) | Giây khối (trung vị) | Giây khối (p90) | Dòng/khối (trung vị) | Hết trần sinh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q4km-lora-pair-f0 | 76.95 | 79.76 | -2.80 | 68.61 | 71.58 | 1/72 | 0.61% | 9.4% | 38.6 | 0.99 | 1.22 | 12.09 | 19.23 | 22 | 0 |
| q5km-lora-pair-f0 | 79.62 | 79.76 | -0.14 | 71.15 | 71.58 | 0/72 | 0.00% | 50.6% | 35.1 | 1.09 | 1.29 | 14.08 | 22.86 | 22 | 1 |

## Kết luận của phiên

- Q4_K_M + LoRA **không đạt** (tụt quá 1 điểm, có khối lệch dòng, có bản rỗng); Q5_K_M + LoRA **đạt**.

## Ghi chú, giới hạn, sự cố

- Đo bằng adapter học fold 1–4 (`pair_f0_p11`, E1); adapter demo `pair_cuoi` chỉ được suy ra là chịu nén tương tự.
- File GGUF ở `ft_run/demo_gguf/` (không liệt kê hash vì nặng).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/demo_do_luong_tu/ket_qua_q4km-lora-pair-f0.json` | 4.249 | `824aaba5338052e0` |
| `ft_run/demo_do_luong_tu/ket_qua_q5km-lora-pair-f0.json` | 4.232 | `759313adf95ee7aa` |
| `ft_run/demo_do_luong_tu/sinh.log` | 208 | `7e82c16b95190457` |
| `ft_run/demo_do_luong_tu/sinh_q4km-lora-pair-f0.jsonl` | 136.446 | `703c7ba2352b05e4` |
| `ft_run/demo_do_luong_tu/sinh_q5.log` | 284 | `023747a55a279fbb` |
| `ft_run/demo_do_luong_tu/sinh_q5km-lora-pair-f0.jsonl` | 138.250 | `743670b93d75aed5` |
