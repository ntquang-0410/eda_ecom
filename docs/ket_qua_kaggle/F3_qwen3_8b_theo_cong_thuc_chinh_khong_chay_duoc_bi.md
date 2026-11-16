# F3. Qwen3-8B theo công thức chính: không chạy được (bị cấp T4)

[← Mục lục](INDEX.md) · [← F2](F2_may_chu_dich_demo_tren_kaggle_thu_13_trang_1688.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | F3 |
| Giai đoạn | F. Demo và việc chưa chạy được |
| Notebook Kaggle | `honghuynanh/nlp-8b-cuoi-cum` |
| Tài khoản | 2 (`honghuynanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 13/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/kaggle_cuoi8b_khoi`, `data/processed/ft_run/kaggle_cuoi8b_khoi2` |

Môi trường và thời lượng, đọc từ file báo cáo của phiên:

| File báo cáo | GPU | VRAM (GB) | torch | transformers | peft | Thời lượng | Đạt / xong | Khóa test |
|---|---|---|---|---|---|---|---|---|
| `ft_run/kaggle_cuoi8b_khoi/bao_cao_thu_cuoi8b.json` | Tesla T4 | – | 2.10.0+cu128 | 5.0.0 | 0.18.1 | 282 giây (0.08 giờ) | không | khong_co_file_test_trong_input |
| `ft_run/kaggle_cuoi8b_khoi2/bao_cao_thu_cuoi8b.json` | Tesla T4 | 14.6 | 2.10.0+cu128 | 5.0.0 | 0.18.1 | 30 giây (0.01 giờ) | không | khong_co_file_test_trong_input |

## Mục đích

Chạy khói Qwen3-8B bản cụm theo công thức chính (rank 64, 2 epoch, `w_title`) trước khi chạy thật, vì ở vòng sàng lọc 8B hơn 4B ở bản cụm.

## Hai lần chạy thử, cả hai bị cấp GPU nhỏ

| Báo cáo | GPU được cấp | VRAM (GB) | Đạt | Thời lượng | Dòng lỗi cuối |
|---|---|---|---|---|---|
| `ft_run/kaggle_cuoi8b_khoi/bao_cao_thu_cuoi8b.json` | Tesla T4 | – | không | 282 giây (0.08 giờ) | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 84.81 MiB is free. Including non-PyTorch memory, this process has 14.48 GiB memory in use. Of the allocated memory 14.` |
| `ft_run/kaggle_cuoi8b_khoi2/bao_cao_thu_cuoi8b.json` | Tesla T4 | 14.6 | không | 30 giây (0.01 giờ) | `RuntimeError: GPU_KHONG_DU_BO_NHO:Tesla T4:14.6GB<80GB` |

## Kết luận của phiên

- Hai lần khói liền được cấp Tesla T4 14,6 GB dù metadata giống notebook từng chạy RTX Pro 6000. Lần 1 hết bộ nhớ khi nạp 8B; sau đó thêm chốt chặn `vram_toi_thieu_gb` nên lần 2 dừng sau 30 giây. **8B theo công thức chính chưa có; phiên thật không được đẩy.**

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/kaggle_cuoi8b_khoi/bao_cao_thu_cuoi8b.json` | 130.740 | `6185efb3594f5565` |
| `ft_run/kaggle_cuoi8b_khoi/nlp-8b-cuoi-cum.log` | 14.645 | `500bf502e77fbf62` |
| `ft_run/kaggle_cuoi8b_khoi/thu_cuoi8b/thu_cuoi8b_trang_thai.json` | 378 | `12b1977545f430bd` |
| `ft_run/kaggle_cuoi8b_khoi2/bao_cao_thu_cuoi8b.json` | 127.251 | `05e8d80d21af2fe4` |
| `ft_run/kaggle_cuoi8b_khoi2/nlp-8b-cuoi-cum.log` | 6.857 | `225bcadd90123ee3` |
