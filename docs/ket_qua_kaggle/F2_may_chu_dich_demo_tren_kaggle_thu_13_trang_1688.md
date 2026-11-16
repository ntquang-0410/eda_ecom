# F2. Máy chủ dịch demo trên Kaggle + thử 13 trang 1688 thật

[← Mục lục](INDEX.md) · [← F1](F1_cong_nen_de_chay_demo_tren_laptop_khong_phai_kag.md) · [F3 →](F3_qwen3_8b_theo_cong_thuc_chinh_khong_chay_duoc_bi.md)

Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.

## Thông tin phiên

| Mục | Giá trị |
|---|---|
| Mã | F2 |
| Giai đoạn | F. Demo và việc chưa chạy được |
| Notebook Kaggle | `trinhnhatanh/demo-may-chu` (2 T4, có Internet, đường hầm Cloudflare) |
| Tài khoản | 1 (`trinhnhatanh`) |
| Bản Kaggle | không ghi trong file kết quả |
| Thời gian | 15/11/2026 |
| Thư mục output trên máy Nhật Anh | `data/processed/ft_run/demo_bo_dem_kaggle` |

## Mục đích

Chạy máy chủ dịch với `pair_cuoi` fp16 trên Kaggle, mở đường hầm cho extension, dịch 13 trang 1688 đã lưu và giữ bộ đệm để demo không cần GPU.

## Output tải về

`may_chu/bo_dem.sqlite`: số dòng theo bảng `{"dem": 37}`. Đây là bộ đệm bản dịch của 13 trang thử (13 title + 22 khối thuộc tính + 2 mục tự kiểm), chép sang `demo/trang_dich_san/` để demo không cần GPU.

Thư mục còn bản chép mã máy chủ (`may_chu/*.py`) đúng như lúc chạy.

## Kết luận của phiên

- 13/13 trang đạt kiểm cấu trúc; đọc tay 8 trang mới khoảng 95% dòng đúng. Lỗi đọc tay và 5 trang chọn demo: `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` mục 9.

## Ghi chú, giới hạn, sự cố

- Không có tham chiếu cho trang thật: đây là quan sát, không phải số đo.
- Đường hầm có 2 lần đứt ngẫu nhiên; extension đã thêm tự thử lại khi lỗi mạng (chưa kiểm trong Chrome thật).

## File gốc trên máy (sha256, 16 ký tự đầu)

| File | Byte | sha256 |
|---|---|---|
| `ft_run/demo_bo_dem_kaggle/demo-may-chu.log` | 35.545 | `c50803892feffe8c` |
