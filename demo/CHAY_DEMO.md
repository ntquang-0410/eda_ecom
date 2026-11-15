# Chạy demo extension dịch 1688 (zh → vi)

Mô hình: Qwen3-4B + adapter `pair_cuoi`. Extension chỉ gửi chữ và nhận chữ; máy chủ dịch (`demo/may_chu/`) giữ lời dẫn và gọi mô hình.

## Cách A (chính): máy chủ trên Kaggle tài khoản 1 (trinhnhatanh), 2 × T4

**Bật TRƯỚC giờ demo ít nhất 40 phút** (nạp mô hình ~27 phút, tùy đĩa Kaggle).

1. Đẩy phiên: `.venv\Scripts\python.exe -X utf8 demo/kaggle/dong_goi_may_chu.py --gio 2`
   (`--gio` = số giờ máy chủ sống sau khi đường hầm mở; Kaggle không có lệnh hủy, hết giờ tự tắt).
   Phải đang dùng token tài khoản 1 (`kaggle config view` ra `trinhnhatanh`).
2. Theo dõi log (ghi ra file, không in khóa ra màn hình):
   `kaggle kernels logs -f trinhnhatanh/demo-may-chu > data/processed/ft_run/demo_may_chu_kaggle3.log`
3. Khi log có dòng `KET_NOI url=https://....trycloudflare.com key=...` và `TU_KIEM {...}` là máy chủ sẵn sàng.
4. Nạp extension (Chrome → `chrome://extensions` → bật *Chế độ nhà phát triển* → *Tải tiện ích đã giải nén* → chọn thư mục `demo/extension`).
5. Bấm biểu tượng extension: dán **URL** và **khóa** từ dòng `KET_NOI` → *Lưu* → *Kiểm tra máy chủ* (đèn xanh = sẵn sàng).
   URL và khóa **đổi mỗi phiên**; khóa là bí mật, không dán vào chat/tài liệu.
6. Mở trang chi tiết 1688 (`detail.1688.com/offer/...`), bấm nút **Dịch sang tiếng Việt** ở góc phải dưới. Bấm lại để xem bản gốc.

Tốc độ đã đo qua đường hầm: tên sản phẩm 2,5–3,5 giây; 20 dòng thuộc tính ~12 giây (trang 35 dòng ≈ 2 lượt ≈ 25 giây). Trang đã dịch rồi lấy từ bộ đệm SQLite trên máy chủ (< 1 giây).

## Cách B (dự phòng): máy chủ trên laptop (RTX 3050 4 GB)

Q5_K_M + LoRA qua llama.cpp (đã đạt ngưỡng ở Giai đoạn 0; VRAM 3,69/4 GB nên đóng các ứng dụng dùng GPU).

1. `demo/llama_bin/vulkan/llama-server.exe -m data/processed/ft_run/demo_gguf/Qwen3-4B-Q5_K_M.gguf --lora data/processed/ft_run/demo_gguf/pair_cuoi.gguf --port 8770 -ngl 99 -c 4096`
   (xem lệnh/cờ đã dùng trong `demo/do_luong_tu.py`).
2. `demo\.venv\Scripts\python.exe demo/may_chu/app.py --llama http://127.0.0.1:8770` (nghe `127.0.0.1:8765`, không cần khóa).
3. Popup extension: địa chỉ `http://127.0.0.1:8765`, để trống khóa.

## Cách C (dự phòng cuối, KHÔNG cần GPU, không cần mạng): bộ đệm dịch sẵn

Bộ đệm `demo/trang_dich_san/bo_dem.sqlite` (37 mục, lấy từ phiên Kaggle ngày 15/11/2026, bản sao ở `data/processed/ft_run/demo_bo_dem_kaggle/may_chu/`) chứa bản dịch THẬT của mô hình (Qwen3-4B fp16 + `pair_cuoi`) cho 13 trang 1688 đã thử. Máy chủ chế độ chỉ-bộ-đệm không nạp mô hình, trả bản dịch đã lưu ngay (< 1 giây/trang) và báo lỗi 503 rõ ràng với trang chưa có:

```
demo\.venv\Scripts\python.exe -X utf8 demo/may_chu/app.py --chi-bo-dem kaggle-fp16-pair_cuoi --dem demo/trang_dich_san/bo_dem.sqlite --cong 8765
```
Popup extension: địa chỉ `http://127.0.0.1:8765`, để trống khóa. Khi demo phải nói rõ đây là bản dịch dịch trước bằng mô hình, không phải dịch trực tiếp.

Trang đã có trong bộ đệm (mã offer 1688, mở `https://detail.1688.com/offer/<mã>.html`):
- **Nên dùng demo (dịch sạch):** 910634322485 (giày), 1074587053581 (áo thun nam), 1070743566317 (sạc dự phòng), 984785688114 (túi xách), 964243788985 (bình tập uống).
- Khá: 1028424261693 (búa thoát hiểm), 1067246811984 (đồ trang trí xe), 743290639213 (đồ chơi), 764758297887 (giày thể thao), 1070297037717 (váy), 1004006341754 (ba lô).
- **Minh họa giới hạn (có lỗi):** 1044015011283 (title phiên âm sai thương hiệu 三只松鼠, nhầm món), 1078813702157 (title sai, thêm "(Thượng Hải)" không có trong nguồn).

Lưu ý: phải dịch lại đúng nội dung trang; nếu người bán sửa trang (title/thuộc tính đổi) thì mục bộ đệm cũ không khớp và trả 503. Lấy thêm bộ đệm: chạy phiên Kaggle (cách A), mở các trang mới, đợi phiên kết thúc rồi `kaggle kernels output trinhnhatanh/demo-may-chu -p <thư mục>` lấy `may_chu/bo_dem.sqlite` (khóa bộ đệm gồm tên mô hình `kaggle-fp16-pair_cuoi`; gộp bộ đệm mới với cũ bằng cách chép các dòng bảng `dem`).
## Trang đang hiện tiếng Anh/Việt (bản 1688 tự dịch)
Nếu trang hiện ô ngôn ngữ "VN-English-VND" và tên sản phẩm bằng tiếng Anh/Việt, 1688 đang trả bản tự dịch của họ (và ẩn bảng thuộc tính sau đăng nhập). Bấm nút của extension: nút đổi thành **Chuyển trang sang tiếng Trung**; bấm tiếp, extension đặt cookie `oversealanguage=zh` (cần quyền `cookies` trong manifest), tải lại trang và tự dịch. Nếu 1688 vẫn trả trang không phải tiếng Trung (ví dụ tài khoản đăng nhập có hồ sơ nước ngoài) thì extension báo hướng dẫn: đổi ô ngôn ngữ sang 中文 hoặc mở bằng cửa sổ ẩn danh. Sau khi sửa manifest/mã extension phải bấm nút **tải lại** (⟳) ở thẻ extension trong `chrome://extensions` rồi F5 trang 1688.

## Phạm vi và giới hạn (nói rõ khi demo)
- Chỉ dịch **tên sản phẩm + bảng thuộc tính** (đúng thứ mô hình đã học). Không dịch ảnh mô tả, đánh giá, tên phân loại SKU, giao diện 1688.
- Điểm số trong báo cáo đo trên dữ liệu đã làm sạch; trang 1688 thật là dữ liệu ngoài phân phối, chất lượng chỉ là quan sát.
- Máy chủ chạy một lượt sinh tại một thời điểm, phục vụ một người dùng.
- Đường hầm là Cloudflare quick tunnel chạy từ notebook Kaggle: chưa kiểm điều khoản dịch vụ của Kaggle/Cloudflare cho việc này; nếu phiên bị chặn thì dùng cách B hoặc C.

## Kiểm tra nhanh trước giờ demo
- `demo\.venv\Scripts\python.exe -X utf8 demo/may_chu/test_may_chu.py` (5 test, không GPU).
- `demo\.venv\Scripts\python.exe -X utf8 demo/extension/test_extension.py` (logic DOM trên 5 trang đã lưu, mô hình giả).
- `... test_extension.py --that data/processed/ft_run/demo_may_chu_kaggle3.log --ra ket_qua.json` (cùng 5 trang nhưng dịch bằng máy chủ thật).
- Lấy thêm trang mẫu: `ecom_crawler\.venv\Scripts\python.exe -X utf8 demo/lay_mau_trang.py URL...` (cửa sổ Chrome hiện ra; nếu gặp thanh trượt xác minh thì người kéo một lần).

