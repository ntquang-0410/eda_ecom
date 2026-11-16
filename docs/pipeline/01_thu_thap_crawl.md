# 01. Thu thập dữ liệu (crawl) và snapshot

[← Tổng quan](00_TONG_QUAN.md) · [02 →](02_eda_va_lam_sach_mo_ta.md)

Thời gian: 09/2026, xong khoảng 01/10/2026. Mã crawler nằm ở repo riêng `ecom_crawler` (`D:\nam4_ky1\e-commerce - crawl\ecom_crawler`), **không** chuyển sang repo EDA. Nguồn: WALKTHROUGH mục 1–4, `docs/ecom_crawler_README.md`, `docs/PROVENANCE.md`.

## 1. Bối cảnh đề tài

- TLCN nhóm 3 người (Quang PM, Nhật Anh, Huy), tháng 9–12/2026: tối ưu dịch máy Việt–Trung cho thương mại điện tử xuyên biên giới.
- Mục tiêu sản phẩm: **tiện ích trình duyệt**. Người dùng mở trang bán hàng tiếng Trung, bấm để thấy bản dịch tiếng Việt của title và bảng thuộc tính.
- Không có bộ song ngữ Việt–Trung miền TMĐT công khai, nên tự crawl là đóng góp chính.

## 2. Hai nguồn

| | 1688 (nguồn chính) | Tiki (nguồn phụ) |
|---|---|---|
| Ngôn ngữ | Trung gốc, kèm bản Việt do **Alibaba dịch máy** ngay trên trang | Việt bản địa (đơn ngữ) |
| Cách lấy | Playwright chạy Chrome thật (đăng nhập, slider), trang tìm kiếm rồi trang chi tiết | JSON API công khai của Tiki, không cần đăng nhập |
| Hàng đợi | Firebase Realtime DB, 3 máy cùng nhận việc | như 1688 |
| Quy mô snapshot | **16.348** sản phẩm song ngữ (thêm `mono_zh` 6.334 sản phẩm không có bản Việt) | **13.200** sản phẩm (6 ngành × 2.200), xong 26/09 |
| Ngành hàng | 9: fashion, electronics, shoes, bags, beauty, mother_baby, food, home, auto | 6: auto, beauty, electronics, food, home, mother_baby |

Mỗi bản ghi gồm title, mô tả dạng `tên: giá trị` (bảng thuộc tính), category và vài cột phụ như `shop`.
- Dữ liệu thô (`data/raw/`, HF `bronze/`) chỉ là bản sao lưu.
- Mọi phân tích đọc từ `data/snapshot/*.parquet`, đã gộp và chuẩn hóa schema. Chuẩn hóa này dùng `textnorm.py`: NFC, đổi chữ và số toàn chiều, bỏ HTML và ký tự vô hình.

## 3. Quy tắc từ điều lệ nhóm còn ảnh hưởng về sau

- **Bản Việt của 1688 là máy dịch**, nên ban đầu chỉ coi là dữ liệu bạc (silver) và nguồn thuật ngữ. Điều lệ ghi test phải do người dịch hoặc người kiểm.
  - Thực tế sau này: chưa có người song ngữ. Ngày 31/10 Nhật Anh quyết để Judge đánh giá và không làm tập vàng (xem file 08, 10).
- Phải lọc thông tin cá nhân trước khi công bố:
  - cột `shop` có ở mọi dòng 1688;
  - một số mô tả chứa số điện thoại hoặc địa chỉ nhà sản xuất.
- Chia tập phải theo sản phẩm hoặc nhóm sản phẩm, không theo dòng.

## 4. Chuyển sang repo EDA (01/10/2026)

- Repo `e-commerce - EDA` tách khỏi `ecom_crawler`.
- 41 file được chép sang, ghi trong `docs/copy_manifest.csv` (33 giống hệt byte, 7 script chỉ sửa đường dẫn).
- Dữ liệu đầu vào đóng băng ở `data/snapshot/`, có SHA256 trong `SNAPSHOT.md`.
- Transcript phiên Claude giai đoạn 15/09–01/10 không còn trên máy (Claude Code đã dọn). Giai đoạn đó chỉ còn ghi lại trong WALKTHROUGH mục 0–7.
