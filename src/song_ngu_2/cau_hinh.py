# Cấu hình chung của pipeline bộ song ngữ thứ hai (Tiki: Việt gốc -> Trung dịch máy). Đổi mô hình, ngưỡng, đường dẫn ở đây.
# Anh Nhật Anh ra lệnh 22/10/2026: dịch bộ Việt của Tiki sang tiếng Trung bằng cùng hai mô hình của bộ 1; chỉ dịch thuộc tính
# (tên + giá trị) của description, không dịch tiêu đề và văn mô tả.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Đầu vào: bản EDA của Tiki trên HF (zh-vie_ecom/data/cleaned/tiki/tiki_eda.parquet), SHA256 ghi trong thẻ repo ---
# Anh Nhật Anh đổi từ tiki_vi_cado.parquet sang tiki_eda.parquet ngày 23/10/2026. Cùng 13.200 sản phẩm, cột attributes_vi giống hệt bản CADO;
# bản EDA thêm cột bt_attributes_text (thuộc tính đã làm sạch emoji, hashtag, gạch đầu dòng, ghép "Tên: giá trị, giá trị; Tên: ..."),
# split (train/dev/test do nhóm chia sẵn), row_hash. Giá trị nhiều phần tử ngăn bằng ", " như bt_attributes_text (và như bộ 1).
FILE_NGUON = ROOT / "data" / "cleaned" / "tiki" / "tiki_eda.parquet"
SHA256_FILE_NGUON = "7af2a2620b8ca2def3b80762cb8f99196bfe7c64baeaeaa8b99d974df570d3bd"
DAU_NGAN_GIA_TRI = ", "
DUYET = ROOT / "data" / "processed" / "song_ngu_2" / "duyet"

# --- Đầu ra ---
THU_MUC_RA = ROOT / "data" / "processed" / "song_ngu_2"
THU_MUC_KAGGLE = THU_MUC_RA / "vao_kaggle"
THU_MUC_MA_KAGGLE = ROOT / "src" / "song_ngu_1" / "kaggle"      # dich.py và cham_kiwi.py dùng chung hai bộ (cờ --huong vi_zh)

# --- Hai mô hình: như bộ 1 (A = Gemma 4 26B-A4B qua Google AI Studio, B = Qwen3-14B-AWQ trên Kaggle T4 x2) ---
MO_HINH = {"A": "api_tuy_chon", "B": "qwen3_14b"}
HUONG = "vi_zh"

# --- Định tuyến giá trị (xem dinh_tuyen.py) ---
# Thuộc tính cất riêng, không dịch: tên và địa chỉ đơn vị chịu trách nhiệm về hàng hóa (tên doanh nghiệp, địa chỉ cụ thể, hay kèm số điện
# thoại; hiến chương mục 6: bỏ thông tin cá nhân/cửa hàng/địa chỉ). Tên chuẩn hoá (chữ thường, bỏ dấu hai chấm cuối).
TEN_BO_QUA = {"tên đơn vị/tổ chức chịu trách nhiệm về hàng hóa", "địa chỉ tổ chức chịu trách nhiệm về hàng hóa"}
# Giá trị của thuộc tính thương hiệu: chép nguyên (tên hãng không dịch, không phiên âm; "OEM", "Vinamilk" giữ nguyên chữ)
TEN_THUONG_HIEU = {"thương hiệu"}
# Thuộc tính mã: giá trị dạng mã (có chữ số, chỉ chữ Latin/số/dấu nối) chép nguyên; giá trị là chữ thường ("Mới nhất") vẫn đi qua mô hình
TEN_MA = {"model", "mã", "mã hàng", "mã sản phẩm", "mã sp", "sku", "mã sku", "part number", "mã pin"}

# Phát hiện thông tin liên hệ (cất riêng, giống bước 2 của bộ 1)
# Thứ tự: số điện thoại Việt (0 hoặc 84 đầu, 9-10 chữ số, cho phép dấu chấm/gạch/khoảng trắng), email, địa chỉ web, kênh liên hệ
PII_REGEX = (r"(?<!\d)(?:\+?84|0)(?:[ .\-]?\d){9,10}(?!\d)"
             r"|[\w.+-]+@[\w-]+\.[\w.]+"
             r"|https?://|www\."
             r"|\b[\w-]+\.(?:com|vn|net|org|info|biz|shop|store|xyz)\b"
             r"|\b(?:zalo|facebook|fb\.com|hotline|messenger|telegram|viber|whatsapp|inbox)\b")

# --- Bước 6c: bản dịch cố định đè lên bản máy (file anh duyệt; file chưa có thì bỏ qua, đọc bằng song_ngu_1.luat_kiem.doc_tu_dien) ---
# Ba bảng, cùng định dạng P6/P7 của bộ 1 (cột anh_duyet, goi_y_claude, <cột chuẩn tạm>):
FILE_TU_DIEN_TEN = DUYET / "P6_ten_thuoc_tinh_tiki.csv"          # khoá ten_vi
FILE_TU_DIEN_GIA_TRI = DUYET / "P7_gia_tri_tiki.csv"             # khoá (ten_vi, gia_tri_vi): giá trị hay gặp (cùng chữ, khác thuộc tính có thể dịch khác)
FILE_CO_KHONG = DUYET / "P8_co_khong_tiki.csv"                   # khoá (ten_vi, gia_tri_vi): Có/Không dịch khác nhau theo thuộc tính (đứng trước P7)

# --- Ngưỡng bước chọn bản: TẠM, phải chọn lại sau khi có điểm CometKiwi của bộ Tiki (ngưỡng bộ 1 không dùng được) ---
# Chọn 23/10/2026 sau khi xem phân bố điểm thật (xem phan_bo_diem và do_nhay_nguong): điểm Kiwi thấp ở Tiki phần lớn do nguồn lộn xộn (sai chính tả, viết tắt),
# không phải do bản dịch sai, nên sàn thấp hơn bộ 1 (0,5/0,35).
TAU = 60.0
KAPPA = 0.40
KAPPA_NGAN = 0.30      # sàn ĐỘ TIN CẬY (A dưới sàn thì thap)
KAPPA_CAT = 0.30       # sàn CẤT RIÊNG: cả hai bản khác nhau và đều dưới sàn này thì bỏ (xem do_nhay_nguong: ~0,6% đơn vị)
KAPPA_CAT_NGAN = 0.22  # như trên cho nguồn ngắn
DELTA = 0.25
UU_TIEN = "A"
CHI_DOI_KHI_TRUOT = True          # như bộ 1: hai bản cùng qua luật thì lấy bản A; B chỉ khi A trượt luật
DO_DAI_NGAN = 3                   # bản Trung (đã chuẩn hoá) từ ngần này ký tự trở xuống: đòi giống hệt, không dùng chrF (chữ Hán ngắn hơn chữ Việt)
DO_DAI_NGUON_NGAN = 12            # chuỗi Việt nguồn từ ngần này ký tự trở xuống dùng điểm sàn KAPPA_NGAN
# luật do_dai chỉ còn đuôi TRÊN (tỉ lệ độ dài Trung/Việt cao hơn phân vị 99,9% của nhóm độ dài nguồn: bản dịch dài bất thường, thường là để nguyên tiếng
# Anh hoặc lặp). Đuôi dưới BỎ (23/10/2026): bản Trung ngắn gọn là bình thường ('Xuất xứ (Made in)' -> 产地 bị bắt oan, 12.672 cặp), việc bị cắt do hết token đã có luật bi_cat.
Q_DO_DAI = (0.0, 0.999)

COT_BAT_BUOC = ["product_id", "category", "attributes_vi", "bt_attributes_text", "split", "row_hash"]
COT_CONG_BO = ["pair_id", "product_id", "category", "name_vi", "gia_tri_vi", "name_zh", "gia_tri_zh", "do_tin_cay"]
