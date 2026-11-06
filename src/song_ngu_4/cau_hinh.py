# Cấu hình bộ song ngữ 4: dịch CẢ CỤM thuộc tính của từng sản phẩm 1688 (Trung -> Việt), mỗi sản phẩm một lần gọi, theo đúng cách v2 của
# bộ 3 (anh duyệt 24/10/2026): tên sản phẩm làm ngữ cảnh, mục KHOÁ (từ điển, quy tắc) không cho mô hình xuất, mục còn lại mang BẢN NHÁP là
# bản dịch đơn vị của bộ 1 (bản đã lên HF) để mô hình hiệu đính. Giá trị dài (danh sách SKU) không cho mô hình xem lại (anh chốt 24/10/2026).
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THU_MUC_BO_1 = ROOT / "data" / "processed" / "song_ngu_1"
NHAN_BO_1 = "20261022_2126"                                          # bản dịch đơn vị đã lên HF (data/description_zh_to_vi/)
FILE_BANG_CAP = THU_MUC_BO_1 / "bang_cap.parquet"                   # cặp của bộ 1 (đã làm sạch, có vi_tri, định tuyến, mã đơn vị)
FILE_BO_1_DAY_DU = THU_MUC_BO_1 / f"bo_song_ngu_1_day_du_{NHAN_BO_1}.parquet"
FILE_DON_VI = THU_MUC_BO_1 / f"don_vi_da_chon_{NHAN_BO_1}.parquet"
FILE_THE_BO_1 = THU_MUC_BO_1 / f"the_nguon_goc_{NHAN_BO_1}.json"     # lấy khoảng độ dài Việt/Trung của luật do_dai (cùng luật bộ 1)
FILE_SAN_PHAM = ROOT / "data" / "snapshot" / "bilingual_zh_vi.parquet"   # title_zh, shop
THU_MUC_RA = ROOT / "data" / "processed" / "song_ngu_4"

SO_SAN_PHAM_THU = 200
HAT_GIONG = 20261024
# Giá trị có hơn ngần này chữ (tính trên gia_tri_zh) không cho mô hình xem lại: giữ bản bộ 1, vẫn nằm trong cụm làm ngữ cảnh. Đo 24/10/2026:
# 6.037 giá trị (2,7% số giá trị cần xem) nhưng 46% số ký tự Việt mô hình phải viết; gần hết là danh sách màu, cỡ, quy cách.
MAX_ZH_XEM = 80
# Một lần gọi tối đa ngần này "trọng số" = chữ Trung (tên + giá trị) + nửa số ký tự Việt (bản khoá, bản nháp). Ngữ cảnh của Qwen trên vLLM là
# 12.288 token gồm cả câu lệnh hệ thống và câu trả lời; sản phẩm 1688 trung vị 287 chữ Trung nên gần như mọi sản phẩm đi một phần.
MAX_TRONG_SO_PHAN = 2000
MAX_KY_TU_TEN_KHAC = 400
# Giá trị KHOÁ quá dài (danh sách SKU tới hàng nghìn chữ) chỉ là ngữ cảnh, mô hình không xuất lại: trong câu lệnh chỉ hiện phần đầu (ngần này chữ
# Trung, gấp đôi ngần này ký tự Việt) kèm " …", để cụm không vượt ngữ cảnh 12.288 token của Qwen (đo 24/10/2026: câu lệnh dài nhất 19.527 ký tự
# trước khi cắt). Bản khoá đầy đủ vẫn nằm ở cap_vao.parquet và là bản cuối.
MAX_KHOA_HIEN = 300
MAX_TIEU_DE = 120

# Thương hiệu vô nghĩa đã bị bộ 1 cất riêng (song_ngu_1.cau_hinh.TEN_HANG_VO_NGHIA): không đưa vào
TEN_HANG_VO_NGHIA = {"是"}
# cach_chon của bộ 1 là bản KHOÁ (từ điển đã duyệt, bảng có/không, quy tắc): mô hình không được sửa
GIA_TRI_KHOA = {"tu_dien_p7", "quy_tac_co_khong", "quy_tac_khoang", "quy_tac_diem_anh", "quy_tac_chep_nguyen", "quy_tac_ten_hang",
                "quy_tac_ma_that", "quy_tac_ngay"}
TEN_KHOA = {"tu_dien_p6", "tu_dien_p6b"}
DANH_MUC_VI = {"bags": "túi xách", "mother_baby": "mẹ và bé", "home": "đồ gia dụng", "shoes": "giày dép", "food": "thực phẩm",
               "fashion": "thời trang", "beauty": "mỹ phẩm, làm đẹp", "electronics": "đồ điện tử", "auto": "phụ kiện ô tô"}
MO_HINH = {"A": "api_tuy_chon", "B": "qwen3_14b"}                    # A = Gemma 4 qua API, B = Qwen3-14B-AWQ (như bộ 1, bộ 3)
