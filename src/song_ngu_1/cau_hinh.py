# Cấu hình chung của pipeline bộ song ngữ thứ nhất. Đổi mô hình, ngưỡng, đường dẫn ở đây, không sửa trong mã xử lý.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Đầu vào: file cặp của bản desc_v1 (cũng nằm trên HF: zh-vie_ecom/data/cleaned/) ---
FILE_CAP = ROOT / "data" / "release" / "desc_v1" / "bilingual_zh_vi_desc_pairs(nhatanh).parquet"
FILE_SAN_PHAM = ROOT / "data" / "release" / "desc_v1" / "bilingual_zh_vi_desc_cleaned(nhatanh).parquet"   # chỉ để lấy nhom_trung
DUYET = ROOT / "data" / "processed" / "desc" / "duyet"
FILE_TU_CHUNG = [DUYET / "P7_tu_chung_ten_hang_v2_editted.csv", DUYET / "P7_tu_chung_ten_hang_v2_editted_bo_sung.csv"]

# --- Đầu ra ---
THU_MUC_RA = ROOT / "data" / "processed" / "song_ngu_1"
THU_MUC_KAGGLE = THU_MUC_RA / "vao_kaggle"          # thư mục tải lên Kaggle thành Dataset riêng tư
THU_MUC_MA_KAGGLE = Path(__file__).resolve().parent / "kaggle"

COT_BAT_BUOC = ["pair_id", "product_id", "category", "vi_tri", "name_zh", "gia_tri_zh", "vai_tro", "nhan", "ly_do"]

# --- Mã loại của bản v1 (anh chốt 09/10/2026: bỏ bản Việt cũ thì mã do lỗi bản Việt không còn lý do giữ) ---
MA_VAN_LOAI = {"PII", "PII_NSX", "RONG", "DS25"}            # nội dung Trung hoặc trùng: vẫn loại
MA_NHAN_LAI = {"T2_CHUA_KIEM", "DS10", "DS17", "DS03", "DS18", "DS07"}   # lỗi nằm ở bản Việt cũ: dùng lại phía Trung
MA_CHO_XEM = {"DS09_dai"}                                   # tên hãng/công ty dài bất thường: cất riêng, chờ anh xem
# mã lạ (không nằm trong ba tập trên) thì chuan_bi dừng và báo, không tự đoán

# --- Hai mô hình (đổi ở đây nếu sau này thay HY-MT) ---
# A đổi từ qwen3_8b sang DeepSeek V4.1 Flash (API NVIDIA) ngày 10/10/2026: Qwen3-8B là baseline huấn luyện về sau, giữ ra ngoài dữ liệu.
# Muốn thử lại Qwen3-8B: đổi "A" về "qwen3_8b" (mô hình vẫn còn trong kaggle/dich.py).
# Ngày 15/10/2026: A đổi từ API miễn phí NVIDIA (deepseek_v41_flash, quá chậm) sang DeepSeek V4.1 Flash trả phí qua Alibaba Cloud Model Studio
# (deepseek_alibaba). Bản gọi thẳng API chính thức của DeepSeek (deepseek_flash_api) vẫn còn trong kaggle/dich.py, đổi lại ở đây nếu cần.
# Ngày 16/10/2026: A đổi sang Gemma 4 26B-A4B (gemma-4-26b-a4b-it) gọi qua Google AI Studio bản miễn phí bằng cấu hình chung
# api_tuy_chon (notebook notebooks/song-ngu-1-gemma-kaggle.ipynb, 5 khoá); thư mục bản dịch trên Kaggle là api_tuy_chon.
# Bản DeepSeek (deepseek_alibaba) vẫn còn trong kaggle/dich.py nếu cần đổi lại.
MO_HINH = {
    "A": "api_tuy_chon",
    # B đổi từ hy_mt15_7b (giấy phép HY-MT hạn chế dùng đầu ra) sang Qwen3-14B bản AWQ chính thức ngày 22/10/2026, sau khi so 4 mô hình
    # trên 200 đơn vị (Qwen3-14B trượt luật 3,5%, lỗi chủ yếu là sót chữ Hán nên luật bắt được; Qwen3-30B-A3B nén cộng đồng bịa nghĩa).
    "B": "qwen3_14b",
}

# --- Ngưỡng bước 6b: TẠM, chọn lại sau khi xem phân bố điểm (không có nhãn đúng sai để hiệu chỉnh) ---
TAU = 60.0      # chrF từ mức này trở lên là "gần giống"
KAPPA = 0.50    # điểm sàn CometKiwi: thấp hơn thì không dùng bản đó
KAPPA_NGAN = 0.35   # điểm sàn riêng cho chuỗi Trung ngắn (từ DO_DAI_ZH_NGAN ký tự trở xuống, xem chon_ban.py); CometKiwi chấm chuỗi ngắn thấp
DELTA = 0.25    # bản của mô hình ưu tiên chỉ bị bỏ khi điểm CometKiwi của bản kia cao hơn từ ngần này (đổi nghĩa ngày 15/10/2026, trước là 0,05)
UU_TIEN = "A"   # mô hình ưu tiên khi hai bản cùng đạt (A = Gemma 26B từ 16/10/2026: trượt luật cứng 6,9% so với HY-MT 19,3% trên cả bộ, sát cấu trúc nguồn hơn)
DO_DAI_NGAN = 6  # chuỗi Việt (đã chuẩn hoá) từ 6 ký tự trở xuống: đòi giống hệt, không dùng chrF
# Anh duyệt 22/10/2026: hai bản cùng qua luật thì luôn lấy bản ưu tiên (A = Gemma); bản B (Qwen) chỉ dùng khi A trượt luật; điểm CometKiwi
# chỉ quyết định độ tin cậy và cất riêng khi cả hai dưới sàn. Lý do: Kiwi chấm bản Qwen sai mà trơn tru cao hơn bản Gemma đúng
# (mẫu 100 cặp: 10/12 lỗi rõ do vậy). False: luật cũ (đổi sang B khi B vượt điểm, khi chỉ B đạt sàn, hoặc khi B ít cờ nhắc hơn).
CHI_DOI_KHI_TRUOT = True

# --- Bước 6c (anh duyệt 21/10/2026): bản dịch cố định đè lên bản máy sau bước chọn bản ---
# Tên thuộc tính có trong P6 và giá trị có trong P7 dùng bản từ điển (chỉ mục anh đã duyệt, đọc bằng luat_kiem.doc_tu_dien);
# giá trị có/không dưới đây dùng bản cố định. Thứ tự ưu tiên của giá trị: CO_KHONG trước, rồi P7 (anh chốt 21/10/2026 để cặp đối nghĩa
# cùng kiểu: P7 ghi 不支持 "Không được hỗ trợ" nhưng dùng "Không hỗ trợ" như 支持 "Hỗ trợ"; không sửa file P7 anh đã duyệt).
FILE_TU_DIEN_TEN = DUYET / "P6_ten_thuoc_tinh_editted.csv"
FILE_TU_DIEN_GIA_TRI = DUYET / "P7_gia_tri_editted.csv"
# Tên bổ sung anh duyệt 22/10/2026 (7 tên bị cất nhiều cặp nhất sau khi đổi B sang Qwen3-14B); cùng định dạng P6, P6 đứng trước khi trùng
FILE_TU_DIEN_TEN_BO_SUNG = DUYET / "P6b_ten_thuoc_tinh_bo_sung_editted.csv"
CO_KHONG = {"是": "Có", "否": "Không", "有": "Có", "无": "Không có", "没有": "Không có", "支持": "Hỗ trợ", "不支持": "Không hỗ trợ"}
# Giá trị của thuộc tính thương hiệu (đi theo quy tắc tên hãng) vô nghĩa làm tên hãng: cất riêng, lý do ten_hang_vo_nghia (anh chốt
# 21/10/2026: thương hiệu "是" bị phiên âm "Shi", mâu thuẫn với 是 "Có" ở các cặp khác; 无 "Không thương hiệu" và 否 "Không có" vẫn giữ)
TEN_HANG_VO_NGHIA = {"是"}

# 9 cột của bản công bố (bản đầy đủ có 18 cột, xem hoan_tat.COT_DAY_DU)
COT_CONG_BO = ["pair_id", "product_id", "category", "nhom_trung", "name_zh", "gia_tri_zh", "name_vi", "gia_tri_vi", "do_tin_cay"]
