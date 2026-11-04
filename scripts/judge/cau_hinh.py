"""Cấu hình judge. Mô hình CỐ ĐỊNH theo yêu cầu của Nhật Anh (29/10/2026); khóa API KHÔNG nằm trong file nào."""
import os
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]

# Bản cụm được chọn để fine-tune (Prompt 569). Từ 30/10 (sau bước 0-4 làm sạch): 25.575 sản phẩm còn lại của bộ chính thức v2.
# Bản v1 cũ (26.038 dòng, dùng cho thí điểm 300 dòng): C:\Users\ADMIN\hf_fd\bo\data\bilingual_official\bilingual_official_context.parquet
FILE_VAO = Path(os.environ.get("JUDGE_FILE_VAO", GOC / "data" / "processed" / "ft_clean" / "ft_buoc4_giu.parquet"))

# Bản PAIR của cùng 25.575 sản phẩm (xuất bằng scripts/ft_clean/xuat_ban_pair.py); dùng khi chấm --doi-tuong mo_ta_pair
FILE_VAO_PAIR = Path(os.environ.get("JUDGE_FILE_VAO_PAIR", GOC / "data" / "processed" / "ft_clean" / "ft_buoc4_giu_pair.parquet"))

THU_MUC_RA = GOC / "data" / "processed" / "judge"

# --- Mô hình (cố định) ---
MO_HINH = "deepseek-v4.1-flash"

# Nhật Anh dùng gói OpenCode GO. Theo trang tài liệu Go (opencode.ai/docs/go, đọc 30/10/2026) gói Go có đường riêng:
#   https://opencode.ai/inference/go/openai/v1/chat/completions   (danh sách mô hình: .../inference/go/v1/models)
# Đường không có "/go" (https://opencode.ai/inference/openai/v1) tính theo SỐ DƯ trả trước, hết số dư thì HTTP 402.
# Đổi bằng biến môi trường OPENCODE_BASE_URL nếu cần.
BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/inference/go/openai/v1").rstrip("/")
TEN_BIEN_KHOA = "OPENCODE_API_KEY"
NHA_CUNG_CAP = "opencode"
RPM = None                    # giới hạn số lời gọi mỗi phút (None = không giới hạn)
GUI_PHIEN_OPENCODE = True     # header x-opencode-session chỉ dành cho OpenCode

# Nhà cung cấp chọn bằng --nha-cung-cap. Cả hai đều là mô hình deepseek-v4.1-flash (theo trang build.nvidia.com/deepseek-ai/
# deepseek-v4.1-flash, đọc 30/10/2026: model id "deepseek-ai/deepseek-v4.1-flash", base_url https://integrate.api.nvidia.com/v1;
# việc hai nơi chạy cùng trọng số/độ lượng tử hóa thì KHÔNG kiểm chứng được). NVIDIA là dịch vụ dùng thử, tuân theo "NVIDIA API
# Trial Terms of Service" (chưa đọc). RPM 35 = dưới mức 40 RPM mà các trang tổng hợp bên thứ ba nêu cho gói miễn phí (chưa xác minh).
# "alibaba" = Alibaba Cloud Model Studio (Bailian), đọc docs 31/10/2026 (alibabacloud.com/help/en/model-studio/deepseek-api): model id
# "deepseek-v4.1-flash" (hybrid, JSON Object được hỗ trợ), tắt suy luận bằng enable_thinking=false (docs chỉ ví dụ giá trị True; false là
# suy ra, cần kiểm bằng --kiem-tra / lần chạy thử). Địa chỉ gốc có WorkspaceId của riêng từng tài khoản nên KHÔNG ghi sẵn: đặt biến
# môi trường ALIBABA_BASE_URL = https://<WorkspaceId>.<vùng>.maas.aliyuncs.com/compatible-mode/v1 (vùng ap-southeast-1 hoặc cn-beijing...).
CAC_NHA_CUNG_CAP = {
    "opencode": {"base_url": BASE_URL, "mo_hinh": MO_HINH, "bien_khoa": "OPENCODE_API_KEY", "rpm": None, "phien": True,
                 "tat_suy_luan": {"thinking": {"type": "disabled"}}},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "mo_hinh": "deepseek-ai/deepseek-v4.1-flash",
               "bien_khoa": "NVIDIA_API_KEY", "rpm": 35, "phien": False,
               "tat_suy_luan": {"thinking": {"type": "disabled"}}},
    "alibaba": {"base_url": None, "bien_url": "ALIBABA_BASE_URL", "mo_hinh": "deepseek-v4.1-flash",
                "bien_khoa": "DASHSCOPE_API_KEY", "rpm": None, "phien": False,
                "tat_suy_luan": {"enable_thinking": False}},
}
THAM_SO_TAT_SUY_LUAN = {"thinking": {"type": "disabled"}}   # đổi theo nhà cung cấp trong chon_nha_cung_cap


def chon_nha_cung_cap(ten):
    global BASE_URL, MO_HINH, TEN_BIEN_KHOA, NHA_CUNG_CAP, RPM, GUI_PHIEN_OPENCODE, THAM_SO_TAT_SUY_LUAN
    c = CAC_NHA_CUNG_CAP[ten]
    url = c["base_url"]
    if url is None:
        url = os.environ.get(c["bien_url"], "").strip().rstrip("/")
        if not url:
            raise SystemExit(f"Nhà cung cấp '{ten}' cần biến môi trường {c['bien_url']} (địa chỉ gốc có WorkspaceId, "
                             f"dạng https://<WorkspaceId>.<vùng>.maas.aliyuncs.com/compatible-mode/v1).")
    NHA_CUNG_CAP, BASE_URL, MO_HINH = ten, url, c["mo_hinh"]
    TEN_BIEN_KHOA, RPM, GUI_PHIEN_OPENCODE = c["bien_khoa"], c["rpm"], c["phien"]
    THAM_SO_TAT_SUY_LUAN = c["tat_suy_luan"]

# --- Chạy ---
# Giá để ƯỚC chi phí (USD / 1 triệu token) khi phản hồi không có usage.opencode. Docs zen ghi 0,30 / 1,20, nhưng phản hồi thật của
# cổng Go (30/10, 37 token vào + 17 token ra) cho list_cost = 1,575e-5 = đúng 37 x 0,15e-6 + 17 x 0,60e-6, tức một nửa giá docs;
# dùng 0,15 / 0,60. Một lời gọi chưa đủ để chắc chắn, nên script ghi list_cost và allowance thật từng dòng.
GIA_VAO, GIA_RA = 0.15, 0.60
NHIET_DO = 0
MAX_TOKEN_RA = 4000           # thí điểm 300 dòng (30/10): 50/300 bị cắt ở 2000 (đều là loại bi_cat_het_token); trung vị token ra 938 dù lời chấm ngắn → nghi có suy luận ẩn tính vào token ra
SO_LUONG_SONG_SONG = 4        # bảo thủ để khỏi dính giới hạn tốc độ; tăng bằng --song-song
THOI_GIAN_CHO = 180           # giây cho mỗi lời gọi
SO_LAN_THU_LAI = 5
PHIEN_BAN_RUBRIC = "v3"       # v2 (30/10): lời chấm ngắn hơn + chỉ liệt kê lỗi thật; v3 (30/10 tối): thêm "hoặc xuống dòng" cho đoạn thuộc tính (bản pair) + rubric riêng cho title
HAT_GIONG = 20261029

# Ngưỡng cắm cờ (đặt tạm, chọn lại sau khi xem phân bố ở bước thí điểm)
NGUONG_CO = 3                 # một chiều <= 3 thì cắm cờ
CHIEU = ["sai_nghia", "thieu_y", "them_y", "thuat_ngu", "so_don_vi", "troi_chay"]
