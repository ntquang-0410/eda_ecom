# Quy tắc mẫu cho giá trị tiếng Việt của Tiki (bước 6c của bộ 2): chuỗi thời hạn "12 Tháng", "3 năm kể từ ngày sản xuất" chiếm ~9% số cặp
# gửi mô hình nên dịch bằng quy tắc cho nhất quán (Gemma viết "12 个月", Qwen viết "12个月", câu có "kể từ NSX" mỗi lần một kiểu).
# Chỉ nhận ĐÚNG mẫu, không đoán: câu dài hơn ("Độ bền trên 10 năm. - Không cong vênh") vẫn do mô hình dịch.
import re

DON_VI = {"tháng": "个月", "năm": "年", "ngày": "天", "tuần": "周", "giờ": "小时", "phút": "分钟"}
THOI_HAN = re.compile(r"^\s*(\d+)\s*(tháng|năm|ngày|tuần|giờ|phút)"
                      r"(?:\s+(?:kể\s+)?(?:từ\s+)?(?:ngày\s+)?(sản xuất|sx|nsx))?\s*$", re.I)


KHOI_LUONG = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(gram|gam|gr|kilogram)\s*$", re.I)      # "500gram", "250 gam": Gemma để nguyên chữ gram (165 cặp)
KHOI_LUONG_DV = {"gram": "克", "gam": "克", "gr": "克", "kilogram": "千克"}


def ban_quy_tac(gia_tri_vi):
    """Trả về (bản Trung, cách chọn) hoặc None. Thời hạn: '12 Tháng' -> '12个月'; '3 năm kể từ ngày sản xuất' -> '自生产之日起3年'
    (cũng nhận 'từ ngày sản xuất', 'kể từ NSX', 'kể từ ngày SX'). Giờ và phút không kèm 'kể từ' vì không bao giờ đi cùng ngày sản xuất."""
    if not isinstance(gia_tri_vi, str):
        return None
    k = KHOI_LUONG.match(gia_tri_vi)
    if k:
        return f"{k.group(1)}{KHOI_LUONG_DV[k.group(2).lower()]}", "quy_tac_khoi_luong"
    m = THOI_HAN.match(gia_tri_vi)
    if not m:
        return None
    n, dv, tu_sx = m.group(1), m.group(2).lower(), m.group(3)
    if tu_sx and dv in ("giờ", "phút", "tuần"):
        return None
    t = f"{int(n)}{DON_VI[dv]}"
    return (f"自生产之日起{t}" if tu_sx else t), "quy_tac_thoi_han"
