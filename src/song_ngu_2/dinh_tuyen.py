# Bước 3a của bộ 2: phân loại giá trị nào KHÔNG cần mô hình (quy tắc), giá trị nào cất riêng, giá trị nào gửi cho mô hình.
# Khác bộ 1: không có mã thật/ngày kiểu Trung/tên hãng phiên âm. Tên hãng chép nguyên; giá trị chỉ có số hoặc số + đơn vị đo chép nguyên.
import re

from . import cau_hinh as C

CHU = re.compile(r"[A-Za-zÀ-ỹ]")
PII = re.compile(C.PII_REGEX, re.I)
CHU_LATIN = re.compile(r"[A-Za-z]+")
# Đơn vị đo viết chữ Latin không dấu; bản Trung giữ nguyên (kg, ml, cm, W, V, mAh, GB ... Trung Quốc cũng viết như vậy)
DON_VI_DO = {"kg", "g", "gr", "mg", "l", "ml", "cl", "dl", "cm", "mm", "m", "km", "w", "kw", "v", "kv", "a", "ma", "mah", "ah", "wh", "kwh",
             "va", "kva", "gb", "mb", "tb", "kb", "hz", "khz", "mhz", "ghz", "db", "lm", "lux", "k", "mp", "inch", "oz", "lb", "lbs",
             "pcs", "ppm", "rpm", "bar", "psi", "pa", "kpa", "mpa", "cc", "hp", "c", "f", "x"}
MA_MODEL = re.compile(r"^(?=.*\d)[A-Za-z0-9][A-Za-z0-9\-_./+# ]{0,38}$")
# Cỡ chữ (S, M, L, XL, 2XL ...) hoặc danh sách cỡ ngăn bằng ; , / khoảng trắng: bản Trung giữ nguyên chữ cỡ
CO_CHU = r"(?:[2-9]?x{0,4}s|m|l|[2-9]?x{1,4}l)"
CO_DANH_SACH = re.compile(rf"^{CO_CHU}(?:\s*[;,/\-]\s*{CO_CHU})*$", re.I)


def chuan_ten(n):
    """Tên thuộc tính đã chuẩn hoá để tra bảng: chữ thường, gọn khoảng trắng, bỏ dấu hai chấm cuối."""
    return re.sub(r"\s+", " ", (n or "").strip().strip(":：").lower())


def co_chu(s):
    return bool(CHU.search(s or ""))


def co_lien_he(s):
    return bool(PII.search(s or ""))


def la_do_luong(v):
    """Giá trị chỉ gồm số và đơn vị đo (10.6*10.2*21.5cm, 12.5 x 7 x 0.5 cm, 10ml; 15ml; 30ml, 1,2kg; 2.4kg): chép nguyên."""
    if not re.search(r"\d", v):
        return False
    chay = CHU_LATIN.findall(v)
    if not chay or any(r.lower() not in DON_VI_DO for r in chay):
        return False
    return not re.search(r"[^\W\d_]", re.sub(r"[A-Za-z]+", "", v))      # không còn chữ cái nào khác (chữ có dấu)


def dinh_tuyen(ten, gia_tri):
    """Trả về (loai, ly_do). loai:
    van_loai (cất riêng, ly_do: to_chuc | lien_he), chep_nguyen (không có chữ cái), ten_hang (thương hiệu, chép nguyên),
    ma_model (giá trị dạng mã của thuộc tính mã, chép nguyên), do_luong (số + đơn vị đo, chép nguyên), co_chu (cỡ chữ S/M/L/XL, chép
    nguyên), mo_hinh (gửi mô hình)."""
    tc = chuan_ten(ten)
    v = (gia_tri or "").strip()
    if tc in C.TEN_BO_QUA:
        return "van_loai", "to_chuc"
    if co_lien_he(v) or co_lien_he(ten):
        return "van_loai", "lien_he"
    if not co_chu(v):
        return "chep_nguyen", ""
    if tc in C.TEN_THUONG_HIEU:
        return "ten_hang", ""
    if tc in C.TEN_MA and MA_MODEL.match(v):
        return "ma_model", ""
    if la_do_luong(v):
        return "do_luong", ""
    if CO_DANH_SACH.match(v):
        return "co_chu", ""
    return "mo_hinh", ""
