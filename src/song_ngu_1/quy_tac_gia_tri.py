# Bước 6c, phần quy tắc theo mẫu (anh duyệt 22/10/2026): vài dạng giá trị lặp lại nhiều mà máy dịch không đều hoặc sai, có cách viết
# chắc chắn nên dịch bằng quy tắc thay cho bản máy. Chỉ khớp NGUYÊN chuỗi theo mẫu chặt; mẫu không chắc thì để mô hình dịch.
#
#   quy_tac_khoang   khoảng có (含)/(不含), 及以上, 及以下, viết theo cách tự nhiên mà máy dịch đúng đã dùng:
#                    "180g(含)-250g(不含)" -> "Từ 180g đến dưới 250g", "91%(含)-95%(含)" -> "Từ 91% đến 95%",
#                    "96%及以上" -> "Từ 96% trở lên", "500g（不含）以下" -> "Dưới 500g", "1年(含)-3年(不含)" -> "Từ 1 năm đến dưới 3 năm".
#                    Lý do: máy dịch 含 thành "chứa", "đầy đủ", và dịch "500g（不含）以下" thành "Từ 500g trở xuống" (sai: 不含 là không
#                    bao gồm), mỗi bản một kiểu viết.
#   quy_tac_diem_anh thuộc tính có chữ 像素 (độ phân giải ảnh), giá trị "N万": N万 điểm ảnh = N/100 MP ("300万" -> "3 MP");
#                    máy dịch lúc "3 MP" lúc "300 vạn".
#
# Không xử lý (để mô hình): "以上"/"以下" không kèm 含 (Trung không nói rõ có bao gồm đầu mút), khoảng chỉ ghi 含 ở một đầu
# ("0cm~3cm(含)"), khoảng nằm trong câu dài ("中跟[3-6cm(含)]"), 万 ở thuộc tính khác (số lượng, giá).
import re

_SO = r"\d+(?:\.\d+)?"
# đơn vị đi sau số: đơn vị Latin giữ nguyên; đơn vị chữ Hán đổi sang tiếng Việt (có dấu cách trước)
_DV_HAN = {"年": "năm", "个月": "tháng", "岁": "tuổi", "小时": "giờ", "天": "ngày"}
_DV = r"%|kg|KG|mg|g|G|ml|mL|ML|l|L|cm|CM|mm|MM|m|mAh|mah|W|w|V|v|" + "|".join(sorted(_DV_HAN, key=len, reverse=True))
_Q = rf"({_SO})\s*({_DV})?"


def _chuan(s):
    """Chuẩn hoá dấu: ngoặc toàn khổ thành ngoặc thường, các dấu nối khoảng thành '-', bỏ khoảng trắng."""
    s = s.strip().replace("（", "(").replace("）", ")")
    s = re.sub(r"\s*(?:-|–|—|~|～|至|到)\s*", "-", s)
    return re.sub(r"\s+", "", s)


def _q(so, dv):
    """Số và đơn vị: '180','g' -> '180g'; '1','年' -> '1 năm'."""
    if not dv:
        return so
    return f"{so} {_DV_HAN[dv]}" if dv in _DV_HAN else f"{so}{dv}"


def _khoang(m, tren):
    a, b = _q(m[1], m[2]), _q(m[3], m[4])
    dau = "Trên" if tren else "Từ"
    return f"{dau} {a} đến {b}" if m[5] == "含" else f"{dau} {a} đến dưới {b}"


MAU_KHOANG = [
    (re.compile(rf"^{_Q}\(含\)-{_Q}\((含|不含)\)$"), lambda m: _khoang(m, False)),
    (re.compile(rf"^{_Q}\(不含\)-{_Q}\((含|不含)\)$"), lambda m: _khoang(m, True)),
    (re.compile(rf"^{_Q}(?:\(含\)以上|及以上|以上\(含\))$"), lambda m: f"Từ {_q(m[1], m[2])} trở lên"),
    (re.compile(rf"^{_Q}(?:\(含\)以下|及以下|以下\(含\))$"), lambda m: f"Từ {_q(m[1], m[2])} trở xuống"),
    (re.compile(rf"^{_Q}(?:\(不含\)以下|以下\(不含\))$"), lambda m: f"Dưới {_q(m[1], m[2])}"),
    (re.compile(rf"^{_Q}(?:\(不含\)以上|以上\(不含\))$"), lambda m: f"Trên {_q(m[1], m[2])}"),
]
DIEM_ANH = re.compile(rf"^({_SO})万(?:像素)?$")


def _mp(n):
    x = round(float(n) / 100, 2)
    return str(int(x)) if x == int(x) else f"{x:g}"


def ban_quy_tac(ten_zh, gia_tri_zh):
    """Bản dịch theo quy tắc mẫu của một giá trị, trả (bản, cách chọn) hoặc None."""
    if not isinstance(gia_tri_zh, str) or not gia_tri_zh.strip():
        return None
    z = _chuan(gia_tri_zh)
    if isinstance(ten_zh, str) and "像素" in ten_zh:
        m = DIEM_ANH.match(z)
        if m:
            return f"{_mp(m[1])} MP", "quy_tac_diem_anh"
    for rx, f in MAU_KHOANG:
        m = rx.match(z)
        if m:
            return f(m), "quy_tac_khoang"
    return None
