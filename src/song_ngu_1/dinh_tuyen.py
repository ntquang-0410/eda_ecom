# Bước 3a: phân loại giá trị nào KHÔNG cần mô hình (quy tắc cố định), giá trị nào gửi cho mô hình.
# Quy tắc lấy lại từ pipeline làm sạch (notebooks/desc-pipeline.ipynb, P7.5 tên hãng và P8 mã thật), áp lên chữ Trung,
# không dùng bản Việt cũ: chép nguyên giá trị không có chữ Hán; mã thật chép nguyên kèm chú thích tỉnh (anh chốt 03/10);
# tên hãng (品牌) chuyển pinyin hoặc theo bảng từ chung anh đã duyệt; ngày 2023年05月24日 -> 24/05/2023.
import re

import pandas as pd
from pypinyin import lazy_pinyin

from .van_ban import CJK, co_han

# ---------- mã thật: chép nguyên, mã còn chữ Hán thêm chú thích tỉnh hoặc "quốc gia" (P8) ----------
TINH = {"粤": "Quảng Đông", "粵": "Quảng Đông", "沪": "Thượng Hải", "浙": "Chiết Giang", "鲁": "Sơn Đông", "陕": "Thiểm Tây",
        "皖": "An Huy", "云": "Vân Nam", "滇": "Vân Nam", "苏": "Giang Tô", "津": "Thiên Tân", "黑": "Hắc Long Giang",
        "赣": "Giang Tây", "湘": "Hồ Nam", "桂": "Quảng Tây", "冀": "Hà Bắc", "鄂": "Hồ Bắc", "吉": "Cát Lâm", "川": "Tứ Xuyên",
        "蜀": "Tứ Xuyên", "豫": "Hà Nam", "京": "Bắc Kinh", "闽": "Phúc Kiến", "辽": "Liêu Ninh", "晋": "Sơn Tây",
        "渝": "Trùng Khánh", "琼": "Hải Nam", "贵": "Quý Châu", "黔": "Quý Châu", "甘": "Cam Túc", "陇": "Cam Túc",
        "宁": "Ninh Hạ", "新": "Tân Cương", "蒙": "Nội Mông", "青": "Thanh Hải", "藏": "Tây Tạng"}
NGAY = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")


def doi_ngay(v):
    m = NGAY.fullmatch((v or "").strip())
    return f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}" if m else None


def chu_thich_ma(v):
    if not co_han(v):
        return v
    ngay = doi_ngay(v)
    if ngay:
        return ngay
    han = CJK.findall(v)
    phan = (["quốc gia"] if han and han[0] == "国" else []) + list(dict.fromkeys(TINH[c] for c in han if c in TINH))
    return f"{v} ({', '.join(phan)})" if phan else v


# ---------- tên hãng (P7.5) ----------
LATIN_KHAC = {"other", "others", "ohter", "other/其他", "other/其它", "其他/other", "其他/others"}
HAU_TO_LOAI = {"汽车用品": "Đồ dùng ô tô", "箱包": "Túi xách", "玩具": "Đồ chơi", "童装": "Thời trang trẻ em", "包装": "Bao bì",
               "茶业": "Trà", "鞋业": "Giày", "茶叶": "Trà", "服饰": "Thời trang", "皮具": "Đồ da", "制衣": "May mặc", "食品": "Thực phẩm"}
LOAI_NGOAC = {"食品": "thực phẩm", "食物": "thực phẩm", "食品类": "thực phẩm", "母婴": "mẹ và bé", "美妆": "mỹ phẩm", "彩妆": "mỹ phẩm",
              "数码": "kỹ thuật số", "鞋": "giày", "箱包": "túi xách", "洗护": "chăm sóc", "收纳整理": "đồ lưu trữ", "服饰": "thời trang",
              "奶嘴": "núm vú", "印度尼西亚": "Indonesia", "布料床单": "vải ga giường", "钓鱼类玩具": "đồ chơi câu cá", "床上用品": "đồ dùng giường ngủ",
              "车载用品": "đồ dùng ô tô", "汽配": "phụ tùng ô tô", "服装鞋帽": "quần áo giày mũ", "清洁工具": "dụng cụ vệ sinh", "电器": "đồ điện"}
CONG_TY = re.compile(r"有限公司|公司|工厂|厂$|厂（|供应链|工艺品$|纺织品|电子产品$")


def doc_tu_chung(cac_file):
    """Bảng từ chung của tên hãng anh đã duyệt: anh_duyet khác rỗng và khác OK thì lấy anh_duyet, còn lại lấy dich_goi_y."""
    bang = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig") for f in cac_file if f.exists()],
                     ignore_index=True)
    out = {}
    for r in bang.itertuples():
        d = r.anh_duyet.strip()
        out[r.tu_chung_zh] = d if d and d.upper() != "OK" else r.dich_goi_y
    return out


def phien_am(s):
    """Phiên âm phần chữ Hán, giữ nguyên chữ số và ký tự khác (70迈 -> 70mai)."""
    out = "".join("".join(lazy_pinyin(t)) if CJK.match(t) else t for t in re.findall(r"[一-鿿]+|[^一-鿿]+", s))
    return out[:1].upper() + out[1:]


def ten_hang(z, tu_chung):
    """Trả về (bản Việt, cách). Bản Việt None nghĩa là không đoán được (tên công ty, chú thích lạ): cặp cất riêng."""
    zs = z.strip()
    if co_han(zs):
        zs = re.sub(r"\s*，\s*", ", ", zs).replace("、", ", ").strip(" -_.·/\\|~*")
    if zs in tu_chung:
        return tu_chung[zs], "tu_chung"
    if zs.lower() in LATIN_KHAC:
        return "Khác", "tu_chung"
    if not co_han(zs):
        return zs, "latin"
    m = re.match(r"^(.*?)[（(]([^）)]*)[）)]?\s*$", zs)
    chinh, ngoac = (m.group(1).strip(), m.group(2).strip()) if m else (zs, "")
    if len(CJK.findall(chinh)) > 7 or CONG_TY.search(chinh):
        return None, "dai"
    latin = re.findall(r"[A-Za-z][A-Za-z0-9&'\-]+(?:\s+[A-Za-z0-9&'\-]+)*", chinh)
    hau_to = next((h for h in HAU_TO_LOAI if chinh.endswith(h) and 1 <= len(chinh) - len(h) <= 4 and co_han(chinh[:-len(h)])), None)
    if latin:
        goc, cach = " ".join(latin).strip(), "giu_latin"
    elif hau_to:
        goc, cach = f"{HAU_TO_LOAI[hau_to]} {phien_am(chinh[:-len(hau_to)])}", "pinyin_loai"
    else:
        goc, cach = phien_am(chinh), "pinyin"
    if ngoac:
        if ngoac in LOAI_NGOAC:
            goc = f"{goc} ({LOAI_NGOAC[ngoac]})"
        elif not co_han(ngoac):
            goc = f"{goc} ({ngoac})"
        else:
            return None, "dai"
    return goc, cach


# ---------- phân loại một cặp ----------
def dinh_tuyen(name_zh, gia_tri_zh, nhan, tu_chung):
    """Trả về (loai, ban_viet). loai: 'mo_hinh' (gửi cho mô hình, ban_viet None), 'chep_nguyen', 'ma_that', 'ten_hang', 'ngay',
    hoặc 'ten_hang_dai' (không đoán được, cất riêng)."""
    z = gia_tri_zh or ""
    if not co_han(z):
        return "chep_nguyen", z
    if "chep_ma" in (nhan or ""):                 # mã thật đã nhận diện ở P4 của pipeline làm sạch
        return "ma_that", chu_thich_ma(z)
    if name_zh == "品牌":
        v, cach = ten_hang(z, tu_chung)
        return ("ten_hang", v) if v is not None else ("ten_hang_dai", None)
    ngay = doi_ngay(z)
    if ngay:
        return "ngay", ngay
    return "mo_hinh", None


# ---------- chuẩn bị chữ gửi mô hình: dấu che của phía Trung đổi sang dạng Việt để mô hình giữ nguyên ----------
DAU_CHE = {"[电话]": "[SĐT]", "[联系方式]": "[LIÊN HỆ]", "[URL]": "[URL]", "[EMAIL]": "[EMAIL]"}


def van_ban_gui(z):
    for a, b in DAU_CHE.items():
        z = z.replace(a, b)
    return z
