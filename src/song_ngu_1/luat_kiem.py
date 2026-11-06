# Bước 5 (laptop, CPU): luật kiểm từng bản dịch. Hai mức:
#   trượt (lỗi nặng, bản đó coi như hỏng) và cờ nhắc (chỉ ghi chú, không loại).
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .van_ban import CJK, chuan_hoa, co_han

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doi_chieu import CAN_VI, GIA_JIN, JIN, canh_bao_don_vi, kiem_jin  # noqa: E402  (src/doi_chieu.py, đã có 30 + 26 ca thử)

LOI_THUA = re.compile(r"^\s*(bản dịch|dịch|translation|tiếng việt|vietnamese|câu trả lời)\s*[:：]|<think>|</think>|```|\*\*", re.I)
# mô hình dịch luôn cả phần ngữ cảnh của câu lệnh (danh mục, tên thuộc tính) hoặc gắn nhãn "Xxx: " trước bản dịch (gặp ở lần chạy thử M0)
NGU_CANH = re.compile(r"tên thuộc tính|thể loại sản phẩm|danh mục sản phẩm|các loại sản phẩm|loại sản phẩm\s*[:：]", re.I)
NHAN_DAU = re.compile(r"^[^:：,]{2,40}[:：]\s")
CAN_GIA_VI = re.compile(r"/\s*cân(?![a-zà-ỹ])|\bmỗi\s+cân(?![a-zà-ỹ])", re.I)   # giá "15 tệ/cân": người Việt đọc là mỗi kg
# "cân" dùng làm đơn vị ở mọi dạng (sửa 22/10/2026 lúc soạn thẻ HF: hai regex trên chỉ bắt "số + cân" và "/cân", sót 半斤 "nửa cân",
# 一斤 "một cân", "130 tệ một cân", "(cân/43 tệ)", "(100-120) cân"). Trừ "cân nặng/bằng/đối/đo/điện tử" và "theo cân" (按斤 bán theo cân).
CAN_DON_VI_VI = re.compile(r"(?<![a-zà-ỹ])(?<!theo )cân(?![a-zà-ỹ])(?!\s*(?:nặng|bằng|đối|đo|điện))", re.I)
SO = re.compile(r"\d+")
HAN_SO = re.compile("[一二三四五六七八九十两半百千万双单]")
MA = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-+./]*")
DAU_CHE = re.compile(r"\[(?:SĐT|LIÊN HỆ|URL|EMAIL)\]")
# âm tiết tiếng Việt viết không dấu: phụ âm đầu + nguyên âm + phụ âm cuối hợp lệ. Chữ Latin không khớp thì nghi là chữ Anh/pinyin.
AM_TIET = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?[aeiouy]{1,3}(ch|ng|nh|[cmnpt])?$")
LATIN_QUEN = {"kg", "g", "mg", "cm", "mm", "m", "ml", "l", "inch", "size", "set", "combo", "usb", "led", "lcd", "pc", "pcs",
              "oem", "logo", "pin", "wifi", "bluetooth", "type", "mini", "max", "pro", "plus", "ok", "cotton", "pvc", "abs", "pu",
              "pp", "pe", "tpu", "eva", "spa", "gel", "vitamin", "collagen", "silicon", "silicone", "nylon", "polyester", "xl", "xxl"}


def so_trong(s):
    return Counter(SO.findall(s))


def lech_so(zh, vi):
    """Số của bản Trung phải còn ở bản Việt (cho phép bỏ dấu tách hàng nghìn, 万 đổi ra số đầy đủ)."""
    can = so_trong(zh)
    if not can:
        return False
    co = so_trong(vi) + so_trong(re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", vi))
    thieu = can - co
    if not thieu:
        return False
    if "万" in zh:   # 5万 -> 50.000 hoặc 50 nghìn
        thieu = Counter({x: n for x, n in thieu.items()
                         if not (str(int(x) * 10000) in co or str(int(x) * 10) in co)})
    return bool(thieu)


def so_muc(zh):
    """Khoảng số mục cho phép của một danh sách nối bằng ', ' (dấu phẩy Trung trong một mục có thể thành dấu phẩy khi dịch)."""
    it = len(zh.split(", "))
    return it, it + zh.count("，") + zh.count("、") + zh.count("；")


def dem_muc_viet(vi):
    return len(re.split(r"(?<!\d),|,(?!\d)", vi))


def latin_la(zh, vi):
    zl = zh.lower()
    ra = []
    for t in re.findall(r"[A-Za-z]{3,}", unicodedata.normalize("NFC", vi)):
        tl = t.lower()
        if tl in zl or tl in LATIN_QUEN or AM_TIET.match(tl):
            continue
        ra.append(tl)
    return ra


def kiem_mot(zh_goc, zh_gui, vi, tu_dien=None):
    """Kiểm một bản dịch. Trả về (danh sách mã trượt, danh sách cờ nhắc). Luật độ dài làm riêng ở do_dai_bat_thuong."""
    vi = vi if isinstance(vi, str) else ""
    truot, nhac = [], []
    v = vi.strip()
    if not v:
        return ["rong"], []
    if CJK.search(v):
        truot.append("han_sot")
    if LOI_THUA.search(v) or ("\n" in v and "\n" not in zh_gui):
        truot.append("loi_thua")
    if (NGU_CANH.search(v) and not re.search("属性|类目|品类|类型|分类", zh_gui)) or \
            (NHAN_DAU.match(v) and not re.search("[:：]", zh_gui)):
        truot.append("lo_ngu_canh")
    tu = v.split()
    if len(tu) >= 8:
        for n in range(1, 5):
            for i in range(0, len(tu) - 4 * n + 1):
                if tu[i:i + n] * 4 == tu[i:i + 4 * n]:
                    truot.append("lap_tu")
                    break
            if "lap_tu" in truot:
                break
        if "lap_tu" not in truot and len(tu) >= 20 and len(set(tu)) / len(tu) < 0.25:
            # Phép tỉ lệ từ không trùng bỏ qua danh sách có số mục khớp nguồn (anh duyệt 21/10/2026): nguồn lặp phần đuôi ở mọi mục
            # (vd "...【全棉磨毛优质A类 更保暖更亲肤】" ở từng mục) nên bản dịch đúng cũng ít từ khác nhau; trên cả bộ phép này bắt
            # nhầm 1.746 bản Gemma. Phép cụm từ lặp liên tiếp ở trên vẫn áp cho mọi bản (bắt kiểu "Ren ren ren ren...").
            lo, hi = so_muc(zh_gui) if ", " in zh_gui else (0, -1)
            if not lo <= dem_muc_viet(v) <= hi:
                truot.append("lap_tu")
    # số và đơn vị 斤
    if "斤" in zh_gui:
        tt, _ = kiem_jin(zh_gui, v)
        if (JIN.search(zh_gui) and CAN_VI.search(v)) or (GIA_JIN.search(zh_gui) and CAN_GIA_VI.search(v)) or CAN_DON_VI_VI.search(v):
            # giữ số 斤 mà viết "cân" (sửa 22/10/2026): người Việt hiểu "cân" là kg nên sai gấp đôi; bản đúng phải đổi ra kg (chia đôi)
            truot.append("jin_can")
        elif tt in ("da_sua", "lech_khac", "gia_theo_can"):
            truot.append("jin_sai")
        elif tt != "ok":
            nhac.append("jin_xem")
        # số ngoài phần 斤 vẫn phải khớp. Xoá cả KHOẢNG "90-100斤" và số Hán (regex JIN), không chỉ "100斤": trước 22/10/2026 số đầu
        # khoảng còn sót lại làm 171 bản Gemma đã đổi đúng ra kg bị trượt lech_so oan
        elif lech_so(JIN.sub("", zh_gui), v):
            truot.append("lech_so")
    elif lech_so(zh_gui, v):
        truot.append("lech_so")
    if not HAN_SO.search(zh_gui):   # số xuất hiện ở bản Việt mà bản Trung không có (kể cả khi bỏ dấu tách hàng nghìn)
        thua = so_trong(v) - so_trong(zh_gui)
        thua_gon = so_trong(re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", v)) - so_trong(zh_gui)
        if thua and thua_gon:
            nhac.append("so_thua")
    # chữ-số và mã Latin của bản Trung: có số hoặc từ 2 chữ hoa thì bắt buộc giữ; từ Anh thường thì chỉ nhắc
    vi_gon = re.sub(r"(?<=\d),(?=\d)", ".", re.sub(r"\s+", "", v).lower())   # 7,5cm của bản Việt khớp 7.5cm của bản Trung
    for t in MA.findall(zh_gui):
        t = t.rstrip("-+./")
        if not re.search(r"[A-Za-z]", t):
            continue
        if t.lower() in vi_gon:
            continue
        if re.search(r"\d", t) or sum(c.isupper() for c in t) >= 2:
            truot.append("mat_ma")
        else:
            nhac.append("mat_tu_latin")
    # danh sách: số mục phải khớp
    if ", " in zh_gui:
        lo, hi = so_muc(zh_gui)
        if not lo <= dem_muc_viet(v) <= hi:
            truot.append("lech_so_muc")
    # dấu che
    if Counter(DAU_CHE.findall(zh_gui)) - Counter(DAU_CHE.findall(v)):
        truot.append("mat_dau_che")
    # cờ nhắc
    nhac += canh_bao_don_vi(zh_gui, v) if re.search("[寸两万丝]", zh_gui) else []
    if latin_la(zh_gui, v):
        nhac.append("latin_la")
    if tu_dien and zh_goc in tu_dien and chuan_hoa(tu_dien[zh_goc]) != chuan_hoa(v):
        nhac.append("khac_tu_dien")
    return sorted(set(truot)), sorted(set(dict.fromkeys(nhac)))


# ---------- luật độ dài: khoảng bình thường lấy từ chính dữ liệu ----------
NHOM_DO_DAI = [(0, 2), (3, 5), (6, 12), (13, 40), (41, 10 ** 9)]


def nhom_do_dai(n):
    return next(i for i, (a, b) in enumerate(NHOM_DO_DAI) if a <= n <= b)


def khoang_do_dai(do_dai_zh, do_dai_vi, q_thap=0.005, q_cao=0.995):
    """Tỉ lệ độ dài Việt/Trung: với mỗi nhóm độ dài chữ Trung, khoảng [phân vị 0,5%, phân vị 99,5%] của mọi bản dịch."""
    df = pd.DataFrame({"n": do_dai_zh, "r": np.asarray(do_dai_vi, float) / np.maximum(np.asarray(do_dai_zh, float), 1)})
    df["nhom"] = df.n.map(nhom_do_dai)
    return {int(k): (float(g.r.quantile(q_thap)), float(g.r.quantile(q_cao))) for k, g in df.groupby("nhom")}


def do_dai_bat_thuong(n_zh, n_vi, khoang):
    lo, hi = khoang.get(nhom_do_dai(n_zh), (0.0, float("inf")))
    r = n_vi / max(n_zh, 1)
    return r < lo or r > hi


def doc_tu_dien(file, khoa, tam):
    """Mục từ điển anh đã duyệt (cột anh_duyet có nội dung): bản dùng là anh_duyet nếu anh ghi tên khác, không thì goi_y_claude,
    không có nữa thì bản chuẩn tạm. P6 tên thuộc tính: khoa='ten_zh', tam='ten_chuan_tam'; P7 giá trị: 'gia_tri_zh', 'gia_tri_chuan_tam'."""
    if not Path(file).exists():
        return {}
    t = pd.read_csv(file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    out = {}
    for r in t[t.anh_duyet.str.strip() != ""].itertuples():
        d = r.anh_duyet.strip()
        out[getattr(r, khoa)] = d if d.upper() != "OK" else (r.goi_y_claude.strip() or getattr(r, tam))
    return out
