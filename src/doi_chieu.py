# -*- coding: utf-8 -*-
"""Luật kiểm bản dịch, bản 2: quy đổi đơn vị.

Vì sao có file này: luật số ở P8 (`lech_so`) coi là "không lệch" ngay khi các con số bản Việt giống hệt bản Trung, và
với 斤 + kg còn chấp nhận cả số gốc lẫn số một nửa. Nên "190-215斤" → "190-215 kg" lọt qua, dù 1 斤 = 0,5 kg
(phải là 95-107.5 kg). Các đơn vị Trung Quốc khác cũng cùng kiểu hở (số giữ nguyên, đơn vị đổi):

  kiem_jin           斤 → kg: kiểm đúng giá trị, tự chia đôi khi chắc chắn (hoặc gắn DS10)
  canh_bao_don_vi    寸 (inch, không phải cm), 两 (50 g), 万 (10.000), 丝 (0,01 mm): chỉ gắn cờ cho người xem, không sửa

Dùng cho pipeline P8 (từng giá trị) và cho các bước đối chiếu bản dịch sau này.
"""
import re

_SO = r"\d+(?:[.,]\d+)?"
_NOI_ZH = r"\s*(?:-|–|—|~|～|至|到)\s*"
_NOI_VI = r"\s*(?:-|–|—|~|～|đến|tới)\s*"
_HAN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# 斤 làm đơn vị: đứng ngay sau số (số Ả Rập, số Hán 1-2 chữ, hoặc 半). "公斤" (= kg) không khớp vì 公 nằm giữa số và 斤;
# "千斤顶" (cái kích xe) bị loại bằng (?!顶).
# "N斤半" = N + 0,5 斤 ("一斤半" = 1,5 斤 = 0,75 kg): nhóm 4 bắt chữ 半 đứng sau 斤.
JIN = re.compile(rf"(?:({_SO})(?:{_NOI_ZH}({_SO}))?|([零〇一二两三四五六七八九十半]{{1,3}}))\s*斤(?!顶)(半)?")
# Giá tính theo 斤: con số là tiền, không được chia đôi. "30元/斤", "每斤", "30块钱一斤", "一斤30元", "5元1斤".
GIA_JIN = re.compile(r"[/／]\s*斤|每\s*斤|(?:元|块钱?|¥|￥|RMB|rmb)\s*[/／]?\s*(?:一|1)\s*斤"
                     r"|(?:一|1)\s*斤\s*[:：]?\s*\d+(?:\.\d+)?\s*(?:元|块)")
KG = re.compile(rf"({_SO})(?:{_NOI_VI}({_SO}))?\s*(?:kg|kilogam|kilôgam|ki-lô-gam)\b", re.I)
GIA_KG = re.compile(r"(?:/|trên|mỗi|một|1)\s*(?:kg|kilogam|kilôgam|ki-lô-gam)\b", re.I)


def han_so(s):
    """半 → 0.5; 一..九十九 → số. Không hiểu (vd '三五') thì trả None."""
    if s == "半":
        return 0.5
    if "十" in s:
        a, _, b = s.partition("十")
        if len(a) > 1 or len(b) > 1 or (a and a not in _HAN) or (b and b not in _HAN):
            return None
        return float((_HAN[a] if a else 1) * 10 + (_HAN[b] if b else 0))
    return float(_HAN[s]) if len(s) == 1 and s in _HAN else None


def _f(x):
    return float(x.replace(",", "."))


def _fmt(x):
    x = round(x, 2)
    return str(int(x)) if x == int(x) else f"{x:g}"


def _luong_zh(z):
    """Danh sách (a, b) các lượng đi với 斤 trong câu Trung, theo thứ tự; None nếu có chỗ không hiểu."""
    ra = []
    for m in JIN.finditer(z):
        ruoi = 0.5 if m.group(4) else 0.0   # "斤半"
        if m.group(3) is not None:
            a = han_so(m.group(3))
            if a is None:
                return None
            ra.append((a + ruoi, None))
        else:
            if m.group(2) and ruoi:   # "1-2斤半": không rõ nửa thuộc về số nào
                return None
            ra.append((_f(m.group(1)) + ruoi, _f(m.group(2)) if m.group(2) else None))
    return ra


def _gan(x, y):
    return abs(x - y) < 1e-6


def kiem_jin(z, v):
    """Kiểm quy đổi 斤 → kg của một giá trị. Trả (trang_thai, v_moi).

    trang_thai:
      khong_ap_dung  không có 斤 làm đơn vị, hoặc bản Việt không dùng kg
      ok             số đã quy đổi đúng (bằng một nửa)
      da_sua         bản Việt giữ nguyên số khi đổi sang kg: đã chia đôi các số đó (v_moi)
      gia_theo_can   giá tính theo 斤 mà bản Việt viết theo kg: không tự sửa được → gắn DS10
      lech_khac      số ở bản Việt không bằng số gốc cũng không bằng một nửa → gắn DS10
      khong_ghep     số lượng 斤 và số lượng kg không khớp nhau để so: không đổi, không gắn mã
    """
    if "斤" not in z:
        return "khong_ap_dung", v
    if GIA_JIN.search(z):
        return ("gia_theo_can", v) if GIA_KG.search(v) else ("khong_ap_dung", v)
    qz = _luong_zh(z)
    qv = list(KG.finditer(v))
    if qz == []:
        return "khong_ap_dung", v
    if not qv:
        return "khong_ap_dung", v
    if qz is None or len(qz) != len(qv):
        return "khong_ghep", v
    sua = []   # (span, số mới) các chỗ cần chia đôi
    for (za, zb), m in zip(qz, qv):
        va, vb = _f(m.group(1)), (_f(m.group(2)) if m.group(2) else None)
        if (zb is None) != (vb is None):
            return "lech_khac", v
        dung = _gan(va, za / 2) and (zb is None or _gan(vb, zb / 2))
        gu = _gan(va, za) and (zb is None or _gan(vb, zb))
        if dung:
            continue
        if not gu:
            return "lech_khac", v
        sua.append((m.span(1), za / 2))
        if zb is not None:
            sua.append((m.span(2), zb / 2))
    if not sua:
        return "ok", v
    for (a, b), x in sorted(sua, reverse=True):
        v = v[:a] + _fmt(x) + v[b:]
    return "da_sua", v


# ---------------------------------------------------------------- cảnh báo đơn vị (chỉ gắn cờ)
_NUM_VI = r"\d+(?:[.,]\d+)?"
CUN_ZH = re.compile(rf"({_SO})\s*寸")
LEN_VI = re.compile(rf"({_NUM_VI})\s*(?:cm|mm|centimet|xentimet|mét|m)\b", re.I)
INCH_VI = re.compile(rf"({_NUM_VI})\s*(?:inch|inches|″|\")", re.I)
TAC_VI = re.compile(r"\b(?:tấc|thốn)\b", re.I)
# 两 làm đơn vị (50 g): đứng sau số, không phải "两个/两件/一两天"... (từ chỉ số lượng, thời gian)
LIANG_ZH = re.compile(rf"(?:({_SO})|([一二三四五六七八九十半]{{1,2}}))\s*两(?![个件天种双张条只套位次年月日对根块把片颗人瓶])")
WEIGHT_VI = re.compile(rf"({_NUM_VI})\s*(g|gram|gam|kg)\b", re.I)
LANG_VI = re.compile(r"\b(?:lạng|lượng|tael)\b", re.I)   # lạng của Việt = 100 g, 两 của Trung = 50 g: dễ nhầm
VAN_ZH = re.compile(rf"({_SO})\s*万")
VAN_VI = re.compile(r"(\d+(?:\.\d+)?)\s*(nghìn|ngàn|triệu|vạn)?", re.I)
TY_ZH = re.compile(rf"({_SO})\s*丝(?![袜绒绸带线网印巾])")
TY_SAI = re.compile(r"\b(?:lụa|tơ)\b", re.I)
TY_DUNG = re.compile(r"\b(?:zem|dem)\b|mm|µm|μm|micron", re.I)


def _tap(rx, s):
    return {_f(m.group(1)) for m in rx.finditer(s)}


def _co(x, tap):
    return any(_gan(x, y) for y in tap)


def _van_dung(n, v):
    """Bản Việt có biểu diễn đúng của n 万 (n × 10.000)?"""
    s = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", v)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    for m in VAN_VI.finditer(s):
        x, w = float(m.group(1)), (m.group(2) or "").lower()
        if (not w and _gan(x, n * 10000)) or (w in ("nghìn", "ngàn") and _gan(x, n * 10)) \
                or (w == "triệu" and _gan(x, n / 100)) or (w == "vạn" and _gan(x, n)):
            return True
    return False


def canh_bao_don_vi(z, v):
    """Mã cảnh báo đơn vị của một giá trị (danh sách, rỗng nếu không có). Chỉ để gắn cờ cho người xem, không sửa, không loại.

      CB_cun    N寸 mà bản Việt viết cùng số N theo cm/mm/m hoặc dịch "tấc/thốn" (寸 trong dữ liệu này chủ yếu là inch)
      CB_luong  N两 (= 50 N gram) mà bản Việt viết N gram, hoặc dùng "lạng/lượng/tael" (lạng của Việt = 100 g)
      CB_van    N万 (= N × 10.000) mà bản Việt giữ nguyên số N, không có dạng đúng (N×10.000, N×10 nghìn, N/100 triệu, N vạn)
      CB_ty     N丝 (độ dày 0,01 mm, dùng cho túi nhựa) mà bản Việt dịch nghĩa "lụa/tơ" thay vì zem/mm
    """
    ma = []
    if "寸" in z:
        zs = _tap(CUN_ZH, z)
        if zs and (any(_co(x, _tap(LEN_VI, v)) and not _co(x, _tap(INCH_VI, v)) for x in zs) or TAC_VI.search(v)):
            ma.append("CB_cun")
    if "两" in z and LIANG_ZH.search(z):
        qs = []
        for m in LIANG_ZH.finditer(z):
            x = _f(m.group(1)) if m.group(1) else han_so(m.group(2))
            if x is not None:
                qs.append(x)
        gram = [(_f(m.group(1)) * (1000 if m.group(2).lower() == "kg" else 1)) for m in WEIGHT_VI.finditer(v)]
        if LANG_VI.search(v) or any(_co(x, gram) and not _co(x * 50, gram) for x in qs):
            ma.append("CB_luong")
    if "万" in z:
        vs = _tap(re.compile(rf"({_NUM_VI})"), re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", v).replace(",", "."))
        if any(not _van_dung(n, v) and _co(n, vs) for n in _tap(VAN_ZH, z)):
            ma.append("CB_van")
    if "丝" in z and TY_ZH.search(z) and TY_SAI.search(v) and not TY_DUNG.search(v):
        ma.append("CB_ty")
    return ma
