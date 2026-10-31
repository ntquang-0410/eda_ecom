"""Rubric v1 + cách dựng lời nhắc và đọc kết quả của judge."""
import json
import re

from cau_hinh import CHIEU, PHIEN_BAN_RUBRIC

HE_THONG = """You are a professional Chinese-Vietnamese e-commerce translation reviewer. You judge the quality of one machine-translated product attribute list. You never translate or rewrite the whole text; you only grade and report errors.

Input: product category, the product title (for context only, do NOT grade it), the SOURCE text and the TRANSLATION. The text is a list of "label: value" attributes separated by "; " or by line breaks. Each message states the translation direction.

Facts about this dataset (do not penalise these):
- Unit conversions such as 斤 -> kg (1 斤 = 0.5 kg), 米 -> m, 寸 -> inch are INTENTIONAL; they are correct if the arithmetic is right.
- Brand names, model codes and size letters may stay unchanged or be transliterated.
- Generic marketplace labels (e.g. "Nơi sản xuất", "产地") may be rendered with any natural equivalent.
- Style differences that do not change meaning are not errors.

Grade six dimensions with integers 1-5:
- sai_nghia (meaning accuracy): 5 every attribute carries the same meaning; 4 trivial nuance; 3 one attribute noticeably off but recoverable; 2 several wrong or one misleading; 1 mostly wrong.
- thieu_y (omission): 5 nothing missing; 4 one trivial item dropped; 3 one real attribute missing; 2 several missing; 1 large parts missing.
- them_y (addition): 5 nothing invented; 4 trivial filler; 3 one invented attribute/claim; 2 several; 1 mostly invented.
- thuat_ngu (terminology): 5 correct domain terms for this category; 4 acceptable; 3 one wrong term; 2 several; 1 pervasive.
- so_don_vi (numbers and units): 5 all numbers/units correct; 3 one number or unit wrong; 1 several wrong or meaning of a measurement lost. If no numbers exist, give 5.
- troi_chay (fluency): 5 natural target language; 4 minor awkwardness; 3 understandable but clumsy; 2 hard to read; 1 unreadable. Leftover untranslated source-language words that are not names/codes count against fluency.

Be strict but fair: do not give 5 to a text that has a real error, and do not invent errors. If you are unsure, say so in ket_luan.

Reply with ONE JSON object and nothing else (no markdown, no explanation outside JSON):
{"sai_nghia":int,"thieu_y":int,"them_y":int,"thuat_ngu":int,"so_don_vi":int,"troi_chay":int,
 "ket_luan":"dat"|"can_xem"|"loi",
 "loi":[{"loai":"sai_nghia|thieu_y|them_y|thuat_ngu|so_don_vi|troi_chay","doan_nguon":"...","doan_dich":"...","goi_y":"..."}]}
"loi" lists ONLY genuine errors that lowered a score (never items you consider acceptable, never limitations of the source text, never "no fix needed" remarks). At most 5 items, most severe first, empty list if none. Keep every field short: doan_nguon and doan_dich at most 60 characters each (quote only the faulty part), goi_y at most 15 words and written in Vietnamese. ket_luan: "dat" = usable as training data; "can_xem" = uncertain or minor issues a human should look at; "loi" = contains an error that should be fixed or removed."""

HE_THONG_TITLE = """You are a professional Chinese-Vietnamese e-commerce translation reviewer. You judge the quality of ONE machine-translated product TITLE. You never translate or rewrite; you only grade and report errors.

Input: product category, the SOURCE title and the TRANSLATION title. Each message states the translation direction.

Facts about product titles (do not penalise these):
- Marketplace titles stack keywords and may look unnatural in the source; a translation that keeps the keyword style is fine.
- Brand names, model codes and size letters may stay unchanged or be transliterated.
- Style differences that do not change meaning are not errors.

Grade six dimensions with integers 1-5:
- sai_nghia (meaning accuracy): 5 same meaning; 4 trivial nuance; 3 one noticeably off but recoverable; 2 several wrong or misleading; 1 mostly wrong.
- thieu_y (omission): 5 nothing missing; 4 one trivial keyword dropped; 3 one real keyword/attribute missing; 2 several; 1 large parts missing.
- them_y (addition): 5 nothing invented; 4 trivial filler; 3 one invented claim, or leaked instruction or placeholder text (e.g. "product code: none, quantity, currency unit"); 2 several; 1 mostly invented.
- thuat_ngu (terminology): 5 correct domain terms; 4 acceptable; 3 one wrong term; 2 several; 1 pervasive.
- so_don_vi (numbers and units): 5 all numbers, units, sizes, codes correct (or none exist); 3 one wrong; 1 several wrong.
- troi_chay (fluency): 5 natural target language; 4 minor awkwardness; 3 clumsy but understandable; 2 hard to read; 1 unreadable. Leftover untranslated source-language words that are not names or codes count against fluency.

Be strict but fair: do not give 5 to a title that has a real error, and do not invent errors. If you are unsure, say so in ket_luan.

Reply with ONE JSON object and nothing else (no markdown, no explanation outside JSON):
{"sai_nghia":int,"thieu_y":int,"them_y":int,"thuat_ngu":int,"so_don_vi":int,"troi_chay":int,
 "ket_luan":"dat"|"can_xem"|"loi",
 "loi":[{"loai":"sai_nghia|thieu_y|them_y|thuat_ngu|so_don_vi|troi_chay","doan_nguon":"...","doan_dich":"...","goi_y":"..."}]}
"loi" lists ONLY genuine errors that lowered a score (never items you consider acceptable). At most 5 items, most severe first, empty list if none. Keep every field short: doan_nguon and doan_dich at most 60 characters each, goi_y at most 15 words and written in Vietnamese. ket_luan: "dat" = usable as training data; "can_xem" = uncertain or minor issues a human should look at; "loi" = contains an error that should be fixed or removed."""

HUONG = {
    # 1688: nguồn gốc là tiếng Trung, bản Việt do mô hình dịch.
    "1688": ("SOURCE is Chinese (original seller text). TRANSLATION is Vietnamese (machine translation).",
             "Chinese", "Vietnamese"),
    # Tiki: nguồn gốc là tiếng Việt, bản Trung do mô hình dịch.
    "tiki": ("SOURCE is Vietnamese (original seller text). TRANSLATION is Chinese (machine translation).",
             "Vietnamese", "Chinese"),
}


def dung_nguoi_dung(hang):
    """Trả về chuỗi lời nhắc người dùng cho một dòng của bản cụm."""
    huong, ten_nguon, ten_dich = HUONG[hang["source"]]
    if hang["source"] == "1688":
        nguon, dich, tieu_de = hang["description_zh"], hang["description_vi"], hang["title_zh"]
    else:
        nguon, dich, tieu_de = hang["description_vi"], hang["description_zh"], hang["title_vi"]
    return (f"Direction: {huong}\nCategory: {hang['category']}\n"
            f"Title (context only, do not grade): {tieu_de}\n\n"
            f"SOURCE ({ten_nguon}):\n{nguon}\n\nTRANSLATION ({ten_dich}):\n{dich}")


def dung_nguoi_dung_title(hang):
    """Lời nhắc cho TITLE: 1688 nguồn là title_zh (bản Việt do máy dịch); Tiki nguồn là title_vi (bản Trung do máy dịch)."""
    huong, ten_nguon, ten_dich = HUONG[hang["source"]]
    nguon, dich = (hang["title_zh"], hang["title_vi"]) if hang["source"] == "1688" else (hang["title_vi"], hang["title_zh"])
    return (f"Direction: {huong}\nCategory: {hang['category']}\n\n"
            f"SOURCE title ({ten_nguon}):\n{nguon}\n\nTRANSLATION title ({ten_dich}):\n{dich}")


# đối tượng chấm -> (lời nhắc hệ thống, hàm dựng lời nhắc người dùng, có chạy luật cứng mô tả không)
# mo_ta      = mô tả của bản CỤM (đoạn nối bằng "; ")
# mo_ta_pair = mô tả của bản PAIR (đoạn nối bằng xuống dòng); luật cứng của mô tả được viết cho định dạng "; " nên bỏ qua
# title      = tiêu đề (giống hệt ở cả hai bản)
def doi_tuong(ten):
    return {"mo_ta": (HE_THONG, dung_nguoi_dung, True),
            "mo_ta_pair": (HE_THONG, dung_nguoi_dung, False),
            "title": (HE_THONG_TITLE, dung_nguoi_dung_title, False)}[ten]


def _lay_json(van_ban):
    """Lấy object JSON đầu tiên trong câu trả lời, chịu được hàng rào ```json và chữ thừa."""
    if not van_ban:
        raise ValueError("tra_loi_rong")
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", van_ban.strip())
    dau = s.find("{")
    if dau < 0:
        raise ValueError("khong_co_json")
    sau, dem, trong_chuoi, thoat = None, 0, False, False
    for i in range(dau, len(s)):
        c = s[i]
        if trong_chuoi:
            if thoat:
                thoat = False
            elif c == "\\":
                thoat = True
            elif c == '"':
                trong_chuoi = False
            continue
        if c == '"':
            trong_chuoi = True
        elif c == "{":
            dem += 1
        elif c == "}":
            dem -= 1
            if dem == 0:
                sau = i
                break
    if sau is None:
        raise ValueError("json_bi_cat")
    return json.loads(s[dau:sau + 1])


def doc_ket_qua(van_ban):
    """Đọc + kiểm tra lời chấm. Ném ValueError nếu sai lược đồ."""
    d = _lay_json(van_ban)
    # Cổng OpenCode Go đã trả message.content là cả chuỗi JSON của message ({"content":"...","role":"assistant",...}):
    # bóc lớp vỏ đó (thấy ở lời gọi thử --kiem-tra 30/10/2026).
    if isinstance(d.get("content"), str) and "sai_nghia" not in d and "role" in d:
        return doc_ket_qua(d["content"])
    kq = {}
    for k in CHIEU:
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not 1 <= int(v) <= 5:
            raise ValueError(f"diem_sai:{k}={v!r}")
        kq[k] = int(v)
    kl = d.get("ket_luan")
    if kl not in ("dat", "can_xem", "loi"):
        raise ValueError(f"ket_luan_sai:{kl!r}")
    kq["ket_luan"] = kl
    loi = d.get("loi", [])
    if not isinstance(loi, list):
        raise ValueError("loi_khong_phai_danh_sach")
    kq["loi"] = [{k: str(x.get(k, ""))[:400] for k in ("loai", "doan_nguon", "doan_dich", "goi_y")}
                 for x in loi[:8] if isinstance(x, dict)]
    return kq


def phien_ban():
    return PHIEN_BAN_RUBRIC
