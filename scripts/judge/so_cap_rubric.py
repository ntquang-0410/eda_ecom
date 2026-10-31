"""Lời nhắc và cách đọc kết quả của bộ SO CẶP MÙ (chọn 1 trong 2 bản dịch Gemma / Qwen).

Nguyên tắc chống thiên vị (có bài thử tự động, xem test_so_cap.py):
 1. MÙ: judge chỉ thấy nguồn + "Translation 1" + "Translation 2"; không có tên mô hình, điểm CometKiwi, id hay bất kỳ cột nào khác
    (hàm dựng lời nhắc chỉ đọc đúng các trường trong DANH_SACH_TRUONG).
 2. ĐỐI XỨNG: mỗi mục luôn hỏi ĐỦ HAI THỨ TỰ (Gemma ở vị trí 1, rồi Qwen ở vị trí 1); hai lời nhắc giống hệt nhau trừ chỗ hoán đổi.
 3. KHÔNG MẶC ĐỊNH: chỉ có kết quả khi hai thứ tự cùng chọn MỘT mô hình; không có luật 'hòa thì lấy Gemma' trong bộ chọn."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rubric as R  # noqa: E402  (dùng lại _lay_json)

PHIEN_BAN = "so_cap_v1"
PHIEN_BAN_NGU_CANH = PHIEN_BAN + "+ngu_canh"          # bản có khối ngữ cảnh sản phẩm (chỉ bộ cụm, 31/10/2026)
DANH_SACH_TRUONG = ("kind", "huong", "category", "nguon", "cand_gemma", "cand_qwen", "ngu_canh")   # TẤT CẢ trường được đưa vào lời nhắc
TIEU_CHI = ["meaning", "completeness", "numbers_codes", "terminology", "fluency", "none"]
TU_CAM = ("gemma", "qwen", "kiwi", "deepseek", "comet")     # không được xuất hiện trong phần khung lời nhắc

HE_THONG = """You are an impartial reviewer of Chinese-Vietnamese e-commerce translations. You will see one SOURCE text and two candidate translations, labelled Translation 1 and Translation 2. Decide which translation is better.

Judge ONLY by these criteria, in this order of importance:
1. meaning: it conveys exactly the same meaning as the source (no wrong, reversed or lost meaning).
2. completeness: nothing from the source is missing and nothing is invented or added (no explanations, no leaked instructions, no placeholder text such as "not available").
3. numbers_codes: numbers, units, model codes, brand names and proper names are preserved or correctly converted.
4. terminology: correct domain terms for the product category.
5. fluency: natural, correct target language; no leftover source-language words (except names and codes).

Rules:
- Do not prefer a translation because it is longer, shorter, more formal, or listed first or second. Position carries no information.
- You do not know who produced either translation and must not guess.
- Minor style differences that do not change meaning are not a reason to prefer one; if both are equally good or equally flawed, answer "tie".
- Unit conversions that are arithmetically correct (e.g. 斤 to kg) are acceptable.
- E-commerce titles may stack keywords; do not penalise that.

Reply with ONE JSON object and nothing else:
{"winner": 1 or 2 or "tie", "criterion": "meaning|completeness|numbers_codes|terminology|fluency|none", "reason": "at most 20 words"}"""

_DOAN_NGU_CANH = """A PRODUCT CONTEXT block may precede the SOURCE: information about the same product in the SOURCE language only (its attribute list and, unless the SOURCE is itself the title, its title). Use it solely to understand the SOURCE text (for example to resolve an ambiguous word). The context is not itself translated or judged, and it does not change the criteria or the rules above."""
HE_THONG_NGU_CANH = HE_THONG.replace("Reply with ONE JSON object", _DOAN_NGU_CANH + "\n\nReply with ONE JSON object")
assert HE_THONG_NGU_CANH != HE_THONG

# ---------------------------------------------------------------- BỘ QUY TẮC v2 (31/10/2026, soạn từ 50 mục "không nhất quán" của tập phát triển)
# Mục tiêu: để judge chọn được trong MỘT lần. Thay vì "so sánh theo cảm nhận", judge phải (1) liệt kê lỗi cụ thể của từng bản, (2) bản ít lỗi
# hơn thắng, (3) không dùng các khác biệt vô hại làm lý do, (4) nếu vẫn tương đương thì khai báo `equivalent` và chọn bản 1 (vị trí được
# ngẫu nhiên hóa ở bên ngoài nên đây là tung đồng xu công bằng, có ghi cờ). Trường JSON đặt theo thứ tự: lỗi -> tương đương -> người thắng.
PHIEN_BAN_V2 = "so_cap_v2"
QUY_TAC_CHINH_SACH = [        # NHỮNG DÒNG NÀY LÀ CHÍNH SÁCH, cần Nhật Anh quyết (mặc định bên dưới do Claude đề xuất, chờ xác nhận)
    "keeping a Latin-script brand, model or product name as in the SOURCE, or writing the SAME brand in the target script (transliteration, official local brand name);",
    "keeping or translating ordinary foreign (for example English) words that already appear in that foreign form in the SOURCE and are not names or codes;",
    "synonyms of equal standing, including a shortened and a full form of the same term, and equally standard attribute labels;",
    "unit and number formatting (for example L and a local unit word, kg and kilogram, decimal point and decimal comma, spacing, punctuation, slashes, brackets);",
]
HE_THONG_V2 = ("""You are an impartial reviewer of Chinese-Vietnamese e-commerce translations. You will see one SOURCE text and two candidate translations, labelled Translation 1 and Translation 2. You must decide which one is better. You may not answer "tie": always choose 1 or 2.

Work in this order and base the decision ONLY on what you find.

STEP 1. List the defects of each translation. Check every key element of the SOURCE one by one (product type, each attribute, each number and unit, brand, model code). Do not claim an omission or an addition unless you can point to the exact words. A defect is one of:
 - WRONG: a wrong or reversed meaning (wrong product type, colour, material, quantity or function), a wrong number, wrong unit conversion or wrong code, or a brand, model or product name turned into a DIFFERENT name (an invented or incorrect transliteration).
 - MISSING: information of the SOURCE that is absent (a feature, attribute, number, brand or code).
 - ADDED: information that is not in the SOURCE and is not supported by the PRODUCT CONTEXT (a new feature, material, product type or nuance). Grammar words, measure words and standard attribute wording are not additions.
 - BROKEN: leaked instructions, placeholder or garbled text, a source word rendered in a wrong language (for example an English word where the target language is required, when the SOURCE word was not already in that foreign form), or unreadable text.
 - TERM: a domain term that is clearly wrong or non-standard for this product category.

STEP 2. The translation with FEWER defects wins. If the defects differ in seriousness, WRONG and BROKEN outweigh MISSING and ADDED, which outweigh TERM.

STEP 3. These differences are NEVER defects and NEVER a reason to prefer a translation:
""" + "\n".join(f" - {r}" for r in QUY_TAC_CHINH_SACH) + """
 - word order that keeps the meaning, and being more or less formal;
 - naturalness or fluency, unless one translation is ungrammatical or hard to read;
 - length, and whether a translation is listed first or second.

STEP 4. If after steps 1 to 3 the two translations are equivalent, set "equivalent" to true and choose Translation 1. Otherwise set "equivalent" to false and choose the translation with fewer or lighter defects.

Reply with ONE JSON object and nothing else, with the fields in exactly this order:
{"defects_1": ["short phrases, at most 4"], "defects_2": ["short phrases, at most 4"], "equivalent": true or false, "winner": 1 or 2, "criterion": "meaning|completeness|numbers_codes|terminology|fluency|none", "reason": "at most 20 words"}""")
_DOAN_NGU_CANH_V2 = """A PRODUCT CONTEXT block may precede the SOURCE: information about the same product in the SOURCE language only (its attribute list and, unless the SOURCE is itself the title, its title). Use it solely to understand the SOURCE text (for example to resolve an ambiguous word). The context is not itself translated or judged, and information that the context supports is not ADDED."""
HE_THONG_V2_NGU_CANH = HE_THONG_V2.replace("Reply with ONE JSON object", _DOAN_NGU_CANH_V2 + "\n\nReply with ONE JSON object")
assert HE_THONG_V2_NGU_CANH != HE_THONG_V2
RUBRIC = "v1"        # đặt bằng set_rubric(); v1 = lời nhắc cũ (có đáp án hòa), v2 = bộ quy tắc mới (bắt buộc chọn)


def set_rubric(ten):
    global RUBRIC
    if ten not in ("v1", "v2"):
        raise ValueError(f"rubric_sai:{ten}")
    RUBRIC = ten


HUONG = {
    "zh->vi": ("SOURCE is Chinese. Both translations are Vietnamese.", "Chinese", "Vietnamese"),
    "vi->zh": ("SOURCE is Vietnamese. Both translations are Chinese.", "Vietnamese", "Chinese"),
}
LOAI = {"title": "a product title",
        "cum_pair": "one product attribute written as 'name: value'",
        "unit_name": "the name (label) of a product attribute",
        "unit_value": "the value of a product attribute"}
THU_TU = {"gq": ("cand_gemma", "cand_qwen"), "qg": ("cand_qwen", "cand_gemma")}     # (vị trí 1, vị trí 2)


def dung_nguoi_dung(muc, thu_tu):
    """Trả về (lời nhắc người dùng, ánh xạ vị trí -> mô hình). Ánh xạ CHỈ dùng nội bộ, không đưa vào lời nhắc."""
    huong, ten_nguon, ten_dich = HUONG[muc["huong"]]
    v1, v2 = THU_TU[thu_tu]
    ngu_canh = muc.get("ngu_canh")
    khoi = f"PRODUCT CONTEXT ({ten_nguon}, for understanding only):\n{ngu_canh}\n\n" if isinstance(ngu_canh, str) and ngu_canh else ""
    prompt = (f"{huong}\nText type: {LOAI[muc['kind']]}\nProduct category: {muc['category']}\n\n{khoi}"
              f"SOURCE ({ten_nguon}):\n{muc['nguon']}\n\n"
              f"Translation 1 ({ten_dich}):\n{muc[v1]}\n\n"
              f"Translation 2 ({ten_dich}):\n{muc[v2]}")
    anh_xa = {1: "gemma" if v1 == "cand_gemma" else "qwen", 2: "gemma" if v2 == "cand_gemma" else "qwen"}
    return prompt, anh_xa


def co_ngu_canh(muc):
    nc = muc.get("ngu_canh")
    return isinstance(nc, str) and bool(nc)


def he_thong(muc):
    if RUBRIC == "v2":
        return HE_THONG_V2_NGU_CANH if co_ngu_canh(muc) else HE_THONG_V2
    return HE_THONG_NGU_CANH if co_ngu_canh(muc) else HE_THONG


def phien_ban_goc():
    """Tiền tố phiên bản của rubric đang dùng (dùng để nhận biết kết quả đã chạy khi chạy tiếp)."""
    return PHIEN_BAN_V2 if RUBRIC == "v2" else PHIEN_BAN


def phien_ban_muc(muc):
    return phien_ban_goc() + ("+ngu_canh" if co_ngu_canh(muc) else "")


def thu_tu_bang_bam(item_id):
    """Thứ tự cho chế độ MỘT LƯỢT: ngẫu nhiên theo băm cố định của item_id (tái lập được; item_id KHÔNG đưa vào lời nhắc).
    Một nửa số mục Gemma đứng trước ('gq'), một nửa Qwen đứng trước ('qg'), nên không mô hình nào được lợi vì vị trí."""
    import hashlib
    return "gq" if int(hashlib.md5(("thu_tu:" + str(item_id)).encode("utf-8")).hexdigest(), 16) % 2 == 0 else "qg"


def doc_ket_qua_v2(van_ban):
    """Đọc câu trả lời của rubric v2: bắt buộc winner là 1 hoặc 2 (không có 'tie')."""
    d = R._lay_json(van_ban)
    w = d.get("winner")
    if isinstance(w, str):
        w = w.strip()
        w = int(w) if w in ("1", "2") else w
    if w not in (1, 2):
        raise ValueError(f"winner_sai:{w!r}")
    tc = d.get("criterion", "none")
    if tc not in TIEU_CHI:
        raise ValueError(f"criterion_sai:{tc!r}")
    tuong_duong = d.get("equivalent", False)
    if isinstance(tuong_duong, str):
        tuong_duong = tuong_duong.strip().lower() == "true"
    dem = lambda k: len(d.get(k) or []) if isinstance(d.get(k), list) else 0      # noqa: E731
    return {"winner": w, "criterion": tc, "reason": str(d.get("reason", ""))[:200], "tuong_duong": bool(tuong_duong),
            "so_loi_1": dem("defects_1"), "so_loi_2": dem("defects_2")}


def doc_ket_qua_hien_hanh(van_ban):
    return doc_ket_qua_v2(van_ban) if RUBRIC == "v2" else doc_ket_qua(van_ban)


def doc_ket_qua(van_ban):
    d = R._lay_json(van_ban)
    w = d.get("winner")
    if isinstance(w, str):
        w = w.strip().lower()
        w = int(w) if w in ("1", "2") else w
    if w not in (1, 2, "tie"):
        raise ValueError(f"winner_sai:{w!r}")
    tc = d.get("criterion", "none")
    if tc not in TIEU_CHI:
        raise ValueError(f"criterion_sai:{tc!r}")
    return {"winner": w, "criterion": tc, "reason": str(d.get("reason", ""))[:200]}


def gop(chon1, chon2):
    """Gộp hai lượt (đã quy về tên mô hình hoặc 'hoa_that'). KHÔNG có mặc định: lệch nhau thì 'khong_nhat_quan'."""
    if chon1 == chon2:
        return chon1
    return "khong_nhat_quan"
