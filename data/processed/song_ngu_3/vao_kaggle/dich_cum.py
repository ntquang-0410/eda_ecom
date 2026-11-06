# -*- coding: utf-8 -*-
"""Bộ song ngữ 3 (Kaggle): dịch CẢ CỤM thuộc tính của một sản phẩm Tiki (Việt -> Trung) trong MỘT lần gọi, để mô hình thấy các thuộc
tính khác của cùng sản phẩm. Dùng lại phần nạp mô hình và gọi API của dich.py (nằm cùng thư mục trong Dataset).

Vào: cum_can_dich.parquet (một dòng một cụm: muc_json = danh sách {"i", "n", "v"}). Mô hình phải trả về MỘT mảng JSON đúng các mục đó,
cùng số thứ tự, mỗi mục {"i", "n" (tên Trung), "v" (giá trị Trung)}. Kiểm cấu trúc: đọc được JSON, đủ và đúng số thứ tự, không rỗng.
Sai cấu trúc hoặc còn sót tiếng Việt thì hỏi lại MỘT lần (lượt hội thoại thứ hai), giữ bản tốt hơn. Ghi từng lô ra
<ra>/<mo_hinh>/lo_*.parquet; chạy tiếp được (bỏ qua cụm đã có, kể cả trong các thư mục --cu). Gọi API lỗi thì không ghi, lần sau dịch lại.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dich as D  # noqa: E402

PHIEN_BAN_CAU_LENH = "cum-vi_zh-v1-2026-10-23"
DANH_MUC_ZH = {"food": "食品", "home": "家居用品", "beauty": "美妆护肤", "mother_baby": "母婴用品", "electronics": "数码电子",
               "auto": "汽车用品"}
VIET_DAU = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
HAN = re.compile("[㐀-䶿一-鿿豈-﫿]")

HE_THONG_CUM = (
    "You translate the attribute list of ONE product from the Vietnamese e-commerce site Tiki.vn into Simplified Chinese, "
    "in the style of Chinese e-commerce listings (Taobao, 1688).\n"
    "You receive a JSON array. Each item has \"i\" (item number), \"n\" (attribute name) and \"v\" (attribute value). All items describe "
    "the same product: use the other items as context, for example to understand what a colour, size, part or short word refers to.\n"
    "Reply with ONLY a JSON array containing exactly the same items in the same order. Each item: {\"i\": the same number, \"n\": the "
    "Chinese translation of n, \"v\": the Chinese translation of v}. No markdown, no code fence, no explanation.\n"
    "Rules:\n"
    "- Translate both n and v of every item. Never merge, split, skip, reorder or add items, and never move text from one item to another.\n"
    "- Use Simplified Chinese characters only. Translate EVERY Vietnamese word: n must contain no Vietnamese word or Vietnamese letter "
    "with diacritics.\n"
    "- Also translate ordinary English words (materials, sizes, colours, nutrients, countries, cities), for example Nylon -> 尼龙, "
    "Free Size -> 均码, PROTEIN -> 蛋白质, Taiwan -> 台湾. Keep unchanged only brand names, product line and model names, model codes, "
    "numbers, units, letter sizes (S, M, XL) and tokens in brackets. Never translate or transliterate a Latin-letter brand name.\n"
    "- Vietnamese place names use their standard Chinese names (Hà Nội -> 河内, Việt Nam -> 越南, Đà Lạt -> 大叻). Medicinal herbs and "
    "plants use their standard Chinese names (Thiên niên kiện -> 千年健). If you are not sure of the Chinese name of a small place, "
    "a herb, a plant or a local dish, keep its Vietnamese name exactly as written instead of guessing (example: 80% Robusta Cầu Đất -> "
    "80% Cầu Đất 罗布斯塔咖啡).\n"
    "- Keep numbers exactly. A list separated by \", \" stays a list with the same number of items in the same order, separated by \", \".\n"
    "- v must not start with the attribute name or a label such as \"颜色：\" unless the Vietnamese value itself starts with that label.\n"
    "Example input:\n"
    '[{"i": 1, "n": "Màu sắc", "v": "Đen, Trắng ngà"}, {"i": 2, "n": "Hạn sử dụng", "v": "24 tháng kể từ ngày sản xuất"}]\n'
    "Example output:\n"
    '[{"i": 1, "n": "颜色", "v": "黑色, 米白色"}, {"i": 2, "n": "保质期", "v": "自生产之日起24个月"}]')


# ---------------- v2 (anh chọn 23/10/2026): tên sản phẩm làm ngữ cảnh + mục khoá + hiệu đính bản nháp ----------------
PHIEN_BAN_CAU_LENH_V2 = "cum-vi_zh-v2-2026-10-23"
HE_THONG_CUM_V2 = (
    "You are a professional e-commerce translator. You receive the attribute list of ONE product from the Vietnamese site Tiki.vn and "
    "must produce its Simplified Chinese version, in the style of Chinese e-commerce listings (Taobao, 1688).\n"
    "Input: the product title (context only, never translate it) and a JSON array of items. Each item has \"i\" (item number), "
    "\"n\" (attribute name, Vietnamese) and \"v\" (attribute value, Vietnamese), and may also have:\n"
    "- \"n_zh\" / \"v_zh\": the FIXED Chinese translation of that field, already reviewed by a person. Never output these fields and "
    "never change them; use them only as context.\n"
    "- \"draft_n\" / \"draft_v\": a DRAFT Chinese translation of that field, made WITHOUT seeing the rest of the product.\n"
    "Your task, for every field that is not fixed:\n"
    "- If it has a draft: check the draft against the Vietnamese and against the WHOLE product (the title and all other items). Keep "
    "the draft EXACTLY as it is when it is correct. Change it ONLY when: the context shows its meaning is wrong (for example a colour, "
    "size, part, flavour, material or product type that the title or the other items make clear); it leaves Vietnamese words or "
    "ordinary English words untranslated (materials, sizes, nutrients, countries, for example Nylon -> 尼龙, Free Size -> 均码, "
    "PROTEIN -> 蛋白质, Taiwan -> 台湾); it writes the gram unit as \"gr\" or \"gram\" (write \"g\"); or it drops or adds meaning. "
    "Never rewrite a correct draft only to change the style or word choice.\n"
    "- If it has no draft: translate it.\n"
    "Output ONLY a JSON array, with no markdown, no code fence and no explanation. Output one object for each item that has at least one "
    "non-fixed field, in the same order: {\"i\": item number, \"n\": Chinese name (only when the name is not fixed), \"v\": Chinese value "
    "(only when the value is not fixed), \"c\": \"\" when you kept the draft(s) unchanged, otherwise a short English reason for the change}.\n"
    "Rules for every field you output:\n"
    "- Simplified Chinese only. No Vietnamese word and no Vietnamese letter with diacritics, except the name of a small place, a herb, "
    "a plant or a local dish whose Chinese name you are not sure of: keep that name exactly as written instead of guessing.\n"
    "- Keep numbers, units, model codes, letter sizes (S, M, XL) and Latin-letter brand names exactly. Never translate or "
    "transliterate a Latin-letter brand name.\n"
    "- A list separated by \", \" stays a list with the same number of items, in the same order, separated by \", \".\n"
    "- Do not put the attribute name or a label before a value unless the Vietnamese value itself starts with it. Do not add "
    "information that is not in the Vietnamese.\n"
    "Example input items:\n"
    '[{"i": 1, "n": "Thương hiệu", "v": "Andes", "n_zh": "品牌", "v_zh": "Andes"}, '
    '{"i": 2, "n": "Loại", "v": "Mũ bảo hiểm 3/4", "n_zh": "类型", "draft_v": "3/4头盔"}, '
    '{"i": 3, "n": "Màu sắc", "v": "Hồng, Đen nhám", "n_zh": "颜色", "draft_v": "红色, 磨砂黑"}]\n'
    "Example output:\n"
    '[{"i": 2, "v": "3/4头盔", "c": ""}, {"i": 3, "v": "粉色, 磨砂黑", "c": "Hồng is pink (粉色), not red"}]')


def mo_ta_can(can):
    return "; ".join(f"item {i}: {', '.join(f)}" for i, f in can.items())


def cau_lenh_cum_v2(cat, tieu_de, muc_json, can_json, ten_khac="", phan=0, so_phan=1):
    dm = DANH_MUC_ZH.get(cat, cat)
    can = json.loads(can_json)
    u = f"Product category: {dm}.\nProduct title (context only, do not translate): {tieu_de or '(none)'}\n"
    if so_phan > 1:
        u += (f"This is part {phan + 1} of {so_phan} of this product's attributes (the product is long). Attribute names in the other "
              f"parts (context only): {ten_khac}\n")
    u += "Items (JSON):\n" + json.dumps(json.loads(muc_json), ensure_ascii=False) + "\n"
    u += f"Output exactly {len(can)} objects, with exactly these fields besides \"i\" and \"c\": {mo_ta_can(can)}."
    return [{"role": "system", "content": HE_THONG_CUM_V2}, {"role": "user", "content": u}]


def phan_tich_v2(text, can):
    """can: {"3": ["v"], "13": ["n"], ...}. Bóc câu trả lời thành {i: {"n"?, "v"?, "c"}} đủ mọi mục và trường phải xuất. Mục thừa (mục
    toàn khoá) bị bỏ qua và đếm ở khoá "_thua". Trả về (dict, "") hoặc (None, lý do)."""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b <= a:
        return None, "khong_co_mang_json"
    try:
        x = json.loads(s[a:b + 1])
    except Exception:
        return None, "json_hong"
    if not isinstance(x, list):
        return None, "khong_phai_mang"
    out, thua = {}, 0
    for m in x:
        if not isinstance(m, dict):
            return None, "muc_khong_phai_doi_tuong"
        try:
            i = str(int(m.get("i")))
        except Exception:
            return None, "so_thu_tu_hong"
        if i not in can:
            thua += 1
            continue
        if i in out:
            return None, f"trung_muc_{i}"
        o = {"c": m.get("c") if isinstance(m.get("c"), str) else ""}
        for f in can[i]:
            g = m.get(f)
            if not isinstance(g, str) or not g.strip():
                return None, f"muc_{i}_thieu_{f}"
            o[f] = g.strip()
        out[i] = o
    thieu = [i for i in can if i not in out]
    if thieu:
        return None, f"thieu_muc_{thieu[0]}"
    out["_thua"] = thua
    return out, ""


def muc_sot_viet_v2(out, can):
    return [i for i in can if ("n" in can[i] and VIET_DAU.search(out[i]["n"])) or ("v" in can[i] and gia_tri_sot_viet(out[i]["v"]))]


def hoi_lai_cum_v2(msgs, tra_loi, can, ly_do=None, sot=None):
    if ly_do:
        yeu_cau = (f"Your reply was not usable ({ly_do}). Reply again with ONLY the JSON array: exactly {len(can)} objects, with exactly "
                   f"these fields besides \"i\" and \"c\": {mo_ta_can(can)}. No text before or after the array.")
    else:
        yeu_cau = (f"Items {', '.join(sot)} still contain Vietnamese words. Translate them fully into Simplified Chinese (keep Vietnamese "
                   f"only for a small place, herb, plant or dish name you are not sure of). Reply again with the FULL JSON array: exactly "
                   f"{len(can)} objects, fields: {mo_ta_can(can)}.")
    return msgs + [{"role": "assistant", "content": tra_loi or ""}, {"role": "user", "content": yeu_cau}]


def dich_lo_v2(be, lo, lap, tran):
    msgs = [cau_lenh_cum_v2(c, t, m, cj, k, p, s) for c, t, m, cj, k, p, s in
            zip(lo.category, lo.tieu_de, lo.muc_json, lo.can_json, lo.ten_khac, lo.phan, lo.so_phan)]
    can = [json.loads(x) for x in lo.can_json]
    toi_da = [int(min(tran, 512 + 2 * (k + n) + 40 * t)) for k, n, t in zip(lo.so_ky_tu, lo.so_ky_tu_nhap, lo.so_truong_can)]
    ra = [None] * len(lo)
    chay = [j for j in range(len(lo)) if can[j]]
    for j in range(len(lo)):
        if not can[j]:      # cụm toàn mục khoá: không gọi mô hình
            ra[j] = {"tho": "", "ly_do_dung": "khong_can", "muc": {"_thua": 0}, "ly_do": "", "sot": [], "hoi_lai": False, "sua_duoc": False}
    kq1 = be.sinh([msgs[j] for j in chay], [toi_da[j] for j in chay], lap)
    for j, (x, r) in zip(chay, kq1):
        x = D.don_dep(x)
        out, ly = phan_tich_v2(x, can[j]) if r not in ("loi_api", "qua_dai") else (None, r)
        ra[j] = {"tho": x, "ly_do_dung": r, "muc": out, "ly_do": ly, "sot": muc_sot_viet_v2(out, can[j]) if out else [],
                 "hoi_lai": False, "sua_duoc": False}
    lai = [j for j in chay if ra[j]["ly_do_dung"] not in ("loi_api", "qua_dai") and (ra[j]["muc"] is None or ra[j]["sot"])]
    if lai:
        kq2 = be.sinh([hoi_lai_cum_v2(msgs[j], ra[j]["tho"], can[j], ra[j]["ly_do"] or None, ra[j]["sot"]) for j in lai],
                      [toi_da[j] for j in lai], lap)
        for j, (x2, r2) in zip(lai, kq2):
            ra[j]["hoi_lai"] = True
            x2 = D.don_dep(x2)
            if r2 in ("loi_api", "qua_dai"):
                continue
            out2, _ = phan_tich_v2(x2, can[j])
            if out2 is None:
                continue
            sot2 = muc_sot_viet_v2(out2, can[j])
            if ra[j]["muc"] is None or len(sot2) < len(ra[j]["sot"]):
                ra[j].update(tho=x2, ly_do_dung=r2, muc=out2, ly_do="", sot=sot2, sua_duoc=True)
    return ra


def cau_lenh_cum(cat, muc_json, ten_khac="", phan=0, so_phan=1):
    dm = DANH_MUC_ZH.get(cat, cat)
    u = f"Product category: {dm}.\n"
    if so_phan > 1:
        u += (f"This is part {phan + 1} of {so_phan} of this product's attributes (the product is long). Attribute names in the other "
              f"parts (context only, do not translate): {ten_khac}\n")
    muc = json.loads(muc_json)
    u += f"Translate these {len(muc)} items. Reply with a JSON array of exactly {len(muc)} items numbered 1 to {len(muc)}:\n"
    u += json.dumps(muc, ensure_ascii=False)
    return [{"role": "system", "content": HE_THONG_CUM}, {"role": "user", "content": u}]


def phan_tich(text, n):
    """Bóc câu trả lời thành danh sách n mục {"i", "n", "v"} theo đúng thứ tự 1..n. Trả về (danh sách, "") hoặc (None, lý do)."""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b <= a:
        return None, "khong_co_mang_json"
    try:
        x = json.loads(s[a:b + 1])
    except Exception:
        return None, "json_hong"
    if not isinstance(x, list):
        return None, "khong_phai_mang"
    out = {}
    for m in x:
        if not isinstance(m, dict):
            return None, "muc_khong_phai_doi_tuong"
        try:
            i = int(m.get("i"))
        except Exception:
            return None, "so_thu_tu_hong"
        ten, gt = m.get("n"), m.get("v")
        if not isinstance(ten, str) or not isinstance(gt, str):
            return None, f"muc_{i}_thieu_n_hoac_v"
        if i in out:
            return None, f"trung_muc_{i}"
        out[i] = {"i": i, "n": ten.strip(), "v": gt.strip()}
    if sorted(out) != list(range(1, n + 1)):
        return None, f"lech_so_muc_{len(out)}_tren_{n}"
    rong = [i for i in out if not out[i]["n"] or not out[i]["v"]]
    if rong:
        return None, f"muc_rong_{rong[0]}"
    return [out[i] for i in range(1, n + 1)], ""


def gia_tri_sot_viet(v):
    """Giá trị Trung còn sót tiếng Việt: có chữ Việt có dấu mà KHÔNG phải tên riêng được giữ (tên riêng = có chữ Hán, không quá 4 từ Việt,
    mỗi từ viết hoa chữ đầu hoặc in hoa; như luật viet_giu của bộ 2)."""
    tu = [t for t in re.findall(r"[^\s,，;；:：()（）/]+", v) if VIET_DAU.search(t)]
    if not tu:
        return False
    if not HAN.search(v) or len(tu) > 4:
        return True
    return any(not (t.istitle() or t.isupper()) for t in tu)


def muc_sot_viet(muc):
    """Số thứ tự các mục còn sót tiếng Việt: tên có bất kỳ chữ Việt có dấu nào, hoặc giá trị sót theo gia_tri_sot_viet."""
    return [m["i"] for m in muc if VIET_DAU.search(m["n"]) or gia_tri_sot_viet(m["v"])]


def hoi_lai_cum(msgs, tra_loi, n, ly_do=None, sot=None):
    if ly_do:
        yeu_cau = (f"Your reply was not usable ({ly_do}). Reply again with ONLY a JSON array of exactly {n} items numbered 1 to {n}, "
                   "each {\"i\", \"n\", \"v\"}, translating every item, with no text before or after the array.")
    else:
        yeu_cau = (f"Items {', '.join(map(str, sot))} still contain Vietnamese words. Translate them fully into Simplified Chinese (keep "
                   f"Vietnamese only for small place, herb or dish names you are not sure of). Reply again with the FULL JSON array of all "
                   f"{n} items, with no text before or after the array.")
    return msgs + [{"role": "assistant", "content": tra_loi or ""}, {"role": "user", "content": yeu_cau}]


def so_token_ra(so_ky_tu, so_muc, tran):
    return int(min(tran, 512 + 2 * so_ky_tu + 30 * so_muc))


def dich_lo(be, lo, lap, tran):
    """Dịch một lô cụm. Trả về danh sách dict kết quả (một dict một cụm, cùng thứ tự với lo)."""
    msgs = [cau_lenh_cum(c, m, t, p, s) for c, m, t, p, s in zip(lo.category, lo.muc_json, lo.ten_khac, lo.phan, lo.so_phan)]
    n = [int(x) for x in lo.so_muc]
    toi_da = [so_token_ra(k, m, tran) for k, m in zip(lo.so_ky_tu, lo.so_muc)]
    kq1 = be.sinh(msgs, toi_da, lap)
    ra = []
    for j, (x, r) in enumerate(kq1):
        x = D.don_dep(x)
        muc, ly = phan_tich(x, n[j]) if r not in ("loi_api", "qua_dai") else (None, r)
        ra.append({"tho": x, "ly_do_dung": r, "muc": muc, "ly_do": ly, "sot": muc_sot_viet(muc) if muc else [], "hoi_lai": False,
                   "sua_duoc": False})
    can = [j for j, k in enumerate(ra) if k["ly_do_dung"] not in ("loi_api", "qua_dai") and (k["muc"] is None or k["sot"])]
    if can:
        msgs2 = [hoi_lai_cum(msgs[j], ra[j]["tho"], n[j], ra[j]["ly_do"] or None, ra[j]["sot"]) for j in can]
        kq2 = be.sinh(msgs2, [toi_da[j] for j in can], lap)
        for j, (x2, r2) in zip(can, kq2):
            ra[j]["hoi_lai"] = True
            x2 = D.don_dep(x2)
            if r2 in ("loi_api", "qua_dai"):
                continue
            muc2, ly2 = phan_tich(x2, n[j])
            if muc2 is None:
                continue
            sot2 = muc_sot_viet(muc2)
            if ra[j]["muc"] is None or len(sot2) < len(ra[j]["sot"]):
                ra[j].update(tho=x2, ly_do_dung=r2, muc=muc2, ly_do="", sot=sot2, sua_duoc=True)
    return ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", required=True, help="cum_can_dich.parquet")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--mo_hinh", required=True, choices=list(D.MO_HINH))
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf", "gia"])
    ap.add_argument("--chi_mau_thu", action="store_true", help="chỉ dịch các cụm có mau_thu = True (200 sản phẩm thử)")
    ap.add_argument("--song_song", type=int, default=None)
    ap.add_argument("--api_base_url", default="")
    ap.add_argument("--api_model", default="")
    ap.add_argument("--api_key_env", default="")
    ap.add_argument("--api_extra", default="")
    ap.add_argument("--rpm", type=int, default=0)
    ap.add_argument("--toi_da_yeu_cau", type=int, default=0)
    ap.add_argument("--ngan_sach_usd", type=float, default=0.0)
    ap.add_argument("--gio_toi_da", type=float, default=8.0)
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--cu", action="append", default=[], help="thư mục đầu ra của phiên bản trước")
    a = ap.parse_args()
    t0 = time.time()
    cfg = D.MO_HINH[a.mo_hinh]
    if "api" in cfg:
        if a.api_base_url.strip():
            cfg = dict(cfg, base_url=a.api_base_url.strip())
        if a.api_model.strip():
            cfg = dict(cfg, api=a.api_model.strip())
        if a.api_key_env.strip():
            cfg = dict(cfg, key_env=a.api_key_env.strip())
    ra = Path(a.ra) / a.mo_hinh
    ra.mkdir(parents=True, exist_ok=True)
    CUM = pd.read_parquet(a.vao)
    if a.chi_mau_thu:
        CUM = CUM[CUM.mau_thu]
    for k, cu in enumerate(a.cu):
        for f in sorted(Path(cu).glob(f"{a.mo_hinh}/lo_*.parquet")):
            dich = ra / f"lo_cu{k}_{f.name}"
            if not dich.exists():
                shutil.copy2(f, dich)
    da = set()
    for f in ra.glob("lo_*.parquet"):
        da |= set(pd.read_parquet(f, columns=["ma_cum"]).ma_cum)
    con = CUM[~CUM.ma_cum.isin(da)].sort_values("so_ky_tu", kind="stable").reset_index(drop=True)
    v2 = "can_json" in CUM.columns          # bản v2: có mục khoá, bản nháp và tên sản phẩm
    D.log(f"{a.mo_hinh}: tổng {len(CUM):,} cụm, đã có {len(CUM) - len(con):,}, còn {len(con):,} | câu lệnh {'v2' if v2 else 'v1'}")
    the = {"mo_hinh": a.mo_hinh, "phien_ban_cau_lenh": PHIEN_BAN_CAU_LENH_V2 if v2 else PHIEN_BAN_CAU_LENH, "huong": "vi_zh", "loai": "cum",
           "bat_dau": time.strftime("%Y-%m-%d %H:%M:%S"), "tong": int(len(CUM)), "da_co_truoc": int(len(CUM) - len(con)),
           "chi_mau_thu": a.chi_mau_thu, "tham_so": {"temperature": 0.0, "repetition_penalty": cfg["lap"],
                                                    "max_tokens": "min(tran, 512 + 2*ky_tu + 30*so_muc)", "hoi_lai": 1}}
    f_the = Path(a.ra) / f"the_{a.mo_hinh}.json"
    if len(con) == 0:
        the.update(ket_thuc=time.strftime("%Y-%m-%d %H:%M:%S"), ghi_chu="đã đủ")
        f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    if cfg.get("gia_usd_1m") and a.backend != "gia" and a.ngan_sach_usd <= 0:
        raise SystemExit(f"{a.mo_hinh} là mô hình có giá: bắt buộc đặt --ngan_sach_usd.")
    song_song = a.song_song if a.song_song is not None else int(cfg.get("song_song", 8))
    be, ten_be, repo, loi_vllm = D.tai_backend(cfg, a.backend, song_song, json.loads(a.api_extra) if a.api_extra.strip() else None)
    if hasattr(be, "tk"):
        if cfg.get("gia_usd_1m"):
            be.ngan_sach = a.ngan_sach_usd
        be.toi_da_goi = a.toi_da_yeu_cau
        be.rpm = a.rpm
        be.han_chot = t0 + a.gio_toi_da * 3600
    tran = D.TOKEN_RA_API if ten_be == "api" else D.DO_DAI_NGU_CANH
    sha = {"gia": "gia", "api": "api (không có mã phiên bản; ghi ngày gọi ở bat_dau)"}.get(ten_be) or D.sha_mo_hinh(repo)
    the.update(backend=ten_be, repo=repo, sha_repo=sha, loi_vllm=loi_vllm, phien_ban=be.phien_ban,
               rpm=a.rpm, toi_da_yeu_cau=a.toi_da_yeu_cau, ngan_sach_usd=a.ngan_sach_usd)
    D.log("Nạp xong sau", round(time.time() - t0), "giây; backend", ten_be)
    so = [int(m.group(1)) for f in ra.glob("lo_*.parquet") for m in [re.fullmatch(r"lo_(\d+)", f.stem)] if m]
    so_lo = max(so) + 1 if so else 0
    dem = {"xong": 0, "loi_cau_truc": 0, "con_sot_viet": 0, "hoi_lai": 0, "sua_duoc": 0, "loi_api": 0}
    t_dich, het_gio = time.time(), False
    for i in range(0, len(con), a.lo):
        if time.time() - t0 > a.gio_toi_da * 3600:
            het_gio = True
            D.log("Hết giờ cho phép, dừng êm để lưu đầu ra")
            break
        lo = con.iloc[i:i + a.lo]
        kq = dich_lo_v2(be, lo, cfg["lap"], tran) if v2 else dich_lo(be, lo, cfg["lap"], tran)
        ok = [k["ly_do_dung"] != "loi_api" for k in kq]
        dem["loi_api"] += len(ok) - sum(ok)
        if not any(ok):
            the["dung_vi_loi_api"] = getattr(be, "tk", {}).get("loi_cuoi", "") or "cả lô lỗi API"
            D.log("Cả lô đều lỗi API, dừng:", the["dung_vi_loi_api"])
            break
        df = pd.DataFrame({"ma_cum": lo.ma_cum.values,
                           "ket_qua": [json.dumps(k["muc"], ensure_ascii=False) if k["muc"] is not None else "" for k in kq],
                           "trang_thai": ["ok" if k["muc"] is not None else "loi_cau_truc" for k in kq],
                           "ly_do": [k["ly_do"] for k in kq], "so_muc_sot_viet": [len(k["sot"]) for k in kq],
                           "hoi_lai": [k["hoi_lai"] for k in kq], "sua_duoc": [k["sua_duoc"] for k in kq],
                           "ly_do_dung": [k["ly_do_dung"] for k in kq],
                           "tho": [k["tho"][:20000] if (k["muc"] is None or v2) else "" for k in kq], "backend": ten_be})[ok]
        tam = ra / f".tam_{so_lo:05d}.parquet"
        df.to_parquet(tam, index=False)
        tam.rename(ra / f"lo_{so_lo:05d}.parquet")
        so_lo += 1
        dem["xong"] += len(df)
        dem["loi_cau_truc"] += int((df.trang_thai != "ok").sum())
        dem["con_sot_viet"] += int((df.so_muc_sot_viet > 0).sum())
        dem["hoi_lai"] += int(df.hoi_lai.sum())
        dem["sua_duoc"] += int(df.sua_duoc.sum())
        td = dem["xong"] / max(time.time() - t_dich, 1)
        D.log(f"{a.mo_hinh}: xong {dem['xong']:,}/{len(con):,} cụm ({td:.2f} cụm/giây) | sai cấu trúc {dem['loi_cau_truc']} | "
              f"còn sót Việt {dem['con_sot_viet']} | hỏi lại {dem['hoi_lai']}, sửa được {dem['sua_duoc']} | lỗi API {dem['loi_api']}")
        the.update(dem=dem, toc_do_cum_giay=round(td, 3))
        if hasattr(be, "tk"):
            the["api"] = be.tk
        f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    the.update(ket_thuc=time.strftime("%Y-%m-%d %H:%M:%S"), dung_vi_het_gio=het_gio, dem=dem,
               con_lai=int(len(con) - dem["xong"]), giay=round(time.time() - t0))
    if hasattr(be, "tk"):
        the["api"] = be.tk
    f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    D.log("Kết thúc", a.mo_hinh, "| còn lại", len(con) - dem["xong"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
