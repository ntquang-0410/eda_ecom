# Bước 3 của bộ 4 (laptop): ghép kết quả hiệu đính cả cụm về từng cặp, theo luật chặt (cùng tinh thần ket_qua_v2 của bộ 3):
#   - mục KHOÁ luôn lấy bản khoá (mô hình lỡ xuất lại cũng bỏ qua);
#   - mỗi trường mô hình xuất ra phải qua TOÀN BỘ luật cứng của bộ 1 (song_ngu_1.luat_kiem.kiem_mot: rỗng, sót chữ Hán, lời thừa, lộ ngữ
#     cảnh, lặp từ, lệch số, 斤 viết "cân", mất mã, lệch số mục danh sách, mất dấu che) và luật độ dài (cùng khoảng Việt/Trung của bộ 1),
#     cộng hai luật thêm cho việc hiệu đính:
#       latin_moi: bản sửa có từ Latin lạ (không phải âm tiết Việt, không có trong chữ Trung, không có trong bản nháp, không phải từ quen như
#                  kg, cotton) mà bản nháp không có: mô hình chèn tiếng Anh hoặc pinyin;
#       mat_viet:  bản sửa còn chưa tới nửa số từ Việt có dấu của bản nháp (bản nháp có từ 2 từ): mô hình quay về chữ Latin của nguồn
#                  hoặc phiên âm (giống luật bo_chu_han của bộ 3);
#     trượt thì LÙI VỀ BẢN NHÁP; không có nháp thì để trống (bước hoàn tất quyết định có dùng bản dịch mới hay không);
#   - bản mô hình giống bản nháp sau chuẩn hoá (hoa thường, kiểu đặt dấu, khoảng trắng) thì giữ NGUYÊN CHỮ bản nháp;
#   - cụm sai cấu trúc (sau một lần hỏi lại): mọi trường của cụm lùi về bản nháp;
#   - chọn hai mô hình: A (Gemma) nếu qua luật, không thì B (Qwen) nếu qua luật, không thì nháp.
import json
import re
import unicodedata

import pandas as pd

from song_ngu_1.luat_kiem import do_dai_bat_thuong, kiem_mot, latin_la
from song_ngu_1.van_ban import chuan_hoa
from song_ngu_3.ket_qua import doc_lo

from . import cau_hinh as C

VIET_DAU = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
HAN = re.compile("[一-鿿]+")
NGUONG_SUA_NHIEU = (3, 0.6)
# Từ mượn quen trong tin đăng bán hàng tiếng Việt: không tính là "từ Latin lạ" ở luật latin_moi (thêm 24/10/2026 sau lần thử 200 sản phẩm:
# luật bắt oan camera, kaki, caro, satin, inox, socola, axit... ở bản Gemma). Chỉ liệt kê từng từ; cụm tiếng Anh (memory foam, flagship,
# dropshipping) vẫn bị bắt.
LATIN_VAY = set("""
camera kaki khaki caro satin inox socola chocolate vani matcha mousse mocha cacao latte caramel pudding cookie snack jelly siro
acid axit amino canxi kali natri oxy retinol niacinamide peptide ceramide keratin salicylic hyaluronic glycolic lactic argan jojoba
serum toner lotion cushion mascara eyeliner primer spf latex acrylic spandex lycra rayon modal viscose linen cashmere flannel fleece
jacquard denim jean jeans kraft chiffon voan oxford canvas tweed nhung raglan polo hoodie legging jogger blazer cardigan bomber
sneaker sandal tote website online laptop tablet smartphone adapter hdmi gps sim nano mah watt volt ampe zircon totem pima flyknit
amin propionat sorbat benzoat citrat glutamat cacbonat photphat sunfat lactat xanthan pectin glucose fructose sucrose maltose lecithin
astaxanthin
""".split())
# Thêm 26/10/2026 (anh duyệt nhóm A+B sau khi đo trên cả bộ: trong 977 trường Gemma trượt latin_moi, nhiều từ bị bắt oan). Nhóm A = từ Việt hoặc
# từ vay quen trong tin đăng (vali, balo, gile là tiếng Việt hẳn); nhóm B = tên nền tảng/hãng. Cụm tiếng Anh (trend, dropshipping, flagship,
# memory foam, livestream, streaming, link) KHÔNG thêm: vẫn bị bắt như quyết định cũ.
LATIN_VAY |= set("""
vali balo gile cskh navy macaron nude retro loafer velcro bodysuit tencel americano truffle arbutin fullerene laser
amazon aliexpress tmall airoha
""".split())


def pinyin_nguon(zh):
    """Mọi cách ghép pinyin (viết liền, chữ thường) của các đoạn chữ Hán liên tiếp dài 1-6 chữ trong nguồn: phiên âm đúng tên riêng của
    nguồn (碧然堂 -> birantang) không tính là từ Latin lạ; phiên âm sai (富安娜 -> fuganna) vẫn bị bắt."""
    from pypinyin import lazy_pinyin
    out = set()
    for doan in HAN.findall(zh or ""):
        py = lazy_pinyin(doan)
        for a in range(len(py)):
            for b in range(a + 1, min(a + 6, len(py)) + 1):
                out.add("".join(py[a:b]).lower())
    return out


def doc_khoang(file_the=C.FILE_THE_BO_1):
    """Khoảng tỉ lệ độ dài Việt/Trung theo nhóm độ dài của bộ 1 (luật do_dai), đọc từ thẻ nguồn gốc của bản đã công bố."""
    k = json.loads(file_the.read_text(encoding="utf-8"))["khoang_do_dai_viet_tren_trung"]
    return {int(a): tuple(b) for a, b in k.items()}


def giong(a, b):
    return isinstance(a, str) and isinstance(b, str) and chuan_hoa(a) == chuan_hoa(b)


def tu_viet(s):
    return sum(1 for t in unicodedata.normalize("NFC", s if isinstance(s, str) else "").split() if VIET_DAU.search(t))


def luat_truong(zh_goc, zh_gui, vi, nhap, khoang):
    """Danh sách mã trượt của một trường mô hình xuất ra (rỗng = qua). nhap không phải chữ (None, NaN đọc từ parquet) = không có nháp."""
    nhap = nhap if isinstance(nhap, str) else None
    tr = list(kiem_mot(zh_goc, zh_gui, vi)[0])
    if isinstance(vi, str) and vi.strip() and do_dai_bat_thuong(len(zh_gui), len(vi), khoang):
        tr.append("do_dai")
    if isinstance(vi, str) and vi.strip():
        moi = set(latin_la(zh_gui, vi)) - set(latin_la(zh_gui, nhap) if isinstance(nhap, str) else []) - \
            set(re.findall(r"[a-z]{3,}", unicodedata.normalize("NFC", nhap or "").lower())) - LATIN_VAY
        if moi:
            moi -= pinyin_nguon(zh_gui)
        if moi:
            tr.append("latin_moi")
        n = tu_viet(nhap)
        if n >= 2 and tu_viet(vi) * 2 < n:
            tr.append("mat_viet")
    return sorted(set(tr))


def chon_truong(khoa, nhap, ung_vien, zh_goc, zh_gui, khoang):
    """ung_vien: danh sách (tên nguồn, bản) theo thứ tự ưu tiên. Trả về (bản cuối, nguồn, luật trượt của ứng viên đầu có bản).
    Nguồn: khoa | giu_nhap | sua_A | sua_B | dich_moi_A | dich_moi_B | lui_nhap (ứng viên trượt hết) | nhap_khong_co_ban_mo_hinh |
    khong_co (không nháp, không ứng viên qua luật)."""
    if isinstance(khoa, str):
        return khoa, "khoa", []
    truot_dau = None
    for ten, z in ung_vien:
        if not isinstance(z, str):
            continue
        tr = luat_truong(zh_goc, zh_gui, z, nhap, khoang)
        if truot_dau is None:
            truot_dau = tr
        if not tr:
            if isinstance(nhap, str):
                return (nhap, "giu_nhap", truot_dau) if giong(z, nhap) else (z, f"sua_{ten}", truot_dau)
            return z, f"dich_moi_{ten}", truot_dau
    if isinstance(nhap, str):
        return nhap, ("lui_nhap" if truot_dau else "nhap_khong_co_ban_mo_hinh"), truot_dau or []
    return None, "khong_co", truot_dau or []


def gan_mo_hinh(P, KQ, k):
    """Thêm cột bản mô hình k (n_k, v_k, c_k) và trạng thái cụm (tt_k: ok | loi_cau_truc | chua_dich)."""
    kq = KQ.set_index("ma_cum")
    cache, out_n, out_v, out_c, tt = {}, [], [], [], []
    for ma, i in zip(P.ma_cum, P.i):
        if ma not in cache:
            if ma not in kq.index:
                cache[ma] = ("chua_dich", {})
            elif kq.at[ma, "trang_thai"] != "ok":
                cache[ma] = ("loi_cau_truc", {})
            else:
                cache[ma] = ("ok", json.loads(kq.at[ma, "ket_qua"]))
        t, d = cache[ma]
        o = d.get(str(i), {})
        out_n.append(o.get("n"))
        out_v.append(o.get("v"))
        out_c.append(o.get("c", "") if o else None)
        tt.append(t)
    P[f"n_{k}"], P[f"v_{k}"], P[f"c_{k}"], P[f"tt_{k}"] = out_n, out_v, out_c, tt
    return P


def lap_bang(P, thu_muc_kaggle, khoang=None):
    """P: cap_vao.parquet (có thể đã lọc theo mẫu thử). thu_muc_kaggle: dict {"A": thư mục Gemma, "B": thư mục Qwen}, hoặc chỉ {"A": ...}
    (anh chốt 25/10/2026: cả bộ chỉ chạy Gemma; khi đó chỉ có cách chọn "A")."""
    khoang = khoang or doc_khoang()
    P = P.copy()
    co = [k for k in C.MO_HINH if k in thu_muc_kaggle]
    assert "A" in co, "thiếu kết quả Gemma (A)"
    for k in co:
        P = gan_mo_hinh(P, doc_lo(thu_muc_kaggle[k], C.MO_HINH[k]), k)
    cac_cach = (("A", ["A"]), ("B", ["B"]), ("AB", ["A", "B"])) if "B" in co else (("A", ["A"]),)
    for cach, uv in cac_cach:
        for f, khoa, nhap, zg, zs in (("n", "n_khoa", "n_nhap", "name_zh", "n_gui"), ("v", "v_khoa", "v_nhap", "gia_tri_zh", "v_gui")):
            kq = [chon_truong(kh, nh, [(x, r[f"{f}_{x}"]) for x in uv], g, s, khoang)
                  for kh, nh, g, s, r in zip(P[khoa], P[nhap], P[zg], P[zs], P[[f"{f}_{x}" for x in uv]].to_dict("records"))]
            P[f"{f}_cuoi_{cach}"] = [x[0] for x in kq]
            P[f"{f}_nguon_{cach}"] = [x[1] for x in kq]
            P[f"{f}_truot_{cach}"] = ["|".join(x[2]) for x in kq]
    # luật riêng từng mô hình (kể cả khi không được chọn), để so hai mô hình
    for k in co:
        for f, nhap, zg, zs in (("n", "n_nhap", "name_zh", "n_gui"), ("v", "v_nhap", "gia_tri_zh", "v_gui")):
            P[f"{f}_luat_{k}"] = ["|".join(luat_truong(g, s, z, nh, khoang)) if isinstance(z, str) and not isinstance(kh, str) else ""
                                  for z, nh, g, s, kh in zip(P[f"{f}_{k}"], P[nhap], P[zg], P[zs], P[f"{f}_khoa"])]
    sua = P.v_nguon_A.str.startswith("sua") | P.n_nguon_A.str.startswith("sua")
    P["sua_khong_ly_do"] = sua & (P.c_A.fillna("").str.strip() == "")
    co_nhap = P.groupby("ma_cum").apply(lambda g: int(g.n_nhap.notna().sum() + g.v_nhap.notna().sum()), include_groups=False)
    so_sua = P.assign(s=sua.astype(int)).groupby("ma_cum").s.sum()
    nhieu = {m for m in co_nhap.index if so_sua.get(m, 0) >= NGUONG_SUA_NHIEU[0] and so_sua.get(m, 0) > NGUONG_SUA_NHIEU[1] * co_nhap[m]}
    P["sua_nhieu"] = P.ma_cum.isin(nhieu)
    return P


def tom_tat(P):
    out = {"cap": int(len(P)), "cum": int(P.ma_cum.nunique()), "san_pham": int(P.product_id.nunique())}
    for k in [k for k in C.MO_HINH if f"tt_{k}" in P.columns]:
        out[k] = {"cum_theo_trang_thai": P.drop_duplicates("ma_cum")[f"tt_{k}"].value_counts().to_dict(),
                  "gia_tri_truot_luat": pd.Series([x for s in P[f"v_luat_{k}"] if s for x in s.split("|")]).value_counts().to_dict(),
                  "ten_truot_luat": pd.Series([x for s in P[f"n_luat_{k}"] if s for x in s.split("|")]).value_counts().to_dict()}
    for cach in [c for c in ("A", "B", "AB") if f"v_nguon_{c}" in P.columns]:
        out[f"chon_{cach}"] = {"ten_nguon": P[f"n_nguon_{cach}"].value_counts().to_dict(),
                               "gia_tri_nguon": P[f"v_nguon_{cach}"].value_counts().to_dict()}
    out["sua_khong_ly_do"] = int(P.sua_khong_ly_do.sum())
    out["cum_sua_nhieu"] = int(P[P.sua_nhieu].ma_cum.nunique())
    return out
