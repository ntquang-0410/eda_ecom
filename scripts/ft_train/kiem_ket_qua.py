"""Bước B2: đọc kết quả chấm (Gemini dán về dạng văn bản thô, Claude dạng jsonl), kiểm tra với lô đã gửi,
tính tỉ lệ lỗi nặng theo nhóm (Wilson 95%), hiệu chuẩn Gemini bằng mẫu chung với Claude, xét cổng.

Thư mục kiem/ (do lay_mau_kiem.py tạo):  gui_gemini/lo_XXX.csv  <->  ket_qua_gemini/lo_XXX.txt
                                          gui_claude/lo_XXX.jsonl <-> ket_qua_claude/lo_XXX.jsonl
Chạy:  python -I -X utf8 scripts/ft_train/kiem_ket_qua.py             (báo cáo đầy đủ -> bao_cao_kiem.json/.md)
       python -I -X utf8 scripts/ft_train/kiem_ket_qua.py --thu20     (chỉ so lô thử, không kết luận kappa)
       python -I -X utf8 scripts/ft_train/kiem_ket_qua.py --dau 9 1   (kiểm định dấu: A thắng 9, B thắng 1)"""
import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung as C  # noqa: E402

KIEM = C.THU_MUC_TEST / "kiem"
HE_SO_PAIR = 0.048          # tham số công thức ước lượng bản pair cho mô tả
NGUONG_QUA = 0.05
NGUONG_XAM = 0.065
KAPPA_TOI_THIEU = 0.4
SOT_TOI_DA = 0.30
GHI_CHU_PAIR = ("Ước lượng bản pair dựa trên các thuộc tính chung; ở 18,3% sp, bản cụm có thêm thuộc tính "
                "mà bản pair không có.")
RE_ID = re.compile(r"^K\d+$", re.I)
MUC_HOP_LE = {"dat": "dat", "nhe": "nhe", "nang": "nang"}


# ---------- thống kê ----------
def wilson(k, n, z=1.96):
    """Khoảng tin cậy Wilson cho tỉ lệ k/n; n = 0 -> (None, None)."""
    if n <= 0:
        return None, None
    p = k / n
    mau = 1 + z * z / n
    tam = (p + z * z / (2 * n)) / mau
    nua = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / mau
    return max(0.0, tam - nua), min(1.0, tam + nua)


def kappa_cohen(a, b, c, d):
    """a: cả hai nặng, b: chỉ A nặng, c: chỉ B nặng, d: cả hai không nặng. Trả kappa (None nếu không xác định)."""
    n = a + b + c + d
    if n == 0:
        return None
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def kiem_dinh_dau(thang_a, thang_b):
    """Kiểm định dấu chính xác (nhị thức p = 0.5, hai phía), bỏ dòng ngang (không đưa vào đây).
    9/10 thắng -> p ≈ 0.0215."""
    n = thang_a + thang_b
    if n == 0:
        return {"n": 0, "a": thang_a, "b": thang_b, "p_hai_phia": 1.0}
    k = max(thang_a, thang_b)
    duoi = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return {"n": n, "a": thang_a, "b": thang_b, "p_hai_phia": min(1.0, 2 * duoi)}


def uoc_luong_pair(ty_cum_m1, ty_m2_pair, ty_m2_cum, he_so=HE_SO_PAIR):
    """ty_le_pair = ty_le_cum(M1) + he_so * (ty_le_M2_pair - ty_le_M2_cum)."""
    return ty_cum_m1 + he_so * (ty_m2_pair - ty_m2_cum)


# ---------- chuẩn hóa ----------
def bo_dau(s):
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").strip(" \t\"'`*.")


def chuan_muc(s):
    return MUC_HOP_LE.get(bo_dau(s))


def doc_ds(raw):
    """ds_manh_nang -> list[int] hoặc ném ValueError. Chấp nhận '1;3', '1, 3', [1, 3], '[2]', '' / None."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        tok = [str(x) for x in raw]
    else:
        tok = [t for t in re.split(r"[;,\s]+", str(raw).strip().strip("[]() ")) if t]
    kq = []
    for t in tok:
        t = t.strip("[]() ")
        if not t.isdigit():
            raise ValueError(f"ds_manh_nang khong phai so: {raw!r}")
        kq.append(int(t))
    return kq


def dung_phan_xet(loai, so_manh, muc=None, so_nang=None, ds=None, loai_loi="", ghi_chu=""):
    """Chuẩn hóa một dòng kết quả. Trả (phan_xet, loi). phan_xet: {muc | so_nang, ds, loai_loi, ghi_chu}."""
    base = {"loai_loi": str(loai_loi or ""), "ghi_chu": str(ghi_chu or "")}
    if loai in ("title", "manh"):
        m = chuan_muc(muc) if muc is not None else None
        if m is None:
            return None, f"muc khong hop le: {muc!r}"
        return dict(base, muc=m), None
    # mo_ta
    try:
        dsl = doc_ds(ds)
    except ValueError as e:
        return None, str(e)
    so = None
    if so_nang is not None and str(so_nang).strip() != "":
        try:
            so = int(float(str(so_nang).strip()))
        except ValueError:
            return None, f"so_manh_nang khong phai so: {so_nang!r}"
    if so is None and not dsl:
        m = chuan_muc(muc) if muc is not None else None
        if m == "dat":
            so = 0
        elif m == "nhe":
            so = 0
        else:
            return None, "thieu so_manh_nang/ds_manh_nang"
    if so is None:
        so = len(dsl)
    if so < 0 or so != len(dsl):
        return None, f"so_manh_nang={so} khong khop ds_manh_nang={dsl}"
    if len(set(dsl)) != len(dsl):
        return None, f"ds_manh_nang co so lap: {dsl}"
    sai = [k for k in dsl if k < 1 or k > so_manh]
    if sai:
        return None, f"ds_manh_nang chua so ngoai 1..{so_manh}: {sai}"
    return dict(base, so_nang=so, ds=frozenset(dsl)), None


def don_vi_nang(loai, so_manh, px):
    """Tập đơn vị bị chấm nặng: title/manh -> {0} hoặc {}; mo_ta -> tập số thứ tự mảnh."""
    if loai in ("title", "manh"):
        return {0} if px["muc"] == "nang" else set()
    return set(px["ds"])


def danh_sach_don_vi(loai, so_manh):
    return [0] if loai in ("title", "manh") else list(range(1, int(so_manh) + 1))


# ---------- parser ----------
def tach_dong_gemini(text):
    """Văn bản thô -> (list (id, cells, so_dong), dòng bỏ qua). Chịu được tiêu đề lặp, khối ```csv, khoảng trắng, ngoặc kép, bảng markdown."""
    rows, bo_qua = [], []
    for n, dong in enumerate(text.replace("﻿", "").splitlines(), 1):
        s = dong.strip()
        if not s or s.startswith("```"):
            continue
        if s.startswith("|"):
            if set(s) <= set("|-: "):
                continue
            cells = [x.strip() for x in s.strip("|").split("|")]
        else:
            try:
                cells = next(csv.reader([s], skipinitialspace=True))
            except (csv.Error, StopIteration):
                bo_qua.append(s[:80])
                continue
        cells = [x.strip() for x in cells]
        dau = cells[0].strip("\"'`* ") if cells else ""
        if dau.lower() == "id":
            continue                                           # dòng tiêu đề (kể cả lặp)
        if RE_ID.match(dau):
            rows.append((dau.upper(), cells[1:], n))
        else:
            bo_qua.append(s[:80])
    return rows, bo_qua


def dong_gemini_thanh_phan_xet(loai, so_manh, cells):
    if loai in ("title", "manh"):
        if not cells:
            return None, "thieu cot muc"
        return dung_phan_xet(loai, so_manh, muc=cells[0], loai_loi=cells[1] if len(cells) > 1 else "",
                             ghi_chu=",".join(cells[2:]))
    if len(cells) < 2:
        return None, "thieu cot so_manh_nang/ds_manh_nang"
    return dung_phan_xet(loai, so_manh, so_nang=cells[0], ds=cells[1], loai_loi=cells[2] if len(cells) > 2 else "",
                         ghi_chu=",".join(cells[3:]))


def tach_dong_claude(text):
    """jsonl -> (list (id, obj, so_dong), dòng lỗi)."""
    rows, loi = [], []
    for n, dong in enumerate(text.replace("﻿", "").splitlines(), 1):
        s = dong.strip().rstrip(",")
        if not s or s.startswith("```"):
            continue
        try:
            o = json.loads(s)
            if not isinstance(o, dict) or "id" not in o:
                raise ValueError("thieu id")
        except ValueError:
            loi.append(f"dong {n} khong phai JSON hop le: {s[:60]!r}")
            continue
        rows.append((str(o["id"]).strip().upper(), o, n))
    return rows, loi


def dong_claude_thanh_phan_xet(loai, so_manh, o):
    return dung_phan_xet(loai, so_manh, muc=o.get("muc"), so_nang=o.get("so_manh_nang"), ds=o.get("ds_manh_nang"),
                         loai_loi=o.get("loai_loi", ""), ghi_chu=o.get("ghi_chu", ""))


def kiem_lo(ten, ids_gui, meta, rows, loi_cu_phap=()):
    """So kết quả với lô đã gửi. rows: list (id, du_lieu, so_dong) ; trả (ket_qua {id: phan_xet}, loi []).
    Lô có bất kỳ lỗi nào -> ket_qua rỗng (không dùng dòng bịa)."""
    loi = list(loi_cu_phap)
    gui = set(ids_gui)
    dem = {}
    for i, _, _ in rows:
        dem[i] = dem.get(i, 0) + 1
    thieu = sorted(gui - set(dem))
    la = sorted(set(dem) - gui)
    trung = sorted(i for i, k in dem.items() if k > 1)
    if thieu:
        loi.append(f"thieu {len(thieu)} id: {thieu[:8]}")
    if la:
        loi.append(f"{len(la)} id la (khong co trong lo gui): {la[:8]}")
    if trung:
        loi.append(f"{len(trung)} id trung: {trung[:8]}")
    kq = {}
    for i, du, n in rows:
        if i not in gui or dem[i] > 1:
            continue
        m = meta.loc[i]
        px, e = (dong_claude_thanh_phan_xet(m.loai, m.so_manh, du) if isinstance(du, dict)
                 else dong_gemini_thanh_phan_xet(m.loai, m.so_manh, du))
        if e:
            loi.append(f"{i} (dong {n}): {e}")
        else:
            kq[i] = px
    if loi:
        return {}, loi
    return kq, []


def doc_ids_gui(path):
    if path.suffix == ".jsonl":
        return [json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)
        return [row[0] for row in r if row]


def doc_cac_lo(kiem, nguon, meta):
    """Trả (ket_qua {id: phan_xet}, trang_thai [dict theo lô])."""
    gui_dir = kiem / f"gui_{nguon}"
    kq_dir = kiem / f"ket_qua_{nguon}"
    hau_gui, hau_kq = (".csv", ".txt") if nguon == "gemini" else (".jsonl", ".jsonl")
    tong, tt = {}, []
    for f in sorted(gui_dir.glob(f"lo_*{hau_gui}")):
        ten = f.stem
        ids_gui = doc_ids_gui(f)
        t = {"lo": ten, "so_gui": len(ids_gui)}
        p = kq_dir / f"{ten}{hau_kq}"
        if not p.exists():
            t["trang_thai"] = "chua_co"
        else:
            txt = p.read_text(encoding="utf-8-sig")
            if nguon == "gemini":
                rows, bo_qua = tach_dong_gemini(txt)
                loi_cp = []
            else:
                rows, loi_cp = tach_dong_claude(txt)
                bo_qua = []
            kq, loi = kiem_lo(ten, ids_gui, meta, rows, loi_cp)
            t["so_nhan"] = len(rows)
            if bo_qua:
                t["dong_bo_qua"] = bo_qua[:5]
            if loi:
                t["trang_thai"] = "loi"
                t["loi"] = loi
                t["huong_dan"] = f"gui lai lo {ten}"
            else:
                t["trang_thai"] = "ok"
                tong.update(kq)
        tt.append(t)
    return tong, tt


# ---------- tỉ lệ theo nhóm ----------
def ty_le_nhom(kq, meta):
    """Nhóm (huong, loai, bo): title = Nang/số title; mo_ta = Σ mảnh nặng/Σ mảnh; manh = Nang/số mục."""
    nhom = {}
    for i, px in kq.items():
        m = meta.loc[i]
        g = nhom.setdefault((m.huong, m.loai, m.bo), {"so_muc": 0, "don_vi": 0, "nang": 0, "nhe": 0})
        g["so_muc"] += 1
        if m.loai == "mo_ta":
            g["don_vi"] += int(m.so_manh)
            g["nang"] += len(px["ds"])
        else:
            g["don_vi"] += 1
            g["nang"] += px["muc"] == "nang"
            g["nhe"] += px["muc"] == "nhe"
    out = []
    for (h, loai, bo), g in sorted(nhom.items()):
        lo, hi = wilson(g["nang"], g["don_vi"])
        out.append({"huong": h, "loai": loai, "bo": bo, "so_muc": g["so_muc"], "don_vi": g["don_vi"], "nang": int(g["nang"]),
                    "nhe": None if loai == "mo_ta" else int(g["nhe"]),
                    "ty_le": g["nang"] / g["don_vi"] if g["don_vi"] else None, "wilson_lo": lo, "wilson_hi": hi})
    return out


def tim(bang, h, loai, bo):
    for r in bang:
        if (r["huong"], r["loai"], r["bo"]) == (h, loai, bo):
            return r
    return None


def uoc_pair_theo_huong(bang, he_so=HE_SO_PAIR):
    out = []
    for h in ("zh->vi", "vi->zh"):
        cum, c2, p2 = tim(bang, h, "mo_ta", "cum"), tim(bang, h, "manh", "cum"), tim(bang, h, "manh", "pair")
        if not (cum and c2 and p2 and cum["ty_le"] is not None and c2["don_vi"] and p2["don_vi"]):
            continue
        tho = uoc_luong_pair(cum["ty_le"], p2["ty_le"], c2["ty_le"], he_so)
        v = lambda r: r["ty_le"] * (1 - r["ty_le"]) / r["don_vi"]       # noqa: E731
        se = math.sqrt(v(cum) + he_so ** 2 * (v(p2) + v(c2)))
        out.append({"huong": h, "loai": "mo_ta", "bo": "pair_uoc_luong", "he_so": he_so, "ty_le_cum_M1": cum["ty_le"],
                    "ty_le_M2_cum": c2["ty_le"], "ty_le_M2_pair": p2["ty_le"], "ty_le_tho": tho,
                    "ty_le": min(1.0, max(0.0, tho)), "se_xap_xi": se,
                    "khoang_95_xap_xi": [max(0.0, tho - 1.96 * se), min(1.0, tho + 1.96 * se)]})
    return out


# ---------- hiệu chuẩn ----------
def hieu_chuan(g, c, meta):
    """Trên mục chung (M1c + M2c) cả hai đã chấm hợp lệ, gộp mọi nhóm, đơn vị nhị phân 'lỗi nặng'."""
    chung = [i for i in g if i in c and (meta.loc[i].thuoc_M1c or meta.loc[i].thuoc_M2c)]
    a = b = cc = d = 0
    for i in chung:
        m = meta.loc[i]
        ng, nc = don_vi_nang(m.loai, m.so_manh, g[i]), don_vi_nang(m.loai, m.so_manh, c[i])
        for u in danh_sach_don_vi(m.loai, m.so_manh):
            gn, cn = u in ng, u in nc
            a += gn and cn
            b += gn and not cn
            cc += cn and not gn
            d += (not gn) and (not cn)
    n = a + b + cc + d
    kp = kappa_cohen(a, b, cc, d)
    nang_c = a + cc
    out = {"so_muc_chung": len(chung), "so_don_vi": int(n), "ca_hai_nang": int(a), "chi_gemini_nang": int(b),
           "chi_claude_nang": int(cc), "ca_hai_khong": int(d), "kappa": kp,
           "ty_le_bat_dong": (b + cc) / n if n else None,
           "bo_sot": max(0, cc - b) / n if n else 0.0,
           "ty_le_claude_nang_gemini_qua": cc / nang_c if nang_c else None,
           "canh_bao": []}
    if n == 0:
        out["canh_bao"].append("chua co muc chung nao duoc ca hai cham hop le: chua hieu chuan, bo_sot = 0")
    if kp is not None and kp < KAPPA_TOI_THIEU:
        out["canh_bao"].append(f"Gemini khong dang tin: kappa {kp:.3f} < {KAPPA_TOI_THIEU}")
    if out["ty_le_claude_nang_gemini_qua"] is not None and out["ty_le_claude_nang_gemini_qua"] > SOT_TOI_DA:
        out["canh_bao"].append(f"Gemini khong dang tin: bo sot {out['ty_le_claude_nang_gemini_qua']:.1%} > {SOT_TOI_DA:.0%}")
    return out


def xet_cong(ty_le_hc, nguong_qua=NGUONG_QUA, nguong_xam=NGUONG_XAM):
    """ty_le_hc: {ten_nhom: ty_le_hieu_chinh}. Mọi nhóm <= qua -> QUA; có nhóm trong (qua, xam] -> VUNG XAM; > xam -> KHONG QUA."""
    if not ty_le_hc:
        return {"ket_luan": "CHUA_CO_DU_LIEU", "nhom_cao_nhat": None, "ty_le_cao_nhat": None}
    ten, cao = max(ty_le_hc.items(), key=lambda kv: kv[1])
    kl = "QUA" if cao <= nguong_qua else ("VUNG XAM" if cao <= nguong_xam else "KHONG QUA")
    return {"ket_luan": kl, "nhom_cao_nhat": ten, "ty_le_cao_nhat": cao, "nguong_qua": nguong_qua, "nguong_xam": nguong_xam}


# ---------- báo cáo ----------
def tao_bao_cao(kiem, he_so=HE_SO_PAIR, nguong_qua=NGUONG_QUA, nguong_xam=NGUONG_XAM, cong_ca_manh=False):
    meta = pd.read_parquet(kiem / "mau_kiem.parquet").set_index("id")
    g, tt_g = doc_cac_lo(kiem, "gemini", meta)
    c, tt_c = doc_cac_lo(kiem, "claude", meta)
    bang = ty_le_nhom(g, meta)
    pair = uoc_pair_theo_huong(bang, he_so)
    hc = hieu_chuan(g, c, meta)
    bo_sot = hc["bo_sot"]
    nhom_cong = {}
    for r in bang:
        if r["ty_le"] is None:
            continue
        r["ty_le_hieu_chinh"] = r["ty_le"] + bo_sot
        if r["loai"] in ("title", "mo_ta") or cong_ca_manh:
            nhom_cong[f"{r['huong']}|{r['loai']}|{r['bo']}"] = r["ty_le_hieu_chinh"]
    for r in pair:
        r["ty_le_hieu_chinh"] = r["ty_le"] + bo_sot
        nhom_cong[f"{r['huong']}|{r['loai']}|{r['bo']}"] = r["ty_le_hieu_chinh"]
    cong = xet_cong(nhom_cong, nguong_qua, nguong_xam)
    canh_bao = list(hc["canh_bao"])
    for ten, tt in (("Gemini", tt_g), ("Claude", tt_c)):
        for t in tt:
            if t["trang_thai"] == "loi":
                canh_bao.append(f"{ten}: lo {t['lo']} loi, {t['huong_dan']} ({'; '.join(t['loi'][:3])})")
    thieu_lo = [t["lo"] for t in tt_g if t["trang_thai"] == "chua_co"]
    if thieu_lo:
        canh_bao.append(f"Gemini: {len(thieu_lo)} lo chua co ket qua; ty le moi tinh tren phan da co")
    if not cong_ca_manh:
        canh_bao.append("Cong chi xet title, mo_ta cum va mo_ta pair (uoc luong); nhom manh M2 chon theo 'khac nhau' nen chi de tham khao")
    return {
        "tham_so": {"he_so_pair": he_so, "nguong_qua": nguong_qua, "nguong_xam": nguong_xam, "kappa_toi_thieu": KAPPA_TOI_THIEU,
                    "sot_toi_da": SOT_TOI_DA, "cong_ca_manh": cong_ca_manh},
        "ghi_chu_uoc_luong_pair": GHI_CHU_PAIR,
        "lo_gemini": tt_g, "lo_claude": tt_c, "ty_le_nhom": bang, "uoc_luong_pair": pair, "hieu_chuan": hc,
        "bo_sot_ap_dung": bo_sot, "cong": cong, "nhom_xet_cong": nhom_cong, "canh_bao": canh_bao,
    }


def pc(x, n=2):
    return "-" if x is None else f"{100 * x:.{n}f}%"


def viet_md(bc):
    L = ["# Báo cáo kiểm chất lượng dữ liệu (B2)", ""]
    ts = bc["tham_so"]
    L += [f"Tham số: hệ số pair {ts['he_so_pair']}, ngưỡng QUA ≤ {pc(ts['nguong_qua'], 1)}, vùng xám ≤ {pc(ts['nguong_xam'], 1)}.", ""]
    L += ["## Kết luận cổng", "", f"**{bc['cong']['ket_luan']}**"
          + (f" (nhóm cao nhất `{bc['cong']['nhom_cao_nhat']}` = {pc(bc['cong']['ty_le_cao_nhat'])} sau hiệu chỉnh)"
             if bc["cong"]["nhom_cao_nhat"] else ""), ""]
    if bc["canh_bao"]:
        L += ["## Cảnh báo", ""] + [f"- {x}" for x in bc["canh_bao"]] + [""]
    L += ["## Tình trạng lô", ""]
    for ten, k in (("Gemini", "lo_gemini"), ("Claude", "lo_claude")):
        dem = {}
        for t in bc[k]:
            dem[t["trang_thai"]] = dem.get(t["trang_thai"], 0) + 1
        L.append(f"- {ten}: {dem}")
        for t in bc[k]:
            if t["trang_thai"] == "loi":
                L.append(f"  - **{t['huong_dan']}**: " + "; ".join(t["loi"][:5]))
    L += ["", "## Tỉ lệ lỗi nặng theo nhóm (Gemini)", "",
          "| hướng | loại | bộ | số mục | đơn vị | nặng | nhẹ | tỉ lệ | Wilson 95% | sau hiệu chỉnh |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in bc["ty_le_nhom"]:
        L.append(f"| {r['huong']} | {r['loai']} | {r['bo']} | {r['so_muc']} | {r['don_vi']} | {r['nang']} | {r['nhe'] if r['nhe'] is not None else '-'} | "
                 f"{pc(r['ty_le'])} | {pc(r['wilson_lo'])} – {pc(r['wilson_hi'])} | {pc(r.get('ty_le_hieu_chinh'))} |")
    L += ["", "(mo_ta: đơn vị là MẢNH; title/manh: đơn vị là mục.)", ""]
    if bc["uoc_luong_pair"]:
        L += ["## Ước lượng bản pair cho mô tả", "", "`ty_le_pair = ty_le_cum(M1) + hệ số × (ty_le_M2_pair − ty_le_M2_cum)`", "",
              "| hướng | cum M1 | M2 cum | M2 pair | ước lượng pair | khoảng 95% xấp xỉ | sau hiệu chỉnh |", "|---|---|---|---|---|---|---|"]
        for r in bc["uoc_luong_pair"]:
            L.append(f"| {r['huong']} | {pc(r['ty_le_cum_M1'])} | {pc(r['ty_le_M2_cum'])} | {pc(r['ty_le_M2_pair'])} | {pc(r['ty_le'])} | "
                     f"{pc(r['khoang_95_xap_xi'][0])} – {pc(r['khoang_95_xap_xi'][1])} | {pc(r['ty_le_hieu_chinh'])} |")
        L.append("")
    h = bc["hieu_chuan"]
    kp = "-" if h["kappa"] is None else f"{h['kappa']:.3f}"
    L += [f"> {bc['ghi_chu_uoc_luong_pair']}", ""]
    L += ["## Hiệu chuẩn Gemini với Claude (mẫu chung, gộp mọi nhóm)", "",
          f"- Số mục chung: {h['so_muc_chung']}, số đơn vị nhị phân: {h['so_don_vi']}",
          f"- Cả hai nặng {h['ca_hai_nang']}, chỉ Gemini nặng {h['chi_gemini_nang']}, chỉ Claude nặng {h['chi_claude_nang']}, cả hai không {h['ca_hai_khong']}",
          f"- Cohen kappa: {kp}",
          f"- Tỉ lệ bất đồng: {pc(h['ty_le_bat_dong'])}",
          f"- Bỏ sót (hiệu chỉnh cộng vào mọi nhóm): {pc(h['bo_sot'])}",
          f"- Claude thấy nặng mà Gemini cho qua / tổng Claude thấy nặng: {pc(h['ty_le_claude_nang_gemini_qua'], 1)}", ""]
    return "\n".join(L) + "\n"


# ---------- --thu20 ----------
def mo_ta_px(loai, so_manh, px):
    if px is None:
        return "(thiếu)"
    if loai == "mo_ta":
        return f"{px['so_nang']}/{so_manh} nặng" + (f" [{';'.join(map(str, sorted(px['ds'])))}]" if px["ds"] else "")
    return px["muc"]


def so_sanh_thu20(kiem):
    meta = pd.read_parquet(kiem / "mau_kiem.parquet").set_index("id")
    ket = {}
    for nguon, hau in (("gemini", ".txt"), ("claude", ".jsonl")):
        f_gui = kiem / f"gui_{nguon}" / f"lo_000_thu20{'.csv' if nguon == 'gemini' else '.jsonl'}"
        f_kq = kiem / f"ket_qua_{nguon}" / f"lo_000_thu20{hau}"
        if not f_kq.exists():
            return {"loi": [f"chua co {f_kq}"]}
        txt = f_kq.read_text(encoding="utf-8-sig")
        rows, extra = tach_dong_gemini(txt) if nguon == "gemini" else tach_dong_claude(txt)
        kq, loi = kiem_lo("lo_000_thu20", doc_ids_gui(f_gui), meta, rows, [] if nguon == "gemini" else extra)
        if loi:
            return {"loi": [f"{nguon}: gui lai lo lo_000_thu20 ({'; '.join(loi[:4])})"]}
        ket[nguon] = kq
    ds = []
    khop = lech = 0
    a = b = cc = d = 0
    for i in sorted(ket["gemini"]):
        m = meta.loc[i]
        pg, pc_ = ket["gemini"][i], ket["claude"][i]
        ng, nc = don_vi_nang(m.loai, m.so_manh, pg), don_vi_nang(m.loai, m.so_manh, pc_)
        bang_nhau = (pg["muc"] == pc_["muc"]) if m.loai != "mo_ta" else (ng == nc)
        khop += bang_nhau
        lech += not bang_nhau
        for u in danh_sach_don_vi(m.loai, m.so_manh):
            a += (u in ng) and (u in nc)
            b += (u in ng) and (u not in nc)
            cc += (u in nc) and (u not in ng)
            d += (u not in ng) and (u not in nc)
        ds.append({"id": i, "huong": m.huong, "loai": m.loai, "gemini": mo_ta_px(m.loai, m.so_manh, pg),
                   "claude": mo_ta_px(m.loai, m.so_manh, pc_), "khop": bool(bang_nhau)})
    return {"muc": ds, "khop": int(khop), "lech": int(lech),
            "don_vi": {"ca_hai_nang": int(a), "chi_gemini_nang": int(b), "chi_claude_nang": int(cc), "ca_hai_khong": int(d)},
            "ghi_chu": "n quá nhỏ để kết luận kappa hay tỉ lệ; chỉ kiểm quy trình và đọc từng ca lệch"}


def in_thu20(kq):
    if "loi" in kq:
        print("LOI:", *kq["loi"], sep="\n  ")
        return
    print(f"{'id':8} {'huong':7} {'loai':6} {'Gemini':22} {'Claude':22} khop")
    for r in kq["muc"]:
        print(f"{r['id']:8} {r['huong']:7} {r['loai']:6} {r['gemini']:22} {r['claude']:22} {'v' if r['khop'] else 'LECH'}")
    print(f"\nkhớp {kq['khop']} / lệch {kq['lech']} (mức mục). Đơn vị nhị phân: {kq['don_vi']}")
    print("n quá nhỏ: không kết luận kappa.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kiem", default=str(KIEM))
    p.add_argument("--thu20", action="store_true")
    p.add_argument("--he-so-pair", type=float, default=HE_SO_PAIR)
    p.add_argument("--nguong-qua", type=float, default=NGUONG_QUA)
    p.add_argument("--nguong-xam", type=float, default=NGUONG_XAM)
    p.add_argument("--cong-ca-manh", action="store_true", help="đưa cả nhóm manh M2 vào cổng")
    p.add_argument("--dau", nargs=2, type=int, metavar=("A_THANG", "B_THANG"), help="kiểm định dấu chính xác")
    a = p.parse_args(argv)
    if a.dau:
        print(json.dumps(kiem_dinh_dau(*a.dau), ensure_ascii=False))
        return 0
    kiem = Path(a.kiem)
    if not (kiem / "mau_kiem.parquet").exists():
        print(f"LOI: chua co {kiem / 'mau_kiem.parquet'} (chay lay_mau_kiem.py truoc)", file=sys.stderr)
        return 2
    if a.thu20:
        kq = so_sanh_thu20(kiem)
        in_thu20(kq)
        if "loi" not in kq:
            (kiem / "bao_cao_thu20.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1 if "loi" in kq else 0
    bc = tao_bao_cao(kiem, a.he_so_pair, a.nguong_qua, a.nguong_xam, a.cong_ca_manh)
    (kiem / "bao_cao_kiem.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, frozenset) else str(o)),
                                            encoding="utf-8")
    md = viet_md(bc)
    (kiem / "bao_cao_kiem.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
