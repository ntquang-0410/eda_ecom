"""B2 bổ sung: kiểm riêng title và báo cáo tỉ lệ lỗi nặng tách hai tầng (Nhật Anh duyệt 05/11/2026).

Rubric giữ nguyên. Chỉ tách cách đếm theo mã lỗi của chính người chấm:
  - nặng lõi: có một trong sai_nghia, so_don_vi, thuong_hieu_ma, sot_chu_nguon;
  - nặng do thiếu: chỉ có thieu_thua (thiếu hoặc thừa thông tin).

  python -I -X utf8 scripts/ft_train/kiem_title.py dung      # dựng 4 lô Gemini (200 title chưa ai chấm) + lô 40 title cho Claude
  python -I -X utf8 scripts/ft_train/kiem_title.py bao-cao   # đọc các lô đã có kết quả, ghi kiem/bao_cao_title.{json,md}

Kết quả Gemini: kiem/ket_qua_gemini_title/lo_tNN.txt; Claude: kiem/ket_qua_claude_title/lo_t_doi_chieu.jsonl.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung as C  # noqa: E402
import kiem_ket_qua as K  # noqa: E402

KIEM = K.KIEM
SEED = 20261105
MA_LOI = {"sai_nghia", "so_don_vi", "thuong_hieu_ma", "sot_chu_nguon"}
COT_CSV = ["id", "huong", "loai", "nguon", "ban_dich"]


def tang(px):
    """Tầng của một phán xét title: None nếu không nặng; 'loi', 'thieu' hoặc 'khac' (mã khác hoặc để trống) nếu nặng."""
    if px["muc"] != "nang":
        return None
    ma = {x.strip() for x in px["loai_loi"].replace(",", "|").split("|") if x.strip()}
    if ma & MA_LOI:
        return "loi"
    return "thieu" if "thieu_thua" in ma else "khac"


def ids_da_cham(kiem=KIEM):
    """Id đã có người chấm hoặc đã gửi đi ở lô thử: không dùng lại trong mẫu title mới."""
    ids = set()
    for p in list(kiem.glob("ket_qua_claude/lo_*.jsonl")) + list(kiem.glob("gui_sonnet/*.jsonl")):
        ids |= {json.loads(x)["id"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
    p = kiem / "gui_gemini" / "lo_000_thu20.csv"
    if p.exists():
        ids |= set(K.doc_ids_gui(p))
    return ids


def chon_mau(meta, da_cham, moi_huong=100, so_claude=20, co_lo=50, seed=SEED):
    """meta có cột id, huong, loai, mau. Trả (danh sách lô id, danh sách id cho Claude). Xáo trộn, không lộ hướng theo thứ tự."""
    rng = np.random.default_rng(seed)
    t = meta[(meta.loai == "title") & (meta.mau == "M1") & ~meta.id.isin(da_cham)].sort_values("id")
    chon, claude = [], []
    for h in sorted(t.huong.unique()):
        ids = t.id[t.huong == h].values
        if len(ids) < moi_huong:
            raise RuntimeError(f"khong_du_title:{h}:{len(ids)}")
        x = rng.permutation(ids)[:moi_huong]
        chon += list(x)
        claude += list(x[:so_claude])
    chon = list(rng.permutation(chon))
    return [chon[i:i + co_lo] for i in range(0, len(chon), co_lo)], list(rng.permutation(claude))


def dung(kiem=KIEM):
    ra_g, ra_c = kiem / "gui_gemini_title", kiem / "gui_claude_title"
    if ra_g.exists() or ra_c.exists():
        sys.exit("TỪ CHỐI: lô title đã có, dựng lại sẽ làm lệch kết quả đã gửi.")
    C.doc_split()
    meta = pd.read_parquet(kiem / "mau_kiem.parquet")
    lo, claude = chon_mau(meta, ids_da_cham(kiem))
    m = meta.set_index("id")
    ra_g.mkdir()
    ra_c.mkdir()
    for k, ids in enumerate(lo, 1):
        with open(ra_g / f"lo_t{k:02d}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(COT_CSV)
            for i in ids:
                w.writerow([i, m.huong[i], "title", m.nguon[i], m.ban_dich[i]])
    with open(ra_c / "lo_t_doi_chieu.jsonl", "w", encoding="utf-8") as f:
        for i in claude:
            f.write(json.dumps({"id": i, "huong": m.huong[i], "loai": "title", "nguon": m.nguon[i],
                                "ban_dich": m.ban_dich[i], "so_manh": 0}, ensure_ascii=False) + "\n")
    tat_ca = [i for ids in lo for i in ids]
    mf = {"seed": SEED, "so_lo": len(lo), "co_lo": [len(x) for x in lo], "theo_huong": m.huong[tat_ca].value_counts().to_dict(),
          "theo_category": m.category[tat_ca].value_counts().to_dict(), "so_claude": len(claude), "sha256_rubric": C.sha256_file(C.GOC / "scripts" / "ft_train" / "rubric_kiem.md")}
    (kiem / "manifest_title.json").write_text(json.dumps(mf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(mf, ensure_ascii=False, indent=1))


def bang_hai_tang(px, meta):
    """px {id: phán xét title} -> {huong: {n, nang, nang_loi, nang_thieu, nang_khac, wilson_nang, wilson_loi}} (kèm 'chung')."""
    bang = {}
    for h, nhom in [(h, [i for i in px if meta.huong[i] == h]) for h in sorted({meta.huong[i] for i in px})] + [("chung", list(px))]:
        t = [tang(px[i]) for i in nhom]
        n, loi, thieu, khac = len(nhom), t.count("loi"), t.count("thieu"), t.count("khac")
        bang[h] = {"n": n, "nang": loi + thieu + khac, "nang_loi": loi, "nang_thieu": thieu, "nang_khac": khac,
                   "wilson_nang": K.wilson(loi + thieu + khac, n), "wilson_loi": K.wilson(loi, n)}
    return bang


def doi_chieu(a, b):
    """So hai người chấm trên các title chung, theo 'nặng' và theo 'nặng lõi'."""
    chung = sorted(set(a) & set(b))
    ra = {"n": len(chung)}
    for ten, ham in [("nang", lambda p: tang(p) is not None), ("nang_loi", lambda p: tang(p) == "loi")]:
        x = [(ham(a[i]), ham(b[i])) for i in chung]
        o = [sum(u and v for u, v in x), sum(u and not v for u, v in x), sum(v and not u for u, v in x), sum(not u and not v for u, v in x)]
        ra[ten] = {"ca_hai": o[0], "chi_a": o[1], "chi_b": o[2], "khong": o[3], "kappa": K.kappa_cohen(*o)}
    ra["lech"] = [i for i in chung if a[i]["muc"] != b[i]["muc"]]
    return ra


def doc_gemini(kiem, thu_muc_gui, thu_muc_kq, meta):
    px, loi = {}, []
    for gui in sorted((kiem / thu_muc_gui).glob("lo_*.csv")):
        kq = kiem / thu_muc_kq / f"{gui.stem}.txt"
        if not kq.exists():
            continue
        rows, _ = K.tach_dong_gemini(kq.read_text(encoding="utf-8"))
        ids = K.doc_ids_gui(gui)
        rows = [r for r in rows if meta.loai.get(r[0]) == "title"]
        ket, e = K.kiem_lo(gui.stem, [i for i in ids if meta.loai[i] == "title"], meta, rows)
        px.update(ket)
        loi += [f"{gui.stem}: {x}" for x in e]
    return px, loi


def doc_claude(paths, meta, chi_title=True):
    px = {}
    for p in paths:
        rows, loi = K.tach_dong_claude(p.read_text(encoding="utf-8"))
        if loi:
            raise RuntimeError(f"{p}: {loi}")
        for i, o, _ in rows:
            if not chi_title or meta.loai[i] == "title":
                x, e = K.dong_claude_thanh_phan_xet(meta.loai[i], meta.so_manh[i], o)
                if e:
                    raise RuntimeError(f"{p} {i}: {e}")
                px[i] = x
    return px


def bao_cao(kiem=KIEM):
    meta = pd.read_parquet(kiem / "mau_kiem.parquet").set_index("id")
    g_moi, loi = doc_gemini(kiem, "gui_gemini_title", "ket_qua_gemini_title", meta)
    o_moi = doc_claude(sorted(kiem.glob("ket_qua_claude_title/*.jsonl")), meta)
    o_cu = doc_claude(sorted(kiem.glob("ket_qua_claude/lo_*.jsonl")), meta)
    s_cu = doc_claude(sorted(kiem.glob("ket_qua_sonnet/*.jsonl")), meta)
    g_thu, loi_thu = doc_gemini(kiem, "gui_gemini", "ket_qua_gemini", meta)
    bc = {
        "dinh_nghia": {"nang_loi": sorted(MA_LOI), "nang_thieu": ["thieu_thua"]},
        "loi_doc": loi + loi_thu,
        "mau_moi_200": {"gemini": bang_hai_tang(g_moi, meta) if g_moi else None,
                        "opus_40": bang_hai_tang(o_moi, meta) if o_moi else None,
                        "gemini_vs_opus": doi_chieu(g_moi, o_moi) if g_moi and o_moi else None},
        "hieu_chuan": {"opus": bang_hai_tang(o_cu, meta), "sonnet": bang_hai_tang(s_cu, meta) if s_cu else None,
                      "opus_vs_sonnet": doi_chieu(o_cu, s_cu) if s_cu else None,
                      "gemini_lo_thu": bang_hai_tang(g_thu, meta) if g_thu else None},
    }
    (kiem / "bao_cao_title.json").write_text(json.dumps(bc, ensure_ascii=False, indent=2), encoding="utf-8")
    dong = ["# Báo cáo title hai tầng (rà soát bằng LLM)", "",
            "| Mẫu, người chấm | Hướng | n | Nặng | Wilson | Nặng lõi | Wilson | Do thiếu | Khác |", "|---|---|---|---|---|---|---|---|---|"]
    for ten, b in [("200 mới, Gemini", bc["mau_moi_200"]["gemini"]), ("40 mới, Opus", bc["mau_moi_200"]["opus_40"]),
                   ("100 hiệu chuẩn, Opus", bc["hieu_chuan"]["opus"]), ("45 đối chiếu, Sonnet", bc["hieu_chuan"]["sonnet"]),
                   ("10 lô thử, Gemini", bc["hieu_chuan"]["gemini_lo_thu"])]:
        for h, x in (b or {}).items():
            w1, w2 = x["wilson_nang"], x["wilson_loi"]
            dong.append(f"| {ten} | {h} | {x['n']} | {x['nang']} | {w1[0]:.1%}–{w1[1]:.1%} | {x['nang_loi']} | {w2[0]:.1%}–{w2[1]:.1%} | {x['nang_thieu']} | {x['nang_khac']} |")
    (kiem / "bao_cao_title.md").write_text("\n".join(dong) + "\n", encoding="utf-8")
    print("\n".join(dong))
    print("doi chieu:", json.dumps({"gemini_vs_opus": bc["mau_moi_200"]["gemini_vs_opus"], "opus_vs_sonnet": bc["hieu_chuan"]["opus_vs_sonnet"]}, ensure_ascii=False))
    if bc["loi_doc"]:
        print("LOI DOC:", bc["loi_doc"])
    return bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["dung", "bao-cao"])
    a = ap.parse_args()
    dung() if a.viec == "dung" else bao_cao()


if __name__ == "__main__":
    main()
