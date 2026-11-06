"""B2 bổ sung: bộ lọc tự động tìm title có dấu hiệu lỗi (Nhật Anh duyệt 05/11/2026). CHỈ ĐỌC, không sửa dữ liệu.

Cờ (mỗi title có thể có nhiều cờ, đều là heuristic, xem giới hạn của từng hàm trong cham_diem.py):
  han_trong_vi     đích Việt còn chữ Hán;
  viet_trong_zh    đích Trung còn chữ Việt có dấu (kể cả tên riêng giữ nguyên, nên có dương tính giả);
  am_tiet_viet     đích Trung còn từ ASCII lấy từ nguồn Việt, viết như âm tiết Việt và không có dáng tên riêng ("Bao Case");
  mat_latin        tên hãng/mã Latin của nguồn không còn trong đích;
  so_lech          số của nguồn bị mất, hoặc đích có số lạ;
  hieu_lech        title nguồn có thương hiệu (lấy từ dòng 品牌/Thương hiệu của mô tả cùng sản phẩm) mà title đích không có bản dịch thương hiệu đó.
Không bắt được lỗi sai nghĩa ("Mỡ Bò" thành 牛脂).

  python -I -X utf8 scripts/ft_train/loc_title.py
Ra: ft_run/loc_title/pool.parquet, ft_run/test_khoa/loc_title_test.parquet (đều chỉ có cờ), ft_run/loc_title/bao_cao.json.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cham_diem as D  # noqa: E402
import chia_tap as H  # noqa: E402
import chung as C  # noqa: E402
import kiem_title as T  # noqa: E402

CO = ["han_trong_vi", "viet_trong_zh", "am_tiet_viet", "mat_latin", "so_lech", "hieu_lech"]
RE_ASCII = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,}(?![A-Za-z0-9])")


def am_tiet_viet(nguon, dich):
    """Từ ASCII trong đích Trung, có trong nguồn Việt, viết như âm tiết Việt, không có dáng tên riêng/mã."""
    trong_nguon = {x for x in RE_ASCII.findall(nguon)}
    return [t for t in RE_ASCII.findall(dich) if t in trong_nguon and D._RE_AM_TIET.match(t.lower()) and not D._nhan_vi(t)]


def cap_hieu(huong, nguon_mo_ta, dich_mo_ta):
    """(thương hiệu nguồn, thương hiệu đích) từ dòng thương hiệu của mô tả bản pair (căn dòng 100%); None nếu không có hoặc chung chung."""
    nhan = "品牌" if huong == "zh->vi" else "thương hiệu"
    for a, b in zip(nguon_mo_ta.split("\n"), dich_mo_ta.split("\n")):
        if a.lower().startswith(nhan + ":") and ":" in b:
            sa, sb = a.split(":", 1)[1].strip(), b.split(":", 1)[1].strip()
            viet = sb if huong == "zh->vi" else sa
            chung = H.chuan_hoa(viet) in H.HIEU_CHUNG or H.chuan_hoa(viet).isdigit()
            return None if chung or not sa or not sb else (sa, sb)
    return None


def hieu_lech(huong, nguon, dich, nguon_mo_ta, dich_mo_ta):
    cap = cap_hieu(huong, nguon_mo_ta, dich_mo_ta)
    if not cap:
        return False
    ns, nd = H.chuan_hoa(nguon), H.chuan_hoa(dich)
    co_o_nguon = any(H.chuan_hoa(x) and H.chuan_hoa(x) in ns for x in cap[0].split("/"))
    co_o_dich = any(H.chuan_hoa(x) and H.chuan_hoa(x) in nd for x in cap[0].split("/") + cap[1].split("/"))
    return co_o_nguon and not co_o_dich


def co_title(huong, nguon, dich, nguon_mo_ta="", dich_mo_ta=""):
    s = D.sai_chu_viet(huong, nguon, dich)
    so = D.giu_so(nguon, dich)
    return {"han_trong_vi": huong == "zh->vi" and s["so_han"] > 0,
            "viet_trong_zh": huong == "vi->zh" and s["so_viet_dau"] > 0,
            "am_tiet_viet": huong == "vi->zh" and bool(am_tiet_viet(nguon, dich)),
            "mat_latin": not D.giu_latin(nguon, dich, huong, "title")["dat"],
            "so_lech": not so["dat"], "hieu_lech": hieu_lech(huong, nguon, dich, nguon_mo_ta, dich_mo_ta)}


def loc(df):
    """df có cột product_id, huong, nguon_title, dich_title. Trả DataFrame cờ theo sản phẩm."""
    hang = [dict(product_id=str(p), huong=h, **co_title(h, n, d, a, b))
            for p, h, n, d, a, b in zip(df.product_id, df.huong, df.nguon_title, df.dich_title, df.nguon_mo_ta, df.dich_mo_ta)]
    ra = pd.DataFrame(hang)
    ra["co_cờ"] = ra[CO].any(axis=1)
    return ra


def danh_gia(c, kiem):
    """So cờ với các title đã có người chấm (Opus, Sonnet, Gemini). 'Lõi' = ít nhất một người chấm xếp nặng lõi."""
    meta = pd.read_parquet(kiem / "mau_kiem.parquet").set_index("id")
    px = {}
    for ten, d in [("opus", T.doc_claude(sorted(kiem.glob("ket_qua_claude/lo_*.jsonl")) + sorted(kiem.glob("ket_qua_claude_title/*.jsonl")), meta)),
                   ("sonnet", T.doc_claude(sorted(kiem.glob("ket_qua_sonnet/*.jsonl")), meta)),
                   ("gemini", {**T.doc_gemini(kiem, "gui_gemini", "ket_qua_gemini", meta)[0], **T.doc_gemini(kiem, "gui_gemini_title", "ket_qua_gemini_title", meta)[0]})]:
        for i, p in d.items():
            px.setdefault(i, {})[ten] = T.tang(p)
    bang = pd.DataFrame([{"id": i, "product_id": meta.product_id[i], "loi": "loi" in v.values(),
                          "thieu": "thieu" in v.values()} for i, v in px.items()])
    bang = bang.merge(c, on="product_id", how="left")
    ra = {"so_title_da_cham": len(bang), "so_title_loi": int(bang.loi.sum()),
          "bat_duoc_title_loi": int((bang.loi & bang["co_cờ"]).sum()),
          "co_co_trong_da_cham": int(bang["co_cờ"].sum()), "trong_do_loi": int((bang.loi & bang["co_cờ"]).sum()),
          "trong_do_khong_ai_cham_nang": int((bang["co_cờ"] & ~bang.loi & ~bang.thieu).sum()),
          "loi_khong_bi_bat": bang.id[bang.loi & ~bang["co_cờ"]].tolist()}
    ra["theo_co"] = {x: {"co": int(bang[x].sum()), "trung_loi": int((bang[x] & bang.loi).sum())} for x in CO}
    return ra


def main():
    df = C.doc_bo("pair")
    df["product_id"] = df.product_id.astype(str)
    sp = C.doc_split()
    c = loc(df).merge(sp[["product_id", "la_test"]], on="product_id")
    (C.FT_RUN / "loc_title").mkdir(exist_ok=True)
    c[~c.la_test].drop(columns="la_test").to_parquet(C.FT_RUN / "loc_title" / "pool.parquet", index=False)
    c[c.la_test].drop(columns="la_test").to_parquet(C.THU_MUC_TEST / "loc_title_test.parquet", index=False)
    bc = {"toan_bo": {h: {"n": len(g), "co_co": int(g["co_cờ"].sum()), **{x: int(g[x].sum()) for x in CO}} for h, g in c.groupby("huong")},
          "danh_gia": danh_gia(c, T.KIEM)}
    (C.FT_RUN / "loc_title" / "bao_cao.json").write_text(json.dumps(bc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(bc, ensure_ascii=False, indent=1))
    d = df.merge(c[~c.la_test], on=["product_id", "huong"])
    for x in ["hieu_lech"]:
        g = d[d[x]]
        for r in g.sample(min(12, len(g)), random_state=1).itertuples():
            print(f"[{x}] {r.nguon_title[:70]} => {r.dich_title[:70]}")


if __name__ == "__main__":
    main()
