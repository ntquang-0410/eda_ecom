# -*- coding: utf-8 -*-
"""Chấm điểm thử chéo pair / cụm trên máy (không GPU). Đọc đầu ra của kaggle_cheo.py:

  python -X utf8 cham_cheo.py --vao <thu_muc_chua_eval_items.jsonl_va_sinh_*.jsonl> --ra <thu_muc_ra> [--sinh_cum_huan_luyen <sinh_ep2.jsonl>] [--sinh_pair_huan_luyen <sinh_ep2.jsonl>]

Cách chấm (chốt TRƯỚC khi xem số, Prompt 956):
- Mỗi bản dịch được CHUẨN HÓA về một chuỗi: dạng pair tách theo dòng, dạng cụm tách theo "; ", rồi nối lại bằng "; ". Nhờ vậy so được hai dạng với nhau.
- chrF++ (sacrebleu) so với CẢ HAI tham chiếu (bản pair và bản cụm cùng sản phẩm); điểm câu lấy tham chiếu cho điểm cao hơn, để không thiên vị dạng có tham chiếu "giống mình".
  Ngoài ra ghi chrF corpus riêng theo từng tham chiếu.
- Chỉ số phụ: đúng số thuộc tính (số dòng với pair, số mảnh "; " với cụm), giữ số, rò lời dẫn, đầu ra rỗng, tỉ lệ độ dài so với tham chiếu.
- So sánh theo cặp (cùng sản phẩm): bootstrap 2.000 lần theo sản phẩm, khoảng 95% của chênh chrF câu trung bình.
Kết luận chỉ dựa trên khoảng tin cậy: chứa 0 là "chưa phân biệt được".
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from sacrebleu.metrics import CHRF

DANG = ("pair", "cum")
ADAPTER = ("cum_f0_p11", "pair_f0_p11", "zero_shot")
TACH = {"cum": "; ", "pair": "\n"}
_SO = re.compile(r"\d+(?:[.,]\d+)?")
_RO_RANG = ("context only", "Translate this", "Product title")
_CHRF = CHRF(word_order=2)           # chrF++ như bộ chấm chính cho zh->vi


def manh(dang, s):
    """Tách bản dịch thành các thuộc tính. pair: theo dòng (bỏ dòng trống); cụm: theo "; "."""
    s = s.strip()
    if not s:
        return []
    if dang == "pair":
        return [x.strip() for x in s.split("\n") if x.strip()]
    return [x.strip() for x in s.split(TACH["cum"])]


def chuan_hoa(dang, s):
    return TACH["cum"].join(manh(dang, s))


def tham_chieu_chuan(x):
    return {"cum": x["ref"]["cum"], "pair": TACH["cum"].join(x["ref"]["pair"].split(TACH["pair"]))}


def giu_so(nguon, hyp):
    """Tỉ lệ các số trong nguồn còn có mặt trong bản dịch (đa tập). None nếu nguồn không có số."""
    a = _SO.findall(nguon)
    if not a:
        return None
    return sum((Counter(a) & Counter(_SO.findall(hyp))).values()) / len(a)


def diem_mot(x, dang, hyp):
    """Các chỉ số của MỘT bản dịch (đầu vào dạng `dang`) cho mục x."""
    ref = tham_chieu_chuan(x)
    h = chuan_hoa(dang, hyp)
    if h:
        c_cum, c_pair = _CHRF.sentence_score(h, [ref["cum"]]).score, _CHRF.sentence_score(h, [ref["pair"]]).score
    else:
        c_cum = c_pair = 0.0
    n_nguon = len(manh(dang, x["nguon"][dang]))
    return {"chrf": max(c_cum, c_pair), "chrf_cum": c_cum, "chrf_pair": c_pair, "chuan": h,
            "dung_so_manh": len(manh(dang, hyp)) == n_nguon, "giu_so": giu_so(x["nguon"][dang], hyp),
            "ro_loi_dan": any(t in hyp for t in _RO_RANG), "rong": not hyp.strip(),
            "ti_le_dai": (len(h) / max(1, len(ref["cum"])))}


def tb(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_chenh(a, b, so_lan=2000, seed=42):
    """a, b: điểm từng sản phẩm của hai điều kiện (cùng thứ tự). Trả chênh TB (a - b) và khoảng 95% bootstrap theo sản phẩm."""
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(d)
    kq = sorted(sum(rng.choices(d, k=n)) / n for _ in range(so_lan))
    return {"chenh": sum(d) / n, "thap": kq[int(0.025 * so_lan)], "cao": kq[int(0.975 * so_lan) - 1], "n": n}


def doc(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def cham_het(vao):
    vao = Path(vao)
    items = doc(vao / "eval_items.jsonl")
    ids = [x["id"] for x in items]
    ket = {}
    for ten in ADAPTER:
        for d in DANG:
            p = vao / f"sinh_{ten}_{d}.jsonl"
            if not p.exists():
                continue
            hyp = {r["id"]: r["hyp"] for r in doc(p)}
            if set(hyp) != set(ids):
                raise RuntimeError(f"{p.name}: thieu/lech id ({len(hyp)} so voi {len(ids)})")
            ket[(ten, d)] = [diem_mot(x, d, hyp[x["id"]]) for x in items]
    return items, ket


def corpus(items, ds, khoa):
    """chrF++ corpus của các bản dịch đã chuẩn hóa so với tham chiếu `khoa` (cum | pair)."""
    hyps = [r["chuan"] for r in ds]
    refs = [tham_chieu_chuan(x)[khoa] for x in items]
    return _CHRF.corpus_score(hyps, [refs]).score


def tong_hop(items, ket):
    bang = []
    for (ten, d), ds in sorted(ket.items()):
        bang.append({"adapter": ten, "dang_dau_vao": d, "n": len(ds), "chrf_cau_tb": tb([r["chrf"] for r in ds]),
                     "chrf_corpus_ref_cum": corpus(items, ds, "cum"), "chrf_corpus_ref_pair": corpus(items, ds, "pair"),
                     "dung_so_manh": tb([float(r["dung_so_manh"]) for r in ds]), "giu_so": tb([r["giu_so"] for r in ds]),
                     "ro_loi_dan": tb([float(r["ro_loi_dan"]) for r in ds]), "rong": tb([float(r["rong"]) for r in ds]),
                     "ti_le_dai_tb": tb([r["ti_le_dai"] for r in ds])})
    diem = lambda k: [r["chrf"] for r in ket[k]]
    cap = []

    def so(ten, a, b, y_nghia):
        if a in ket and b in ket:
            cap.append({"ten": ten, "y_nghia": y_nghia, "a": f"{a[0]}/{a[1]}", "b": f"{b[0]}/{b[1]}", **boot_chenh(diem(a), diem(b))})

    so("pair_adapter_pair_vs_cum", ("pair_f0_p11", "pair"), ("pair_f0_p11", "cum"), "Adapter pair: đưa dạng pair hay dạng cụm tốt hơn?")
    so("cum_adapter_cum_vs_pair", ("cum_f0_p11", "cum"), ("cum_f0_p11", "pair"), "Adapter cụm: đưa dạng cụm hay dạng pair tốt hơn?")
    so("tron_goi_pair_vs_cum", ("pair_f0_p11", "pair"), ("cum_f0_p11", "cum"), "Cách dùng đúng dạng: pair + adapter pair so với cụm + adapter cụm")
    so("mot_adapter_pair_cho_cum", ("pair_f0_p11", "cum"), ("cum_f0_p11", "cum"), "Dạng cụm: adapter pair (sai dạng) so với adapter cụm (đúng dạng)")
    so("mot_adapter_cum_cho_pair", ("cum_f0_p11", "pair"), ("pair_f0_p11", "pair"), "Dạng pair: adapter cụm (sai dạng) so với adapter pair (đúng dạng)")
    so("pair_ft_vs_zero_shot", ("pair_f0_p11", "pair"), ("zero_shot", "pair"), "Adapter pair so với không adapter (dạng pair)")
    so("cum_ft_vs_zero_shot", ("cum_f0_p11", "cum"), ("zero_shot", "cum"), "Adapter cụm so với không adapter (dạng cụm)")
    return bang, cap


def doi_chieu_huan_luyen(items, ket, sinh_huan_luyen, ten, dang):
    """Điểm của bản dịch do chính lần huấn luyện sinh (cùng mẫu cố định, cùng bộ chấm này) so với bản sinh lại ở phép thử chéo, trên các sản phẩm 'cố định'."""
    cd = {x["id"]: i for i, x in enumerate(items) if x.get("co_dinh")}
    if (ten, dang) not in ket or not cd:
        return None
    hyp = {r["id"]: r["hyp"] for r in doc(sinh_huan_luyen)}
    thieu = [i for i in cd if i not in hyp]
    chung = [i for i in cd if i in hyp]
    cu = [diem_mot(items[cd[i]], dang, hyp[i])["chrf"] for i in chung]
    moi = [ket[(ten, dang)][cd[i]]["chrf"] for i in chung]
    giong = sum(1 for i in chung if chuan_hoa(dang, hyp[i]) == ket[(ten, dang)][cd[i]]["chuan"])
    return {"n": len(chung), "thieu_o_ban_huan_luyen": len(thieu), "chrf_ban_huan_luyen": tb(cu), "chrf_sinh_lai": tb(moi),
            "so_ban_dich_giong_het": giong, "boot": boot_chenh(moi, cu) if chung else None}


def viet_md(bang, cap, doi, tk_mau):
    f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"
    pc = lambda v: "-" if v is None else f"{100 * v:.1f}%"
    L = ["# Thử chéo pair và cụm: mô tả zh→vi, Qwen3-4B + adapter fold 0", "",
         f"Mẫu: {tk_mau.get('so_chon')} sản phẩm (fold 0, mô tả nguyên, nguồn hai nhánh giống hệt); {tk_mau.get('so_co_dinh_trong_chon')} sản phẩm thuộc mẫu cố định của các lần huấn luyện.", "",
         "## Điểm từng điều kiện", "",
         "| Adapter | Dạng đầu vào | chrF câu TB (2 tham chiếu) | chrF corpus vs ref cụm | chrF corpus vs ref pair | Đúng số thuộc tính | Giữ số | Rò lời dẫn | Rỗng | Tỉ lệ dài |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in bang:
        L.append(f"| {r['adapter']} | {r['dang_dau_vao']} | {f(r['chrf_cau_tb'])} | {f(r['chrf_corpus_ref_cum'])} | {f(r['chrf_corpus_ref_pair'])} | {pc(r['dung_so_manh'])} | "
                 f"{pc(r['giu_so'])} | {pc(r['ro_loi_dan'])} | {pc(r['rong'])} | {f(r['ti_le_dai_tb'])} |")
    L += ["", "## So sánh theo cặp (chênh chrF câu TB, khoảng 95% bootstrap theo sản phẩm)", "", "| So sánh | A | B | Chênh A−B | Khoảng 95% | Kết luận |", "|---|---|---|---|---|---|"]
    for c in cap:
        kl = "chưa phân biệt được" if c["thap"] <= 0 <= c["cao"] else ("A tốt hơn" if c["thap"] > 0 else "B tốt hơn")
        L.append(f"| {c['y_nghia']} | {c['a']} | {c['b']} | {c['chenh']:+.2f} | [{c['thap']:+.2f}; {c['cao']:+.2f}] | {kl} |")
    if doi:
        L += ["", "## Đối chiếu với lần huấn luyện (kiểm quy trình)", ""]
        for k, v in doi.items():
            if v:
                L.append(f"- {k}: n={v['n']}, chrF của bản dịch lúc huấn luyện {f(v['chrf_ban_huan_luyen'])}, sinh lại ở phép thử chéo {f(v['chrf_sinh_lai'])}, "
                         f"{v['so_ban_dich_giong_het']}/{v['n']} bản dịch giống hệt; thiếu ở bản huấn luyện: {v['thieu_o_ban_huan_luyen']}.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", required=True)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--sinh_cum_huan_luyen", default=None)
    ap.add_argument("--sinh_pair_huan_luyen", default=None)
    a = ap.parse_args()
    items, ket = cham_het(a.vao)
    bang, cap = tong_hop(items, ket)
    doi = {}
    if a.sinh_cum_huan_luyen:
        doi["cum_f0_p11/cum"] = doi_chieu_huan_luyen(items, ket, a.sinh_cum_huan_luyen, "cum_f0_p11", "cum")
    if a.sinh_pair_huan_luyen:
        doi["pair_f0_p11/pair"] = doi_chieu_huan_luyen(items, ket, a.sinh_pair_huan_luyen, "pair_f0_p11", "pair")
    tk = {"so_chon": len(items), "so_co_dinh_trong_chon": sum(1 for x in items if x.get("co_dinh"))}
    ra = Path(a.ra)
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "ket_qua.json").write_text(json.dumps({"bang": bang, "so_sanh": cap, "doi_chieu": doi, "mau": tk}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = viet_md(bang, cap, doi, tk)
    (ra / "bang.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
