"""Tổng hợp lần đo test MỘT lần (B10): fine-tune (adapter fold 0, ~11% title) so với zero-shot, cho từng nhánh (cụm, pair) và từng hướng. 0 GPU, chạy trên máy.
Đọc CHỈ kết quả phiên Kaggle (do_test_trang_thai.json, sinh_diem_*.jsonl: điểm từng mục, không có nguồn hay tham chiếu); KHÔNG đọc ft_run/test_khoa, split.parquet, ds_test.txt.

LUẬT KHÓA (Nhật Anh duyệt 07/11/2026, Prompt 806, ghi vào WALKTHROUGH TRƯỚC khi đo; không đổi sau khi thấy số):
- Kết luận mô hình theo chốt chặn C5 và Plan mục 5: với MỖI nhánh và MỖI hướng, fine-tune "thắng zero-shot" khi cận dưới khoảng 95% của
  chênh chrF theo mục (fine-tune trừ zero-shot, trung bình theo mục của cả hai loại title và mô tả) > 0. Bootstrap 2.000 lần, lấy mẫu lại THEO SẢN PHẨM (mỗi sản phẩm
  mang theo cả hai mục), seed 42. Hướng nào không thắng: báo ngay, KHÔNG kết luận "ổn".
- Cảnh báo (phải báo, không đổi kết luận; ngưỡng [MẶC ĐỊNH]): (i) loại title hoặc mô tả của một hướng có cận trên chênh < 0; (ii) tỉ lệ sinh lặp của fine-tune cao hơn
  zero-shot quá 1 điểm phần trăm; (iii) tỉ lệ giữ số hoặc giữ thương hiệu của fine-tune thấp hơn zero-shot quá 3 điểm phần trăm.
- Không đổi tỉ lệ title : mô tả, lr hay adapter sau khi thấy số. chrF chỉ đo độ giống bản dịch máy dùng làm tham chiếu, chưa phải chất lượng do người chấm.
Chạy:  python -I -X utf8 scripts/ft_train/tong_hop_do_test.py [--thu-muc <.../do_test>] [--ra <thu_muc>]"""
import argparse
import json
import random
import statistics
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
FT = GOC / "data" / "processed" / "ft_run"
SO_LAN, SEED = 2000, 42
NGUONG_LAP, NGUONG_GIU = 0.01, 0.03       # [MẶC ĐỊNH], xem docstring


def boot_theo_sp(rows, so_lan=SO_LAN, seed=SEED):
    """rows: [(product_id, chênh chrF của mục)]. Khoảng 95% của trung bình chênh theo mục, lấy mẫu lại theo sản phẩm. Trả {chenh, thap, cao, n_muc, n_sp}."""
    theo = {}
    for p, d in rows:
        theo.setdefault(p, []).append(d)
    sp = sorted(theo)
    tong = sum(len(v) for v in theo.values())
    rng = random.Random(seed)
    kq = []
    for _ in range(so_lan):
        s = n = 0
        for p in rng.choices(sp, k=len(sp)):
            v = theo[p]
            s += sum(v)
            n += len(v)
        kq.append(s / n)
    kq.sort()
    return {"chenh": sum(sum(v) for v in theo.values()) / tong, "thap": kq[int(0.025 * so_lan)], "cao": kq[int(0.975 * so_lan) - 1], "n_muc": tong, "n_sp": len(sp)}


def doc(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def so_nhanh(thu_muc, nhanh):
    """So fine-tune với zero-shot trên cùng mục. Trả dict kết quả của một nhánh."""
    zs = {r["id"]: r for r in doc(thu_muc / f"sinh_diem_{nhanh}_zero_shot.jsonl")}
    ft = {r["id"]: r for r in doc(thu_muc / f"sinh_diem_{nhanh}_ft.jsonl")}
    if set(zs) != set(ft):
        raise SystemExit(f"{nhanh}: hai mô hình chấm khác tập mục")
    ra = {"n_muc": len(zs), "huong": {}, "canh_bao": []}
    for h in sorted({r["huong"] for r in zs.values()}):
        ids = sorted(i for i in zs if zs[i]["huong"] == h)
        tong = boot_theo_sp([(zs[i]["product_id"], ft[i]["chrf"] - zs[i]["chrf"]) for i in ids])
        tong["chrf_zs"] = statistics.fmean(zs[i]["chrf"] for i in ids)
        tong["chrf_ft"] = statistics.fmean(ft[i]["chrf"] for i in ids)
        tong["thang_zero_shot"] = bool(tong["thap"] > 0)
        ra["huong"][h] = {"chung": tong, "loai": {}, "kiem_rieng": {}}
        for loai in ("title", "mo_ta"):
            li = [i for i in ids if zs[i]["loai"] == loai]
            b = boot_theo_sp([(zs[i]["product_id"], ft[i]["chrf"] - zs[i]["chrf"]) for i in li])
            b["chrf_zs"], b["chrf_ft"] = statistics.fmean(zs[i]["chrf"] for i in li), statistics.fmean(ft[i]["chrf"] for i in li)
            ra["huong"][h]["loai"][loai] = b
            if b["cao"] < 0:
                ra["canh_bao"].append(f"{nhanh} {h} {loai}: cận trên chênh chrF {b['cao']:+.2f} < 0 (fine-tune kém zero-shot rõ)")
        tl = lambda ds, c: statistics.fmean(1.0 if r[c] else 0.0 for r in ds)  # noqa: E731
        sub_zs, sub_ft = [zs[i] for i in ids], [ft[i] for i in ids]
        k = {c: {"zero_shot": tl(sub_zs, c), "fine_tune": tl(sub_ft, c)} for c in ("sinh_lap", "giu_so_dat", "giu_latin_dat", "sai_chu_viet", "da_cat_loi_dan")}
        k["ref"] = {c: tl(sub_zs, c) for c in ("ref_giu_so_dat", "ref_giu_latin_dat", "ref_sai_chu_viet")}
        ra["huong"][h]["kiem_rieng"] = k
        if k["sinh_lap"]["fine_tune"] - k["sinh_lap"]["zero_shot"] > NGUONG_LAP:
            ra["canh_bao"].append(f"{nhanh} {h}: sinh lặp fine-tune {k['sinh_lap']['fine_tune']:.3f} cao hơn zero-shot {k['sinh_lap']['zero_shot']:.3f} quá {NGUONG_LAP}")
        for c in ("giu_so_dat", "giu_latin_dat"):
            if k[c]["zero_shot"] - k[c]["fine_tune"] > NGUONG_GIU:
                ra["canh_bao"].append(f"{nhanh} {h}: {c} fine-tune {k[c]['fine_tune']:.3f} thấp hơn zero-shot {k[c]['zero_shot']:.3f} quá {NGUONG_GIU}")
    return ra


def viet_md(kq, st):
    L = ["# Đo test MỘT lần: fine-tune (adapter fold 0, ~11% title) so với zero-shot", "",
         "Đây là phần đọc số chrF, không phải đánh giá của người. chrF đo độ giống bản dịch máy dùng làm tham chiếu.", ""]
    thua = [f"{n} {h}" for n, r in kq.items() for h, v in r["huong"].items() if not v["chung"]["thang_zero_shot"]]
    L.append("**Kết luận theo luật khóa: " + ("MỌI nhánh, mọi hướng fine-tune thắng zero-shot." if not thua else "KHÔNG ĐẠT ở: " + ", ".join(thua) + ". Không kết luận 'ổn'.") + "**")
    L.append("")
    L.append("| Nhánh | Hướng | chrF zero-shot | chrF fine-tune | Chênh [95%, theo sản phẩm] | Thắng |")
    L.append("|---|---|---|---|---|---|")
    for n, r in kq.items():
        for h, v in r["huong"].items():
            c = v["chung"]
            L.append(f"| {n} | {h} | {c['chrf_zs']:.2f} | {c['chrf_ft']:.2f} | {c['chenh']:+.2f} [{c['thap']:+.2f}; {c['cao']:+.2f}] | {'có' if c['thang_zero_shot'] else 'KHÔNG'} |")
    L += ["", "| Nhánh | Hướng | Loại | Chênh [95%] |", "|---|---|---|---|"]
    for n, r in kq.items():
        for h, v in r["huong"].items():
            for loai, b in v["loai"].items():
                L.append(f"| {n} | {h} | {loai} | {b['chenh']:+.2f} [{b['thap']:+.2f}; {b['cao']:+.2f}] |")
    L += ["", "| Nhánh | Hướng | Kiểm riêng | zero-shot | fine-tune | mốc tham chiếu |", "|---|---|---|---|---|---|"]
    for n, r in kq.items():
        for h, v in r["huong"].items():
            k = v["kiem_rieng"]
            for c, ref in (("giu_so_dat", "ref_giu_so_dat"), ("giu_latin_dat", "ref_giu_latin_dat"), ("sai_chu_viet", "ref_sai_chu_viet"), ("sinh_lap", None), ("da_cat_loi_dan", None)):
                L.append(f"| {n} | {h} | {c} | {k[c]['zero_shot']:.3f} | {k[c]['fine_tune']:.3f} | {'' if ref is None else format(k['ref'][ref], '.3f')} |")
    L += ["", "**Cảnh báo:** " + ("không có." if not any(r["canh_bao"] for r in kq.values()) else "")]
    L += [f"- {w}" for r in kq.values() for w in r["canh_bao"]]
    L += ["", "Mẫu: " + ", ".join(f"{n} {m.get('so_mau')} mục / {m.get('so_sp', '?')} sản phẩm" for n, m in st.get("mau_do", {}).items()),
          "Adapter: " + ", ".join(f"{n} {a['ten']} (sha256 {a['sha256'][:12]}…)" for n, a in st.get("adapter", {}).items()), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thu-muc")
    ap.add_argument("--ra")
    a = ap.parse_args()
    tm = Path(a.thu_muc) if a.thu_muc else sorted((FT / "kaggle_do_test").glob("phien_*/do_test"))[-1]
    st = json.loads((tm / "do_test_trang_thai.json").read_text(encoding="utf-8"))
    if not st.get("xong") or st.get("nguon_do") != "test":
        raise SystemExit(f"Không phải lần đo test thật đã xong: xong={st.get('xong')} nguon_do={st.get('nguon_do')}")
    kq = {n: so_nhanh(tm, n) for n in ("cum", "pair")}
    ra = Path(a.ra) if a.ra else tm.parent / "tong_hop"
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "tong_hop_do_test.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    md = viet_md(kq, st)
    (ra / "tong_hop_do_test.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
