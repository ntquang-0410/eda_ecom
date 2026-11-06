# -*- coding: utf-8 -*-
"""Kết quả chấm mù mẫu 160 mục của lần chạy thật bộ 3 (Claude chấm 24/10/2026; chấm xong mới mở khoá).
Mỗi mục: (danh sách đánh giá theo thứ tự Bản 1, Bản 2, ...: D đúng, N sai nhẹ, S sai, - không có bản; bản tốt nhất: chuỗi chỉ số, '=' là ngang nhau,
None là mục chỉ có một bản). Khoá cho biết Bản i là nhap (bản dịch đơn vị bộ 2), A (Gemma) hay B (Qwen).

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\tinh_cham_that.py <thư mục cham_that_...>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
J = {
    1: ("D", None), 2: ("DD", "="), 3: ("DN", "1"), 4: ("D", None), 5: ("DN", "1"), 6: ("D", None), 7: ("-DD", "="), 8: ("D", None),
    9: ("DN", "1"), 10: ("DN", "1"), 11: ("DD", "="), 12: ("D", None), 13: ("DN", "1"), 14: ("D", None), 15: ("DN", "1"), 16: ("N", None),
    17: ("DD", "="), 18: ("DD", "="), 19: ("D", None), 20: ("DD", "="), 21: ("DN", "1"), 22: ("NDD", "23"), 23: ("SDN", "2"), 24: ("D", None),
    25: ("ND", "2"), 26: ("NDN", "2"), 27: ("NNN", "="), 28: ("ND", "2"), 29: ("DN", "1"), 30: ("DNN", "1"), 31: ("DN", "1"), 32: ("ND", "2"),
    33: ("DS", "1"), 34: ("DD", "="), 35: ("-ND", "3"), 36: ("D", None), 37: ("N", None), 38: ("DN", "1"), 39: ("NNN", "="), 40: ("DD", "="),
    41: ("D", None), 42: ("DD", "="), 43: ("ND", "2"), 44: ("DN", "1"), 45: ("DD", "="), 46: ("DN", "1"), 47: ("DNN", "1"), 48: ("DD", "="),
    49: ("D", None), 50: ("D", None), 51: ("DN", "1"), 52: ("D", None), 53: ("ND", "2"), 54: ("DD", "="), 55: ("D", None), 56: ("D", None),
    57: ("DD", "="), 58: ("DN", "1"), 59: ("DN", "1"), 60: ("D", None), 61: ("DN", "1"), 62: ("NDN", "2"), 63: ("ND", "2"), 64: ("ND", "2"),
    65: ("ND", "2"), 66: ("DND", "13"), 67: ("NDN", "2"), 68: ("ND", "2"), 69: ("DD", "="), 70: ("ND", "2"), 71: ("ND", "2"), 72: ("D", None),
    73: ("DD", "="), 74: ("DN", "1"), 75: ("NN", "="), 76: ("DN", "1"), 77: ("D", None), 78: ("DN", "1"), 79: ("DS", "1"), 80: ("D", None),
    81: ("D", None), 82: ("DS", "1"), 83: ("DS", "1"), 84: ("DD", "="), 85: ("ND", "2"), 86: ("D", None), 87: ("NN-", "="), 88: ("D", None),
    89: ("DD", "="), 90: ("ND", "2"), 91: ("ND", "2"), 92: ("DND", "13"), 93: ("D", None), 94: ("DN", "1"), 95: ("DN", "1"), 96: ("DD", "="),
    97: ("NS", "1"), 98: ("ND", "2"), 99: ("DN", "1"), 100: ("DD", "="), 101: ("D", None), 102: ("N-D", "3"), 103: ("DD", "="), 104: ("D", None),
    105: ("DD", "="), 106: ("D", None), 107: ("DD", "="), 108: ("NDD", "23"), 109: ("NDD", "23"), 110: ("DDN", "12"), 111: ("ND", "2"),
    112: ("SD", "2"), 113: ("SD", "2"), 114: ("D", None), 115: ("D", None), 116: ("D", None), 117: ("SN", "2"), 118: ("DD", "="), 119: ("DD", "="),
    120: ("DD", "="), 121: ("DDN", "12"), 122: ("D", None), 123: ("DN", "1"), 124: ("DNN", "1"), 125: ("SD", "2"), 126: ("ND", "2"),
    127: ("DN", "1"), 128: ("DD", "="), 129: ("D-D", "="), 130: ("SD", "2"), 131: ("DD", "="), 132: ("ND", "2"), 133: ("DDN", "12"),
    134: ("DN", "1"), 135: ("ND", "2"), 136: ("D", None), 137: ("DD", "="), 138: ("DN", "1"), 139: ("DDN", "12"), 140: ("ND", "2"),
    141: ("NS", "1"), 142: ("-DN", "2"), 143: ("SSD", "3"), 144: ("D", None), 145: ("D", None), 146: ("D", None), 147: ("DD", "="),
    148: ("DD", "="), 149: ("DN", "1"), 150: ("DD", "="), 151: ("DD", "="), 152: ("D", None), 153: ("D", None), 154: ("DN", "1"),
    155: ("DN", "1"), 156: ("D", None), 157: ("SD", "2"), 158: ("D", None), 159: ("ND", "2"), 160: ("DN", "1"),
}
thu_muc = Path(sys.argv[1])
khoa = {int(k): v for k, v in json.loads((thu_muc / "khoa_KHONG_XEM.json").read_text(encoding="utf-8")).items()}
assert sorted(J) == sorted(khoa), "số mục chấm khác số mục trong khoá"
rows = []
for k, (g, tot) in J.items():
    kk = khoa[k]
    thu_tu = kk["thu_tu"]
    assert len(g) == len(thu_tu), (k, g, thu_tu)
    cham = {x: g[i] for i, x in enumerate(thu_tu)}
    if tot in (None, "="):
        best = set(thu_tu) if tot == "=" else None
    else:
        best = {thu_tu[int(c) - 1] for c in tot}
    rows.append({"stt": k, "tang": kk["tang"], "pair_id": kk["pair_id"], "truong": kk["truong"], "cham": cham, "tot_nhat": sorted(best) if best else None,
                 "ly_do": kk["ly_do"]})
(thu_muc / "ket_qua_cham.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
by = defaultdict(list)
for r in rows:
    by[r["tang"]].append(r)
MO_TA = {"S1": "chỉ Gemma sửa (bản cuối = Gemma)", "S2": "cả hai sửa giống nhau (bản cuối = Gemma)", "S3": "cả hai sửa khác nhau (bản cuối = Gemma)",
         "S4": "chỉ Qwen sửa (bản cuối = nháp)", "S5": "cả hai giữ nháp (bản cuối = nháp)", "S6": "tên thuộc tính bị Gemma sửa (bản cuối = Gemma)"}
tong_cuoi = Counter()
for t in ("S1", "S2", "S3", "S4", "S5", "S6"):
    R = by[t]
    cuoi = "A" if t in ("S1", "S2", "S3", "S6") else "nhap"
    dc = Counter(r["cham"][cuoi] for r in R)
    tong_cuoi.update({(t, x): n for x, n in dc.items()})
    print(f"\n{t} {MO_TA[t]}: {len(R)} mục | bản cuối: Đúng {dc['D']}, Sai nhẹ {dc['N']}, Sai {dc['S']}")
    if t == "S5":
        continue
    kh = "nhap" if cuoi == "A" else "B"
    hon = sum(1 for r in R if r["tot_nhat"] and kh in r["tot_nhat"] and cuoi not in r["tot_nhat"])
    kem = sum(1 for r in R if r["tot_nhat"] and cuoi in r["tot_nhat"] and kh not in r["tot_nhat"])
    ngang = sum(1 for r in R if r["tot_nhat"] is not None and cuoi in r["tot_nhat"] and kh in r["tot_nhat"])
    if cuoi == "A":      # so bản cuối (Gemma) với nháp
        print(f"   Gemma hơn nháp {kem}, nháp hơn Gemma {hon}, ngang {ngang}")
    else:                # so Qwen với nháp (bản cuối là nháp)
        print(f"   Qwen hơn nháp {hon}, nháp hơn Qwen {kem}, ngang {ngang}")
    if t == "S3":
        b = Counter()
        for r in R:
            tn = r["tot_nhat"] or []
            for x in ("nhap", "A", "B"):
                if x in tn and len(tn) == 1:
                    b[x] += 1
        print("   S3 tốt nhất duy nhất:", dict(b), "| đánh giá nháp", dict(Counter(r["cham"]["nhap"] for r in R)),
              "| Gemma", dict(Counter(r["cham"]["A"] for r in R)), "| Qwen", dict(Counter(r["cham"]["B"] for r in R)))
# tổng hợp các chỗ mô hình sửa (S1, S2, S3, S6): Gemma so với nháp
sua = by["S1"] + by["S2"] + by["S3"]
gv = sum(1 for r in sua if "A" in r["tot_nhat"] and "nhap" not in r["tot_nhat"])
nv = sum(1 for r in sua if "nhap" in r["tot_nhat"] and "A" not in r["tot_nhat"])
ng = sum(1 for r in sua if "A" in r["tot_nhat"] and "nhap" in r["tot_nhat"])
print(f"\nTỔNG các chỗ Gemma sửa giá trị (S1+S2+S3, {len(sua)} mục): Gemma hơn {gv}, nháp hơn {nv}, ngang {ng}; "
      f"Gemma Sai {sum(r['cham']['A'] == 'S' for r in sua)}, nháp Sai {sum(r['cham']['nhap'] == 'S' for r in sua)}")
print("Sai nhẹ + Sai của bản Gemma trong chỗ sửa:", sum(r["cham"]["A"] in "NS" for r in sua), "| của bản nháp tại đúng các chỗ đó:",
      sum(r["cham"]["nhap"] in "NS" for r in sua if r["cham"]["nhap"] != "-"))
print("\nDanh sách chỗ Gemma sửa mà bị chấm kém hơn nháp hoặc Sai:")
for r in sua:
    if ("nhap" in r["tot_nhat"] and "A" not in r["tot_nhat"]) or r["cham"]["A"] == "S":
        print(f"  #{r['stt']} {r['tang']} A={r['cham']['A']} nháp={r['cham']['nhap']} | lý do Gemma: {str(r['ly_do']['A'])[:150]}")
