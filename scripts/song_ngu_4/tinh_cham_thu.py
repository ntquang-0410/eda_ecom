# -*- coding: utf-8 -*-
"""Tính kết quả CHẤM MÙ lần thử 200 sản phẩm của bộ 4 (24/10/2026). Người chấm: Claude (trợ lý AI), chấm trên cham_mu.txt TRƯỚC khi mở
khoa_KHONG_XEM.json. Nhãn từng bản: D = đúng, N = sai nhẹ (không tự nhiên, sót nhẹ, dịch sát chữ), S = sai nghĩa. "hon": bản tốt hơn
(số thứ tự trong mục) hoặc "ngang".

    .venv\\Scripts\\python.exe -X utf8 scripts\\song_ngu_4\\tinh_cham_thu.py data\\processed\\song_ngu_4\\cham_thu_20261024
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# mục: (nhãn bản 1, nhãn bản 2 hoặc None, bản tốt hơn)
CHAM = {
    1: ("D", "D", "ngang"), 2: ("D", None, "-"), 3: ("N", "D", 2), 4: ("D", None, "-"), 5: ("N", "D", 2), 6: ("D", "N", 1), 7: ("D", "N", 1),
    8: ("D", "N", 1), 9: ("D", "S", 1), 10: ("N", None, "-"), 11: ("N", "S", 1), 12: ("D", None, "-"), 13: ("N", "D", 2),
    14: ("D", "D", "ngang"), 15: ("D", None, "-"), 16: ("D", "D", "ngang"), 17: ("D", "D", "ngang"), 18: ("D", "D", "ngang"),
    19: ("D", None, "-"), 20: ("D", "D", 2), 21: ("N", None, "-"), 22: ("S", "N", 2), 23: ("D", "D", "ngang"), 24: ("D", None, "-"),
    25: ("N", "S", 1), 26: ("D", None, "-"), 27: ("D", "N", 1), 28: ("D", None, "-"), 29: ("S", "D", 2), 30: ("D", "D", "ngang"),
    31: ("S", "D", 2), 32: ("D", None, "-"), 33: ("D", None, "-"), 34: ("D", "D", "ngang"), 35: ("D", None, "-"), 36: ("D", "N", 1),
    37: ("D", "D", "ngang"), 38: ("D", "S", 1), 39: ("D", None, "-"), 40: ("N", None, "-"), 41: ("N", "D", 2), 42: ("N", "D", 2),
    43: ("N", "D", 2), 44: ("N", "D", 2), 45: ("D", "D", "ngang"), 46: ("D", None, "-"), 47: ("D", "S", 1), 48: ("D", "D", "ngang"),
    49: ("D", "N", 1), 50: ("D", "N", 1), 51: ("D", "D", "ngang"), 52: ("D", "N", 1), 53: ("D", None, "-"), 54: ("N", "S", 1),
    55: ("D", None, "-"), 56: ("D", "D", "ngang"), 57: ("N", "D", 2), 58: ("S", "D", 2), 59: ("S", "D", 2), 60: ("N", None, "-"),
    61: ("N", None, "-"), 62: ("D", "N", 1), 63: ("D", "D", "ngang"), 64: ("N", "D", 2), 65: ("D", "D", "ngang"), 66: ("N", "D", 2),
    67: ("N", "D", 2), 68: ("N", "D", 2), 69: ("N", "D", 2), 70: ("D", "D", "ngang"), 71: ("N", "S", 1), 72: ("D", "D", "ngang"),
    73: ("D", "N", 1), 74: ("D", None, "-"), 75: ("D", "N", 1), 76: ("D", "D", "ngang"), 77: ("D", "N", 1), 78: ("S", "D", 2),
    79: ("D", None, "-"), 80: ("D", "D", "ngang"), 81: ("N", "D", 2), 82: ("D", "D", "ngang"), 83: ("D", None, "-"), 84: ("N", None, "-"),
    85: ("D", "N", 1), 86: ("D", "D", "ngang"), 87: ("S", "S", "ngang"), 88: ("N", "N", "ngang"), 89: ("N", "N", 2), 90: ("D", None, "-"),
    91: ("D", "N", 1), 92: ("D", "N", 1), 93: ("D", "D", "ngang"), 94: ("N", "D", 2), 95: ("D", "D", "ngang"), 96: ("N", "D", 2),
    97: ("D", "S", 1), 98: ("N", "D", 2), 99: ("D", "D", "ngang"), 100: ("D", "N", 1), 101: ("N", "D", 2), 102: ("S", "D", 2),
    103: ("D", "D", "ngang"), 104: ("D", "S", 1), 105: ("D", "D", "ngang"), 106: ("D", None, "-"), 107: ("D", "N", 1), 108: ("D", "S", 1),
    109: ("D", "D", "ngang"), 110: ("D", "S", 1), 111: ("N", "S", 1), 112: ("D", "N", 1), 113: ("D", "D", "ngang"), 114: ("D", "D", "ngang"),
    115: ("N", "D", 2), 116: ("N", "D", 2), 117: ("D", "N", 1), 118: ("N", "S", 1), 119: ("D", "N", 1), 120: ("N", "N", "ngang"),
    121: ("D", "D", "ngang"), 122: ("D", "D", "ngang"), 123: ("D", "S", 1), 124: ("D", "N", 1), 125: ("D", "D", "ngang"),
    126: ("D", "D", "ngang"), 127: ("D", None, "-"), 128: ("D", "N", 1), 129: ("N", "D", 2), 130: ("D", None, "-"), 131: ("D", None, "-"),
    132: ("D", "D", "ngang"), 133: ("N", "D", 2), 134: ("N", None, "-"), 135: ("D", "N", 1), 136: ("D", "D", "ngang"), 137: ("D", "N", 1),
}


def main():
    d = Path(sys.argv[1])
    khoa = {int(k): v for k, v in json.loads((d / "khoa_KHONG_XEM.json").read_text(encoding="utf-8")).items()}
    assert set(khoa) == set(CHAM), "số mục chấm khác số mục khoá"
    nhan = defaultdict(Counter)        # (tầng, nguồn) -> nhãn
    hon = defaultdict(Counter)         # tầng -> nguồn thắng
    ra = []
    for k, (n1, n2, h) in CHAM.items():
        kk = khoa[k]
        tt = kk["thu_tu"]
        assert len(tt) == (1 if n2 is None else 2), f"mục {k}: số bản chấm khác số bản trong khoá"
        for nguon, n in zip(tt, (n1, n2)):
            nhan[(kk["tang"], nguon)][n] += 1
        if n2 is not None:
            hon[kk["tang"]]["ngang" if h == "ngang" else tt[h - 1]] += 1
        ra.append({"muc": k, **kk, "nhan": dict(zip(tt, (n1, n2))), "hon": "ngang" if h in ("ngang", "-") else tt[h - 1]})
    (d / "ket_qua_cham.json").write_text(json.dumps(ra, ensure_ascii=False, indent=0), encoding="utf-8")
    ten = {"S1": "chỉ Gemma sửa (bản cuối = Gemma)", "S2": "cả hai sửa giống", "S3": "cả hai sửa khác", "S4": "chỉ Qwen sửa (bản cuối = nháp)",
           "S5": "cả hai giữ nháp", "S6": "tên bị sửa", "S7": "dịch mới (không nháp)", "S8": "bản Gemma bị luật loại (bản cuối = nháp)"}
    for t in sorted({kk["tang"] for kk in khoa.values()}):
        print(f"\n{t} {ten[t]}: {sum(1 for kk in khoa.values() if kk['tang'] == t)} mục")
        for (tt, nguon), c in sorted(nhan.items()):
            if tt == t:
                print(f"   {nguon:5s} D {c['D']:3d} | N {c['N']:3d} | S {c['S']:3d}")
        if hon[t]:
            print("   tốt hơn:", dict(hon[t]))


if __name__ == "__main__":
    main()
