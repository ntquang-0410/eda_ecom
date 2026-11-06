# -*- coding: utf-8 -*-
"""Kết quả chấm mù 40 trường bản v2 khác bản đơn vị (Claude, 23/10/2026; chấm xong mới mở khoá). Mỗi mục: thắng (1/2/=), đánh giá
bản 1, bản 2 (D đúng, N sai nhẹ, S sai, - không có), nhãn lý do: nc ngữ cảnh, en tiếng Anh/đơn vị, pv bộ 2 không có, kh khác.

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\tinh_cham_mau_v2.py <thư mục cham_mau_v2_...>
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
S = """1 = - - pv|2 1 N S en|3 2 N D kh|4 1 D N kh|5 2 D D kh|6 2 N D en|7 2 N D kh|8 = D D kh|9 = D D kh|10 2 N D en|11 1 D N kh|12 1 D S nc|
13 1 D N en|14 2 N D nc|15 = D D kh|16 1 D N en|17 2 N D nc|18 2 N D kh|19 1 D N en|20 1 D N kh|21 = D D kh|22 1 D N kh|23 2 N D nc|
24 = D D kh|25 1 D N nc|26 1 D N en|27 2 N D nc|28 1 D - pv|29 1 D - pv|30 2 D D nc|31 1 D N kh|32 2 N D en|33 1 D - pv|34 2 N D nc|
35 = D D en|36 = D D en|37 1 D D en|38 = D D kh|39 1 D N en|40 1 N S en"""
J = {}
for x in S.replace("\n", "").split("|"):
    k, w, a, b, t = x.split()
    J[int(k)] = (w, a, b, t)
thu_muc = Path(sys.argv[1])
khoa = {int(k): v for k, v in json.loads((thu_muc / "khoa_KHONG_XEM.json").read_text(encoding="utf-8")).items()}
assert sorted(J) == sorted(khoa), "số mục chấm khác số mục trong khoá"
rows = []
for k, (w, a, b, t) in J.items():
    v2_la_1 = khoa[k]["khoa"] == "1=v2"
    v2, dv = (a, b) if v2_la_1 else (b, a)
    thang = "ngang" if w == "=" else ("v2" if (w == "1") == v2_la_1 else "don_vi")
    rows.append({"stt": k, **{x: khoa[k][x] for x in ("pair_id", "truong", "nguon_v2", "ly_do_sua")}, "thang": thang, "v2": v2, "don_vi": dv,
                 "nhan": t})
(thu_muc / "ket_qua_cham.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")


def bao(r, ten):
    c = Counter(x["thang"] for x in r)
    print(f"{ten}: {len(r)} | v2 thắng {c['v2']}, đơn vị thắng {c['don_vi']}, ngang {c['ngang']} | v2 {dict(Counter(x['v2'] for x in r))}"
          f" | đơn vị {dict(Counter(x['don_vi'] for x in r))}")


bao(rows, "Tất cả")
for t in ["nc", "en", "pv", "kh"]:
    bao([x for x in rows if x["nhan"] == t], f"  nhãn {t}")
print("\nv2 thua hoặc bị chấm Sai:")
for x in rows:
    if x["thang"] == "don_vi" or x["v2"] == "S":
        print(f"  #{x['stt']} {x['truong']} {x['nguon_v2']} thắng={x['thang']} v2={x['v2']} dv={x['don_vi']} | lý do mô hình: {x['ly_do_sua']}")
