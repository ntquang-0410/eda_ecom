# Chấm mù 188 cặp giá trị khác nhau (cả cụm Gemma vs đơn vị bộ 2), Claude 23/10/2026. Mỗi mục: thắng (1/2/=), đánh giá bản 1, bản 2
# (D đúng, N sai nhẹ, S sai, - không có), nhãn lý do: nc ngữ cảnh, en từ/đơn vị chưa dịch, dt chỗ bộ 2 có từ điển/quy tắc (sẽ đè ở bản
# thật), pv bộ 2 không có cặp này (đã bị cất), kh khác.
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
S = """1 = D D kh|2 1 D - pv|3 = D D dt|4 = D D kh|5 = D D kh|6 = D D dt|7 1 D D kh|8 1 D S nc|9 2 N N kh|10 2 D D kh|11 1 D N kh|12 = D D kh|
13 2 N D dt|14 = D D kh|15 1 D N kh|16 = D D kh|17 = D D kh|18 2 D D kh|19 = D D dt|20 = D D dt|21 = D D kh|22 = D D kh|23 2 N D en|
24 1 D N en|25 2 N D kh|26 = D D kh|27 = D D kh|28 1 D D nc|29 = D D kh|30 = D D kh|31 = D D dt|32 = D D dt|33 = N N en|34 = D D kh|
35 = S S kh|36 = D D kh|37 = D D kh|38 = D D kh|39 = D D dt|40 = D D kh|41 2 N D dt|42 = D D kh|43 = D D kh|44 = D D kh|45 = D D kh|
46 = D D dt|47 2 N D en|48 = D D dt|49 = D D kh|50 2 N D en|51 = D D kh|52 = D D kh|53 2 N D dt|54 1 D N dt|55 = D D kh|56 = D D kh|
57 = D D kh|58 = D D kh|59 = D D kh|60 2 N D en|61 1 D N kh|62 = N N kh|63 = D D kh|64 = D D kh|65 2 N D kh|66 = D D kh|67 = D D dt|
68 = D D kh|69 = D D kh|70 = D D dt|71 = D D kh|72 = D D kh|73 = D D kh|74 = D D dt|75 1 D N kh|76 = D D kh|77 = D D kh|78 = D D kh|
79 = D D kh|80 2 N D kh|81 2 N D kh|82 1 D N kh|83 = D D dt|84 = D D kh|85 = D D kh|86 = D D kh|87 = D D kh|88 = D D kh|89 = D D kh|
90 = D D kh|91 = D D dt|92 = N N kh|93 2 D D en|94 = D D dt|95 = D D dt|96 = D D kh|97 = D D dt|98 = D D kh|99 = N N kh|100 1 D N kh|
101 = D D kh|102 = N N kh|103 = D D dt|104 = D D kh|105 = D D kh|106 1 D - pv|107 1 D - pv|108 2 D D kh|109 = D D kh|110 = D D kh|
111 = D D dt|112 = D D kh|113 = D D kh|114 = D D kh|115 2 D D nc|116 = D D kh|117 = D D kh|118 = D D kh|119 = D D kh|120 = D D kh|
121 1 D N en|122 = D D kh|123 = D D kh|124 = D D kh|125 = D D kh|126 = D D kh|127 1 D N en|128 1 D - pv|129 1 D N nc|130 = D D kh|
131 2 N D dt|132 = D D dt|133 1 D N kh|134 2 N D nc|135 = D D kh|136 = D D kh|137 = D D kh|138 = D D kh|139 = D D kh|140 = D D dt|
141 = D D kh|142 = D D kh|143 2 N D kh|144 = D D kh|145 2 N D dt|146 = D D kh|147 = D D kh|148 1 D N kh|149 = D D kh|150 = D D kh|
151 = D D kh|152 2 N D dt|153 = N N kh|154 = D D kh|155 = D D kh|156 = D D kh|157 1 N S nc|158 = D D kh|159 2 N D kh|160 = D D kh|
161 = D D kh|162 = D D kh|163 = D D dt|164 = D D dt|165 = D D kh|166 2 N D kh|167 = D D kh|168 2 D D kh|169 = D D kh|170 = D D dt|
171 = D D dt|172 = D D kh|173 = D D kh|174 = D D kh|175 = D D kh|176 = D D kh|177 = D D kh|178 = D D kh|179 = D D kh|180 = D D dt|
181 = D D kh|182 = D D kh|183 = D D kh|184 = D D kh|185 = N N en|186 = D D kh|187 = D D dt|188 = D D dt"""
J = {}
for x in S.replace("\n", "").split("|"):
    k, w, a, b, t = x.split()
    J[int(k)] = (w, a, b, t)
assert sorted(J) == list(range(1, 189)), "thiếu mục"
khoa = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
rows = []
for k, (w, a, b, t) in J.items():
    cum_la_1 = khoa[k]["khoa"] == "1=cum"
    cum, dv = (a, b) if cum_la_1 else (b, a)
    thang = "ngang" if w == "=" else ("cum" if (w == "1") == cum_la_1 else "don_vi")
    rows.append({"stt": k, "pair_id": khoa[k]["pair_id"], "thang": thang, "cum": cum, "don_vi": dv, "nhan": t})
json.dump(rows, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)


def bao(r, ten):
    c = Counter(x["thang"] for x in r)
    print(f"{ten}: {len(r)} cặp | cả cụm thắng {c['cum']}, đơn vị thắng {c['don_vi']}, ngang {c['ngang']}"
          f" | đánh giá cả cụm {dict(Counter(x['cum'] for x in r))} | đơn vị {dict(Counter(x['don_vi'] for x in r))}")


bao(rows, "Tất cả 188")
bao([x for x in rows if x["nhan"] not in ("dt", "pv")], "Bỏ dt (từ điển/quy tắc sẽ đè) và pv (bộ 2 không có)")
for t in ["nc", "en", "dt", "pv", "kh"]:
    bao([x for x in rows if x["nhan"] == t], f"  nhãn {t}")
