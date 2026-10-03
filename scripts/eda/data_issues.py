# -*- coding: utf-8 -*-
"""Consolidate every data problem found so far into one ranked table + per-row evidence.
Outputs -> data/eda/data_issues/
  data_issues_summary.csv   one line per problem, most severe first
  data_issues_evidence.csv  one line per (problem, product) with the evidence snippet"""
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDA = os.path.join(ROOT, "data", "eda")
OUT = os.path.join(EDA, "data_issues")
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(os.path.join(ROOT, "data", "snapshot", "bilingual_zh_vi.parquet")).reset_index(drop=True)
for c in ["title_zh", "title_vi", "description_zh", "description_vi"]:
    df[c] = df[c].fillna("")
N = len(df)
pid = df["product_id"].astype(str)
sim = pd.read_csv(os.path.join(EDA, "title_similarity_full.csv"), encoding="utf-8-sig", dtype={"product_id": str})
flags = pd.read_csv(os.path.join(EDA, "title_audit", "title_issues_rows.csv"), encoding="utf-8-sig", dtype={"product_id": str}).fillna("")
toks = pd.read_csv(os.path.join(EDA, "title_audit", "title_latin_tokens_editted.csv"), encoding="utf-8-sig").fillna("")
pairs = pd.read_parquet(os.path.join(ROOT, "data", "interim", "attr_pairs.parquet"))

ev = []          # (ma, product_id, bang_chung)
def add(ma, idx, proof):
    for i, p in zip(idx, proof):
        ev.append((ma, pid.iat[i], str(p)[:300]))

HAN = re.compile("[\u4e00-\u9fff]+")
pos = {p: i for i, p in enumerate(pid)}

# ---- D01 gross mistranslation: title similarity below the per-category 'surely wrong' threshold
# same thresholds as notebook 5.2: p50 / p95 of the same-category wrong pairs, over all categories
HARD = float(sim["sim_samecat"].median()); GREY = float(sim["sim_samecat"].quantile(0.95))
bad = sim[sim["sim"] < HARD].sort_values("sim")
add("D01", [pos[p] for p in bad["product_id"]], [f"sim={a:.3f} < ngưỡng 'chắc sai' {HARD:.3f}" for a in bad["sim"]])

# ---- D02 untranslated / Chinese left in title_vi
m = flags["chu_han_trong_vi"].astype(str).eq("True") | flags["chua_dich"].astype(str).eq("True")
sub = flags[m]
add("D02", [pos[p] for p in sub["product_id"]], ["chữ Hán còn lại: " + " ".join(HAN.findall(v)) for v in sub["title_vi"]])

# ---- D03 MT misunderstood a word / name (found while reviewing the token table)
WRONG = {
    "sydney": "雪梨 (lê tuyết) -> 'Sydney'", "iceland": "冰岛 (làng trà Băng Đảo) -> 'Iceland'",
    "tinkerbell": "小叮当 (Doraemon) -> 'Tinkerbell'", "presbyopic": "老花 (họa tiết monogram) -> 'Presbyopic'",
    "luzhou": "浓香型 (hương đậm) -> 'Luzhou'", "akita": "莆田 (Bồ Điền) -> 'Akita'",
    "burberry": "柏贝妮 (Babani) -> 'Burberry'", "woodruff": "木耳边 (viền bèo) -> 'Woodruff'",
    "pull-back": "回力 (Warrior) -> 'Pull-back'", "jerry": "杰里 (chip Jieli) -> 'Jerry'",
    "lanxess": "兰精 (Lenzing) -> 'Lanxess'", "seedling": "积雪草 (rau má) -> 'Seedling'",
    "kosa": "小种 (Tiểu Chủng) -> 'Kosa'", "daquan": "大全 (đủ loại) -> 'Daquan'",
    "net red": "网红 (nổi tiếng trên mạng) -> 'Net Red'",
    "hộp mật khẩu": "密码箱 (vali khóa số) -> 'hộp mật khẩu'", "kho báu sạc": "充电宝 (sạc dự phòng) -> 'kho báu sạc'",
    "rạp chiếu phim": "膜 (mặt nạ/màng) -> 'phim' -> 'rạp chiếu phim'",
}
W = r"0-9A-Za-zÀ-ỹĐđ"
low = df["title_vi"].str.lower()
for k, why in WRONG.items():
    p = re.compile(rf"(?<![{W}]){re.escape(k)}(?![{W}])")
    idx = [i for i, t in enumerate(low) if p.search(t)]
    add("D03", idx, [why] * len(idx))
# 苹果 = apple (fruit) rendered as the brand Apple
fruit = df["title_zh"].str.contains("苹果") & df["title_vi"].str.contains(r"\bApple\b") & \
        ~df["title_zh"].str.contains("适用|手机|iPhone|iphone|ipad|耳机|数据线|充电|投屏|安卓", regex=True)
add("D03", list(np.where(fruit)[0]), ["苹果 (quả táo) -> 'Apple' (hãng)"] * int(fruit.sum()))

# ---- D04 under-translation (vi far too short for the zh)
sub = flags[flags["do_dai_bat_thuong"] == "vi_qua_ngan"]
add("D04", [pos[p] for p in sub["product_id"]], [f"tỉ lệ từ vi/chữ zh = {r}" for r in sub["ti_le_tu_vi_tren_chu_zh"]])

# ---- D05 numbers lost
sub = flags[flags["so_zh_mat_trong_vi"] != ""]
add("D05", [pos[p] for p in sub["product_id"]], ["số trong zh không có ở vi: " + v for v in sub["so_zh_mat_trong_vi"]])

# ---- D06 Chinese left in description_vi
idx = [i for i, t in enumerate(df["description_vi"]) if HAN.search(t)]
add("D06", idx, ["chữ Hán còn lại: " + " ".join(HAN.findall(df.at[i, "description_vi"]))[:120] for i in idx])

# ---- D07 numbering artifact in attribute values ('Không1', 'Có2')
NUM = re.compile(r"(?:Không|Có|Là|Khác)\.?\d+\s*$")
pa = pairs[pairs["val_vi"].str.contains(NUM)]
g = pa.groupby("row").apply(lambda x: f"{len(x)} giá trị, vd '{x['val_zh'].iat[0]}' -> '{x['val_vi'].iat[0]}'")
add("D07", list(g.index), list(g.values))

# ---- D08 attribute list cannot be aligned (different number of 'name: value' segments)
def nseg(t): return len([x for x in str(t).split("; ") if x.strip()])
mis = [i for i in range(N) if nseg(df.at[i, "description_zh"]) != nseg(df.at[i, "description_vi"])]
add("D08", mis, [f"zh {nseg(df.at[i,'description_zh'])} đoạn / vi {nseg(df.at[i,'description_vi'])} đoạn" for i in mis])

# ---- D09 literal / inconsistent terminology
sub = flags[flags["dich_sat_chu"] != ""]
add("D09", [pos[p] for p in sub["product_id"]], sub["dich_sat_chu"])

# ---- D10 duplicates and train/test leakage risk
dup = df.duplicated(["title_zh", "title_vi"], keep=False)
same_zh = df.groupby("title_zh")["title_vi"].transform("nunique") > 1
add("D10", list(np.where(dup)[0]), ["cặp (zh, vi) trùng hoàn toàn với dòng khác"] * int(dup.sum()))
add("D10", list(np.where(same_zh & ~dup)[0]), ["cùng title_zh với dòng khác nhưng title_vi khác"] * int((same_zh & ~dup).sum()))

# ---- D11 English / pinyin to fix (token table: anything not GIU)
act = toks[toks["hanh_dong"] != "GIU"]
TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[.\-+'’][0-9A-Za-zÀ-ỹĐđ]+)*")
need = dict(zip(act["token"], act["hanh_dong"]))
for i, t in enumerate(df["title_vi"]):
    hit = sorted({w.lower() for w in TOKEN.findall(t) if w.lower() in need})
    if hit:
        add("D11", [i], [", ".join(f"{h}[{need[h]}]" for h in hit)])

# ---- D12 whole English phrases left untranslated
sub = flags[flags["cum_tieng_anh_dai"] != ""]
add("D12", [pos[p] for p in sub["product_id"]], sub["cum_tieng_anh_dai"])

# ---- D13 repeated phrase (two Chinese terms collapsed into one Vietnamese word)
sub = flags[flags["lap_tu_lien_tiep"] != ""]
add("D13", [pos[p] for p in sub["product_id"]], ["lặp: '" + v + "'" for v in sub["lap_tu_lien_tiep"]])

# ---- D14 grey zone of title similarity (suspicious, needs reading)
gz = sim[(sim["sim"] >= HARD) & (sim["sim"] < GREY)]
add("D14", [pos[p] for p in gz["product_id"]], [f"sim={a:.3f} trong vùng xám [{HARD:.3f}, {GREY:.3f})" for a in gz["sim"]])

# ---- D15-D17 formatting
low_flags = [("D15", flags["viet_hoa_moi_tu"].astype(str).eq("True"), lambda r: f"titlecase_ratio={r['titlecase_ratio']}"),
             ("D16", flags["ngoac_ky_hieu"] != "", lambda r: "ký hiệu: " + r["ngoac_ky_hieu"]),
             ("D17", flags["ket_thuc_do_dang"].astype(str).eq("True"), lambda r: "kết thúc: '..." + r["title_vi"][-25:] + "'")]
# viet_hoa_moi_tu rows were not all exported to title_issues_rows.csv -> recompute on the full data
WORDS = re.compile(r"[^\W\d_]+", re.U)
def tc(v):
    w = [x for x in WORDS.findall(v) if not x.isupper() or len(x) == 1]
    return sum(x[0].isupper() for x in w[1:]) / max(1, len(w) - 1)
r15 = df["title_vi"].map(tc)
add("D15", list(np.where(r15 >= 0.6)[0]), [f"{x:.0%} số từ viết hoa chữ đầu" for x in r15[r15 >= 0.6]])
for code, mask, fn in low_flags[1:]:
    sub = flags[mask]
    add(code, [pos[p] for p in sub["product_id"]], [fn(r) for _, r in sub.iterrows()])

META = [
    ("D01", 1, "Nghiêm trọng", "title", "Bản dịch sai nghĩa nặng", "Similarity bge-m3 dưới ngưỡng 'chắc sai' của category (trung vị điểm các cặp sai cố ý cùng category)"),
    ("D02", 1, "Nghiêm trọng", "title", "Chưa dịch / còn chữ Hán trong title_vi", "Regex chữ Hán trên title_vi"),
    ("D03", 1, "Nghiêm trọng", "title", "Máy dịch hiểu sai nghĩa từ / tên riêng", "Duyệt tay bảng từ Latin + mẫu dịch sát chữ, đếm dòng chứa từ lỗi"),
    ("D04", 1, "Nghiêm trọng", "title", "Dịch thiếu ý (bản Việt quá ngắn)", "Tỉ lệ số từ vi / số chữ Hán < phân vị 1%"),
    ("D05", 1, "Nghiêm trọng", "title", "Mất số liệu (năm, dung tích, số lượng)", "So khớp các con số giữa zh và vi"),
    ("D06", 2, "Cao", "description", "Description còn chữ Hán", "Regex chữ Hán trên description_vi"),
    ("D07", 2, "Cao", "description", "Giá trị thuộc tính dính số thứ tự ('Không1', 'Có2')", "Regex trên từng cặp thuộc tính"),
    ("D08", 2, "Cao", "description", "Không ghép cặp được thuộc tính zh-vi", "Số đoạn 'tên: giá trị' của zh khác vi"),
    ("D09", 2, "Cao", "title", "Thuật ngữ 1688 dịch sát chữ / không thống nhất", "Từ điển mẫu dịch sát chữ + bảng title_term_consistency.csv"),
    ("D10", 2, "Cao", "title", "Trùng lặp và nguy cơ rò rỉ train/test", "Trùng cặp (zh, vi) hoặc cùng zh khác vi"),
    ("D11", 3, "Trung bình", "title", "Tiếng Anh thừa / pinyin chưa Việt hoá", "Bảng quyết định title_latin_tokens_editted.csv (hành động khác GIU)"),
    ("D12", 3, "Trung bình", "title", "Cả cụm tiếng Anh chưa dịch", ">= 3 từ tiếng Anh liền nhau"),
    ("D13", 3, "Trung bình", "title", "Lặp cụm liền nhau (2 từ gốc bị dịch thành 1)", "Regex cụm 2-3 từ lặp liền"),
    ("D14", 3, "Trung bình", "title", "Nghi ngờ (vùng xám similarity)", "Similarity trong khoảng [p50, p95) điểm cặp sai cùng category"),
    ("D15", 4, "Thấp", "title", "Viết Hoa Mỗi Từ", ">= 60% số từ viết hoa chữ đầu"),
    ("D16", 4, "Thấp", "title", "Ký hiệu / ngoặc thừa (【】[]~|*)", "Regex ký hiệu"),
    ("D17", 4, "Thấp", "title", "Kết thúc dở dang", "Kết thúc bằng 'và/cho/của...' hoặc dấu phẩy"),
]
evd = pd.DataFrame(ev, columns=["ma_van_de", "product_id", "bang_chung"]).drop_duplicates(["ma_van_de", "product_id"])
evd = evd.merge(df[["product_id", "category", "title_zh", "title_vi"]].astype({"product_id": str}), on="product_id", how="left")
order = {m[0]: k for k, m in enumerate(META)}
evd["_o"] = evd["ma_van_de"].map(order)
evd = evd.sort_values(["_o", "category", "product_id"]).drop(columns="_o")
evd.to_csv(os.path.join(OUT, "data_issues_evidence.csv"), index=False, encoding="utf-8-sig")

rows = [{"ma_van_de": "D00", "muc_do": "0 - Vấn đề gốc", "pham_vi": "toàn bộ", "van_de": "Phía tiếng Việt là bản máy dịch của Alibaba (không phải người dịch)",
          "so_dong": N, "ti_le_%": 100.0, "cach_phat_hien": "Cách 1688 sinh bản tiếng Việt (README mục nguồn dữ liệu); mọi lỗi D01-D17 là hệ quả",
          "top_category": "", "vi_du_product_ids": "(tất cả)", "vi_du_bang_chung": "Xem D01-D05: lỗi đặc trưng của máy dịch"}]
for ma, lv, lvname, scope, name, how in META:
    e = evd[evd["ma_van_de"] == ma]
    rows.append({"ma_van_de": ma, "muc_do": f"{lv} - {lvname}", "pham_vi": scope, "van_de": name,
                 "so_dong": len(e), "ti_le_%": round(100 * len(e) / N, 2), "cach_phat_hien": how,
                 "top_category": ", ".join(f"{c}:{n}" for c, n in e["category"].value_counts().head(3).items()),
                 "vi_du_product_ids": ", ".join(e["product_id"].head(5)),
                 "vi_du_bang_chung": e["bang_chung"].iat[0] if len(e) else ""})
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "data_issues_summary.csv"), index=False, encoding="utf-8-sig")
print(summ[["ma_van_de", "muc_do", "van_de", "so_dong", "ti_le_%", "top_category"]].to_string(index=False))
print("rows with >=1 severe (level 1):", evd[evd["ma_van_de"].isin(["D01", "D02", "D03", "D04", "D05"])]["product_id"].nunique())
print("rows with >=1 problem of level 1-3:", evd[~evd["ma_van_de"].isin(["D15", "D16", "D17"])]["product_id"].nunique())
print("D03 breakdown:"); print(evd[evd.ma_van_de == "D03"]["bang_chung"].value_counts().to_string())
for ma in ["D01", "D03", "D04", "D05", "D07", "D08"]:
    e = evd[evd.ma_van_de == ma].head(2)
    for _, r in e.iterrows():
        print(f"[{ma}] {r.product_id} | {r.bang_chung[:80]}\n     ZH: {r.title_zh[:60]}\n     VI: {str(r.title_vi)[:100]}")
