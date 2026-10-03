# -*- coding: utf-8 -*-
"""Audit title_vi of the bilingual zh-vi corpus:
  (1) Latin-script (non-Vietnamese) tokens, grouped + counted per token, with a suggested action;
  (2) other title problems (row flags), term-consistency table.
Outputs -> data/eda/title_audit/"""
import re, os
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "eda", "title_audit")
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(os.path.join(ROOT, "data", "snapshot", "bilingual_zh_vi.parquet")).reset_index(drop=True)
tiki = pd.read_parquet(os.path.join(ROOT, "data", "snapshot", "tiki_vi.parquet"))
df["title_vi"] = df["title_vi"].fillna("")
df["title_zh"] = df["title_zh"].fillna("")

# ---------------------------------------------------------------- tokenisation
TOKEN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[.\-+'’][0-9A-Za-zÀ-ỹĐđ]+)*")
ONSET = r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
RHYME = (r"(?:a|ai|ao|au|ay|an|am|ang|anh|e|eo|en|em|eng|i|ia|iu|in|im|inh|o|oi|on|om|ong|oong|"
         r"oa|oai|oay|oan|oang|oanh|oe|oeo|oen|u|ua|ui|uy|un|um|ung|uya|uyu|uynh|y)")
VN_ASCII = re.compile(rf"^{ONSET}{RHYME}$")   # toneless syllable; p/t/c/ch codas need a tone mark

def latin_foreign(tok):
    letters = re.sub(r"[^A-Za-zÀ-ỹĐđ]", "", tok)
    if not letters or not letters.isascii():
        return False
    if re.search(r"\d", tok):
        return True
    return not VN_ASCII.match(tok.lower())

# ---------------------------------------------------------------- reference lists
UNITS = set("""cm mm m km ml l lít kg g mg w kw v mah ah gb tb mb hz khz ghz inch db lm oz lb cc k p x
mw kv va kwh mbps ma a nm um""".split())
TECH = set("""wifi bluetooth usb led lcd oled hdmi gps nfc tws anc enc hifi hd fhd uhd 2k 4k 8k ai app tv pc
ios android carplay magsafe type-c typec otg sim cpu ram rom rgb ic ups dc ac adas dvr etc obd tpms 3d 5d
pvc abs pp pe pet tpu tpr eva ppsu pom pu pa pc nbr epe pla uv spf pa+++ oem odm diy qr id mp3 mp4 mp5 ipx
ip67 ip68 wi-fi 4g 5g 3g lte sd tf""".split())
# Loanwords already written the Vietnamese way (not English spelling) -> keep
VN_LOAN = set("""vali balo axit amin inox caro gile oxy amoniac sandal vest nylon silicon camera logo laptop video
robot micro nano gel vitamin protein internet game piano laser radar polo jean short tivi ô-tô karaoke
mascara album""".split())

# Hand-reviewed suggestions for the frequent translator-produced English tokens.
SUGGEST = {
    "cotton": ("tu_muon_pho_bien", "Giữ (người bán VN dùng 'cotton' rất phổ biến); nếu muốn thuần Việt: 'vải bông'"),
    "casual": ("tieng_anh_thua", "Thay: 'thường ngày' / 'dạo phố'"),
    "retro": ("tu_muon_pho_bien", "Giữ hoặc thay 'cổ điển' (thống nhất 1 cách)"),
    "vintage": ("tu_muon_pho_bien", "Giữ hoặc thay 'cổ điển'"),
    "hot": ("tieng_anh_thua", "Bỏ, hoặc 'bán chạy' (爆款); 'Douyin Hot Model' -> 'mẫu bán chạy trên Douyin'"),
    "trend": ("tieng_anh_thua", "Thay: 'xu hướng'; 'hot trend' -> 'đang thịnh hành'"),
    "tote": ("tu_muon_pho_bien", "Giữ 'túi tote' (phổ biến ở VN)"),
    "sneaker": ("tu_muon_pho_bien", "Giữ (phổ biến ở VN)"),
    "ins": ("tieng_anh_thua", "ins风 -> 'phong cách Instagram' hoặc bỏ; KHÔNG để 'ins' trơ trọi"),
    "instagram": ("tieng_anh_thua", "Giữ nếu là 'phong cách Instagram'; 'ins Internet người nổi tiếng' là dịch hỏng -> dịch lại"),
    "internet": ("tieng_anh_thua", "网红 -> 'nổi tiếng trên mạng'; 'ins Internet' là dịch hỏng"),
    "mini": ("tu_muon_pho_bien", "Giữ (phổ biến ở VN)"),
    "size": ("tu_muon_pho_bien", "Giữ ('size lớn' phổ biến); thuần Việt: 'cỡ lớn'"),
    "unisex": ("tieng_anh_thua", "Thay: 'nam nữ'"),
    "niche": ("tieng_anh_thua", "Thay: 'độc lạ' / 'ít đụng hàng', hoặc bỏ"),
    "dad": ("tieng_anh_thua", "老爹鞋 -> 'giày đế thô' (người bán VN hay ghi 'giày ulzzang/dad shoes')"),
    "shoes": ("tieng_anh_thua", "Thay: 'giày'"),
    "set": ("tieng_anh_thua", "Thay: 'bộ'"),
    "new": ("tieng_anh_thua", "Thay: 'mới' (trừ tên riêng New Balance, New York)"),
    "top": ("tieng_anh_thua", "'Top 10' -> '10 ... hàng đầu', hoặc bỏ"),
    "girl": ("tieng_anh_thua", "Thay: 'cô gái' / 'nữ'"),
    "little": ("tieng_anh_thua", "Thay: 'nhỏ'"),
    "body": ("tieng_anh_thua", "'ôm body' -> 'ôm dáng'"),
    "black": ("tieng_anh_thua", "Thay: 'đen' (trừ tên riêng)"),
    "white": ("tieng_anh_thua", "Thay: 'trắng' (trừ tên riêng)"),
    "red": ("tieng_anh_thua", "'Net Red' = 网红 -> 'nổi tiếng trên mạng'"),
    "full": ("tieng_anh_thua", "'full box' -> 'nguyên thùng'"),
    "box": ("tieng_anh_thua", "Thay: 'hộp' / 'thùng'"),
    "big": ("tieng_anh_thua", "Thay: 'lớn' (trừ tên riêng)"),
    "old": ("tieng_anh_thua", "Thay: 'cũ/lâu năm' (trừ tên riêng)"),
    "light": ("tieng_anh_thua", "Thay: 'nhẹ' (trừ tên riêng)"),
    "series": ("tieng_anh_thua", "'3 Series' của BMW giữ; còn lại -> 'dòng'"),
    "mask": ("tieng_anh_thua", "Thay: 'mặt nạ' (cụm tiếng Anh dài -> dịch lại)"),
    "tea": ("tieng_anh_thua", "Thay: 'trà'"),
    "cream": ("tieng_anh_thua", "Thay: 'kem'"),
    "lip": ("tieng_anh_thua", "Thay: 'môi' (Lip Gloss -> 'son bóng')"),
    "essence": ("tieng_anh_thua", "Thay: 'tinh chất'"),
    "moisturizing": ("tieng_anh_thua", "Thay: 'dưỡng ẩm'"),
    "hydrating": ("tieng_anh_thua", "Thay: 'cấp ẩm'"),
    "cleansing": ("tieng_anh_thua", "Thay: 'làm sạch'"),
    "cleanser": ("tieng_anh_thua", "Thay: 'sữa rửa mặt'"),
    "facial": ("tieng_anh_thua", "Thay: 'mặt' ('Facial mask' -> 'mặt nạ')"),
    "beauty": ("tieng_anh_thua", "Thay: 'làm đẹp' (trừ tên hãng)"),
    "water": ("tieng_anh_thua", "Thay: 'nước'"),
    "silk": ("tieng_anh_thua", "Thay: 'lụa'"),
    "ice": ("tieng_anh_thua", "Thay: 'băng/lạnh'"),
    "snack": ("tieng_anh_thua", "Thay: 'đồ ăn vặt'"),
    "spicy": ("tieng_anh_thua", "Thay: 'cay'"),
    "sound": ("tieng_anh_thua", "Thay: 'âm thanh' (trừ tên riêng)"),
    "matte": ("tieng_anh_thua", "Thay: 'lì'"),
    "mist": ("tieng_anh_thua", "Thay: 'sương' / 'lì mờ'"),
    "mud": ("tieng_anh_thua", "Thay: 'bùn'"),
    "gradient": ("tieng_anh_thua", "Thay: 'chuyển màu'"),
    "bucket": ("tieng_anh_thua", "Thay: 'túi xô'"),
    "stereo": ("tieng_anh_thua", "'3D Stereo' -> '3D nổi'"),
    "jelly": ("tieng_anh_thua", "Thay: 'thạch' / 'nhựa dẻo'"),
    "preppy": ("tieng_anh_thua", "Thay: 'phong cách học đường'"),
    "sweatshirt": ("tieng_anh_thua", "Thay: 'áo nỉ'"),
    "romper": ("tieng_anh_thua", "Thay: 'bộ liền thân'"),
    "sequin": ("tieng_anh_thua", "Thay: 'kim sa'"),
    "air": ("tieng_anh_thua", "'Air Vent' -> 'cửa gió'; 'Air Cushion' -> 'cushion/đệm khí'"),
    "acid": ("tieng_anh_thua", "Thay: 'axit' (Hyaluronic acid -> 'axit hyaluronic')"),
    "amino": ("tieng_anh_thua", "Thay: 'amin' (Amino Acid -> 'axit amin')"),
    "oolong": ("tieng_anh_thua", "Thay: 'ô long'"),
    "snail": ("tieng_anh_thua", "Thay: 'ốc sên'"),
    "centella": ("tieng_anh_thua", "Thay: 'rau má' (Centella Asiatica)"),
    "asiatica": ("tieng_anh_thua", "Thay: 'rau má' (Centella Asiatica)"),
    "asiatium": ("tieng_anh_thua", "Thay: 'rau má'"),
    "camellia": ("tieng_anh_thua", "Thay: 'hoa trà'"),
    "astragalus": ("tieng_anh_thua", "Thay: 'hoàng kỳ'"),
    "poria": ("tieng_anh_thua", "Thay: 'phục linh'"),
    "cucumber": ("tieng_anh_thua", "Thay: 'dưa leo'"),
    "strawberry": ("tieng_anh_thua", "Thay: 'dâu tây'"),
    "mint": ("tieng_anh_thua", "Thay: 'bạc hà'"),
    "rose": ("tieng_anh_thua", "Thay: 'hoa hồng' (trừ tên hãng ROSE)"),
    "dropshipping": ("tieng_anh_thua", "一件代发: thay 'nhận dropship' / 'giao hộ từng đơn', hoặc bỏ (là chữ quảng cáo)"),
    "dropship": ("tieng_anh_thua", "Như 'dropshipping'"),
    "tiktok": ("ten_rieng", "Giữ (tên nền tảng)"),
    "amazon": ("ten_rieng", "Giữ (tên nền tảng) hoặc bỏ cả cụm quảng cáo"),
    "douyin": ("ten_rieng", "Giữ (tên nền tảng TQ)"),
    "forrest": ("ten_rieng", "阿甘鞋: người bán VN gọi 'giày Forrest Gump' -> giữ"),
    "gump": ("ten_rieng", "Như 'forrest'"),
    "mary": ("ten_rieng", "Giữ 'Mary Jane' (kiểu giày); 'Tom và Mary' là tên riêng"),
    "jane": ("ten_rieng", "Giữ 'Mary Jane'"),
    "martin": ("ten_rieng", "Giữ 'bốt Martin'"),
    "kelly": ("ten_rieng", "Giữ 'túi Kelly'"),
    "boston": ("ten_rieng", "Giữ 'túi Boston'"),
    "oxford": ("tu_muon_pho_bien", "Giữ 'vải Oxford'"),
    "hepburn": ("ten_rieng", "Giữ 'phong cách Hepburn'"),
    "chanel": ("ten_rieng", "Giữ (phong cách Chanel)"),
    "muji": ("ten_rieng", "Giữ"),
    "apple": ("ten_rieng", "Giữ khi là hãng Apple; kiểm tra 'Đồ trang trí xe hơi Apple' (có thể là 苹果 = quả táo -> dịch sai)"),
    "huawei": ("ten_rieng", "Giữ"), "samsung": ("ten_rieng", "Giữ"), "xiaomi": ("ten_rieng", "Giữ"),
    "iphone": ("ten_rieng", "Giữ"), "pigeon": ("ten_rieng", "Giữ (hãng bình sữa)"), "lego": ("ten_rieng", "Giữ"),
    "hegen": ("ten_rieng", "Giữ"), "balance": ("ten_rieng", "Giữ 'New Balance'"),
    "harajuku": ("ten_rieng", "Giữ hoặc 'phong cách Harajuku'"),
}
for _w in """acrylic jacquard denim canvas modal satin tencel lyocell flannel spandex velvet hoodie cardigan legging
polyester lycra kaki jeans""".split():
    SUGGEST.setdefault(_w, ("tu_muon_pho_bien", "Giữ (tên chất liệu/kiểu đồ người bán VN dùng nguyên)"))
for _w in """collagen hyaluronic niacinamide nicotinamide peptide polypeptide retinol arbutin astaxanthin fullerene
salicylic ceramide squalane probiotic""".split():
    SUGGEST.setdefault(_w, ("tu_muon_pho_bien", "Giữ (tên hoạt chất mỹ phẩm, VN dùng nguyên)"))
# Chinese proper nouns left as pinyin -> Sino-Vietnamese reading (âm Hán Việt) the Vietnamese market uses.
HANVIET = {
    "wuyishan": "Vũ Di Sơn", "wuyi": "Vũ Di", "dahongpao": "Đại Hồng Bào", "da hong pao": "Đại Hồng Bào",
    "tieguanyin": "Thiết Quan Âm", "jinjunmei": "Kim Tuấn Mi", "junmei": "Kim Tuấn Mi (Jin Junmei)",
    "zhengshan": "Chính Sơn (Tiểu Chủng)", "xiaozhong": "Tiểu Chủng", "souchong": "Chính Sơn Tiểu Chủng (Lapsang Souchong)",
    "lapsang": "Chính Sơn Tiểu Chủng", "biluochun": "Bích Loa Xuân", "maojian": "Mao Tiêm", "pu'er": "Phổ Nhĩ",
    "anxi": "An Khê", "fuding": "Phúc Đỉnh", "shoumei": "Thọ Mi", "menghai": "Mãnh Hải", "banzhang": "Ban Chương",
    "dianhong": "Điền Hồng", "xinhui": "Tân Hội", "tongmuguan": "Đồng Mộc Quan", "huaqiangbei": "Hoa Cường Bắc",
    "putian": "Phủ Điền", "nantong": "Nam Thông", "luzhou": "Lư Châu", "mingqian": "Minh Tiền (trà trước Thanh Minh)",
    "piaoxue": "Phiêu Tuyết", "hengxian": "Hoành huyện", "anyang": "An Dương", "fude": "Phúc Đức",
}

# ---------------------------------------------------------------- pinyin detector
PY_INIT = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w", ""]
PY_FIN = ["iang", "iong", "uang", "ueng", "ang", "eng", "ing", "ong", "ian", "iao", "uai", "uan", "van", "ai", "ei", "ao", "ou",
          "an", "en", "in", "un", "vn", "ia", "ie", "iu", "ua", "uo", "ui", "ue", "ve", "er", "a", "o", "e", "i", "u", "v"]
def _py_ok(i, f):
    # drop the initial+final combos that do not exist in Mandarin, so English words stop segmenting as pinyin
    if f in ("v", "ve", "van", "vn"):
        return i in ("n", "l") and f in ("v", "ve")
    if f == "o":
        return i in ("b", "p", "m", "f", "w", "")
    if f == "er":
        return i == ""
    if f == "e":
        return i not in ("b", "p", "f", "j", "q", "x", "w")
    if f == "i":
        return i not in ("f", "g", "k", "h", "w", "")
    if f in ("in", "ing", "ie", "iao", "ian", "iu"):
        return i in ("b", "p", "m", "d", "t", "n", "l", "j", "q", "x", "")
    if f in ("ia", "iang", "iong"):
        return i in ("j", "q", "x", "l", "n", "")
    if f in ("ue", "un") and i in ("b", "p", "m", "f"):
        return False
    if i in ("j", "q", "x"):
        return f.startswith("i") or f in ("u", "ue", "uan", "un")
    return True

PINYIN = {i + f for i in PY_INIT for f in PY_FIN if _py_ok(i, f)}
PINYIN -= {""}

def pinyin_split(w):
    w = w.lower().replace("'", "").replace("’", "")
    n = len(w); best = [None] * (n + 1); best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 1, min(n, i + 6) + 1):
            if i > 0 and w[i] in "aoe":
                break                                   # vowel-initial syllable only word-initially
            if w[i:j] in PINYIN and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + [w[i:j]]
    return best[n]

# ---------------------------------------------------------------- Tiki document frequency (real Vietnamese e-commerce usage)
tiki_df = Counter()
for t in (tiki["title_vi"].fillna("") + " " + tiki["description_vi"].fillna("")):
    tiki_df.update({x.lower() for x in TOKEN.findall(t)})

# ---------------------------------------------------------------- per-token stats
rows_of = defaultdict(list); variants = defaultdict(Counter); in_zh = Counter()
row_tokens = []
for i, (vi, zh) in enumerate(zip(df["title_vi"], df["title_zh"])):
    zl = zh.lower(); seen = []
    for tok in TOKEN.findall(vi):
        if not latin_foreign(tok):
            continue
        k = tok.lower(); variants[k][tok] += 1
        if k not in seen:
            seen.append(k); rows_of[k].append(i)
            if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", zl):
                in_zh[k] += 1
    row_tokens.append(seen)

UNIT_NUM = re.compile(r"^\d+(?:[.,]\d+)?(?:" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")$")

def classify(k):
    n = len(rows_of[k]); zshare = in_zh[k] / n
    if k in SUGGEST:
        return SUGGEST[k]
    if k in HANVIET:
        return ("phien_am_pinyin", f"Chuyển âm Hán Việt: '{HANVIET[k]}'")
    if re.search(r"\d", k):
        if UNIT_NUM.match(k):
            return ("don_vi_so", "Giữ (số + đơn vị)")
        return ("ma_model", "Giữ (mã/model)" if zshare >= 0.5 else "Kiểm tra: mã không có trong bản gốc")
    if k in UNITS:
        return ("don_vi_so", "Giữ (đơn vị)")
    if k in TECH:
        return ("thuat_ngu_ky_thuat", "Giữ (thuật ngữ/viết tắt kỹ thuật)")
    if k in VN_LOAN:
        return ("tu_muon_pho_bien", "Giữ (từ mượn đã Việt hoá)")
    if len(k) <= 2 and zshare >= 0.5:
        return ("ky_hieu", "Giữ (ký hiệu/chữ cái có trong bản gốc: loại C, chữ H, size M...)")
    if zshare >= 0.5:
        return ("co_trong_ban_goc", "Giữ (bản gốc TQ cũng viết Latin: thương hiệu/mã/thuật ngữ)")
    seg = pinyin_split(k)
    if seg and (len(seg) >= 2 or k in {"jin", "mei", "bei", "wei", "fu", "jun", "zan", "xiu", "qi", "zhi"}) and tiki_df[k] < 5:
        return ("phien_am_pinyin", "Tên riêng TQ để nguyên pinyin: địa danh/tên trà -> âm Hán Việt; tên hãng -> giữ pinyin (thống nhất)")
    if tiki_df[k] >= 20:
        return ("tu_muon_pho_bien", f"Người bán VN có dùng (Tiki: {tiki_df[k]} sp) -> giữ hoặc thay tuỳ anh")
    return ("tieng_anh_can_xem", "Tên hãng/model -> giữ; từ tiếng Anh thường -> dịch sang tiếng Việt")

recs = []
for k, idx in rows_of.items():
    grp, sug = classify(k)
    i0 = idx[0]
    cap = sum(c for v, c in variants[k].items() if v[0].isupper()) / sum(variants[k].values())
    hint = ""
    if grp == "tieng_anh_can_xem":
        hint = "co_the_ten_hang (luôn viết hoa)" if cap == 1 and df.loc[idx, "shop"].nunique() <= 3 else \
               ("tu_thuong (có lúc viết thường)" if cap < 1 else "")
    recs.append({
        "token": k, "bien_the": " | ".join(v for v, _ in variants[k].most_common(4)),
        "ti_le_viet_hoa": round(cap, 2), "so_shop": df.loc[idx, "shop"].nunique(), "goi_y_them": hint,
        "so_dong": len(idx), "so_dong_zh_cung_co": in_zh[k], "ti_le_zh_cung_co": round(in_zh[k] / len(idx), 2),
        "so_sp_tiki_dung": tiki_df[k], "nhom": grp, "de_xuat": sug,
        "category_chinh": df.loc[idx, "category"].value_counts().idxmax(),
        "vi_du_title_vi": df.at[i0, "title_vi"], "vi_du_title_zh": df.at[i0, "title_zh"],
        "product_ids": ";".join(df.loc[idx[:50], "product_id"].astype(str)),
    })
ORDER = {"tieng_anh_thua": 0, "phien_am_pinyin": 1, "tieng_anh_can_xem": 2, "tu_muon_pho_bien": 3, "ten_rieng": 4,
         "thuat_ngu_ky_thuat": 5, "co_trong_ban_goc": 6, "ma_model": 7, "ky_hieu": 8, "don_vi_so": 9}
tok = pd.DataFrame(recs)
tok["_o"] = tok["nhom"].map(ORDER)
tok = tok.sort_values(["_o", "so_dong", "token"], ascending=[True, False, True]).drop(columns="_o")
tok.to_csv(os.path.join(OUT, "title_latin_tokens.csv"), index=False, encoding="utf-8-sig")
grp_of = dict(zip(tok["token"], tok["nhom"]))
KEEP = {"don_vi_so", "thuat_ngu_ky_thuat", "ky_hieu", "co_trong_ban_goc", "ma_model", "ten_rieng", "tu_muon_pho_bien"}

print("=== Latin tokens in title_vi ===")
print("distinct tokens:", len(tok), "| rows with >=1 Latin token:", sum(1 for s in row_tokens if s))
g = tok.groupby("nhom").agg(so_token=("token", "size"), tong_luot_dong=("so_dong", "sum"))
need_rows = [i for i, s in enumerate(row_tokens) if any(grp_of[k] not in KEEP for k in s)]
print(g.sort_values("tong_luot_dong", ascending=False).to_string())
print("rows with >=1 token needing action (tieng_anh_thua / phien_am_pinyin):", len(need_rows))

# ---------------------------------------------------------------- long English spans (whole phrase left in English)
def english_spans(vi):
    spans, cur = [], []
    for m in TOKEN.finditer(vi):
        t = m.group()
        if latin_foreign(t) and grp_of.get(t.lower()) in ("tieng_anh_thua", "tieng_anh_can_xem"):
            cur.append(t)
        else:
            if len(cur) >= 3: spans.append(" ".join(cur))
            cur = []
    if len(cur) >= 3: spans.append(" ".join(cur))
    return spans

df["english_spans"] = df["title_vi"].map(english_spans)

lat_rows = pd.DataFrame({
    "product_id": df["product_id"], "category": df["category"],
    "title_zh": df["title_zh"], "title_vi": df["title_vi"],
    "token_can_xu_ly": [", ".join(k for k in s if grp_of[k] not in KEEP) for s in row_tokens],
    "nhom": [", ".join(sorted({grp_of[k] for k in s if grp_of[k] not in KEEP})) for s in row_tokens],
    "token_giu": [", ".join(k for k in s if grp_of[k] in KEEP) for s in row_tokens],
    "cum_tieng_anh_dai": df["english_spans"].map(" || ".join),
})
lat_rows = lat_rows[lat_rows["token_can_xu_ly"] != ""].copy()
lat_rows["xu_ly"] = ""; lat_rows["ghi_chu"] = ""
lat_rows.to_csv(os.path.join(OUT, "title_latin_rows_can_xu_ly.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- other title problems
NUM = re.compile(r"\d+(?:[.,]\d+)?")
def nums(s):
    s = re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", s)          # 100.000 -> 100000
    return Counter(x.replace(",", ".") for x in NUM.findall(s))

def missing_nums(a, b):
    miss = a - b
    return [x for x in miss.elements() if not (len(x) == 2 and ("20" + x) in b)]   # 26年 -> 2026

CALQUES = {   # literal translations of 1688 jargon; pattern in vi -> (zh term, better Vietnamese)
    r"máy ghi (?:âm|hình) lái xe|ghi âm lái xe|ghi hình lái xe": ("行车记录仪", "camera hành trình"),
    r"hộp mật khẩu|khóa mật khẩu": ("密码箱/密码锁", "vali khóa số / khóa số"),
    r"kho báu sạc": ("充电宝", "sạc dự phòng"),
    r"bùng nổ|kiểu nổ|mẫu nổ|hot model": ("爆款", "mẫu bán chạy"),
    r"net red|lưới đỏ|người nổi tiếng internet|internet người nổi tiếng": ("网红", "nổi tiếng trên mạng / hot trend"),
    r"tất cả tiếng anh": ("全英文", "bao bì tiếng Anh"),
    r"thay mặt|một mảnh đại diện|một mảnh thay": ("一件代发", "nhận dropship / bỏ"),
    r"rạp chiếu phim": ("医美/敷料 (膜 bị dịch 'phim')", "dịch lại theo ngữ cảnh"),
    r"túi đựng bánh bao|túi bánh bao": ("饺子包/包子包", "túi dáng bánh bao (kiểm tra)"),
}
PROMO = {"bán buôn": "批发", "xuyên biên giới": "跨境", "nhà máy|nhà sản xuất": "厂家/工厂", "có sẵn|hàng có sẵn": "现货",
         "dropship": "一件代发", "amazon|temu|shein|tiktok": "平台名", "phân phối": "分销/代发"}
END_BAD = re.compile(r"(?:[,\-/+&(]|\b(?:và|cho|của|với|bằng|hoặc|các|là|có|dành|từ))\s*$", re.I)
WORDS = re.compile(r"[^\W\d_]+", re.U)

REP = re.compile(r"(?<!\w)(\w+ \w+(?: \w+)?)[ ,;]+\1(?!\w)")   # same 2-3 word phrase twice in a row

def rep_ngram(vi):
    m = REP.search(vi.lower())
    return m.group(1) if m else ""

def titlecase_ratio(vi):
    w = [x for x in WORDS.findall(vi) if not x.isupper() or len(x) == 1]
    return sum(x[0].isupper() for x in w[1:]) / max(1, len(w) - 1)

vi, zh = df["title_vi"], df["title_zh"]
flags = pd.DataFrame({"product_id": df["product_id"], "category": df["category"], "title_zh": zh, "title_vi": vi})
flags["chu_han_trong_vi"] = vi.str.contains("[\u4e00-\u9fff]")
NO_VI = re.compile("[\\s\\W\\d\u4e00-\u9fff]*")
flags["chua_dich"] = (vi.str.strip() == zh.str.strip()) | vi.map(lambda s: bool(NO_VI.fullmatch(s)))
nz, nv = zh.map(nums), vi.map(nums)
flags["so_zh_mat_trong_vi"] = [", ".join(sorted(missing_nums(a, b))) for a, b in zip(nz, nv)]
flags["so_vi_thua_so_voi_zh"] = [", ".join(sorted((b - a).elements())) for a, b in zip(nz, nv)]
flags["cum_tieng_anh_dai"] = df["english_spans"].map(" || ".join)
calq = []
for s in vi.str.lower():
    calq.append(", ".join(f"{v[0]}->{v[1]}" for p, v in CALQUES.items() if re.search(p, s)))
flags["dich_sat_chu"] = calq
flags["lap_tu_lien_tiep"] = vi.map(rep_ngram)
SYM = re.compile(r"[【】\[\]~|*★☆♥●◆#]")
flags["ngoac_ky_hieu"] = vi.map(lambda s: "".join(sorted(set(SYM.findall(s)))))
flags["ket_thuc_do_dang"] = vi.map(lambda s: bool(END_BAD.search(s)))
ratio = vi.map(lambda s: len(WORDS.findall(s))) / zh.map(lambda s: max(1, len(re.findall(r"[\u4e00-\u9fff]", s))))
flags["ti_le_tu_vi_tren_chu_zh"] = ratio.round(3)
lo, hi = ratio.quantile(0.01), ratio.quantile(0.99)
flags["do_dai_bat_thuong"] = np.where(ratio < lo, "vi_qua_ngan", np.where(ratio > hi, "vi_qua_dai", ""))
flags["titlecase_ratio"] = vi.map(titlecase_ratio).round(2)
flags["viet_hoa_moi_tu"] = flags["titlecase_ratio"] >= 0.6
dup_pair = df.duplicated(["title_zh", "title_vi"], keep=False)
zh_multi_vi = df.groupby("title_zh")["title_vi"].transform("nunique") > 1
vi_multi_zh = df.groupby("title_vi")["title_zh"].transform("nunique") > 1
flags["trung_cap"] = dup_pair
flags["cung_zh_khac_vi"] = zh_multi_vi
flags["cung_vi_khac_zh"] = vi_multi_zh
flags["quang_cao"] = [", ".join(k for k in PROMO if re.search(k, s)) for s in vi.str.lower()]

summary = []
def add(name, mask, meaning, action):
    summary.append({"van_de": name, "so_dong": int(mask.sum()), "ti_le_%": round(100 * mask.mean(), 2),
                    "y_nghia": meaning, "de_xuat": action})
add("chu_han_trong_vi", flags["chu_han_trong_vi"], "title_vi còn chữ Hán", "Dịch lại hoặc bỏ dòng")
add("chua_dich", flags["chua_dich"], "title_vi trùng/không có chữ Việt", "Bỏ dòng")
add("so_zh_mat_trong_vi", flags["so_zh_mat_trong_vi"] != "", "Số trong zh không xuất hiện ở vi", "Đọc lại: mất thông tin (năm, dung tích, số lượng)")
add("so_vi_thua_so_voi_zh", flags["so_vi_thua_so_voi_zh"] != "", "vi có số zh không có (thường do 三→3, OK)", "Chỉ xem nhanh")
add("cum_tieng_anh_dai", flags["cum_tieng_anh_dai"] != "", ">=3 từ tiếng Anh liền nhau", "Dịch lại cụm đó sang tiếng Việt, giữ tên hãng")
add("dich_sat_chu", flags["dich_sat_chu"] != "", "Thuật ngữ 1688 bị dịch từng chữ", "Thay hàng loạt bằng từ điển thuật ngữ (có sẵn cột đề xuất)")
add("lap_tu_lien_tiep", flags["lap_tu_lien_tiep"] != "", "Từ/cụm lặp liền nhau (vd 'dưỡng ẩm dưỡng ẩm')", "Xem: thường do 补水保湿 bị dịch cùng 1 từ -> mất nghĩa")
add("ngoac_ky_hieu", flags["ngoac_ky_hieu"] != "", "Có 【】[]~|* ...", "Chuẩn hoá: bỏ ký hiệu, giữ nội dung")
add("ket_thuc_do_dang", flags["ket_thuc_do_dang"], "Kết thúc bằng 'và/cho/của' hoặc dấu phẩy", "Kiểm tra bị cắt cụt")
add("vi_qua_ngan", flags["do_dai_bat_thuong"] == "vi_qua_ngan", f"Tỉ lệ từ vi / chữ zh < p1={lo:.2f}", "Đọc: thường là dịch THIẾU ý -> bỏ hoặc dịch lại")
add("vi_qua_dai", flags["do_dai_bat_thuong"] == "vi_qua_dai", f"Tỉ lệ từ vi / chữ zh > p99={hi:.2f}", "Đọc: thường là dịch diễn giải dài, ít khi sai")
add("viet_hoa_moi_tu", flags["viet_hoa_moi_tu"], "Viết Hoa Mỗi Từ (>=60% từ)", "Chuẩn hoá chữ thường (giữ tên riêng) cho nhất quán")
add("trung_cap", flags["trung_cap"], "Cặp (zh, vi) trùng hoàn toàn", "Giữ 1")
add("cung_zh_khac_vi", flags["cung_zh_khac_vi"], "Cùng zh nhưng vi khác nhau", "Giữ 1 bản dịch tốt nhất")
add("cung_vi_khac_zh", flags["cung_vi_khac_zh"], "Cùng vi nhưng zh khác nhau", "Xem, thường OK")
add("quang_cao", flags["quang_cao"] != "", "Có chữ quảng cáo 1688 (bán buôn, xuyên biên giới, nhà máy...)", "Dịch đúng, không phải lỗi; cân nhắc nếu muốn title giống sàn VN")
summ = pd.DataFrame(summary)
summ.to_csv(os.path.join(OUT, "title_issues_summary.csv"), index=False, encoding="utf-8-sig")
bool_cols = ["chu_han_trong_vi", "chua_dich", "ket_thuc_do_dang", "trung_cap", "cung_zh_khac_vi"]
str_cols = ["do_dai_bat_thuong", "so_zh_mat_trong_vi", "cum_tieng_anh_dai", "dich_sat_chu", "lap_tu_lien_tiep", "ngoac_ky_hieu"]
anyf = flags[bool_cols].any(axis=1) | (flags[str_cols] != "").any(axis=1)
out_flags = flags[anyf].copy(); out_flags["xu_ly"] = ""; out_flags["ghi_chu"] = ""
out_flags.to_csv(os.path.join(OUT, "title_issues_rows.csv"), index=False, encoding="utf-8-sig")
print("\n=== other problems ===")
print(summ[["van_de", "so_dong", "ti_le_%"]].to_string(index=False))
print("rows with >=1 real problem flag:", int(anyf.sum()))

# ---------------------------------------------------------------- term consistency: same zh term, different vi renderings
TERMS = {
    "行车记录仪": ["camera hành trình", "máy ghi âm lái xe", "máy ghi hình lái xe", "ghi âm lái xe", "ghi hình lái xe"],
    "充电宝": ["sạc dự phòng", "kho báu sạc", "pin dự phòng"],
    "密码箱": ["vali khóa số", "hộp mật khẩu", "vali mật khẩu", "vali mã"],
    "爆款": ["bán chạy", "bùng nổ", "kiểu nổ", "mẫu nổ", "hot"],
    "网红": ["nổi tiếng trên mạng", "net red", "người nổi tiếng", "internet"],
    "一件代发": ["dropship", "một mảnh", "thay mặt", "giao hàng"],
    "老爹鞋": ["dad", "giày bố", "giày ông già", "đế thô", "chunky"],
    "拉杆箱": ["vali kéo", "vali xe đẩy", "hộp xe đẩy", "xe đẩy"],
    "双肩包": ["ba lô", "balo", "túi đeo vai"],
    "补水": ["cấp nước", "cấp ẩm", "dưỡng ẩm", "bổ sung nước"],
    "面膜": ["mặt nạ", "mask"],
}
tc = []
for term, cands in TERMS.items():
    m = zh.str.contains(term)
    sub = vi[m].str.lower()
    row = {"thuat_ngu_zh": term, "so_dong_zh_co": int(m.sum())}
    for c in cands:
        row[c] = int(sub.str.contains(re.escape(c)).sum())
    row["khong_khop_mau_nao"] = int((~sub.apply(lambda s: any(c in s for c in cands))).sum())
    tc.append({"thuat_ngu_zh": term, "so_dong_zh_co": row["so_dong_zh_co"],
               "cach_dich_vi (so_dong)": "; ".join(f"{c}: {row[c]}" for c in cands),
               "khong_khop_mau_nao": row["khong_khop_mau_nao"]})
tcd = pd.DataFrame(tc)
tcd.to_csv(os.path.join(OUT, "title_term_consistency.csv"), index=False, encoding="utf-8-sig")
print("\n=== term consistency ===")
print(tcd.to_string(index=False))

# ---------------------------------------------------------------- spot checks
ap = vi.str.contains(r"\bApple\b")
fruit = zh.str.contains("苹果") & ~zh.str.contains("适用|手机|iPhone|iphone|苹果1|苹果X|苹果手|苹果耳|ipad|耳机|数据线|充电", regex=True)
print(f"\nApple in vi: {int(ap.sum())} | of which zh looks like 苹果=fruit/decor: {int((ap & fruit).sum())}")
for _, r in df[ap & fruit].head(6).iterrows():
    print("   ZH:", r.title_zh[:60], "\n   VI:", r.title_vi[:110])
print("\n=== rows needing Latin action, by category ===")
print(lat_rows["category"].value_counts().to_string())

# ---------------------------------------------------------------- samples for the report
print("\n=== top tokens per group ===")
for gname, sub in tok.groupby("nhom"):
    print(f"[{gname}]", ", ".join(f"{a}({b})" for a, b in zip(sub["token"].head(25), sub["so_dong"].head(25))))
for col in ["cum_tieng_anh_dai", "so_zh_mat_trong_vi", "lap_tu_lien_tiep", "ket_thuc_do_dang", "do_dai_bat_thuong", "cung_zh_khac_vi"]:
    m = flags[col] if flags[col].dtype == bool else flags[col] != ""
    print(f"\n--- sample {col} ---")
    for _, r in flags[m].sample(min(5, int(m.sum())), random_state=0).iterrows():
        print(f"  [{r[col] if not isinstance(r[col], (bool,)) else ''}] ZH: {r.title_zh[:70]}\n      VI: {r.title_vi[:140]}")
