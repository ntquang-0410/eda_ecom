"""B4 bộ chấm tự động cho dịch Trung-Việt thương mại điện tử (title + danh sách thuộc tính).

Chỉ cần thư viện chuẩn + numpy/pandas/sacrebleu (chạy được trên Kaggle tắt internet); COMET/CometKiwi là tùy chọn.
Chạy:
  python -I -X utf8 scripts/ft_train/cham_diem.py --tham-chieu REF.jsonl --du-doan HYP.jsonl --ra-thu-muc DIR
         [--nhanh cum|pair] [--mau-huan-luyen A.jsonl B.jsonl --fold-kiem k] [--comet CKPT] [--kiwi CKPT]
Từ notebook:  tung_muc_df, tong_hop = cham(tham_chieu_df, du_doan_df, ...)

Tham chiếu: id, huong ('zh->vi'|'vi->zh'), loai ('title'|'mo_ta'), nhanh ('cum'|'pair'), category, nguon, dich [, fold5].
Dự đoán: id, hyp [, hyp_tho]. Mục không có hyp tính là hyp rỗng và đếm riêng (thieu_hyp).

Các kiểm riêng đều là HEURISTIC, giới hạn ghi ở docstring từng hàm và ở GIOI_HAN cuối tong_hop.json.
Không đụng đến data/processed/ft_run/test_khoa/: đường dẫn do người gọi truyền vào."""
import argparse
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from chung import TACH_DONG
except Exception:                                    # Kaggle không có chung.py: dùng bản sao
    TACH_DONG = {"cum": "; ", "pair": "\n"}

PHIEN_BAN = "1"
HUONG_HOP_LE = ("zh->vi", "vi->zh")
# ngưỡng mặc định; sẽ chốt sau khi xem đầu ra zero-shot (ghi vào tong_hop.json mỗi lần chấm)
MIN_DOAN, MIN_LAN, HE_SO_DAI, NGUONG_LATIN = 8, 4, 3.0, 0.3
GIOI_HAN = [
    "giu_so: số chữ Hán (三, 二十) và số viết chữ tiếng Việt không được nhận; dạng '1.500' mơ hồ (1500 hay 1,5) coi là 1500; "
    "'25款' <-> '2025' được bỏ qua bằng quy tắc hai chữ số <-> 20xx; quy đổi đơn vị (斤 -> kg) làm số lệch vẫn bị báo",
    "giu_latin: chỉ là đại diện cho thương hiệu/mã model; nguồn Việt chỉ lấy token có chữ số, có -._/+, có chữ hoa giữa từ, hoặc "
    "từ viết hoa đầu không phải âm tiết tiếng Việt không dấu (loại nhãn trước ':' ở mô tả); còn dương tính giả với từ mượn "
    "(Cotton, Polyester...) dù đã có danh sách loại nhỏ",
    "sai_chu_viet: đích Trung chứa chữ Việt có dấu bị cờ kể cả khi đó là tên riêng giữ nguyên; đích Việt chứa chữ Hán bị cờ kể cả tên hãng Trung",
    "dung_so_dong: tách thuần theo TACH_DONG, '; ' nằm trong giá trị tự do cũng bị tách",
    "sinh_lap: ngưỡng chưa chốt, đoạn lặp chu kỳ ngắn cần chạy dài >= min_doan*min_lan ký tự mới bị bắt",
    "cat_loi_dan: một lượt, chỉ các quy tắc ghi rõ; think chưa đóng thẻ thì không cắt",
]

_FULL = {ord(chr(0xFF10 + i)): str(i) for i in range(10)}
_FULL[ord("．")] = "."


# ---------------------------------------------------------------- tiện ích
def chuan_chuoi(x):
    """None/NaN -> ''; còn lại ép str."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def tach(text, nhanh):
    """Tách danh sách thuộc tính theo TACH_DONG[nhanh], bỏ đoạn rỗng."""
    return [p.strip() for p in chuan_chuoi(text).split(TACH_DONG[nhanh]) if p.strip()]


def chuan_dong(s):
    """Khóa so khớp dòng: gộp khoảng trắng."""
    return " ".join(chuan_chuoi(s).split())


# ---------------------------------------------------------------- chrF
def tao_chrf(huong):
    """Đích Việt: chrF++ (word_order=2); đích Trung: chrF ký tự (word_order=0)."""
    if huong not in HUONG_HOP_LE:
        raise ValueError(f"huong_la:{huong}")
    return CHRF(word_order=2 if huong.endswith("vi") else 0)


def chrf_cau(huong, hyp, ref):
    return float(tao_chrf(huong).sentence_score(hyp, [ref]).score)


def chrf_corpus(huong, hyps, refs):
    if len(hyps) == 0:
        return None
    return float(tao_chrf(huong).corpus_score(list(hyps), [list(refs)]).score)


# ---------------------------------------------------------------- COMET (tùy chọn)
def _tai_comet(duong_dan):
    if not duong_dan:
        return None
    try:
        from comet import load_from_checkpoint
    except Exception as e:
        print(f"[comet] không có thư viện comet, bỏ qua: {e}")
        return None
    try:
        return load_from_checkpoint(duong_dan)
    except Exception as e:
        print(f"[comet] không nạp được mô hình {duong_dan}: {e}")
        return None


def _du_doan_comet(mo_hinh, data, batch):
    try:
        import torch
        gpus = 1 if torch.cuda.is_available() else 0
    except Exception:
        gpus = 0
    try:
        out = mo_hinh.predict(data, batch_size=batch, gpus=gpus, progress_bar=False)
        return [float(x) for x in out.scores]
    except Exception as e:
        print(f"[comet] lỗi khi chấm: {e}")
        return None


def cham_comet(nguon, hyp, ref, duong_dan_mo_hinh, batch=16):
    """COMET có tham chiếu. Trả danh sách điểm theo câu, hoặc None nếu thiếu thư viện/mô hình/lỗi."""
    mo_hinh = _tai_comet(duong_dan_mo_hinh)
    if mo_hinh is None:
        return None
    return _du_doan_comet(mo_hinh, [{"src": s, "mt": h, "ref": r} for s, h, r in zip(nguon, hyp, ref)], batch)


def cham_kiwi(nguon, hyp, duong_dan_mo_hinh, batch=16):
    """CometKiwi (không cần tham chiếu). Trả danh sách điểm theo câu, hoặc None như cham_comet."""
    mo_hinh = _tai_comet(duong_dan_mo_hinh)
    if mo_hinh is None:
        return None
    return _du_doan_comet(mo_hinh, [{"src": s, "mt": h} for s, h in zip(nguon, hyp)], batch)


# ---------------------------------------------------------------- giữ số
_RE_SO = re.compile(r"\d+(?:[.,]\d+)*")


def _chuan_so(tok):
    """Một cụm số thô -> danh sách số chuẩn (chuỗi). Dấu phẩy thập phân -> chấm; bỏ dấu phân nghìn khi nhóm đúng 3 chữ số."""
    seps = re.findall(r"[.,]", tok)
    parts = re.split(r"[.,]", tok)
    khong0 = lambda s: s.lstrip("0") or "0"            # noqa: E731
    if not seps:
        return [khong0(tok)]
    if len(set(seps)) == 2:                            # '1.234,56' / '1,234.56': dấu cuối là thập phân
        ngan = seps[-1]
        phan = tok.replace("." if ngan == "," else ",", "")
        nguyen, le = phan.split(ngan)
        le = le.rstrip("0")
        return [khong0(nguyen) + ("." + le if le else "")]
    if len(seps) == 1:
        nguyen, le = parts
        if len(le) == 3 and nguyen != "0" and len(nguyen) <= 3:        # 2,500 / 1.250 -> hàng nghìn
            return [khong0(nguyen + le)]
        le = le.rstrip("0")
        return [khong0(nguyen) + ("." + le if le else "")]
    if all(len(p) == 3 for p in parts[1:]) and len(parts[0]) <= 3:      # 1.000.000
        return [khong0("".join(parts))]
    return [khong0(p) for p in parts]                  # '1,2,3' hay số phiên bản: tách rời


def so_tu_text(text):
    """Tập số chuẩn hóa trong chuỗi (chữ số toàn chiều -> ASCII)."""
    t = chuan_chuoi(text).translate(_FULL)
    kq = set()
    for m in _RE_SO.findall(t):
        kq.update(_chuan_so(m))
    return kq


def giu_so(nguon, hyp):
    """Số trong nguồn phải có trong hyp và ngược lại. Trả dict(dat, mat, la).
    Giới hạn: không nhận số chữ (三, 'hai mươi'); '1.500' coi là 1500; '25款' ~ '2025' được bỏ qua bằng quy tắc 2 chữ số <-> 20xx."""
    a, b = so_tu_text(nguon), so_tu_text(hyp)
    mat, la = a - b, b - a
    for m in list(mat):
        if m.isdigit() and len(m) == 2 and "20" + m in la:
            mat.discard(m)
            la.discard("20" + m)
    for x in list(la):
        if x.isdigit() and len(x) == 2 and "20" + x in mat:
            la.discard(x)
            mat.discard("20" + x)
    key = lambda s: (len(s), s)                        # noqa: E731
    return {"dat": not mat and not la, "mat": sorted(mat, key=key), "la": sorted(la, key=key)}


# ---------------------------------------------------------------- giữ token Latin
_NL = "À-ɏḀ-ỿ"                     # chữ Latin có dấu: token ASCII không được dính vào chúng
_RE_LATIN = re.compile(rf"(?<![A-Za-z0-9{_NL}])[A-Za-z0-9][A-Za-z0-9\-\._/+]{{1,}}(?![A-Za-z0-9{_NL}])")
_DON_VI = ("kg|g|mg|ml|l|cm|mm|m|km|w|v|a|hz|khz|mhz|ghz|mah|gb|tb|mb|kb|oz|lb|lbs|pcs|pc|cl|dl|ah|kw|kv|ma|mw|db|cc|gr")
_RE_DON_VI = re.compile(rf"^[\d.,x*×/\-+]*(?:{_DON_VI})$")
# âm tiết tiếng Việt không dấu: phụ âm đầu + 1-3 nguyên âm + phụ âm cuối
_RE_AM_TIET = re.compile(r"^(?:ngh|ng|nh|ph|th|tr|ch|kh|gh|gi|qu|[bcdghklmnpqrstvx])?[aeiouy]{1,3}(?:ng|nh|ch|[cmnpt])?$")
_TU_MUON = {"cotton", "polyester", "nylon", "spandex", "inox", "silicone", "plastic", "metal", "leather", "latex", "acrylic",
            "combo", "freeship", "size"}


def _bo_nhan(nguon, nhanh):
    """Mô tả: bỏ phần nhãn trước dấu ':' đầu tiên của mỗi thuộc tính, chỉ giữ giá trị."""
    kq = []
    for dong in chuan_chuoi(nguon).split(TACH_DONG[nhanh]):
        i = min([k for k in (dong.find(":"), dong.find("：")) if k >= 0], default=-1)
        kq.append(dong[i + 1:] if i >= 0 else dong)
    return " ".join(kq)


def _nhan_vi(t):
    """Nguồn Việt: token ASCII nào đáng coi là tên riêng/mã (heuristic, xem GIOI_HAN)."""
    if any(c.isdigit() for c in t) or re.search(r"[\-\._/+]", t):
        return True
    if any(c.isupper() for c in t[1:]):                # ALLCAPS hay iPhone
        return True
    return t[0].isupper() and len(t) >= 3 and not _RE_AM_TIET.match(t.lower()) and t.lower() not in _TU_MUON


def token_latin(nguon, huong="zh->vi", loai="title", nhanh="cum"):
    """Token Latin/số-chữ đại diện thương hiệu/mã model, chữ thường, không trùng lặp, theo thứ tự xuất hiện."""
    text = chuan_chuoi(nguon).translate(_FULL)
    viet = huong.startswith("vi")
    if viet and loai == "mo_ta":
        text = _bo_nhan(text, nhanh)
    kq = []
    for m in _RE_LATIN.finditer(text):
        t = m.group(0).rstrip("-._/+")
        if len(t) < 2 or not any(c.isalpha() for c in t) or _RE_DON_VI.match(t.lower()):
            continue
        if viet and not _nhan_vi(t):
            continue
        if t.lower() not in kq:
            kq.append(t.lower())
    return kq


def _gon(s):
    return re.sub(r"[\s\-\._/+]", "", s)


def giu_latin(nguon, hyp, huong="zh->vi", loai="title", nhanh="cum"):
    """Mọi token Latin của nguồn phải có trong hyp (không phân biệt hoa thường, so cả dạng bỏ khoảng trắng/gạch). Dict(dat, mat, tong).
    Heuristic: thương hiệu/mã model; dương tính giả với từ mượn được dịch hẳn sang chữ khác."""
    toks = token_latin(nguon, huong, loai, nhanh)
    h = chuan_chuoi(hyp).translate(_FULL).lower()
    hg = _gon(h)
    mat = [t for t in toks if t not in h and _gon(t) not in hg]
    return {"dat": not mat, "mat": mat, "tong": len(toks)}


# ---------------------------------------------------------------- sai chữ viết
_RE_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_VIET_DAU = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
_VIET_DAU |= {c.upper() for c in _VIET_DAU}


def sai_chu_viet(huong, nguon, hyp, nguong_latin=NGUONG_LATIN):
    """Đích Việt chứa chữ Hán -> cờ. Đích Trung chứa chữ Việt có dấu, hoặc tỉ lệ chữ Latin (không tính từ ASCII đã có trong nguồn)
    trên tổng ký tự chữ > nguong_latin (và >= 4 chữ) -> cờ. Dict(co, ly_do, so_han, so_viet_dau, ti_le_latin)."""
    h = chuan_chuoi(hyp)
    so_han = len(_RE_HAN.findall(h))
    if huong.endswith("vi"):
        return {"co": so_han > 0, "ly_do": "co_chu_han" if so_han else "", "so_han": so_han, "so_viet_dau": 0, "ti_le_latin": 0.0}
    so_dau = sum(1 for c in h if c in _VIET_DAU)
    trong_nguon = {r.lower() for r in re.findall(r"[A-Za-z]{2,}", chuan_chuoi(nguon))}
    con_lai = sum(len(r) for r in re.findall(r"[A-Za-z]{2,}", h) if r.lower() not in trong_nguon)
    tong = sum(1 for c in h if c.isalpha())
    ti_le = con_lai / tong if tong else 0.0
    ly = []
    if so_dau:
        ly.append("chu_viet_dau")
    if con_lai >= 4 and ti_le > nguong_latin:
        ly.append("nhieu_latin")
    return {"co": bool(ly), "ly_do": "+".join(ly), "so_han": so_han, "so_viet_dau": so_dau, "ti_le_latin": ti_le}


# ---------------------------------------------------------------- số dòng, sinh lặp, độ dài
def dung_so_dong(hyp, ref, nhanh):
    """Số thuộc tính hyp == số thuộc tính tham chiếu (tách theo TACH_DONG[nhanh]). Chỉ dùng cho loai == 'mo_ta'."""
    return len(tach(hyp, nhanh)) == len(tach(ref, nhanh))


@functools.lru_cache(maxsize=None)
def _re_lap(min_doan, min_lan):
    return re.compile(rf"(.{{{min_doan},{max(200, min_doan)}}}?)\1{{{min_lan - 1},}}")


def sinh_lap(hyp, ref, min_doan=MIN_DOAN, min_lan=MIN_LAN, he_so_dai=HE_SO_DAI):
    """Lặp vô hạn: đoạn >= min_doan ký tự (bỏ khoảng trắng) lặp liên tiếp >= min_lan lần, hoặc len(hyp) > he_so_dai*len(ref)+20.
    Dict(co, ly_do)."""
    h = chuan_chuoi(hyp)
    s = re.sub(r"\s+", "", h)
    ly = []
    if len(s) >= min_doan * min_lan and _re_lap(min_doan, min_lan).search(s):
        ly.append("doan_lap")
    if len(h) > he_so_dai * len(chuan_chuoi(ref)) + 20:
        ly.append("qua_dai")
    return {"co": bool(ly), "ly_do": "+".join(ly)}


def ti_le_do_dai(hyp, ref):
    n = len(chuan_chuoi(ref))
    return len(chuan_chuoi(hyp)) / n if n else float("nan")


# ---------------------------------------------------------------- dòng đã thấy / mới
def xay_tap_da_thay(mau, fold_kiem=None, nhanh=None):
    """Tập (category, dòng_nguồn) từ mẫu HUẤN LUYỆN (DataFrame cột loai, category, nguon, nhanh, fold5).
    Có fold5: chỉ lấy fold5 >= 0, và nếu fold_kiem != None thì bỏ fold5 == fold_kiem. Chỉ loai == 'mo_ta'."""
    d = mau[mau.loai == "mo_ta"]
    if "fold5" in d.columns:
        d = d[d.fold5 >= 0]
        if fold_kiem is not None:
            d = d[d.fold5 != fold_kiem]
    tap = set()
    for r in d.itertuples(index=False):
        nh = getattr(r, "nhanh", None) or nhanh
        for dong in tach(r.nguon, nh):
            tap.add((chuan_chuoi(getattr(r, "category", "")), chuan_dong(dong)))
    return tap


# ---------------------------------------------------------------- cắt lời dẫn
_INTRO = (r"(?:(?:here\s+is|here's|here\s+are|below\s+is|this\s+is)\s+(?:the\s+)?(?:(?:vietnamese|chinese|english|final)\s+)?"
          r"(?:translation|translated\s+text)(?:\s+of\s+[^:：\n]{1,60})?"
          r"|(?:vietnamese\s+|chinese\s+)?translation|translated\s+text"
          r"|(?:(?:đây\s+là|dưới\s+đây\s+là)\s+)?bản\s+dịch(?:\s+tiếng\s+(?:việt|trung))?|dịch"
          r"|(?:以下是)?翻[译譯](?:结果)?|译文)")
_RE_DONG_DAN = re.compile(rf"^\s*{_INTRO}\s*[:：]?\s*$", re.I)
_RE_TIEN_TO = re.compile(rf"^\s*{_INTRO}\s*[:：]\s*(\S.*)$", re.I | re.S)
_RE_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_RE_RAO = re.compile(r"^\s*```[^\n`]*\n(.*?)\n?[ \t]*```\s*$", re.S)
_RE_RAO_DONG = re.compile(r"^\s*```(.*?)```\s*$", re.S)
_CAP_NGOAC = {'"': '"', "“": "”", "‘": "’", "'": "'", "«": "»", "「": "」", "『": "』", "„": "“"}


def cat_loi_dan(text):
    """Cắt lời dẫn, một lượt, theo thứ tự (tên quy tắc ghi trong chuỗi trả về, ngăn bằng ','):
      think        bỏ mọi khối <think>...</think> (kể cả rỗng);
      rao_code     bỏ rào ``` bao toàn bộ (kể cả dòng ```lang);
      dong_dan     bỏ dòng đầu (không rỗng) nếu CẢ DÒNG là lời dẫn: 'Here is the translation:', 'Translation:', 'Bản dịch:', 'Dịch:',
                   '翻译：', '译文：' (không phân biệt hoa thường, có/không dấu hai chấm);
      tien_to_dan  nếu lời dẫn nằm cùng dòng với nội dung thì phải CÓ dấu hai chấm ('Translation: Áo thun' -> 'Áo thun');
                   'Dịch vụ bảo hành: Có' giữ nguyên vì sau 'Dịch' không phải dấu hai chấm;
      ngoac_kep    bỏ cặp ngoặc kép bao toàn bộ nếu bên trong không còn ngoặc cùng loại;
      rồi strip hai đầu (không tính là 'đã cắt').
    Trả (sạch, đã_cắt, quy_tắc)."""
    t = chuan_chuoi(text)
    qt = []
    t2 = _RE_THINK.sub("", t)
    if t2 != t:
        qt.append("think")
        t = t2
    t = t.strip()
    m = _RE_RAO.match(t) or _RE_RAO_DONG.match(t)
    if m:
        t = m.group(1).strip()
        qt.append("rao_code")
    dong = t.split("\n")
    i = next((k for k, d in enumerate(dong) if d.strip()), None)
    if i is not None:
        if _RE_DONG_DAN.match(dong[i]):
            t = "\n".join(dong[i + 1:]).strip()
            qt.append("dong_dan")
        else:
            m = _RE_TIEN_TO.match(dong[i])
            if m:
                t = "\n".join([m.group(1)] + dong[i + 1:]).strip()
                qt.append("tien_to_dan")
    if len(t) >= 2 and t[0] in _CAP_NGOAC and t[-1] == _CAP_NGOAC[t[0]]:
        trong = t[1:-1]
        if t[0] not in trong and _CAP_NGOAC[t[0]] not in trong:
            t = trong
            qt.append("ngoac_kep")
    return t.strip(), bool(qt), ",".join(qt)


# ---------------------------------------------------------------- chấm
def _chuan_tham_chieu(tc, nhanh):
    d = tc.copy()
    for c in ("id", "huong", "loai", "nguon", "dich"):
        if c not in d.columns:
            raise ValueError(f"thieu_cot_tham_chieu:{c}")
    d["id"] = d["id"].astype(str)
    if d["id"].duplicated().any():
        raise ValueError("id_tham_chieu_trung")
    if "category" not in d.columns:
        d["category"] = ""
    d["category"] = d["category"].map(chuan_chuoi)
    if "nhanh" in d.columns:
        if nhanh:
            d = d[d["nhanh"] == nhanh]
    else:
        if nhanh is None and (d["loai"] == "mo_ta").any():
            raise ValueError("thieu_nhanh: truyền nhanh hoặc thêm cột nhanh")
        d["nhanh"] = nhanh or ""
    bad = set(d["huong"]) - set(HUONG_HOP_LE)
    if bad:
        raise ValueError(f"huong_la:{bad}")
    return d.reset_index(drop=True)


def _lay_du_doan(dd):
    """id -> hyp; trả (dict, số id trùng). hyp thiếu/NaN không vào dict."""
    kq, trung = {}, 0
    for r in dd.to_dict("records"):
        k = str(r["id"])
        if k in kq:
            trung += 1
        h = r.get("hyp")
        if h is None or (isinstance(h, float) and np.isnan(h)):
            kq.pop(k, None)
            continue
        kq[k] = str(h)
    return kq, trung


def cham(tham_chieu, du_doan, nhanh=None, tap_da_thay=None, comet=None, kiwi=None, batch=16, min_doan=MIN_DOAN,
         min_lan=MIN_LAN, he_so_dai=HE_SO_DAI, nguong_latin=NGUONG_LATIN, cat_loi_dan_bat=True):
    """Chấm toàn bộ. Trả (tung_muc_df, tong_hop_dict).
    tap_da_thay: tập (category, dòng_nguồn) từ xay_tap_da_thay(); None thì bỏ phần dòng đã thấy/mới.
    comet / kiwi: đường dẫn checkpoint (tùy chọn). Cột bắt đầu bằng '_' là phụ trợ, không ghi ra file."""
    ref = _chuan_tham_chieu(tham_chieu, nhanh)
    hyps, n_trung = _lay_du_doan(du_doan)
    ids_ref = set(ref["id"])
    n_thua = len(set(hyps) - ids_ref)
    rows = []
    for r in ref.to_dict("records"):
        huong, loai, nh = r["huong"], r["loai"], r["nhanh"]
        nguon, dich = chuan_chuoi(r["nguon"]), chuan_chuoi(r["dich"])
        thieu = r["id"] not in hyps
        hyp = hyps.get(r["id"], "")
        if cat_loi_dan_bat:
            hyp, da_cat, qt = cat_loi_dan(hyp)
        else:
            hyp, da_cat, qt = hyp, False, ""
        so, la = giu_so(nguon, hyp), giu_latin(nguon, hyp, huong, loai, nh)
        sc, lap = sai_chu_viet(huong, nguon, hyp, nguong_latin), sinh_lap(hyp, dich, min_doan, min_lan, he_so_dai)
        row = {"id": r["id"], "huong": huong, "loai": loai, "category": r["category"], "nhanh": nh, "thieu_hyp": thieu,
               "hyp_cham": hyp, "da_cat_loi_dan": da_cat, "quy_tac_cat": qt, "chrf": chrf_cau(huong, hyp, dich),
               "giu_so_dat": so["dat"], "so_mat": so["mat"], "so_la": so["la"],
               "giu_latin_dat": la["dat"], "latin_mat": la["mat"], "latin_tong": la["tong"],
               "sai_chu_viet": sc["co"], "sai_chu_viet_ly_do": sc["ly_do"], "so_han": sc["so_han"],
               "so_viet_dau": sc["so_viet_dau"], "ti_le_latin": sc["ti_le_latin"],
               "dung_so_dong": None, "so_dong_hyp": None, "so_dong_ref": None,
               "sinh_lap": lap["co"], "sinh_lap_ly_do": lap["ly_do"], "ti_le_do_dai": ti_le_do_dai(hyp, dich),
               "_nguon": nguon, "_ref": dich}
        if loai == "mo_ta":
            hl, rl, sl = tach(hyp, nh), tach(dich, nh), tach(nguon, nh)
            row.update(dung_so_dong=len(hl) == len(rl), so_dong_hyp=len(hl), so_dong_ref=len(rl))
            if tap_da_thay is not None:
                if len(hl) == len(rl) == len(sl):
                    thay = [(r["category"], chuan_dong(x)) in tap_da_thay for x in sl]
                    row.update(can_dong=True, n_dong_da_thay=sum(thay), n_dong_moi=len(thay) - sum(thay),
                               _dong=list(zip(thay, hl, rl)))
                else:
                    row.update(can_dong=False, n_dong_da_thay=None, n_dong_moi=None, _dong=None)
        rows.append(row)
    df = pd.DataFrame(rows)
    ghi_chu_comet = {}
    for ten, duong_dan in (("comet", comet), ("kiwi", kiwi)):
        if not duong_dan:
            continue
        if ten == "comet":
            diem = cham_comet(df["_nguon"].tolist(), df["hyp_cham"].tolist(), df["_ref"].tolist(), duong_dan, batch)
        else:
            diem = cham_kiwi(df["_nguon"].tolist(), df["hyp_cham"].tolist(), duong_dan, batch)
        ghi_chu_comet[ten] = "da_chay" if diem is not None else "khong_chay_duoc (thieu thu vien comet hoac mo hinh)"
        if diem is not None:
            df[ten] = diem
    cau_hinh = {"min_doan": min_doan, "min_lan": min_lan, "he_so_dai": he_so_dai, "nguong_latin": nguong_latin,
                "cat_loi_dan": cat_loi_dan_bat, "nhanh": nhanh, "co_dong_da_thay_moi": tap_da_thay is not None,
                "so_dong_da_thay_trong_tap": None if tap_da_thay is None else len(tap_da_thay)}
    return df, tong_hop(df, cau_hinh, n_thua, n_trung, ghi_chu_comet)


# ---------------------------------------------------------------- tổng hợp
def _tom_tat(d, huong, day_du=True):
    out = {"n": int(len(d)), "thieu_hyp": int(d["thieu_hyp"].sum()),
           "chrf_corpus": chrf_corpus(huong, d["hyp_cham"], d["_ref"]), "chrf_cau_tb": float(d["chrf"].mean())}
    if not day_du:
        return out
    out.update(
        ti_le_giu_so_dat=float(d["giu_so_dat"].mean()), ti_le_giu_latin_dat=float(d["giu_latin_dat"].mean()),
        ti_le_sai_chu_viet=float(d["sai_chu_viet"].mean()), ti_le_sinh_lap=float(d["sinh_lap"].mean()),
        ti_le_da_cat_loi_dan=float(d["da_cat_loi_dan"].mean()),
        ti_le_do_dai_tb=float(d["ti_le_do_dai"].mean()), ti_le_do_dai_trung_vi=float(d["ti_le_do_dai"].median()))
    if d["dung_so_dong"].notna().any():
        out["ti_le_dung_so_dong"] = float(d["dung_so_dong"].dropna().astype(bool).mean())
    for c in ("comet", "kiwi"):
        if c in d.columns:
            out[f"{c}_tb"] = float(d[c].mean())
    return out


def tong_hop(df, cau_hinh, n_hyp_thua=0, n_trung=0, ghi_chu_comet=None):
    """Nhóm (huong, loai), (huong, loai, category); điểm chính = macro theo hướng (không trọng số theo số mục/token)."""
    nhom = {}
    for (h, l), d in df.groupby(["huong", "loai"], sort=True):
        nhom[f"{h}/{l}"] = _tom_tat(d, h)
    theo_cat = {}
    for (h, l, c), d in df.groupby(["huong", "loai", "category"], sort=True):
        theo_cat[f"{h}/{l}/{c}"] = _tom_tat(d, h, day_du=False)
    theo_huong = {}
    for h in sorted(df["huong"].unique()):
        loai_co = [l for l in ("title", "mo_ta") if f"{h}/{l}" in nhom]
        theo_huong[h] = {"loai_gom": loai_co,
                         "chrf_corpus": float(np.mean([nhom[f"{h}/{l}"]["chrf_corpus"] for l in loai_co])),
                         "chrf_cau_tb": float(np.mean([nhom[f"{h}/{l}"]["chrf_cau_tb"] for l in loai_co]))}
    diem_chinh = {"cach_tinh": "chrF corpus: mỗi hướng = trung bình đều của title và mo_ta; điểm chính = trung bình đều các hướng "
                               "(zh->vi dùng chrF++, vi->zh dùng chrF ký tự; không trọng số theo số mục/token)",
                  "huong_gom": sorted(theo_huong),
                  "chrf_macro_huong": float(np.mean([v["chrf_corpus"] for v in theo_huong.values()])),
                  "chrf_cau_tb_macro_huong": float(np.mean([v["chrf_cau_tb"] for v in theo_huong.values()]))}
    cat = {}
    for (h, l), d in df.groupby(["huong", "loai"], sort=True):
        cat[f"{h}/{l}"] = {"so_cat": int(d["da_cat_loi_dan"].sum()), "ti_le": float(d["da_cat_loi_dan"].mean()),
                           "theo_quy_tac": {k: int(v) for k, v in
                                            pd.Series([x for q in d["quy_tac_cat"] for x in q.split(",") if x]).value_counts().items()}}
    tm = {"phien_ban": {"cham_diem": PHIEN_BAN, "sacrebleu": sacrebleu.__version__},
          "cau_hinh": cau_hinh, "so_muc": int(len(df)), "thieu_hyp": int(df["thieu_hyp"].sum()),
          "thieu_hyp_theo_nhom": {k: v["thieu_hyp"] for k, v in nhom.items()},
          "hyp_thua_khong_co_trong_tham_chieu": int(n_hyp_thua), "id_du_doan_trung": int(n_trung),
          "diem_chinh": diem_chinh, "theo_huong": theo_huong, "theo_nhom": nhom, "theo_category": theo_cat,
          "cat_loi_dan": cat, "comet": ghi_chu_comet or {}, "gioi_han": GIOI_HAN}
    if "can_dong" in df.columns:
        dm = {}
        for h, d in df[df["loai"] == "mo_ta"].groupby("huong", sort=True):
            can = d[d["can_dong"] == True]       # noqa: E712
            dong = [x for l in can["_dong"] for x in l]
            ph = {}
            for ten, co in (("da_thay", True), ("moi", False)):
                p = [(hy, rf) for t, hy, rf in dong if t == co]
                ph[ten] = {"n_dong": len(p), "chrf_corpus": chrf_corpus(h, [a for a, _ in p], [b for _, b in p])}
            ph["so_muc_khong_can_duoc"] = int(len(d) - len(can))
            ph["so_muc_mo_ta"] = int(len(d))
            dm[h] = ph
        tm["dong_da_thay_moi"] = dm
    return tm


# ---------------------------------------------------------------- vào/ra
def _sach(o):
    if isinstance(o, dict):
        return {str(k): _sach(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_sach(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) or np.isinf(o) else float(o)
    return o


def doc_jsonl(path, cot=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for dong in f:
            if dong.strip():
                o = json.loads(dong)
                rows.append({k: o.get(k) for k in cot} if cot else o)
    return pd.DataFrame(rows)


def ghi_ket_qua(df, tm, thu_muc):
    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    cot = [c for c in df.columns if not c.startswith("_")]
    with open(thu_muc / "diem_tung_muc.jsonl", "w", encoding="utf-8") as f:
        for r in df[cot].to_dict("records"):
            f.write(json.dumps(_sach(r), ensure_ascii=False) + "\n")
    (thu_muc / "tong_hop.json").write_text(json.dumps(_sach(tm), ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tham-chieu", required=True)
    ap.add_argument("--du-doan", required=True)
    ap.add_argument("--ra-thu-muc", required=True)
    ap.add_argument("--nhanh", choices=["cum", "pair"])
    ap.add_argument("--mau-huan-luyen", nargs="+")
    ap.add_argument("--fold-kiem", type=int)
    ap.add_argument("--comet")
    ap.add_argument("--kiwi")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-doan", type=int, default=MIN_DOAN)
    ap.add_argument("--min-lan", type=int, default=MIN_LAN)
    ap.add_argument("--he-so-dai", type=float, default=HE_SO_DAI)
    ap.add_argument("--nguong-latin", type=float, default=NGUONG_LATIN)
    ap.add_argument("--khong-cat-loi-dan", action="store_true")
    a = ap.parse_args(argv)
    tc, dd = doc_jsonl(a.tham_chieu), doc_jsonl(a.du_doan, ["id", "hyp"])
    tap = None
    if a.mau_huan_luyen:
        mau = pd.concat([doc_jsonl(p, ["id", "category", "loai", "nhanh", "fold5", "nguon"]) for p in a.mau_huan_luyen],
                        ignore_index=True)
        tap = xay_tap_da_thay(mau, a.fold_kiem, a.nhanh)
    df, tm = cham(tc, dd, nhanh=a.nhanh, tap_da_thay=tap, comet=a.comet, kiwi=a.kiwi, batch=a.batch, min_doan=a.min_doan,
                  min_lan=a.min_lan, he_so_dai=a.he_so_dai, nguong_latin=a.nguong_latin, cat_loi_dan_bat=not a.khong_cat_loi_dan)
    if a.fold_kiem is not None:
        tm["cau_hinh"]["fold_kiem"] = a.fold_kiem
    ghi_ket_qua(df, tm, a.ra_thu_muc)
    dc = tm["diem_chinh"]
    print(f"chrF macro theo hướng: {dc['chrf_macro_huong']:.2f}  (số mục {tm['so_muc']}, thiếu hyp {tm['thieu_hyp']})")
    for h, v in tm["theo_huong"].items():
        print(f"  {h}: {v['chrf_corpus']:.2f}  ({'+'.join(v['loai_gom'])})")
    return df, tm


if __name__ == "__main__":
    main()
