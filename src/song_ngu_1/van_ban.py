# Hàm xử lý chữ dùng chung: nhận diện chữ Hán, chuẩn hoá để so hai bản Việt, chrF.
import re
import unicodedata
from collections import Counter

CJK = re.compile("[一-鿿]")


def co_han(s):
    return bool(CJK.search(s or ""))


# Kiểu đặt dấu thanh cũ và mới coi như nhau: hòa = hoà, thủy = thuỷ, khỏe = khoẻ
_DAU_CU = {}
for _nguyen, _phu in (("o", "a"), ("o", "e"), ("u", "y")):
    for _thanh in "̣̀́̉̃":   # huyền, sắc, hỏi, ngã, nặng
        _cu = unicodedata.normalize("NFC", _nguyen + _thanh + _phu)
        _moi = unicodedata.normalize("NFC", _nguyen + _phu + _thanh)
        _DAU_CU[_cu] = _moi
_RX_DAU_CU = re.compile("|".join(map(re.escape, _DAU_CU)))


def chuan_hoa(s):
    """Chuẩn hoá một bản Việt để so với bản kia (không dùng làm bản xuất ra)."""
    s = unicodedata.normalize("NFC", s or "").lower().strip()
    s = _RX_DAU_CU.sub(lambda m: _DAU_CU[m.group(0)], s)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)                 # 7,5 = 7.5
    s = re.sub(r"\s*[×xX*]\s*(?=\d)", "x", s)             # 3 × 4, 3x4, 3*4 như nhau
    s = re.sub(r"(?<=\d)\s+(?=[a-zà-ỹ])", "", s)          # 30 kg = 30kg
    s = re.sub(r"[\s]+", " ", s)
    s = s.strip(" .。,，;；:：!！")
    return s


def _ngram(s, n):
    s = "".join(s.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(a, b, n_max=6, beta=2.0):
    """chrF (n-gram ký tự 1..6, bỏ khoảng trắng, beta = 2), thang 0-100. Bản tự viết, gần với sacrebleu nhưng không giống hệt."""
    p_ds, r_ds = [], []
    for n in range(1, n_max + 1):
        ga, gb = _ngram(a, n), _ngram(b, n)
        if not ga or not gb:
            continue
        chung = sum((ga & gb).values())
        p_ds.append(chung / sum(ga.values()))
        r_ds.append(chung / sum(gb.values()))
    if not p_ds:
        return 100.0 if a == b else 0.0
    p, r = sum(p_ds) / len(p_ds), sum(r_ds) / len(r_ds)
    if p + r == 0:
        return 0.0
    return 100.0 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
