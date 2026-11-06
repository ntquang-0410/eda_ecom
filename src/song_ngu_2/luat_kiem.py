# Bước 5 của bộ 2 (laptop, CPU): luật kiểm từng bản dịch Việt -> Trung. Bản gương của song_ngu_1.luat_kiem, hai mức:
# trượt (lỗi nặng, bản đó coi như hỏng) và cờ nhắc (chỉ ghi chú). Điểm khác chính so với chiều Trung -> Việt:
#   - "còn sót ngôn ngữ nguồn" là chữ Việt có dấu (viet_sot), chữ Hán trong bản dịch là điều đúng;
#   - số kiểu Việt (1.000 = một nghìn, 1,5 = một phẩy năm) phải đối chiếu được với số kiểu Trung (1000, 1.5);
#   - mã Latin chỉ bắt buộc giữ khi thật sự là mã (không nhầm chữ Việt in hoa không dấu như XANH, MAU);
#   - bẫy đơn vị đảo chiều: ở Việt "cân" là kg nên bản Trung viết 斤 là sai gấp đôi (jin_sai);
#   - lặp ký tự Hán (không có dấu cách) đếm bằng ký tự, không đếm từ.
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from song_ngu_1.luat_kiem import AM_TIET, doc_tu_dien, do_dai_bat_thuong, khoang_do_dai  # noqa: E402,F401  (dùng lại: không phụ thuộc ngôn ngữ)
from song_ngu_1.van_ban import chuan_hoa  # noqa: E402,F401

from .dinh_tuyen import DON_VI_DO  # noqa: E402

HAN = re.compile("[㐀-䶿一-鿿豈-﫿]")
VIET_DAU = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
LOI_THUA = re.compile(r"^\s*(译文|翻译|中文|简体中文|简体|答案|回答|translation|chinese|simplified chinese)\s*[:：]|<think>|</think>|```", re.I)
MARKDOWN_DAM = "**"      # chỉ là lời thừa khi nguồn không có (nguồn Tiki hay có "***", "**" làm dấu nhấn, bản dịch giữ nguyên là đúng)
# mô hình dịch luôn phần ngữ cảnh của câu lệnh (danh mục, tên thuộc tính) hoặc gắn nhãn "Xxx：" trước bản dịch
NGU_CANH_ZH = re.compile(r"产品类别|商品类别|属性名|属性值|属性名称|product category|attribute (?:name|value)", re.I)
NGU_CANH_NGUON = re.compile(r"thuộc tính|danh mục|ngành hàng|loại sản phẩm|product category|attribute", re.I)
NHAN_DAU = re.compile(r"^[^:：,，;；]{1,20}[:：]\s*")
NHAN_NGAN = 40       # giá trị nguồn ngắn hơn ngần này ký tự mà bản dịch mở đầu bằng "Xxx：" là mô hình in tên thuộc tính ra (lỗi cứng)
LAP_HAN = re.compile(r"([一-鿿]{1,6})\1{3,}")
SO = re.compile(r"\d+")
SO_CHU_VIET = re.compile(r"\b(?:một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười|trăm|nghìn|ngàn|nửa|rưỡi|triệu)\b", re.I)
TU_LATIN = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+")
HANG_NGHIN = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")
CHUNK = re.compile(r"[\s,;，；()（）\[\]【】{}<>|:：]+")
MA_LATIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.+/*#&]*")
LATIN_QUEN = {"kg", "g", "mg", "cm", "mm", "m", "ml", "l", "inch", "size", "set", "combo", "usb", "led", "lcd", "pc", "pcs", "oem", "logo",
              "pin", "wifi", "bluetooth", "type", "mini", "max", "pro", "plus", "ok", "cotton", "pvc", "abs", "pu", "pp", "pe", "tpu", "eva",
              "spa", "gel", "vitamin", "collagen", "silicon", "silicone", "nylon", "polyester", "xl", "xxl", "gps", "hdmi", "sim", "app", "diy"}
CHU_SO_HAN = {"0": "零半", "1": "一单独", "2": "二两双对", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八", "9": "九", "10": "十"}
# Chữ Hán phồn thể phổ biến mà giản thể không dùng (bản dịch phải là giản thể). Chỉ những chữ chắc chắn không còn dùng trong giản thể.
PHON_THE = set("國這個們說對點裡後來麼長開關門顏設計體種類電氣車買賣區質線號時間東萬與專業務價優據產選擇護養藥級環練圖圓帶幫載輪動適應無麗學發現"
               "實經過還進遠運農鐘錶鐵銅鋁鋼銀針織絲紅綠藍黃陽陰雲風飛馬魚鳥龍貝見親觀覺讀寫語話請認識讓該試訂記論許訊調談謝證變競豐彈強歸當錄師帥幣"
               "廣廠異鬆輕鬥齊歲歷壓聽聲響麵廚藝樂標檔樣橫機條構權構殺氫淺濕滿滾潔澤濾熱爲燈爐牆環現瑪產畫異療盤盞確碼礦禮禱稱穩筆範築簡籃糧級細終結給統維網羅罰羣職腳臺與興舊艱莊葉蓋薑藥蟲蠟衛補裝褲複觸課負貨貼貿費資賓趕躍軟轉軸輕載較輸邊達遞還鄉鑰鑽門閃開閉間閒關陣階際隨險雙雜離難電霧靜頭頸顆願類顯飲餅館餘騎驗體髮鬍魚鮮鹽麥黨")


def _dem(s):
    """Số có trong chuỗi, mỗi số quy về chuỗi số nguyên không chữ số 0 đầu (05 = 5)."""
    return Counter(str(int(x)) for x in SO.findall(s))


DAI_SO = 100          # giá trị nguồn dài hơn ngần này ký tự: cho phép mất một ít số khi dịch văn bản (1 ly, 2 cạnh, hậu tố hoá chất)


def so_thieu(vi, zh):
    """Số lượng số của bản Việt không còn ở bản Trung (lấy cách đếm ít thiếu nhất). Cho phép: dấu tách hàng nghìn kiểu Việt
    (1.000 = 1000), 万 (10.000 = 1万), số Hán thay chữ số nhỏ (1 = 一/单, 2 = 二/两/双, 0 = 零/半)."""
    goc = _dem(vi)
    if not goc:
        return 0
    gon = _dem(HANG_NGHIN.sub("", vi))
    co = _dem(zh) | _dem(HANG_NGHIN.sub("", zh))
    tot = None
    for can in (goc, gon):
        thieu = can - co
        for x in list(thieu):
            n = int(x)
            if x in CHU_SO_HAN and any(c in zh for c in CHU_SO_HAN[x]):
                del thieu[x]
            elif n >= 10000 and n % 10000 == 0 and "万" in zh and str(n // 10000) in co:
                del thieu[x]
        k = sum(thieu.values())
        tot = k if tot is None else min(tot, k)
    return tot


def lech_so(vi, zh):
    """Mất số của bản Việt thì trượt. Giá trị dài (> DAI_SO ký tự): cho phép mất 1 số, hoặc 10% tổng số số (làm tròn xuống)."""
    k = so_thieu(vi, zh)
    if k == 0:
        return False
    if len(vi) > DAI_SO:
        return k > max(1, sum(_dem(vi).values()) // 10)
    return True


MUC_NGUON = 40      # mỗi mục ngắn hơn ngần này ký tự và ngần này từ (MUC_TU) mới coi là danh sách; dấu ", " trong câu văn không phải dấu ngăn mục
MUC_TU = 3
DAU_MUC_ZH = re.compile(r"\s*(?:(?<!\d),|,(?!\d)|[，、;；])\s*")      # dấu ngăn mục bản Trung (dấu phẩy có chữ số ở CẢ HAI bên là số thập phân, không tách)
DAU_PHAY_VIET = re.compile(r"(?<!\d),|,(?!\d)")            # mọi dấu phẩy ở bản Việt trừ số thập phân


def muc_nguon(vi):
    """Các mục của danh sách ngăn bằng ', ' (bt_attributes_text của Tiki và bộ 1 đều ngăn như vậy)."""
    return [x for x in re.split(r",\s+", vi.strip()) if x.strip()]


def muc_khop(vi, zh):
    """Giá trị Việt là danh sách mục ngắn ngăn bằng ', ' thì bản Trung phải có cùng số mục (ngăn bằng , ， 、 ; ；).
    Câu văn dài có dấu phẩy (mục dài) không phải danh sách nên không kiểm."""
    ms = muc_nguon(vi)
    if len(ms) <= 1 or any(len(x) > MUC_NGUON or len(x.split()) > MUC_TU for x in ms):
        return True
    if len(ms) == 2 and any(len(x.split()) > 2 for x in ms):
        return True          # hai cụm ("Vải nỉ mỏng, nhẹ", "Bảo quản sạch sẽ, khô ráo"): dịch gộp thành một cụm là bình thường
    n_zh = len([x for x in DAU_MUC_ZH.split(zh.strip()) if x.strip()])
    # nguồn có mục dính dấu phẩy không cách ("Hồng,0-6m Lime"): mô hình tách hay giữ đều chấp nhận, nên đếm cả theo mọi dấu phẩy
    n_het = len([x for x in DAU_PHAY_VIET.split(vi.strip()) if x.strip()])
    return n_zh in (len(ms), n_het)


DON_VI_MA = DON_VI_DO | {"h", "hr", "hrs", "min", "mins"}      # thêm đơn vị thời gian (3h, 30min): dịch ra 3小时 là đúng, không phải mã (không thêm "s": S11 là mã)
MA_DAI = 80          # chỉ kiểm mã Latin ở giá trị ngắn: văn bản dài hay bị mô hình bỏ mã phụ (C10-30 trong tên hoá chất) khi dịch đúng nghĩa
NIN = re.compile(r"\d+in\d+", re.I)       # 4in1, 2in1: dịch thành 4合1 là đúng
KICH_THUOC = re.compile(r"(?:[DRC]\d+(?:[.,]\d+)?(?:[xX×*]|$))+", re.I)      # D60, R28, C5: Dài, Rộng, Cao viết tắt (dịch ra 长, 宽, 高)


def la_don_vi(run):
    """Chuỗi chữ chỉ gồm đơn vị đo nối bằng x (m, mx, cmxcm): "2mx2" là 2m x 2."""
    return all(p == "" or p.lower() in DON_VI_MA for p in re.split(r"[xX×]", run))


def la_chu_ma(run):
    """Phần chữ Latin trong một mã: chữ in hoa (ABC), chữ rất ngắn (A, XL), hoặc chữ lẫn hoa thường kiểu iPhone. Từ thường
    (Blue, pro) và từ Viết Hoa Đầu (Blue) là từ chứ không phải mã, dịch ra là đúng."""
    return run.isupper() or len(run) <= 2 or not (run.islower() or run.istitle())


def ma_bat_buoc(vi):
    """Các mã Latin của bản Việt phải còn nguyên ở bản Trung: chữ-số có phần chữ là mã (ABC-123, A4, S11, iPhone15; không phải
    20cm, 3h, 1PCS-Blue, 4in1). Chuỗi dính gạch nối chỉ đòi phần là mã ("S11-Kaki" chỉ đòi "S11", "Kaki" dịch ra 卡其色 là đúng); chuỗi
    toàn mã ("ABC-123") đòi nguyên chuỗi. Từ in hoa (USB, SX, NSX, BH, TB) KHÔNG còn bị đòi: ở dữ liệu Tiki phần lớn là chữ viết tắt
    tiếng Việt (SX = sản xuất, BH = bảo hành, TB = trung bình), lần chạy thật 23/10/2026 chúng làm trượt oan hàng trăm đơn vị.
    Giá trị dài hơn MA_DAI ký tự không kiểm."""
    if len(vi) > MA_DAI:
        return []
    ra = []
    for c in CHUNK.split(vi):
        c = c.strip("-_.+/*'\"!?")
        if not c or not MA_LATIN.fullmatch(c) or NIN.fullmatch(c) or KICH_THUOC.fullmatch(c) or not re.search(r"\d", c):
            continue
        parts = [x for x in re.split(r"[-_/]", c) if x]

        def la_ma(x):
            chu = re.findall(r"[A-Za-z]+", re.sub(r"\d+(?:[.,]\d+)?", " ", x))
            return bool(re.search(r"\d", x)) and any(not la_don_vi(r) and la_chu_ma(r) for r in chu)

        co_tu = any(re.fullmatch(r"[A-Za-z]{3,}", x) and not la_chu_ma(x) and x.lower() not in DON_VI_MA for x in parts)
        if co_tu:
            ra += [x for x in parts if la_ma(x)]
        elif any(la_ma(x) or (re.fullmatch(r"[A-Za-z]+", x) and la_chu_ma(x) and x.lower() not in DON_VI_MA) for x in parts):
            ra.append(c)
    return ra


def go_nhan(zh, vi_gui, ten_zh, ten_vi=None):
    """Gỡ nhãn "tên thuộc tính: " mà mô hình tự thêm vào đầu bản dịch (ca 'Ý' -> '品牌产地: 意大利', 12,7% giá trị ở lần thử Gemma 23/10/2026).
    Chỉ gỡ khi nguồn không có dấu hai chấm VÀ phần nhãn trùng hoặc chứa/được chứa trong bản dịch của chính tên thuộc tính (ten_zh).
    Trả về bản đã gỡ, hoặc None khi không phải ca này (không đoán)."""
    if not isinstance(zh, str) or not isinstance(ten_zh, str) or re.search("[:：]", vi_gui) or not ten_zh.strip():
        return None
    if ten_vi and chuan_hoa(vi_gui).startswith(chuan_hoa(ten_vi)):
        return None      # nguồn mở đầu bằng chính tên thuộc tính ("Thành phần Bột mì ...") thì "成分：" là bản dịch của nguồn, giữ
    m = re.match(r"^\s*([^:：]{1,30})[:：]\s*(.+)$", zh.strip(), re.S)
    if not m:
        return None
    nhan, phan = m.group(1).strip(), m.group(2).strip()
    gon = lambda t: re.sub(r"[\s()（）]+", "", t).lower()
    a, b = gon(nhan), gon(ten_zh)
    if a and b and (a == b or a in b or b in a) and phan:
        return phan
    return None


MAU_ANH = re.compile(r"\b(?:black|white|pink|red|grey|gray|green|yellow|blue|silver|gold|purple|orange|brown|beige|navy)\b", re.I)


def mau_anh_chua_dich(vi, zh):
    """Nguồn có tên màu tiếng Anh (Black, Blue...) mà bản dịch còn giữ nguyên chữ đó. Lần chạy thật 23/10/2026: Gemma để nguyên ở 168 đơn vị
    (Qwen chỉ 24), ví dụ 'Blue, Champagne Gold, Green' -> 'Blue, Champagne Gold, Green'; Qwen dịch 蓝色，香槟金，绿色. Dùng ở bước hoàn tất:
    chỉ khi bản của mô hình kia đã dịch và qua luật thì bản này mới bị coi là trượt (không cất riêng nếu cả hai cùng để nguyên)."""
    return bool(MAU_ANH.search(vi or "")) and bool(MAU_ANH.search(zh or ""))


def viet_giu_ten_rieng(v, loai):
    """Bản Trung còn chữ Việt có dấu nhưng chỉ là tên riêng: có chữ Hán, không quá 4 từ Việt, mỗi từ Việt viết hoa chữ đầu (Nhà Chíp,
    Thống Nhất, Đen Đá, ĐIỀN TRANG). Chữ Việt viết thường ("đỏ") là sót chưa dịch. Tên thuộc tính (loai = ten) thì không bao giờ cho phép."""
    if loai == "ten" or not HAN.search(v):
        return False
    tu = [w for w in TU_LATIN.findall(v) if VIET_DAU.search(w)]
    return 0 < len(tu) <= 4 and all(w[0].isupper() for w in tu)


def la_ma_viet(vi, v):
    """Giá trị toàn mã có chữ Việt (ĐQ EMR09, CVG35Đ, BTĐ): bản dịch giữ nguyên là đúng (mã không dịch). Mỗi từ có chữ: có chữ số, hoặc
    in hoa không quá 3 chữ; không có chữ Hán trong bản dịch và bản dịch trùng nguồn."""
    if HAN.search(v) or chuan_hoa(v) != chuan_hoa(vi):
        return False
    tu = TU_LATIN.findall(vi)
    return bool(tu) and all(re.search(r"\d", w) or (w.isupper() and len(w) <= 3) for w in tu)


def kiem_mot(vi_goc, vi_gui, zh, tu_dien=None, loai="gia_tri"):
    """Kiểm một bản dịch Việt -> Trung. Trả về (danh sách mã trượt, danh sách cờ nhắc). Luật độ dài làm riêng ở do_dai_bat_thuong."""
    zh = zh if isinstance(zh, str) else ""
    truot, nhac = [], []
    v = zh.strip()
    if not v:
        return ["rong"], []
    if VIET_DAU.search(v):
        if loai != "ten" and la_ma_viet(vi_gui, v):
            nhac.append("ma_viet")       # mã có chữ Việt (Đ) giữ nguyên
        elif viet_giu_ten_rieng(v, loai):
            nhac.append("viet_giu")      # tên riêng tiếng Việt (tên hãng, tên người, địa danh) giữ nguyên chữ trong bản Trung
        else:
            truot.append("viet_sot")
    if LOI_THUA.search(v) or ("\n" in v and "\n" not in vi_gui) or (MARKDOWN_DAM in v and MARKDOWN_DAM not in vi_gui):
        truot.append("loi_thua")
    nhan_dau = bool(NHAN_DAU.match(v)) and not re.search("[:：]", vi_gui)
    if (NGU_CANH_ZH.search(v) and not NGU_CANH_NGUON.search(vi_gui)) or (nhan_dau and len(vi_gui) <= NHAN_NGAN):
        truot.append("lo_ngu_canh")
    elif nhan_dau:
        nhac.append("nhan_dau")      # giá trị dài: nguồn thường mở đầu bằng tiêu đề ("Thành phần ...", "Cách dùng ..."), dịch ra "成分：" là đúng
    # lặp: cụm chữ Hán lặp liên tiếp ≥ 4 lần (không tính nếu nguồn cũng lặp), hoặc cụm từ Latin lặp (nguồn không có dấu cách giữa chữ Hán)
    if LAP_HAN.search(v) and not LAP_HAN.search(vi_gui):
        truot.append("lap_ky_tu")
    tu = v.split()
    tu_nguon = vi_gui.split()
    nguon_lap = bool(tu_nguon) and max(Counter(tu_nguon).values()) >= 4      # nguồn tự lặp một từ (tên hãng ×4): bản dịch lặp theo là trung thành
    if len(tu) >= 8 and not nguon_lap:
        for n in range(1, 5):
            if any(tu[i:i + n] * 4 == tu[i:i + 4 * n] for i in range(0, len(tu) - 4 * n + 1)):
                truot.append("lap_ky_tu")
                break
    # chữ giản thể
    if any(c in PHON_THE for c in v):
        truot.append("phon_the")
    # đơn vị: ở Việt "cân" là kg, 斤 trong bản Trung (luôn do mô hình tự thêm vì nguồn tiếng Việt không có 斤) là sai gấp đôi
    if re.search(r"(?<!公)斤", v.replace("千斤顶", "").replace("千斤", "")):          # 公斤 (kg), 千斤顶 (cái kích) là đúng
        truot.append("jin_sai")
    if re.search(r"\d\s*两", v) and "lạng" not in vi_gui.lower():
        nhac.append("luong_xem")
    # số
    if lech_so(vi_gui, v):
        truot.append("lech_so")
    elif so_thieu(vi_gui, v):
        nhac.append("so_thieu")          # giá trị dài mất vài số (1 ly, hậu tố hoá chất): chấp nhận, ghi chú
    # số ở bản Trung mà nguồn không có (bỏ qua khi nguồn có số viết bằng chữ như "hai cái" -> 2个, hoặc 万 quy đổi từ số lớn)
    thua = _dem(v) - (_dem(vi_gui) | _dem(HANG_NGHIN.sub("", vi_gui)))
    if thua and not SO_CHU_VIET.search(vi_gui):
        if "万" in v:
            thua = Counter({x: c for x, c in thua.items() if not any(int(k) == int(x) * 10000 for k in _dem(HANG_NGHIN.sub("", vi_gui)))})
        if thua:
            nhac.append("so_thua")
    # mã Latin
    gon = re.sub(r"(?<=\d),(?=\d)", ".", re.sub(r"\s+", "", unicodedata.normalize("NFKC", v)).lower())
    for t in ma_bat_buoc(vi_gui):
        if t.lower() not in gon:
            truot.append("mat_ma")
            break
    # từ Latin thường (tên hãng, tên thiết bị) bị bỏ: chỉ nhắc
    for t in re.findall(r"[A-Za-z]{3,}", vi_gui):
        tl = t.lower()
        if tl not in LATIN_QUEN and not AM_TIET.match(tl) and tl not in unicodedata.normalize("NFKC", v).lower():
            nhac.append("mat_tu_latin")
            break
    # danh sách: số mục phải khớp
    if not muc_khop(vi_gui, v):
        truot.append("lech_so_muc")
    # cờ nhắc
    zl = vi_gui.lower()
    thu_la = [t.lower() for t in re.findall(r"[A-Za-z]{3,}", unicodedata.normalize("NFKC", v))
              if t.lower() not in zl and t.lower() not in LATIN_QUEN]
    if thu_la:
        nhac.append("latin_la")
    if not HAN.search(v) and re.search(r"[A-Za-z]{3,}", vi_gui) and chuan_hoa(v) == chuan_hoa(vi_gui):
        nhac.append("khong_dich")
    if tu_dien and vi_goc in tu_dien and chuan_hoa(tu_dien[vi_goc]) != chuan_hoa(v):
        nhac.append("khac_tu_dien")
    return sorted(set(truot)), sorted(set(dict.fromkeys(nhac)))
