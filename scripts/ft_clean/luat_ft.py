"""Luật làm sạch bộ chính thức v2 để chuẩn bị fine-tune (bước 1-4 của kế hoạch 30/10/2026).

Quy ước: cờ CỨNG làm loại dòng (ghi mã lý do, không xóa khỏi bảng ứng viên); cờ MỀM chỉ ghi lại để bước judge và bước
chọn tập sạch dùng sau. Không viết lại nội dung nào."""
import hashlib
import re
import sys
import unicodedata
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
from luat_kiem import CHUOI_HAN, HAN_HOP_LE  # noqa: E402  (tái dùng luật Hán sót + số đăng ký mỹ phẩm)

# ---- B1a. Rò lời nhắc / ảo giác ở title_vi của 1688 (đo trên v2: 36 ca theo luật lõi, 35 ca theo điểm >= 3) ----
LEAK_LOI = re.compile(
    r"đơn vị tiền tệ|mã model|mã mô hình|chưa được đề cập|mã (?:sản phẩm|sp)[:, ]+(?:không|không rõ)"
    r"|số lượng[:, ]+(?:không|không xác định|không rõ)|đơn giá[:, ]+không|tên thương hiệu tiếng anh"
    r"|mã sản phẩm, số lượng|(?:không có|không rõ)[,.]? *(?:số lượng|đơn vị)|giá tiền (?:chưa|không)"
    r"|giá[:, ]+(?:không có|không rõ)")
LEAK_GIU_CHO = re.compile(r"(?:mã (?:sản phẩm|sp)|giá(?: bán)?)\s*[:：]?\s*(?:không\b|12345\d*|123\.45|0 đồng)")
LEAK_TU_KHOA = ["mã sản phẩm", "mã model", "mã mô hình", "số lượng", "đơn vị tiền tệ", "đơn vị", "đơn giá", "giá",
                "thương hiệu tiếng anh", "chữ số", "giữ nguyên", "không có", "không rõ", "không xác định",
                "chưa được đề cập", "tiền tệ"]
LEAK_DIEM_TOI_THIEU = 3        # 3 từ khóa khác nhau trong cùng title; 2 từ khóa thì còn lẫn tiêu đề thật ("giá sỉ số lượng lớn")


def ro_loi_nhac(title_vi):
    s = (title_vi or "").lower()
    if LEAK_LOI.search(s) or LEAK_GIU_CHO.search(s):
        return True
    return sum(1 for k in LEAK_TU_KHOA if k in s) >= LEAK_DIEM_TOI_THIEU


# ---- B1b/B4. Đoạn thuộc tính và số ----
SO = re.compile(r"\d+(?:[.,]\d+)?")
DAU_VI = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
LATIN = re.compile(r"[A-Za-z]{4,}")


# Chỉ tách ở "; " khi phần đứng sau có dạng "nhãn: ..." (không chứa ";" trước dấu ":"). Nếu tách ở mọi "; " thì giá trị tự do có
# chứa "; " (vd. hướng dẫn bảo quản của Tiki) bị đếm thành đoạn mới và bản Trung (dùng "；" toàn góc) bị đếm lệch oan.
TACH_DOAN = re.compile(r"; (?=[^;:：]{1,80}[:：])")


def doan(mo_ta):
    return TACH_DOAN.split(mo_ta or "")


def _so(s):
    return sorted(x.replace(",", ".") for x in SO.findall(s))


def can_hang(zh, vi):
    """(so_doan_lech, so_doan_so_khac): số đoạn lệch thì cờ cứng; số khác từng đoạn chỉ là tín hiệu mềm
    (có thể do quy đổi đơn vị như 斤 -> kg)."""
    a, b = doan(zh), doan(vi)
    if len(a) != len(b):
        return True, None
    return False, sum(1 for x, y in zip(a, b) if _so(x) != _so(y))


def han_sot(source, vi):
    """Chữ Hán còn trong bản Việt của 1688, trừ số đăng ký mỹ phẩm (粤妆网备字...)."""
    if source != "1688":
        return 0
    return sum(len(h) for h in CHUOI_HAN.findall(vi or "") if not set(h) <= HAN_HOP_LE)


def latin_la(source, zh, vi):
    """Tiki (bản Trung do máy dịch): từ Latin >= 4 chữ trong bản Trung mà không có trong bản Việt gốc
    (vd. 'Stabilizer', 'Emulsifier' thay vì 稳定剂). Mềm."""
    if source != "tiki":
        return 0
    v = (vi or "").casefold()
    return sum(1 for w in LATIN.findall(zh or "") if w.casefold() not in v)


def viet_sot_title_zh(source, title_zh):
    """Tiki: title bản Trung còn >= 4 chữ cái tiếng Việt có dấu (phần chưa dịch). Mềm."""
    return source == "tiki" and len(DAU_VI.findall(title_zh or "")) >= 4


# ---- B2. Khuôn mô tả và nhóm mô tả trùng ----
def khuon(source, zh, vi):
    """Chuỗi nhãn thuộc tính của bản GỐC (1688: tiếng Trung, Tiki: tiếng Việt)."""
    goc = zh if source == "1688" else vi
    nhan = []
    for seg in doan(goc):
        p = re.split(r"[:：]", seg, maxsplit=1)
        nhan.append(p[0].strip().casefold() if len(p) > 1 else "")
    return source + "#" + "|".join(nhan)


def khuon_id(source, zh, vi):
    return hashlib.md5(khuon(source, zh, vi).encode("utf-8")).hexdigest()[:12]


def chuan(s):
    return " ".join(unicodedata.normalize("NFKC", s or "").casefold().split())


def nhom_mo_ta_id(zh, vi):
    """Cùng mô tả (cả hai bản) thì cùng nhóm; dùng để KHÔNG tách các dòng này sang hai phía khi chia tập sau này."""
    return hashlib.md5((chuan(zh) + "\n" + chuan(vi)).encode("utf-8")).hexdigest()[:12]


def khoa_thu_tu(product_id):
    return (zlib.crc32(str(product_id).encode("utf-8")), str(product_id))
