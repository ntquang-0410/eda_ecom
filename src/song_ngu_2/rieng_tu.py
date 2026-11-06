# Lọc thông tin riêng tư còn sót ở bước ghép cặp của bộ 2 (hiến chương mục 6: bỏ tên cửa hàng, số điện thoại, địa chỉ). Bước định tuyến đã cất
# riêng số điện thoại, email, địa chỉ web, tên và địa chỉ đơn vị chịu trách nhiệm; quét bản công bố 23/10/2026 còn sót (đếm theo cặp):
# địa chỉ đường phố cụ thể của cửa hàng ~6, tên cửa hàng nêu trong dấu ngoặc kép ~8, thương hiệu thực chất là tên cửa hàng (xxx Store) ~10.
# Làm ở bước ghép cặp (không đổi bước chuẩn bị) để KHÔNG làm lệch mã đơn vị đã dịch. Chỉ nhận mẫu chắc chắn, không đoán: "shop" nói chung
# ("shop cập nhật thường xuyên", "nhắn tin cho shop") là lời văn bình thường nên giữ; tên doanh nghiệp (Công ty TNHH ...) cũng giữ (tên doanh nghiệp
# không phải thông tin cá nhân, như bộ 1 giữ tên xưởng).
import re

DIA_CHI = [
    re.compile(r"(?i)\bđ/c\s*[:：]"),                                              # "Đ/c: 63 Nguyễn Thị Căn, ..."
    re.compile(r"(?i)\bphường\s+\d+\b.*\bquận\s+\d+\b"),                           # "165 Pasteur, Phường 6, Quận 3"
    re.compile(r"(?i)\b(?:phường|xã)\s+[A-ZĐÀ-Ỵ][\wÀ-ỹ]+(?:\s+[A-ZĐÀ-Ỵ][\wÀ-ỹ]+)?\s*,\s*(?:quận|huyện|thị xã|tp|thành phố)\b"),
]
# tên cửa hàng nêu cụ thể: shop "Nhà Chíp"; "Cung cấp bởi HT SYS OFFICIAL STORE"
TEN_SHOP = [
    re.compile(r"(?i)\bshop\s*[\"“”']\s*[^\"“”']{2,40}[\"“”']"),
    re.compile(r"(?i)\b(?:bởi|của|tại)\s+[A-ZĐ][\w &.\-]{1,40}?\s+(?:official\s+|oficial\s+)?store\b"),
    # thêm 23/10/2026 khi soát lại mẫu 200 cặp (cửa hàng tự xưng tên trong lời văn; thử trên toàn bộ 89.300 cặp nguồn, không bắt nhầm):
    re.compile(r"(?i)\bshop(?!ee|ping|house|s\b)[a-z0-9]{4,}\b"),                     # tên shop viết liền "shopquyetnam" (91 cặp)
    re.compile(r"\bSHOP\s+[A-Z0-9]{5,}\b"),                                           # "PET SHOP UYTINPRO" (4 cặp)
    re.compile(r"(?:nhà bán hàng|[Hh]ệ [Tt]hống)\s+(?:[A-ZĐ][\wÀ-ỹ]*\s+){0,2}(?:HEAD|Head)\s+[A-ZĐ][\wÀ-ỹ]+"),  # đại lý "Honda Head Phát Tiến" (2 cặp)
]
THUONG_HIEU_CUA_HANG = re.compile(r"(?i)\b(?:store|shop|official store)\b")        # thương hiệu "Anh Lam Store", "USA Store"


def rieng_tu(ten_vi, gia_tri_vi):
    """Trả về lý do cất riêng ('dia_chi' | 'ten_shop') hoặc None. Chỉ nhìn bản Việt gốc (bản Trung là bản dịch của chính chữ đó)."""
    v = gia_tri_vi or ""
    if any(p.search(v) for p in DIA_CHI):
        return "dia_chi"
    if any(p.search(v) for p in TEN_SHOP):
        return "ten_shop"
    if " ".join((ten_vi or "").split()).lower() == "thương hiệu" and THUONG_HIEU_CUA_HANG.search(v):
        return "ten_shop"
    return None
