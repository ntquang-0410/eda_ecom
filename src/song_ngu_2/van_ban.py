# Hàm chữ riêng của bộ 2: chuẩn hoá bản dịch TIẾNG TRUNG để so hai bản (Gemma và Qwen) có giống nhau không.
# Lần chạy thật 23/10/2026: hai bản hay chỉ khác dấu câu toàn/nửa góc và khoảng trắng ("白色，灰色，黑色" và "白色, 灰色, 黑色", chrF chỉ 20;
# "100% 有机" và "100%有机"), nên so hai bản phải bỏ những khác biệt đó. Chỉ dùng để SO SÁNH, không dùng làm bản xuất ra.
import re
import unicodedata

from song_ngu_1.van_ban import chrf  # noqa: F401  (chrF đã bỏ khoảng trắng, dùng chung)

HAN = re.compile("[㐀-䶿一-鿿豈-﫿]")
_BIEN_DOI = str.maketrans({"×": "x", "＊": "*", "‘": "'", "’": "'", "“": '"', "”": '"', "·": "", "、": ",", "﹑": ","})


def chuan_hoa_zh(s):
    """NFKC (toàn góc thành nửa góc: ，→, （）→(), ：→:, ％→%, ℃→°C), chữ thường, bỏ mọi khoảng trắng, đồng nhất × và 、 với x và ",",
    bỏ dấu câu ở hai đầu, 7,5 = 7.5."""
    s = unicodedata.normalize("NFKC", s or "").translate(_BIEN_DOI).lower()
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = re.sub(r"\s+", "", s)
    return s.strip(".。,，;；:：!！?？-")
