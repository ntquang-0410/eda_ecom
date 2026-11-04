"""Kiểm tra bằng quy tắc cứng, miễn phí, chạy được trên cả 26.038 dòng (không gọi mô hình).

Đây là tầng lọc thứ nhất; judge LLM là tầng thứ hai. Các ngưỡng là đặt tạm, chọn lại từ phân bố khi chạy thật."""
import re

SO = re.compile(r"\d+(?:[.,]\d+)?")
HAN = re.compile(r"[一-鿿]")
# chữ cái tiếng Việt có dấu (không có trong tiếng Trung)
DAU_VI = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)

# Số đăng ký mỹ phẩm (vd. 粤妆网备字, 国妆特字) được GIỮ NGUYÊN trong bản Việt: một chuỗi Hán chỉ gồm các chữ này là hợp lệ.
# Đo 29/10/2026: 1.460/14.996 sản phẩm 1688 có Hán trong bản Việt, gần như toàn loại này (beauty 1.272, mother_baby 184).
HAN_HOP_LE = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤粵青藏川宁琼国妆妝网备進进出特字")
CHUOI_HAN = re.compile(r"[一-鿿]+")

TI_LE_DAI_MIN = 1.2    # độ dài Việt / độ dài Trung ở bản 1688 (trung bình đo được ~2,5)
TI_LE_DAI_MAX = 5.0


def _so(van_ban):
    return sorted(s.replace(",", ".") for s in SO.findall(van_ban))


def kiem(hang):
    """Trả về dict các co cho một dòng bản cụm."""
    zh, vi = hang["description_zh"] or "", hang["description_vi"] or ""
    co = {}
    so_zh, so_vi = _so(zh), _so(vi)
    co["lech_so"] = so_zh != so_vi          # mềm: có thể do quy đổi đơn vị (斤 -> kg) nên chỉ là tín hiệu
    co["so_chi_o_mot_ben"] = len(set(so_zh) ^ set(so_vi))
    co["so_doan_lech"] = len(zh.split("; ")) != len(vi.split("; "))
    if hang["source"] == "1688":
        co["han_sot"] = sum(len(h) for h in CHUOI_HAN.findall(vi)       # Hán còn trong bản Việt,
                            if not set(h) <= HAN_HOP_LE)                   # trừ số đăng ký mỹ phẩm
        co["dau_viet_sot"] = 0
    else:
        co["han_sot"] = 0
        co["dau_viet_sot"] = len(DAU_VI.findall(zh))  # chữ Việt còn trong bản Trung (tên riêng có thể hợp lệ)
    t = len(vi) / max(len(zh), 1)
    co["ti_le_dai"] = round(t, 3)
    co["ti_le_la"] = bool(t < TI_LE_DAI_MIN or t > TI_LE_DAI_MAX)
    co["rong"] = not zh.strip() or not vi.strip()
    co["co_luat"] = bool(co["rong"] or (co["han_sot"] > 0) or co["so_doan_lech"] or co["ti_le_la"])
    return co
