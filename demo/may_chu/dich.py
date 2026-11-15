# -*- coding: utf-8 -*-
"""Dịch title và khối thuộc tính. Hàm thuần theo một hàm sinh(prompt_chat, tran) nên thử được bằng mô hình giả."""
import loi_dan as L

KHOI_TOI_DA = 40       # số dòng tối đa một lượt (extension chia khối ~20 dòng)


def lam_sach(s):
    return s.strip("\n").strip()


def dich_title(sinh, zh):
    zh = zh.strip()
    if not zh:
        return ""
    return lam_sach(sinh(L.khuon(L.loi_dan_title(zh)), L.chon_tran(zh)))


def tach_cap(dong):
    """'tên: giá trị' -> (tên, giá trị) ở dấu ': ' đầu tiên; không có thì (dong, None)."""
    t, sep, g = dong.partition(": ")
    return (t, g) if sep else (dong, None)


def dich_khoi(sinh, title_zh, cap):
    """cap: [(tên, giá trị)] -> (list (tên_vi, giá_trị_vi), thong_ke). Khối ra lệch số dòng thì dịch lại từng dòng (như lúc đo), không đoán dòng."""
    cap = [(str(t).strip(), str(g).strip()) for t, g in cap if str(t).strip() or str(g).strip()]
    if not cap:
        return [], {"so_dong": 0, "lech_dong": False}
    if len(cap) > KHOI_TOI_DA:
        raise ValueError(f"khoi_qua_dai:{len(cap)}")
    u, khoi = L.loi_dan_khoi(title_zh, cap)
    hyp = lam_sach(sinh(L.khuon(u), L.chon_tran(khoi)))
    dong = hyp.split("\n")
    lech = len(dong) != len(cap)
    if lech:
        dong = []
        for t, g in cap:
            u1, k1 = L.loi_dan_khoi(title_zh, [(t, g)])
            dong.append(lam_sach(sinh(L.khuon(u1), L.chon_tran(k1))).split("\n")[0])
    return [tach_cap(d) for d in dong], {"so_dong": len(cap), "lech_dong": lech}
