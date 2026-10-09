# -*- coding: utf-8 -*-
"""Thử luật quy đổi 斤 → kg (src/doi_chieu.py) bằng các câu tổng hợp, không đọc dữ liệu thật.

    python scripts/eda/test_doi_chieu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from doi_chieu import canh_bao_don_vi, han_so, kiem_jin  # noqa: E402

CA = [
    # (zh, vi, trạng thái mong đợi, vi mới mong đợi hoặc None = giữ nguyên)
    ("3XL【建议190-215斤】", "3XL [khuyến nghị 190-215 kg]", "da_sua", "3XL [khuyến nghị 95-107.5 kg]"),
    ("S【建议90斤以下】", "S [Nên dưới 90 kg]", "da_sua", "S [Nên dưới 45 kg]"),
    ("4XL推荐（180-200斤）", "4XL khuyến nghị (180-200 kg)", "da_sua", "4XL khuyến nghị (90-100 kg)"),
    ("S【建议90-115斤】", "S [Khuyến nghị 45-57,5 kg]", "ok", None),
    ("S【建议80-100斤】, M【建议100-120斤】", "S [gợi ý 40-50kg], M [gợi ý 50-60kg]", "ok", None),
    ("S【建议80-100斤】, M【建议100-120斤】", "S [gợi ý 80-100kg], M [gợi ý 50-60kg]", "da_sua", "S [gợi ý 40-50kg], M [gợi ý 50-60kg]"),
    ("约2斤", "khoảng 1kg", "ok", None),
    ("一斤装", "1kg", "da_sua", "0.5kg"),
    ("半斤", "0.5kg", "da_sua", "0.25kg"),
    ("两斤", "2 kg", "da_sua", "1 kg"),
    ("30元/斤绿版龙井", "30 nhân dân tệ/kg phiên bản màu xanh lá cây Long Tỉnh", "gia_theo_can", None),
    ("30元/斤绿版龙井", "30 nhân dân tệ/cân bản xanh Long Tỉnh", "khong_ap_dung", None),
    ("5公斤", "5 kg", "khong_ap_dung", None),
    ("千斤顶", "Kích nâng 2 kg", "khong_ap_dung", None),
    ("5斤装", "bao 500g", "khong_ap_dung", None),
    ("100-120斤", "70 kg", "lech_khac", None),
    ("100-120斤", "100 kg", "lech_khac", None),
    ("S【建议80-100斤】, M【建议100-120斤】", "S [40-50kg]", "khong_ghep", None),
    ("三五斤", "3-5 kg", "khong_ghep", None),
    ("红色", "Màu đỏ", "khong_ap_dung", None),
    # lỗi phát hiện khi rà ngày 06/10: 斤半 và giá viết "tiền + một斤"
    ("一斤半", "1.5 kg", "da_sua", "0.75 kg"),
    ("一斤半", "0.75 kg", "ok", None),
    ("一斤半", "1 kg", "lech_khac", None),
    ("两斤半", "2.5 kg", "da_sua", "1.25 kg"),
    ("1-2斤半", "1-2 kg", "khong_ghep", None),
    ("30块钱一斤", "30 tệ 1 kg", "gia_theo_can", None),
    ("一斤30元", "1 kg 30 tệ", "gia_theo_can", None),
    ("5元1斤", "5 nhân dân tệ/kg", "gia_theo_can", None),
    ("30元5斤", "30 tệ 5 kg", "da_sua", "30 tệ 2.5 kg"),   # 30 tệ cho 5 斤: số 5 là cân nặng, được chia đôi
    ("10元半斤", "10 tệ 0.5 kg", "da_sua", "10 tệ 0.25 kg"),
]

loi = 0
for z, v, tt, moi in CA:
    got_tt, got_v = kiem_jin(z, v)
    mong_v = v if moi is None else moi
    ok = got_tt == tt and got_v == mong_v
    loi += not ok
    print(("OK  " if ok else "LỖI"), z, "|", v, "→", got_tt, "|", got_v, "" if ok else f"   (mong đợi {tt} | {mong_v})")
assert han_so("半") == 0.5 and han_so("十五") == 15 and han_so("二十") == 20 and han_so("十") == 10 and han_so("三五") is None
print("han_so OK")
print(f"{len(CA) - loi}/{len(CA)} ca quy đổi 斤 đạt")

# cảnh báo đơn vị 寸, 两, 万, 丝 (chỉ gắn cờ): (zh, vi, danh sách mã mong đợi)
CB = [
    ("20寸行李箱", "Vali 20 cm", ["CB_cun"]),
    ("磨泥盘5寸", "Khay nghiền bùn 5 inch", []),
    ("20寸【送箱套】", "20 inch [tặng áo trùm vali]", []),
    ("20寸", "Vali 20 thốn", ["CB_cun"]),
    ("20寸", "Vali 50 cm", []),                       # đã quy đổi (khác số): không cảnh báo
    ("大尺寸", "Cỡ lớn", []),                          # 尺寸 không có số đứng trước 寸
    ("14件/15寸", "14 miếng/ 15inch", []),
    ("二两装", "Gói 2 g", ["CB_luong"]),
    ("二两装", "Gói 100g", []),
    ("3两", "3 lạng", ["CB_luong"]),
    ("半两", "0.5 gram", ["CB_luong"]),
    ("两个装", "Gói 2 chiếc", []),
    ("一两天", "Một hai ngày", []),
    ("500克", "500g", []),
    ("5万次", "5 lần", ["CB_van"]),
    ("5万次", "50.000 lần", []),
    ("5万次", "50 nghìn lần", []),
    ("100万", "1 triệu", []),
    ("2万毫安", "2 vạn mAh", []),
    ("1.5万", "15000", []),
    ("万向轮", "Bánh xe xoay", []),
    ("5*5*5（25丝）", "5*5*5 (25 lụa)", ["CB_ty"]),
    ("25丝", "dày 0.25mm", []),
    ("25丝", "25 zem", []),
    ("10丝袜", "10 tất lụa", []),                     # 丝袜 (tất) không phải độ dày
    ("红色", "Màu đỏ", []),
]
loi2 = 0
for z, v, mong in CB:
    got = canh_bao_don_vi(z, v)
    ok = got == mong
    loi2 += not ok
    print(("OK  " if ok else "LỖI"), z, "|", v, "→", got, "" if ok else f"   (mong đợi {mong})")
print(f"{len(CB) - loi2}/{len(CB)} ca cảnh báo đơn vị đạt")
sys.exit(1 if (loi or loi2) else 0)
