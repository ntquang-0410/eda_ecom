"""Thử chia_tap trên dữ liệu nhỏ tự dựng. Chạy: python -I -X utf8 scripts/ft_train/test_chia_tap.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import chia_tap as T  # noqa: E402
import kiem_ro_ri as K  # noqa: E402


def test_noi_tap_chinh_xac_bang_vet_can():
    random.seed(0)
    for thu in range(30):
        sets = [frozenset(random.sample(range(25), random.randint(0, 8))) for _ in range(120)]
        for ng in (0.5, 0.6, 0.7, 1.0):
            a = {(min(i, j), max(i, j)) for i, j, _ in T.cap_gan_trung(sets, ng)}
            b = {(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))
                 if sets[i] and sets[j] and len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= ng - 1e-9}
            assert a == b, (thu, ng)


def test_thuong_hieu():
    assert T.thuong_hieu("Thương hiệu: Dược Liệu Tuệ Tâm\nXuất xứ thương hiệu: Việt Nam") == "dượcliệutuệtâm"
    for chung in ("Khác", "OEM", "Không thương hiệu", "1", "Thương hiệu riêng"):
        assert T.thuong_hieu(f"Thương hiệu: {chung}") == "", chung
    assert T.thuong_hieu("Xuất xứ thương hiệu: Việt Nam") == ""


def _df():
    hang = []
    for k in range(200):                                     # sp riêng lẻ, mỗi sp một title khác hẳn
        src = "1688" if k % 2 else "tiki"
        hang.append(dict(product_id=str(1000 + k), source=src, category=["a", "b"][k % 3 == 0],
                         title_zh=f"产品{k}号型{k * 7}", title_vi=f"San pham so {k} loai {k * 7} mau {k * 13}",
                         description_zh=f"品牌: 牌{k}\n型号: X{k}", description_vi=f"Thương hiệu: Hieu{k}\nMã: X{k}"))
    # tin đăng anh em: title ngắn nằm gọn trong title dài, cùng thương hiệu
    hang.append(dict(product_id="9001", source="tiki", category="a", title_zh="枸杞茶", title_vi="Trà kỷ tử",
                     description_zh="品牌: 慧心\n保质期: 12个月", description_vi="Thương hiệu: Tuệ Tâm\nHạn: 12 tháng"))
    hang.append(dict(product_id="9002", source="tiki", category="a", title_zh="优质枸杞茶 - 枸杞 - 高级款 (500g)",
                     title_vi="Trà kỷ tử - Câu kỷ tử - Loại cao cấp (500gr)",
                     description_zh="品牌: 慧心\n成分: 枸杞", description_vi="Thương hiệu: Tuệ Tâm\nThành phần: kỷ tử"))
    # trùng title tiếng Trung chéo hướng
    hang.append(dict(product_id="9003", source="1688", category="b", title_zh="蓝牙耳机无线运动款", title_vi="Tai nghe A",
                     description_zh="颜色: 黑", description_vi="Màu: đen"))
    hang.append(dict(product_id="9004", source="tiki", category="b", title_zh="蓝牙耳机无线运动款", title_vi="Tai nghe B khác",
                     description_zh="颜色: 白", description_vi="Màu: trắng"))
    return pd.DataFrame(hang)


def test_chia_khong_ro_ri():
    df = _df()
    sp, tk = T.chia(df, seed=7)
    T.kiem_split(sp)
    g = sp.set_index("product_id").nhom_id
    assert g["9001"] == g["9002"], "tin đăng anh em cùng thương hiệu phải cùng nhóm"
    assert g["9003"] == g["9004"], "trùng title zh chéo hướng phải cùng nhóm"
    assert 0 < sp.la_test.sum() < len(sp)
    assert sp[~sp.la_test].fold5.between(0, 4).all() and (sp[sp.la_test].fold5 == -1).all()
    sp2, _ = T.chia(df, seed=7)
    assert sp.equals(sp2), "cùng seed phải ra cùng phép chia"
    bc, loi = K.kiem(sp, df, df.assign(description_zh=df.description_zh.str.replace("\n", "; "),
                                       description_vi=df.description_vi.str.replace("\n", "; ")),
                     {"title": T.NGUONG_TITLE, "mo_ta": T.NGUONG_MO_TA, "chua": T.NGUONG_CHUA})
    assert not loi, bc["loi"]


def test_kiem_ro_ri_bat_duoc_vi_pham():
    df = _df()
    sp, _ = T.chia(df, seed=7)
    xau = sp.copy()
    i, j = xau.index[xau.product_id == "9001"][0], xau.index[xau.product_id == "9002"][0]
    xau.loc[i, ["la_test", "fold5", "nhom_id"]] = [True, -1, "x1"]          # cố tình tách tin đăng anh em
    xau.loc[j, ["la_test", "fold5", "nhom_id"]] = [False, 0, "x2"]
    cum = df.assign(description_zh=df.description_zh.str.replace("\n", "; "),
                    description_vi=df.description_vi.str.replace("\n", "; "))
    _, loi = K.kiem(xau, df, cum, {"title": T.NGUONG_TITLE, "mo_ta": T.NGUONG_MO_TA, "chua": T.NGUONG_CHUA})
    assert loi, "bộ kiểm phải bắt được"


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_"):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
