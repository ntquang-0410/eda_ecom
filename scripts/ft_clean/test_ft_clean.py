"""Thử luật làm sạch trên dữ liệu nhỏ tự dựng. Chạy: python -I -X utf8 scripts/ft_clean/test_ft_clean.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import luat_ft as L  # noqa: E402
import xay_ft_clean as X  # noqa: E402


def test_ro_loi_nhac():
    xau = ["Túi xách nữ da thật, mã sản phẩm: không có, thương hiệu: không có, giá: không có",
           "Áo thun nam 100-300kg, thương hiệu Fat Brother, mã sản phẩm không có, giá không có, đơn vị không có",
           "Ba lô thể thao, mã mô hình chữ số, số lượng, đơn vị tiền tệ",
           "Giày lê nữ đi biển mã sản phẩm 1234567890 giá 123.45 USD",
           "Hộp đựng quần áo gia dụng, mã sản phẩm: 12345, giá: 250.000 VND",
           "Trà đen Wuyi Shan, mã SP: không có, trọng lượng 500g"]
    tot = ["Trà Thiết Quan Âm bán buôn đóng gói rời, giá rẻ số lượng lớn, xưởng Phúc Kiến",
           "Mặt nạ dưỡng ẩm cấp nước, hộp 10 miếng, model A12",
           "Bánh mì hoa cúc Mini Harrys 210gr PHÁP", "", None]
    for s in xau:
        assert L.ro_loi_nhac(s), s
    for s in tot:
        assert not L.ro_loi_nhac(s), s


def test_can_hang_va_han_sot():
    assert L.can_hang("a: 1; b: 2斤", "A: 1; B: 1kg") == (False, 1)        # quy đổi đơn vị: chỉ là tín hiệu mềm
    assert L.can_hang("a: 1; b: 2", "A: 1; B: 2") == (False, 0)
    assert L.can_hang("a: 1; b: 2", "A: 1")[0] is True
    # "; " nằm bên trong giá trị tự do không phải ranh giới thuộc tính (Tiki: bản Trung dùng "；" toàn góc)
    zh = "品牌: A; 保存方法: 存放在阴凉干燥处；开封后冷藏; 注意事项: 包装可能变化"
    vi = "Thương hiệu: A; Hướng dẫn bảo quản: Bảo quản nơi mát; sau khi mở nắp bảo quản lạnh; Lưu ý: Bao bì có thể thay đổi"
    assert len(L.doan(zh)) == 3 and len(L.doan(vi)) == 3 and L.can_hang(zh, vi)[0] is False
    assert L.can_hang("a: 1; b: 2; c: 3", "A: 1; B: 2")[0] is True                 # mất một thuộc tính vẫn bị bắt
    assert L.han_sot("1688", "Số đăng ký: 粤妆网备字2023; Chất liệu: 材质") == 2
    assert L.han_sot("tiki", "材质") == 0


def test_latin_la_va_viet_sot():
    assert L.latin_la("tiki", "稳定剂 Stabilizer, 乳化剂 Emulsifier (471), BrandX", "Chất ổn định; Chất nhũ hóa (471); BrandX") == 2
    assert L.latin_la("1688", "x", "y") == 0
    assert L.viet_sot_title_zh("tiki", "智能时尚桌面化妆镜 GƯƠNG TRANG ĐIỂM") is True
    assert L.viet_sot_title_zh("tiki", "迷你雏菊面包 Harrys 210克") is False


def test_khuon_va_nhom():
    a = L.khuon_id("tiki", "x", "Thương hiệu: A; Xuất xứ: VN")
    b = L.khuon_id("tiki", "y", "Thương hiệu: B; Xuất xứ: Nhật")       # cùng nhãn, khác giá trị -> cùng khuôn
    c = L.khuon_id("tiki", "y", "Thương hiệu: B; Màu: Đỏ")
    assert a == b != c
    assert L.nhom_mo_ta_id("A  b", "x") == L.nhom_mo_ta_id("a b", "X")   # chuẩn hóa NFKC/casefold/khoảng trắng


def mau(n, nguon="tiki", nhan="Thương hiệu", tien_to="t"):
    return [{"product_id": f"{tien_to}{i}", "source": nguon, "category": "home", "title_zh": "标题", "title_vi": "Tiêu đề",
             "description_zh": f"品牌: B{i}; 产地: 越南", "description_vi": f"{nhan}: B{i}; Xuất xứ: Việt Nam",
             "title_nguon_dich": "gemma4", "desc_nguon_dich": "tu_dien_quy_tac"} for i in range(n)]


def test_pipeline_nho():
    rows = mau(10) + mau(3, "1688", "Thương hiệu", "b")
    rows[0]["title_vi"] = "Áo, mã sản phẩm: không có, giá: không có"                 # rò lời nhắc
    rows[1]["description_vi"] = "Thương hiệu: B1"                                    # lệch số đoạn
    rows[10]["description_vi"] = "品牌: B0; Xuất xứ: 材质"                            # Hán sót ở 1688
    df = pd.DataFrame(rows)
    d = X.loai_tuan_tu(X.gan_co(df), tran=4)
    ly = d.set_index("product_id").ly_do_loai_ft
    assert ly["t0"] == "ro_loi_nhac_title" and ly["t1"] == "so_doan_lech" and ly["b0"] == "han_sot_ban_viet"
    # 10 dòng Tiki cùng khuôn, 2 dòng bị loại sớm -> còn 8 dòng cùng khuôn, trần 4 -> loại 4 dòng cuối theo thứ tự crc32
    tiki_con = d[(d.source == "tiki") & d.ly_do_loai_ft.isna()]
    assert len(tiki_con) == 4 and (d.ly_do_loai_ft == "vuot_tran_khuon").sum() == 4
    assert d[d.source == "1688"].giu_ft.sum() == 2                                   # 1688 khác khuôn, không bị trần
    f = X.pheu(d)
    assert f["vao"] == 13 and f["con_lai"] == 6 and sum(b["loai"] for b in f["theo_buoc"]) == 7


def test_chay_het_tren_file_gia():
    df = pd.DataFrame(mau(6) + mau(2, "1688", "Thương hiệu", "b"))
    with tempfile.TemporaryDirectory() as tmp:
        vao = Path(tmp) / "vao.parquet"
        df.to_parquet(vao)
        assert X.main(["--vao", str(vao), "--ra", str(Path(tmp) / "ra"), "--bo-qua-sha"]) == 0
        giu = pd.read_parquet(Path(tmp) / "ra" / "ft_buoc4_giu.parquet")
        loai = pd.read_parquet(Path(tmp) / "ra" / "ft_buoc4_loai.parquet")
        assert len(giu) + len(loai) == 8 and set(giu.columns) >= set(X.COT_GOC)
        for c in X.COT_GOC[1:]:                                                      # không sửa nội dung
            assert giu.set_index("product_id")[c].equals(df.set_index("product_id").loc[giu.product_id, c])


if __name__ == "__main__":
    n = 0
    for ten, f in list(globals().items()):
        if ten.startswith("test_"):
            f()
            n += 1
            print("ok", ten)
    print(f"{n}/{n} bài thử đạt")
