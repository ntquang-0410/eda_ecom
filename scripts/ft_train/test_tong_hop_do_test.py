"""Thử tong_hop_do_test.py trên số tự dựng (không GPU, không dữ liệu thật). Chạy: python -I -X utf8 scripts/ft_train/test_tong_hop_do_test.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tong_hop_do_test as T  # noqa: E402


def test_boot_theo_sp():
    ro = T.boot_theo_sp([(f"p{i}", 1.0) for i in range(30)])
    assert ro["chenh"] == 1.0 and ro["thap"] == 1.0 and ro["cao"] == 1.0 and ro["n_sp"] == 30 and ro["n_muc"] == 30
    huy = T.boot_theo_sp([(f"p{i}", 1.0 if i % 2 else -1.0) for i in range(40)])
    assert huy["thap"] < 0 < huy["cao"]                  # lẫn tăng giảm: khoảng chứa 0
    # hai mục chung một sản phẩm thì được lấy theo cặp: 2 mục/sp, n_muc = 2 * n_sp
    cap = T.boot_theo_sp([(f"p{i}", d) for i in range(20) for d in (0.5, 1.5)])
    assert cap["n_muc"] == 40 and cap["n_sp"] == 20 and abs(cap["chenh"] - 1.0) < 1e-12
    assert T.boot_theo_sp([(f"p{i}", 0.3) for i in range(30)], seed=1) == T.boot_theo_sp([(f"p{i}", 0.3) for i in range(30)], seed=1)


def _ghi(td, nhanh, mh, rows):
    (td / f"sinh_diem_{nhanh}_{mh}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _dong(i, h, loai, chrf, **k):
    d = {"id": f"p{i}:{loai}:{h}", "product_id": f"p{i}", "huong": h, "loai": loai, "chrf": chrf, "sinh_lap": False, "giu_so_dat": True, "giu_latin_dat": True,
         "sai_chu_viet": False, "da_cat_loi_dan": False, "ref_giu_so_dat": True, "ref_giu_latin_dat": True, "ref_sai_chu_viet": False}
    d.update(k)
    return d


def test_so_nhanh_thang_thua_va_canh_bao():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zs, ft = [], []
        for i in range(60):
            for loai in ("title", "mo_ta"):
                # zh->vi: fine-tune hơn rõ cả hai loại; vi->zh: fine-tune kém rõ, sinh lặp và mất số tăng
                zs.append(_dong(i, "zh->vi", loai, 50.0 + (i % 3)))
                ft.append(_dong(i, "zh->vi", loai, 55.0 + (i % 3)))
                zs.append(_dong(i, "vi->zh", loai, 60.0 + (i % 3)))
                ft.append(_dong(i, "vi->zh", loai, 55.0 + (i % 3), sinh_lap=(i % 10 == 0), giu_so_dat=(i % 5 != 0)))
        zs = [dict(r, id=r["id"]) for r in zs]
        _ghi(td, "cum", "zero_shot", zs)
        _ghi(td, "cum", "ft", ft)
        kq = T.so_nhanh(td, "cum")
        assert kq["huong"]["zh->vi"]["chung"]["thang_zero_shot"] is True and abs(kq["huong"]["zh->vi"]["chung"]["chenh"] - 5.0) < 1e-9
        assert kq["huong"]["vi->zh"]["chung"]["thang_zero_shot"] is False and kq["huong"]["vi->zh"]["chung"]["cao"] < 0
        cb = " | ".join(kq["canh_bao"])
        assert "vi->zh title" in cb and "vi->zh mo_ta" in cb and "sinh lặp" in cb and "giu_so_dat" in cb and "zh->vi" not in cb, cb
        md = T.viet_md({"cum": kq}, {"mau_do": {"cum": {"so_mau": 240, "so_sp": 60}}, "adapter": {"cum": {"ten": "cum_f0_p11", "sha256": "a" * 64}}})
        assert "KHÔNG ĐẠT ở: cum vi->zh" in md and "không phải đánh giá của người" in md
        # lệch tập mục thì dừng
        _ghi(td, "pair", "zero_shot", zs)
        _ghi(td, "pair", "ft", ft[:-2])
        try:
            T.so_nhanh(td, "pair")
        except SystemExit:
            pass
        else:
            raise AssertionError("phai dung khi lech tap muc")


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_") and callable(ham):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
