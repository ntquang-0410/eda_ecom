"""Thử phần thuần của dich_bang_adapter.py (không cần GPU): lời dẫn đúng chữ như khi huấn luyện, trần sinh giống kaggle_train. Chạy: python -I -X utf8 scripts/ft_train/test_dich_bang_adapter.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dich_bang_adapter as A  # noqa: E402
import kaggle_train as K  # noqa: E402


def test_tran_giong_kaggle_train():
    for n in (0, 1, 20, 500, 5000):
        assert A.chon_tran(n) == K.chon_tran_sinh(n)


def test_loi_dan_giong_mau_huan_luyen():
    """Lấy vài mẫu thật của POOL (không phải test): lời dẫn dựng lại phải trùng nguyên văn nội dung 'user' lưu trong mẫu."""
    pool = Path(__file__).resolve().parents[2] / "data" / "processed" / "ft_run" / "mau"
    da = 0
    for nhanh in ("cum", "pair"):
        p = pool / f"pool_{nhanh}.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            ds = [json.loads(x) for _, x in zip(range(400), f)]
        title = {m["product_id"]: m["nguon"] for m in ds if m["loai"] == "title"}
        for m in ds:
            if m["loai"] == "title":
                assert A.dung_loi_dan(m["huong"], "title", nhanh, m["nguon"]) == m["messages"][0]["content"]
                da += 1
            elif m["product_id"] in title and m.get("phan", 0) == 0:
                assert A.dung_loi_dan(m["huong"], "mo_ta", nhanh, m["nguon"], title[m["product_id"]]) == m["messages"][0]["content"]
                da += 1
    assert da > 0, "khong co mau pool de so"


def test_mo_ta_can_title():
    try:
        A.dung_loi_dan("zh->vi", "mo_ta", "pair", "品牌: x")
    except ValueError:
        pass
    else:
        raise AssertionError("mo_ta thieu title nguon phai loi")


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_") and callable(ham):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
