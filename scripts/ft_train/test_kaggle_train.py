"""Thử các hàm thuần của kaggle_train (không cần torch). Chạy: python -I -X utf8 scripts/ft_train/test_kaggle_train.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_train as K  # noqa: E402


class TokGia:
    """Mỗi ký tự là một token; khuôn chat giống Qwen3 khi tắt suy luận."""

    def apply_chat_template(self, tin, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert tokenize is False and add_generation_prompt and enable_thinking is False
        return f"<u>{tin[0]['content']}</u><a>"

    def __call__(self, s, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in s]}


def _m(i, u, a, f=1):
    return {"id": str(i), "fold5": f, "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}


def test_xay_mau_loss_chi_tren_dich():
    ra, bo = K.xay_mau([_m(1, "ab", "xy"), _m(2, "a" * 50, "b" * 50)], TokGia(), 40)
    assert bo == 1 and len(ra) == 1
    r = ra[0]
    tinh = [x for x in r["labels"] if x != -100]
    assert "".join(map(chr, tinh)) == "xy" + K.NHAC_EOS + "\n", "loss phải gồm đích và <|im_end|>"
    assert len(r["ids"]) == len(r["labels"]) and r["labels"][0] == -100 and r["ids"][-len(tinh):] == tinh


def test_chia_lo():
    mau = [{"ids": [0] * ((i * 7) % 90 + 1)} for i in range(1000)]
    lo = K.chia_lo(mau, 8, 42)
    assert sum(len(x) for x in lo) == 1000 and all(len(x) <= 8 for x in lo)
    dai = [len(m["ids"]) for m in lo[0]]
    assert dai == sorted(dai), "trong một lô phải gần cùng độ dài"
    assert [len(m["ids"]) for x in lo for m in x] == [len(m["ids"]) for x in K.chia_lo(mau, 8, 42) for m in x], "cùng seed cùng thứ tự"


def test_ngan_sach_token():
    mau = [{"ids": [0] * n} for n in [2000] * 6 + [10] * 20]
    for lo in K.chia_lo(mau, 8, 1, ngan_sach=8192):
        assert len(lo) <= 8 and len(lo) * max(len(m["ids"]) for m in lo) <= 8192, "lô dài phải bị giới hạn theo token"
    assert sum(len(x) for x in K.chia_lo(mau, 8, 1, ngan_sach=8192)) == 26, "không mất mẫu"
    assert [len(x) for x in K.cat_ngan_sach([{"ids": [0] * 5000}], 8, 100)] == [1], "mẫu đơn vượt ngân sách vẫn thành lô một mẫu"


def test_gom_buoc():
    lo = [[{"ids": [0]}] * n for n in (32, 20, 20, 5, 30, 4)]
    b = K.gom_buoc(lo, 32)
    assert [[len(k) for k in x] for x in b] == [[32], [20, 20], [5, 30], [4]], "gom tới khi đủ, phần dư cuối thành một bước"
    assert sum(len(k) for x in b for k in x) == 111, "không mất mẫu"
    mau = [{"ids": [0] * ((i * 13) % 1500 + 5)} for i in range(5000)]
    a = K.gom_buoc(K.chia_lo(mau, 32, 7, 4096), 32)
    assert sum(len(k) for x in a for k in x) == 5000
    assert all(sum(len(k) for k in x) >= 32 for x in a[:-1]), "mọi bước trừ bước cuối đủ batch"
    assert K.lich_lr(len(a) - 1, len(a)) < 0.01, "lịch lr về gần 0 đúng ở bước cuối khi tổng số bước đếm từ gom_buoc"


def test_cau_hinh_b6():
    ten = [c["ten"] for c in K.CAU_HINH_B6]
    assert len(ten) == 7 and len(set(ten)) == 7
    assert {(c["lr"], c["rank"]) for c in K.CAU_HINH_B6 if "muc_tieu" not in c} == {(l, r) for l in (5e-5, 1e-4, 2e-4) for r in (16, 64)}, "đủ lưới 6 cấu hình"
    a = [c for c in K.CAU_HINH_B6 if "muc_tieu" in c][0]
    assert a["lr"] == 1e-4 and a["rank"] == 16 and a["muc_tieu"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_diem_macro_theo_huong():
    d = {"zh->vi|title": 1.0, "zh->vi|mo_ta": 3.0, "vi->zh|title": 0.0, "vi->zh|mo_ta": 2.0}
    assert K.diem_macro(d) == 1.5, "mỗi hướng trung bình đều hai loại, rồi trung bình đều hai hướng"


def test_chon_vong2():
    assert K.chon_vong2({"a": 1.0, "b": 1.5, "c": 2.0}) == ["a", "b"], "mặc định hai cấu hình đầu"
    assert K.chon_vong2({"a": 1.0, "b": 1.005, "c": 1.008, "d": 1.009}) == ["a", "b", "c"], "thêm cấu hình sát đầu bảng, trần 3"
    assert K.chon_vong2({"a": 1.0, "b": 1.5, "c": 1.6}) == ["a", "b"], "cấu hình thứ 3 lệch quá 1% thì không vào"
    assert K.chon_vong2({"a": float("nan"), "b": 1.0, "c": 1.1}) == ["b", "c"], "bỏ loss không hữu hạn"
    assert K.chon_vong2({"a": float("inf"), "b": 1.0}) == ["b"]


def test_lay_phan_tang():
    mau = [{"id": str(i), "huong": "zh->vi" if i % 2 else "vi->zh", "loai": "title" if i % 3 else "mo_ta"} for i in range(400)]
    a = K.lay_phan_tang(mau, 10, 1)
    assert len(a) == 40 and len({m["id"] for m in a}) == 40
    assert a == K.lay_phan_tang(list(reversed(mau)), 10, 1), "không phụ thuộc thứ tự đầu vào"
    assert len(K.lay_phan_tang(mau, 1000, 1)) == 400, "nhóm ít hơn thì lấy hết"


def test_bootstrap_chenh():
    n = ["zh->vi|title"] * 200
    r = K.bootstrap_chenh([50.0 + (i % 5) for i in range(200)], [40.0 + (i % 5) for i in range(200)], n, so_lan=200)
    assert abs(r["chenh"] - 10) < 1e-9 and r["thap"] <= 10 <= r["cao"] and r["thap"] > 0, "chênh rõ thì khoảng không chứa 0"
    r = K.bootstrap_chenh([float(i % 7) for i in range(200)], [float((i + 3) % 7) for i in range(200)], n, so_lan=200)
    assert r["thap"] < 0 < r["cao"], "không chênh thì khoảng chứa 0"


def _mau_b78(n=400, nhanh="cum"):
    return [{"id": f"{i}_{nhanh if i % 50 == 0 else 'chung'}", "fold5": i % 5, "phan": 0, "nguon": f"{nhanh}{i}", "huong": "zh->vi" if i % 2 else "vi->zh", "loai": "title" if i % 3 else "mo_ta"} for i in range(n)]


def test_chon_mau_sinh():
    cum, pair = _mau_b78(), _mau_b78(nhanh="pair")
    a, thieu = K.chon_mau_sinh(cum, cum, 0, 5, 42)
    assert thieu == 0 and a == K.lay_phan_tang([m for m in cum if m["fold5"] == 0], 5, 43), "nhánh cụm phải đúng cách chọn của B6: seed + 1"
    b, thieu_b = K.chon_mau_sinh(cum, pair, 0, 5, 42)
    assert all(m["nguon"].startswith("pair") for m in b), "lấy nội dung của nhánh được chấm"
    co_id_rieng = [m for m in a if m["id"].endswith("_cum")]
    assert thieu_b == len(co_id_rieng) and len(b) == len(a) - thieu_b, "id chỉ có ở cụm phải đếm là thiếu, không im lặng bỏ"
    assert [m["id"] for m in b] == [m["id"] for m in a if not m["id"].endswith("_cum")], "giữ thứ tự để so theo cặp"
    gon, _ = K.chon_mau_sinh(cum, cum, 0, 5, 42, rut_gon=2)
    assert {m["id"] for m in gon} <= {m["id"] for m in a}, "chạy khói phải là tập con của mẫu thật"
    assert len(gon) == 2 * len({f"{m['huong']}|{m['loai']}" for m in a})


def test_gop_loss():
    nhom, dm = K.gop_loss({"zh->vi|title|a": (10.0, 10), "zh->vi|title|b": (30.0, 10), "zh->vi|mo_ta|a": (1.0, 1), "vi->zh|title|a": (4.0, 2), "vi->zh|mo_ta|a": (8.0, 2)})
    assert nhom == {"vi->zh|mo_ta": 4.0, "vi->zh|title": 2.0, "zh->vi|mo_ta": 1.0, "zh->vi|title": 2.0}, "gộp trong nhóm theo token"
    assert dm == (3.0 + 1.5) / 2


def test_viec_b78():
    assert K.CAU_HINH_CHOT == {"ten": "lr2e-4_r64", "lr": 2e-4, "rank": 64} and K.CAU_HINH_CHOT["ten"] in {c["ten"] for c in K.CAU_HINH_B6}
    assert K.VIEC_B78 == [("cum_f0", "cum", 0), ("pair_f0", "pair", 0), ("cum_f1", "cum", 1), ("cum_f2", "cum", 2)]
    assert set(K.CAP_ATTN) <= {c["ten"] for c in K.CAU_HINH_B6}


def _mau_tl(loai, so_token):
    return {"loai": loai, "labels": [-100] * 3 + [1] * so_token}


def test_ti_trong_title_tinh_tay():
    # bước 1: lô 2 title (10 token) và lô 2 mô tả (90 token); bước 2: lô trộn 1 title 10 token, 1 mô tả 30 token
    ds = [[[_mau_tl("title", 5), _mau_tl("title", 5)], [_mau_tl("mo_ta", 45), _mau_tl("mo_ta", 45)]],
          [[_mau_tl("title", 10), _mau_tl("mo_ta", 30)]]]
    tat = K.tom_tat_buoc(ds)
    assert tat == [[(2, 0, 10, 0), (0, 2, 0, 90)], [(1, 1, 10, 30)]]
    assert abs(K.ti_trong_title(tat, 1.0) - (0.5 + 0.25) / 2) < 1e-12, "bước 1 chia đều theo mẫu, bước 2 theo token trong lô trộn"
    assert abs(K.ti_trong_title(tat, 3.0) - (0.75 + 0.5) / 2) < 1e-12


def test_tim_w_title():
    ds = [[[_mau_tl("title", 5), _mau_tl("title", 5)], [_mau_tl("mo_ta", 45), _mau_tl("mo_ta", 45)]], [[_mau_tl("title", 10), _mau_tl("mo_ta", 30)]]]
    tat = K.tom_tat_buoc(ds)
    assert K.ti_trong_title(tat, 1e-4) < 0.01 and K.ti_trong_title(tat, 1e4) > 0.99, "bước toàn title, toàn mô tả không được làm tỉ lệ kẹt giữa chừng"
    for p in (0.1074, 0.5, 0.75):
        assert abs(K.ti_trong_title(tat, K.tim_w_title(tat, p)) - p) < 1e-9
    mau = [_mau_tl("title", 1)] * 3 + [_mau_tl("mo_ta", 1)]
    assert K.he_chuan(mau, 1.0) == 1.0 and abs(K.he_chuan(mau, 3.0) * sum(K.trong_so_lo(mau, 3.0)) / len(mau) - 1.0) < 1e-12, "trọng số trung bình giữ bằng 1"
    for sai in (0, 1, -0.1):
        try:
            K.tim_w_title(tat, sai)
        except ValueError:
            continue
        raise AssertionError("p ngoài khoảng phải báo lỗi")
    assert K.trong_so_lo([_mau_tl("title", 1), _mau_tl("mo_ta", 1)], 0.2) == [0.2, 1.0]


def test_kiem_bat_thuong():
    c = lambda d, l: {"chrf": {"chrf_macro_huong": d}, "sinh_lap": l, "so_mau": 1000}    # noqa: E731
    assert K.kiem_bat_thuong(c(75.0, 3), 65.0, 0.05) is None
    assert K.kiem_bat_thuong(c(65.0, 50), 65.0, 0.05) is None, "đúng ngưỡng thì không dừng"
    assert "chrF" in K.kiem_bat_thuong(c(64.9, 0), 65.0, 0.05)
    assert "sinh lap" in K.kiem_bat_thuong(c(75.0, 51), 65.0, 0.05)
    assert K.kiem_bat_thuong(c(None, 0), 65.0, 0.05) is not None, "thiếu điểm không được coi là ổn"


def test_chon_trang_thai():
    cu = ("kho", {"viec": {"a": {"buoc": 1102, "cham": None}, "b": {"buoc": 1843, "cham": {"x": 1}}}})
    moi = ("notebook", {"viec": {"a": {"buoc": 1500, "cham": None}, "b": {"buoc": 1843, "cham": {"x": 1}}}})
    assert K.chon_trang_thai([cu, moi])[0] == "notebook" and K.chon_trang_thai([moi, cu])[0] == "notebook", "không phụ thuộc thứ tự"
    xong = ("c", {"viec": {"a": {"buoc": 1800, "cham": {"x": 1}}, "b": {"buoc": 1843, "cham": {"x": 1}}}})
    assert K.chon_trang_thai([moi, xong])[0] == "c"


def test_lich_lr_noi_dung_khi_dung_giua_chung():
    tong = 300
    assert K.lich_lr(100, tong) > K.lich_lr(200, tong) > K.lich_lr(299, tong), "lịch cosine của cả 3 epoch nên vòng 2 nối tiếp đúng"
    assert K.lich_lr(100, tong) > 0.5


def test_lich_lr():
    assert K.lich_lr(0, 100) < K.lich_lr(2, 100) <= 1.0
    assert abs(K.lich_lr(3, 100) - 1.0) < 1e-9
    assert K.lich_lr(99, 100) < 0.01
    assert all(0 <= K.lich_lr(b, 100) <= 1 for b in range(100))


def test_chon_tran_sinh():
    """Trần theo độ dài nguồn (07/11/2026): 3,5 lần + 64, tối đa 2048; luôn đủ cho tỉ lệ đích/nguồn lớn nhất đo trên pool (2,94 ở title zh->vi)."""
    assert K.chon_tran_sinh(20) == 134 and K.chon_tran_sinh(1) == 68
    assert K.chon_tran_sinh(10000) == 2048
    for n in range(1, 2000):
        assert K.chon_tran_sinh(n) >= min(2048, 2.94 * n) and K.chon_tran_sinh(n) <= 2048


def test_kiem_ro_ri_test():
    pool = [{"id": "a:title", "product_id": "a"}, {"id": "b:title", "product_id": "b"}]
    ok = [{"id": "c:title", "product_id": "c"}, {"id": "c:mo_ta", "product_id": "c"}]
    assert K.kiem_ro_ri_test(ok, pool) == (1, 2)
    for xau in ([{"id": "x:title", "product_id": "a"}], [{"id": "b:title", "product_id": "z"}]):
        try:
            K.kiem_ro_ri_test(xau, pool)
        except RuntimeError as e:
            assert str(e).startswith("TEST_ROI_VAO_POOL")
        else:
            raise AssertionError("phai bat ro ri")


def test_kiem_khoa_test_do():
    import tempfile
    cu = K.VAO
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            K.VAO = td
            (td / "pool").mkdir()
            (td / "pool" / "pool_cum.jsonl.gz").write_bytes(b"x")
            for cam in ("test_cum.jsonl", "split.parquet", "ds_test.txt"):
                (td / "pool" / cam).write_bytes(b"x")
                try:
                    K.kiem_khoa_test_do()
                except RuntimeError as e:
                    assert "CO_FILE_TEST_LA" in str(e) or "THIEU_DO_KHOA" in str(e), (cam, e)
                else:
                    raise AssertionError(f"phai bat {cam}")
                (td / "pool" / cam).unlink()
            try:
                K.kiem_khoa_test_do()
            except RuntimeError as e:
                assert str(e) == "THIEU_DO_KHOA"          # không có do_khoa thì không phải lần đo test
            else:
                raise AssertionError("thieu do_khoa phai loi")
            (td / "do_khoa").mkdir()
            (td / "do_khoa" / "do_khoa_cum.jsonl.gz").write_bytes(b"x")
            K.kiem_khoa_test_do()                          # đúng tên thì qua
            (td / "do_khoa" / "test_pair.jsonl").write_bytes(b"x")
            try:
                K.kiem_khoa_test_do()
            except RuntimeError as e:
                assert "CO_FILE_TEST_LA" in str(e)
            else:
                raise AssertionError("file test_ la phai bi bat du co do_khoa")
            (td / "do_khoa" / "test_pair.jsonl").unlink()
            # mọi chế độ KHÁC đo test: guard thường phải chặn cả file do_khoa_*, kẻo gắn nhầm Dataset test vào phiên huấn luyện
            try:
                K.kiem_khoa_test()
            except RuntimeError as e:
                assert "CO_FILE_TEST_TRONG_INPUT" in str(e) and "do_khoa_cum" in str(e)
            else:
                raise AssertionError("guard thuong phai chan do_khoa_")
            (td / "do_khoa" / "do_khoa_cum.jsonl.gz").unlink()
            K.kiem_khoa_test()                              # không còn gì thì qua
    finally:
        K.VAO = cu


if __name__ == "__main__":
    for ten, ham in list(globals().items()):
        if ten.startswith("test_"):
            ham()
            print("ok", ten)
    print("TẤT CẢ ĐẠT")
