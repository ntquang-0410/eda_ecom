# -*- coding: utf-8 -*-
"""Test các hàm thuần của bộ bao job và tiến trình theo dõi (không cần Colab).
Chạy:  python -I -X utf8 -m unittest scripts/colab/test_colab.py   (hoặc cd scripts/colab && python -I -X utf8 -m unittest test_colab -v)"""
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cau_hinh_tlc as CH  # noqa: E402
import giam_sat_vm as G  # noqa: E402
import theo_doi_may_nha as T  # noqa: E402


class TestGiamSat(unittest.TestCase):
    def test_quyet_dinh(self):
        self.assertEqual(G.quyet_dinh_sau_luot(0, {"xong": True}, 0, 5, 9), ("XONG", 0))
        self.assertEqual(G.quyet_dinh_sau_luot(0, {"bat_thuong": "x"}, 0, 5, 9)[0], "BAT_THUONG")
        self.assertEqual(G.quyet_dinh_sau_luot(1, {}, 0, 5, 5), ("TIEP", 1))        # lỗi lần 1: thử lại
        self.assertEqual(G.quyet_dinh_sau_luot(1, {}, 1, 5, 5), ("LOI", 2))         # lỗi lần 2 liên tiếp
        self.assertEqual(G.quyet_dinh_sau_luot(0, {}, 0, 5, 9), ("TIEP", 0))        # chạy bình thường, có tiến bước
        self.assertEqual(G.quyet_dinh_sau_luot(0, {}, 2, 9, 9), ("LOI", 3))         # kẹt ba lượt liền
        self.assertEqual(G.quyet_dinh_sau_luot(1, {"xong": True}, 0, 5, 9)[0], "TIEP")   # xong mà rc!=0: không tin là xong

    def test_tong_buoc(self):
        self.assertEqual(G.tong_buoc_da_chay({"viec": {"a": {"buoc": 3}, "b": {"buoc": None}, "c": {}}}), 3)
        self.assertEqual(G.tong_buoc_da_chay({}), 0)

    def test_tao_goi_va_hash(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            goc = d / "goc"
            (goc / "v" / "ep2").mkdir(parents=True)
            a, b = goc / "v" / "sinh_ep2.jsonl", goc / "v" / "ep2" / "adapter_model.safetensors"
            a.write_text("xin chao\n", encoding="utf-8")
            b.write_bytes(b"\x00" * 1000)
            mf = G.tao_goi("ket_qua_v", goc, [a, b], thu_muc_ra=d / "kq", them={"loai": "ket_qua"}, kich_phan=700)
            self.assertEqual(mf["so_tep"], 2)
            # các phần nối lại phải đúng bằng tar, mỗi phần đúng hash
            ghep = b""
            for ph in mf["phan"]:
                nd = (d / "kq" / "ket_qua_v.parts" / ph["ten"]).read_bytes()
                self.assertEqual(len(nd), ph["byte"])
                self.assertEqual(G.hashlib.sha256(nd).hexdigest(), ph["sha256"])
                ghep += nd
            self.assertGreater(len(mf["phan"]), 1)
            self.assertEqual(ghep, (d / "kq" / "ket_qua_v.tar").read_bytes())
            self.assertEqual(G.hashlib.sha256(ghep).hexdigest(), mf["sha256"])
            self.assertEqual(mf["sha256"], G.sha256_file(d / "kq" / "ket_qua_v.tar"))
            self.assertTrue((d / "kq" / "ket_qua_v.json").exists())
            self.assertFalse((d / "kq" / "ket_qua_v.tar.part").exists())
            with tarfile.open(d / "kq" / "ket_qua_v.tar") as t:
                self.assertEqual(sorted(t.getnames()), ["v/ep2/adapter_model.safetensors", "v/sinh_ep2.jsonl"])
            # ghi đè: .json cũ bị xóa trước, rồi ghi lại với hash mới
            a.write_text("doi\n", encoding="utf-8")
            mf2 = G.tao_goi("ket_qua_v", goc, [a], thu_muc_ra=d / "kq")
            self.assertNotEqual(mf["sha256"], mf2["sha256"])

    def test_tep_ket_qua_va_viec_xong(self):
        with tempfile.TemporaryDirectory() as d:
            ra = Path(d)
            (ra / "v" / "ep2").mkdir(parents=True)
            (ra / "v" / "sinh_ep2.jsonl").write_text("{}", encoding="utf-8")
            (ra / "v" / "ep2" / "adapter_model.safetensors").write_bytes(b"x")
            (ra / "v" / "ep2" / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(len(G.tep_ket_qua_viec(ra, "v", True)), 3)
            self.assertEqual(len(G.tep_ket_qua_viec(ra, "v", False)), 2)      # bỏ safetensors
            self.assertFalse(G.viec_xong({"viec": {"v": {"cham": None}}}, "v", ra))
            self.assertTrue(G.viec_xong({"viec": {"v": {"cham": {"chrf": 1}}}}, "v", ra))
            self.assertFalse(G.viec_xong({"viec": {}}, "v", ra))


class TestTheoDoi(unittest.TestCase):
    def test_tach_json(self):
        self.assertEqual(T.tach_json("canh bao\n<<JSON>>{\"a\": 1}<<FIN>>\nrac"), {"a": 1})
        self.assertIsNone(T.tach_json("khong co"))
        self.assertIsNone(T.tach_json("<<JSON>>{hong<<FIN>>"))

    def test_phien_con(self):
        s = "[tlc] gpu-g4-s-abc | Hardware: G4 | Shape: Standard | Variant: GPU\n[khac] x"
        self.assertTrue(T.phien_con_tren_server(s, "tlc"))
        self.assertFalse(T.phien_con_tren_server("[colab] No active sessions found on server.", "tlc"))
        self.assertFalse(T.phien_con_tren_server("[tlc2] x", "tlc"))

    def test_can_tai_ckpt(self):
        mf = {"sha256": "b"}
        self.assertTrue(T.can_tai_ckpt(mf, "a", 0, 2000, 1800))
        self.assertFalse(T.can_tai_ckpt(mf, "b", 0, 2000, 1800))       # đã tải bản này
        self.assertFalse(T.can_tai_ckpt(mf, "a", 1000, 2000, 1800))    # chưa đủ chu kỳ
        self.assertFalse(T.can_tai_ckpt(None, "a", 0, 2000, 1800))

    def test_goi_can_tai_va_da_tai_du(self):
        kq = [{"ten": "ket_qua_a", "sha256": "1", "loai": "ket_qua"}, {"ten": "chung", "sha256": "2", "loai": "chung"}, {"ten": "ckpt", "sha256": "3", "loai": "ckpt"}]
        self.assertEqual([m["ten"] for m in T.goi_can_tai(kq, {})], ["ket_qua_a", "chung"])
        self.assertEqual([m["ten"] for m in T.goi_can_tai(kq, {"ket_qua_a": "1", "chung": "9"})], ["chung"])   # chung đổi hash thì tải lại
        done = {"goi": ["ckpt", "chung", "ket_qua_a"]}
        self.assertFalse(T.da_tai_du(done, {"ket_qua_a": "1"}, kq))
        self.assertTrue(T.da_tai_du(done, {"ket_qua_a": "1", "chung": "2"}, kq))
        self.assertFalse(T.da_tai_du(None, {}, kq))

    def test_giai_nen_an_toan(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ok = d / "ok.tar"
            with tarfile.open(ok, "w") as t:
                data = b"abc"
                ti = tarfile.TarInfo("v/x.txt")
                ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))
            T.giai_nen_an_toan(ok, d / "ra")
            self.assertEqual((d / "ra" / "v" / "x.txt").read_bytes(), b"abc")
            xau = d / "xau.tar"
            with tarfile.open(xau, "w") as t:
                ti = tarfile.TarInfo("../thoat.txt")
                ti.size = 1
                t.addfile(ti, io.BytesIO(b"x"))
            with self.assertRaises(RuntimeError):
                T.giai_nen_an_toan(xau, d / "ra2")
            self.assertFalse((d / "thoat.txt").exists())


class TestCauHinh(unittest.TestCase):
    def test_job_that(self):
        j = CH.tao_job("that")
        ten = [v["ten"] for v in j["cfg"]["viec"]]
        self.assertEqual(ten, ["cum_f0_p11", "pair_f0_p11", "cum_f0_p25", "cum_f0_p35", "pair_f0_p25", "pair_f0_p35"])
        self.assertEqual(j["viec"], ten)
        p = {v["ten"]: v["p_title"] for v in j["cfg"]["viec"]}
        self.assertEqual(p["cum_f0_p11"], 0.1074)
        self.assertEqual(p["pair_f0_p11"], 0.1086)
        self.assertTrue(all(v["fold"] == 0 for v in j["cfg"]["viec"]))
        c = j["cfg"]
        for k, v in {"micro_bs": 32, "ngan_sach_token": 4096, "max_len": 2048, "seed": 42, "gc": False, "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250}.items():
            self.assertEqual(c[k], v, k)
        self.assertNotIn("dung_sau_buoc", c)
        self.assertNotIn("so_train", c)
        self.assertEqual(len(j["pool_sha256"]), 2)
        self.assertEqual(j["mo_hinh"]["revision"], "1cfa9a7208912126459214e8b04321603b3df60c")
        self.assertEqual(j["ten_trang_thai"], "tlc")

    def test_job_khoi(self):
        j = CH.tao_job("khoi")
        self.assertEqual(j["ten_trang_thai"], "thu_tlc")
        self.assertEqual(j["cfg"]["dung_sau_buoc"], 3)
        self.assertIsNone(j["cfg"]["chrf_toi_thieu"])
        self.assertLessEqual(j["tran_gio"], 1.0)

    def test_khong_co_duong_dan_test(self):
        s = json.dumps(CH.tao_job("that")) + json.dumps(CH.tao_job("khoi"))
        for cam in ("test_khoa", "do_khoa", "split.parquet", "ds_test"):
            self.assertNotIn(cam, s)


if __name__ == "__main__":
    unittest.main()
