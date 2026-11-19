# -*- coding: utf-8 -*-
"""Test nho cho P1 (du lieu gia, khong doc du lieu that)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_phan_tich as P  # noqa: E402
import p1_tai_lap as L  # noqa: E402


class TestBootstrap(unittest.TestCase):
    def test_boot_chenh_co_ban(self):
        r = P.boot_chenh({"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 2.0}, n=50, seed=1)
        self.assertAlmostEqual(r["chenh"], 0.5)
        self.assertEqual(r["n_sp"], 2)
        self.assertLessEqual(r["thap"], r["cao"])

    def test_boot_chenh_khong_giao(self):
        r = P.boot_chenh({"a": 1.0}, {"b": 2.0}, n=10, seed=1)
        self.assertIsNone(r)


class TestKep(unittest.TestCase):
    def test_kep_bang_nhau(self):
        a = {"p1": 10.0, "p2": 10.0, "q1": 10.0, "q2": 10.0}
        b = {"p1": 0.0, "p2": 0.0, "q1": 0.0, "q2": 0.0}
        nhom = {"p1": "gemma", "p2": "gemma", "q1": "qwen", "q2": "qwen"}
        r = P._kep(a, b, nhom, "M", "B", n_boot=20)
        self.assertAlmostEqual(r["kep"]["chenh"], 0.0)
        self.assertAlmostEqual(r["qwen"]["chenh"], 10.0)
        self.assertAlmostEqual(r["gemma"]["chenh"], 10.0)

    def test_kep_thieu_nhom(self):
        a = {"p1": 10.0}
        b = {"p1": 0.0}
        r = P._kep(a, b, {"p1": "gemma"}, "M", "B", n_boot=10)
        self.assertIsNone(r)


class TestTaiLap(unittest.TestCase):
    def test_so_sanh_khop(self):
        nhom = {g: 1.0 for g in L.T.NHOM}
        cu = {"m": {"tb_chinh": 1.0, "tb_macro": 2.0,
                    "cac_lan": {"s42": {"chrf_corpus": nhom}}}}
        moi = {"tb_chinh": 1.005, "tb_macro": 2.0,
               "cac_lan": [{"lan": "s42", "chrf_corpus": nhom}]}
        r = L.so_sanh("m", moi, cu)
        self.assertTrue(all(x["dat"] for x in r))

    def test_so_sanh_lech(self):
        nhom = {g: 1.0 for g in L.T.NHOM}
        cu = {"m": {"tb_chinh": 1.0, "tb_macro": 2.0,
                    "cac_lan": {"s42": {"chrf_corpus": nhom}}}}
        moi = {"tb_chinh": 1.5, "tb_macro": 2.0,
               "cac_lan": [{"lan": "s42", "chrf_corpus": nhom}]}
        r = L.so_sanh("m", moi, cu)
        self.assertFalse(all(x["dat"] for x in r))


if __name__ == "__main__":
    unittest.main()
