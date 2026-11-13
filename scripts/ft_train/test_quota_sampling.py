# -*- coding: utf-8 -*-
"""Test cong thuc quota: chia deu trong nhom, group share, tinh tat dinh."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from quota_sampling import CONFIGS, THU_TU, lay_nhom  # noqa: E402


def test_config_sum():
    for tap in ["cum", "pair"]:
        for cname, cfg in CONFIGS[tap].items():
            if cfg is None:
                continue
            assert abs(sum(cfg.values()) - 100) < 1e-9, (tap, cname, sum(cfg.values()))
    print("OK config sum 100")


def test_nhom_deu():
    # 2 nhom, moi nhom 1 dong; e=10/20; muon 50/50 -> quota phai lay nhieu dong 10 hon
    e = np.array([10.0, 20.0])
    idx_a = np.array([0])
    idx_b = np.array([1])
    T = 2_000_000
    rng = np.random.default_rng(42)
    a = lay_nhom(rng, idx_a, e, T)
    b = lay_nhom(rng, idx_b, e, T)
    share_a = e[a].sum() / (e[a].sum() + e[b].sum())
    print("share A", round(float(share_a), 4), "n_a", len(a), "n_b", len(b))
    assert abs(share_a - 0.5) < 0.01, share_a


def test_tat_dinh():
    e = np.array([5.0, 10.0, 20.0])
    idx = np.array([0, 1, 2])
    a = lay_nhom(np.random.default_rng(42), idx, e, 100_000)
    b = lay_nhom(np.random.default_rng(42), idx, e, 100_000)
    assert np.array_equal(a, b)
    print("OK tat dinh")


def test_rong():
    a = lay_nhom(np.random.default_rng(42), np.array([0]), np.array([10.0]), 0)
    assert len(a) == 0
    print("OK T=0 rong")


if __name__ == "__main__":
    test_config_sum()
    test_nhom_deu()
    test_tat_dinh()
    test_rong()
    print("TAT CA DAT")
