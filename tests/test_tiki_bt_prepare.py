# -*- coding: utf-8 -*-
"""Test Plan A: key map, BT input, cache, routing, product state."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cado_core.tiki_bt_prepare import (
    build_tiki_key_map,
    build_translation_cache,
    build_product_state,
    prepare_bt_input,
)


def units_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "pair_id": [
            "tiki:p1:title",
            "tiki:p1:attr:0:0",
            "tiki:p2:title",
            "tiki:p2:attr:0:0",
            "tiki:p2:attr:1:0",
            "tiki:p3:title",
            "tiki:p3:attr:0:0",
        ],
        "product_id": ["p1", "p1", "p2", "p2", "p2", "p3", "p3"],
        "unit_type": ["title", "attribute", "title", "attribute", "attribute", "title", "attribute"],
        "key_vi": [None, "Chất liệu", None, "Màu sắc", "Thương hiệu", None, "Kích cỡ"],
        "key_norm": [None, "chất liệu", None, "màu sắc", "thương hiệu", None, "kích cỡ"],
        "key_class": [None, "text", None, "text", "admin", None, "numeric"],
        "key_zh": [None, None, None, None, None, None, None],
        "value_vi": [None, "Nhựa", None, "Đen", "LocknLock", None, "20 cm"],
        "text_vi": [
            "Hộp nhựa 1L", "Chất liệu: Nhựa", "Bình nước", "Màu sắc: Đen",
            "Thương hiệu: LocknLock", "Sản phẩm", "Kích cỡ: 20 cm",
        ],
        "value_class": ["text", "text", "text", "text", "admin", "text", "numeric"],
        "train_eligible": [True, True, True, True, False, True, False],
        "expected_zh_value_chars": [5, 3, 4, 3, 3, 4, 4],
        "zh_value_chars_p05": [3, 1, 2, 1, 1, 2, 2],
        "zh_value_chars_p95": [8, 6, 7, 6, 6, 7, 7],
        "key_len_source": [None, "estimated_key_class", None, "estimated_key_class", "estimated_key_class", None, "estimated_key_class"],
        "length_prior_soft_qc": [True, False, True, False, True, True, True],
    })


def products_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "title_vi": ["Hộp nhựa 1L", "Bình nước", "Sản phẩm"],
        "category": ["home", "home", "home"],
        "split": ["train", "train", "test"],
        "n_units": [2, 3, 2],
        "n_train_eligible_units": [2, 2, 1],
        "bt_status": ["pending", "pending", "pending"],
        "zh_origin": ["synthetic_tiki", "synthetic_tiki", "synthetic_tiki"],
    })


class PlanATest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.alignment_path = self.root / "alignment.csv"
        pd.DataFrame({
            "key_zh_norm": ["材质", "颜色", "品牌"],
            "key_vi_norm": ["chất liệu", "màu sắc", "thương hiệu"],
            "freq": [10, 10, 8],
            "share": [0.9, 0.9, 0.8],
            "top_vi": ["chất liệu", "màu sắc", "thương hiệu"],
            "top_share": [0.9, 0.9, 0.8],
            "mapped": [True, True, True],
        }).to_csv(self.alignment_path, index=False, encoding="utf-8-sig")
        self.key_map_path = self.root / "key_map.csv"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_build_key_map_and_pending(self) -> None:
        key_map = build_tiki_key_map(
            units_frame(),
            alignment_map_path=self.alignment_path,
            existing_map_path=self.key_map_path,
            min_key_freq=1,
            min_key_share=0.6,
        )
        self.assertEqual(key_map.mapping["chất liệu"], "材质")
        self.assertEqual(key_map.mapping["màu sắc"], "颜色")
        self.assertEqual(key_map.mapping["thương hiệu"], "品牌")
        self.assertNotIn("kích cỡ", key_map.mapping)
        self.assertEqual(key_map.stats["n_mapped_keys"], 3)
        self.assertEqual(key_map.stats["n_pending_keys"], 1)
        self.assertEqual(key_map.stats["n_attribute_units"], 4)
        self.assertEqual(key_map.stats["n_mapped_attribute_units"], 3)

    def test_prepare_bt_input_uses_value_only_for_attribute(self) -> None:
        key_map = build_tiki_key_map(
            units_frame(),
            alignment_map_path=self.alignment_path,
            existing_map_path=self.key_map_path,
            min_key_freq=1,
            min_key_share=0.6,
        )
        result = prepare_bt_input(
            units_frame(), products_frame(), key_map
        )
        units = result.units

        material = units[units["pair_id"] == "tiki:p1:attr:0:0"].iloc[0]
        self.assertEqual(material["source_text_bt"], "Nhựa")
        self.assertNotIn("Chất liệu", material["source_text_bt"])
        self.assertEqual(material["key_zh_effective"], "材质")
        self.assertEqual(material["bt_route"], "mt")
        self.assertTrue(bool(material["merge_ready"]))
        self.assertEqual(
            int(material["expected_zh_text_chars"]),
            int(material["expected_zh_value_chars"]) + 2 + 2,
        )

        title = units[units["pair_id"] == "tiki:p1:title"].iloc[0]
        self.assertEqual(title["source_text_bt"], "Hộp nhựa 1L")
        self.assertEqual(title["bt_route"], "mt")

        brand = units[units["pair_id"] == "tiki:p2:attr:1:0"].iloc[0]
        self.assertEqual(brand["bt_route"], "dictionary")
        self.assertEqual(brand["bt_route_fallback"], "mt")
        self.assertTrue(bool(brand["length_prior_soft_qc"]))

        numeric = units[units["pair_id"] == "tiki:p3:attr:0:0"].iloc[0]
        self.assertEqual(numeric["bt_route"], "rule")
        self.assertEqual(numeric["merge_block_reason"], "key_unmapped")
        self.assertFalse(bool(numeric["merge_ready"]))

        self.assertEqual(result.stats["plan"], "A")
        self.assertEqual(result.stats["n_units"], 7)
        self.assertGreater(result.stats["n_bt_ready"], 0)
        self.assertEqual(result.stats["n_product_merge_eligible"], 0)

    def test_cache_dedup_and_product_state(self) -> None:
        key_map = build_tiki_key_map(
            units_frame(),
            alignment_map_path=self.alignment_path,
            existing_map_path=self.key_map_path,
            min_key_freq=1,
            min_key_share=0.6,
        )
        result = prepare_bt_input(
            units_frame(), products_frame(), key_map
        )
        cache = result.cache
        self.assertGreater(len(cache), 0)
        self.assertIn("cache_key", cache.columns)
        self.assertEqual(
            int(cache["n_units"].sum()),
            len(result.bt_input),
        )

        states = result.product_state.set_index("product_id")
        self.assertEqual(states.loc["p1", "product_bt_status"], "ready_for_bt")
        self.assertEqual(states.loc["p3", "product_bt_status"], "blocked_key_map")
        self.assertFalse(bool(states.loc["p1", "merge_eligible"]))
        self.assertTrue(bool(states.loc["p1", "plan_a_ready"]))
        self.assertFalse(bool(states.loc["p3", "plan_a_ready"]))


if __name__ == "__main__":
    unittest.main()
