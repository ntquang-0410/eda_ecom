# -*- coding: utf-8 -*-
"""Test Phase 1 Tiki: profile + BT units (không cần model)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cado_core.tiki_units import (
    build_tiki_units,
    classify_key,
    classify_value,
    normalize_key,
    profile_tiki,
    serialize_attributes,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["1", "2", "3", "4"],
        "category": ["home", "home", "home", "home"],
        "brand": ["LocknLock", "OEM", "", ""],
        "title_vi": [
            "Hộp cơm LocknLock 750ml",
            "SP",
            "Bình nước 1L",
            "Khăn giấy 100 tờ",
        ],
        "description_vi": [
            (
                "Thương hiệu: LocknLock; Xuất xứ: Việt Nam; "
                "Chất liệu: Nhựa; Kích thước: 20 x 30 cm; "
                "Màu sắc: Đen, Trắng"
            ),
            "Thương hiệu: OEM",
            "",
            "Chất liệu: Giấy; phần không có dấu hai chấm",
        ],
        "attributes_vi": [
            [
                {"name": "Thương hiệu", "values": ["LocknLock"]},
                {"name": "Xuất xứ", "values": ["Việt Nam"]},
                {"name": "Chất liệu", "values": ["Nhựa"]},
                {"name": "Kích thước", "values": ["20 x 30 cm"]},
                {"name": "Màu sắc", "values": ["Đen", "Trắng"]},
            ],
            [],
            None,
            [],
        ],
        # Cột prose chỉ để chắc chắn Phase 1 không đọc/không xuất nó.
        "description_prose_vi": [
            "Đây là prose mô tả sản phẩm, không được dùng cho BT.",
            "",
            "",
            "",
        ],
    })


class KeyTest(unittest.TestCase):
    def test_normalize_key_and_alias(self) -> None:
        self.assertEqual(normalize_key(" Màu  Sắc "), "màu sắc")
        self.assertEqual(normalize_key("Màu"), "màu sắc")
        self.assertEqual(normalize_key("Hạn dùng"), "hạn sử dụng")

    def test_classify_key(self) -> None:
        self.assertEqual(classify_key("Thương hiệu"), "admin")
        self.assertEqual(classify_key("Xuất xứ (Made in)"), "admin")
        self.assertEqual(classify_key("Model"), "model")
        self.assertEqual(classify_key("Kích thước"), "numeric")
        self.assertEqual(classify_key("Chất liệu"), "text")

    def test_classify_value_overrides_numeric_key_with_text_value(self) -> None:
        self.assertEqual(classify_value("20 x 30 cm", "numeric"), "numeric")
        self.assertEqual(classify_value("Nhiều size", "numeric"), "text")
        self.assertEqual(classify_value("100% Cotton", "text"), "text")
        self.assertEqual(classify_value("2025", "text"), "numeric")

    def test_serialize_attributes_escapes_semicolon(self) -> None:
        text = serialize_attributes([
            {"name": "Màu sắc", "values": ["Đen; Trắng"]},
        ])
        self.assertEqual(text, "Màu sắc: Đen, Trắng")


class ProfileTest(unittest.TestCase):
    def test_profile_counts(self) -> None:
        result = profile_tiki(sample_frame())
        stats = result.stats

        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["title"]["n_3_40"], 3)
        self.assertEqual(stats["title"]["n_short_lt3"], 1)
        self.assertEqual(stats["title"]["duplicate_rows"], 0)
        self.assertEqual(stats["description"]["n_empty"], 1)
        self.assertEqual(stats["description"]["n_malformed_parts"], 1)
        self.assertEqual(stats["attributes"]["n_pairs"], 7)
        self.assertEqual(stats["attributes"]["n_values"], 8)
        self.assertEqual(stats["attributes"]["parse_sources"]["attributes_vi"], 1)
        self.assertEqual(stats["attributes"]["parse_sources"]["description_vi"], 2)
        self.assertEqual(stats["attributes"]["parse_sources"]["none"], 1)
        self.assertEqual(stats["value_classes"]["admin"]["n_values"], 3)
        self.assertEqual(stats["value_classes"]["numeric"]["n_values"], 1)
        self.assertEqual(stats["value_classes"]["text"]["n_values"], 4)
        self.assertEqual(stats["key_inventory_rows"], 5)
        self.assertIn(
            stats["top_keys"][0]["key_norm"],
            {"thương hiệu", "chất liệu"},
        )

        inventory = result.key_inventory
        self.assertEqual(len(inventory), 5)
        self.assertTrue((inventory["key_map_status"] == "pending_phase2").all())
        brand_row = inventory[inventory["key_norm"] == "thương hiệu"].iloc[0]
        self.assertEqual(brand_row["key_class"], "admin")
        self.assertEqual(int(brand_row["n_pairs"]), 2)
        self.assertEqual(int(brand_row["n_products"]), 2)

        self.assertEqual(len(result.parse_errors), 1)
        self.assertEqual(result.parse_errors.iloc[0]["product_id"], "4")
        self.assertEqual(result.parse_errors.iloc[0]["reason"], "missing_colon")


class UnitsTest(unittest.TestCase):
    def test_build_units_no_prose(self) -> None:
        result = build_tiki_units(sample_frame())
        products = result.products
        units = result.units
        stats = result.stats

        self.assertEqual(len(products), 4)
        self.assertEqual(len(units), 12)  # 4 title + 8 attribute values
        self.assertEqual(stats["title_units"], 4)
        self.assertEqual(stats["attribute_units"], 8)
        self.assertEqual(stats["train_eligible_units"], 7)  # 3 title + 4 text values
        self.assertEqual(stats["value_classes"], {"admin": 3, "text": 4, "numeric": 1})
        self.assertEqual(stats["bt_methods"], {"dictionary": 3, "mt": 4, "rule": 1})
        self.assertEqual(stats["pending_key_map"], 8)
        self.assertEqual(stats["empty_descriptions"], 1)
        self.assertEqual(stats["malformed_description_parts"], 1)

        self.assertFalse([column for column in products.columns if "prose" in column])
        self.assertFalse([column for column in units.columns if "prose" in column])

        product = products[products["product_id"] == "1"].iloc[0]
        self.assertEqual(
            product["description_vi"],
            (
                "Thương hiệu: LocknLock; Xuất xứ: Việt Nam; "
                "Chất liệu: Nhựa; Kích thước: 20 x 30 cm; "
                "Màu sắc: Đen, Trắng"
            ),
        )
        self.assertIsNone(product["title_zh"])
        self.assertIsNone(product["description_zh"])
        self.assertEqual(list(product["attributes_zh"]), [])
        self.assertEqual(product["n_attributes_vi"], 5)
        self.assertEqual(product["n_values_vi"], 6)
        self.assertEqual(product["bt_status"], "pending")

        title_unit = units[units["pair_id"] == "tiki:1:title"].iloc[0]
        self.assertEqual(title_unit["unit_type"], "title")
        self.assertEqual(title_unit["text_vi"], "Hộp cơm LocknLock 750ml")
        self.assertTrue(bool(title_unit["train_eligible"]))
        self.assertIsNone(title_unit["text_zh"])
        self.assertEqual(title_unit["key_map_status"], "not_applicable")

        material = units[
            (units["product_id"] == "1") & (units["key_norm"] == "chất liệu")
        ].iloc[0]
        self.assertEqual(material["value_class"], "text")
        self.assertEqual(material["bt_method"], "mt")
        self.assertTrue(bool(material["train_eligible"]))
        self.assertIn("key_unmapped", material["qc_flags"])
        self.assertEqual(material["text_vi"], "Chất liệu: Nhựa")

        size = units[
            (units["product_id"] == "1") & (units["key_norm"] == "kích thước")
        ].iloc[0]
        self.assertEqual(size["value_class"], "numeric")
        self.assertEqual(size["bt_method"], "rule")
        self.assertFalse(bool(size["train_eligible"]))

        short_title = units[units["pair_id"] == "tiki:2:title"].iloc[0]
        self.assertFalse(bool(short_title["train_eligible"]))
        self.assertIn("title_short", short_title["qc_flags"])

        malformed_product = products[products["product_id"] == "4"].iloc[0]
        self.assertIn("description_malformed", malformed_product["qc_flags"])
        self.assertEqual(malformed_product["description_parse_source"], "description_vi")

    def test_key_map_is_optional_and_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "key_map.csv"
            pd.DataFrame({
                "key_vi": ["Thương hiệu", "Chất liệu", "Màu sắc"],
                "key_zh": ["品牌", "材质", "颜色"],
            }).to_csv(path, index=False, encoding="utf-8-sig")

            result = build_tiki_units(sample_frame(), key_map_path=path)
            stats = result.stats
            units = result.units

            self.assertEqual(stats["key_map_entries"], 3)
            self.assertEqual(stats["mapped_keys"], 6)
            self.assertEqual(stats["pending_key_map"], 2)

            material = units[
                (units["product_id"] == "1") & (units["key_norm"] == "chất liệu")
            ].iloc[0]
            self.assertEqual(material["key_zh"], "材质")
            self.assertEqual(material["key_map_status"], "mapped")
            self.assertNotIn("key_unmapped", material["qc_flags"])

    def test_duplicate_product_id_raises(self) -> None:
        frame = sample_frame()
        frame.loc[1, "product_id"] = "1"
        with self.assertRaises(ValueError):
            build_tiki_units(frame)


if __name__ == "__main__":
    unittest.main()
