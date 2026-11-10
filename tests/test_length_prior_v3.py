# -*- coding: utf-8 -*-
"""Test Phase C: hierarchical title smoothing trên prior v3."""

from __future__ import annotations

import unittest

import pandas as pd

from cado_core.length_prior_v3 import (
    LengthPriorV3Config,
    apply_length_prior_v3,
    evaluate_length_prior_v3,
    fit_length_prior_v3,
)
from cado_core.length_prior_v2 import (
    LengthPriorV2Config,
    extract_human_value_pairs,
)


def human_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3", "p4"],
        "category": ["home", "home", "beauty", "home"],
        "title_vi": [
            "Hộp nhựa 1L",
            "Bình nước",
            "Kem dưỡng da",
            "Hộp thủy tinh 500ml",
        ],
        "title_zh": ["塑料盒1L", "水瓶", "护肤霜", "玻璃盒500ml"],
        "description_vi": [
            "Chất liệu: Nhựa; Màu sắc: Đen, Trắng",
            "Chất liệu: Kim loại; Màu sắc: Bạc",
            "Thương hiệu: ABC; Dung tích: 50ml",
            "Chất liệu: Thủy tinh; Màu sắc: Trong suốt",
        ],
        "description_zh": [
            "材质: 塑料; 颜色: 黑, 白",
            "材质: 金属; 颜色: 银",
            "品牌: ABC; 容量: 50ml",
            "材质: 玻璃; 颜色: 透明",
        ],
    })


class TitleSmoothingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v2_config = LengthPriorV2Config(
            min_group_samples=1,
            min_category_samples=1,
            min_value_class_samples=1,
            min_value_class_category_samples=1,
            fit_split="all",
        )
        self.v3_config = LengthPriorV3Config(
            v2=self.v2_config,
            title_smoothing_k=20,
            title_interval_inflation=0.5,
        )
        extracted = extract_human_value_pairs(
            human_frame(),
            config=self.v2_config,
        )
        self.pairs = extracted.pairs
        self.prior = fit_length_prior_v3(
            self.pairs,
            config=self.v3_config,
        )

    def test_fit_adds_smoothed_title_columns(self) -> None:
        table = self.prior.table
        self.assertEqual(self.prior.stats["version"], 3)
        self.assertIn("expected_zh_chars_smooth", table.columns)
        self.assertIn("zh_chars_p05_smooth", table.columns)
        self.assertIn("zh_chars_p95_smooth", table.columns)
        self.assertIn("title_smoothing", self.prior.stats)

        title_rows = table[
            (table["grain"] == "title")
            & (table["scope"] == "unit_type+bucket")
        ]
        self.assertFalse(title_rows.empty)
        self.assertTrue(
            title_rows["expected_zh_chars_smooth"].notna().all()
        )
        self.assertTrue(
            (title_rows["zh_chars_p05_smooth"] <= title_rows["expected_zh_chars_smooth"]).all()
        )
        self.assertTrue(
            (title_rows["expected_zh_chars_smooth"] <= title_rows["zh_chars_p95_smooth"]).all()
        )

    def test_apply_uses_hierarchical_title_and_keeps_attribute_prior(self) -> None:
        units = pd.DataFrame({
            "text_vi": ["Hộp nhựa 1L", "Nhựa", "Đen, Trắng"],
            "value_vi": ["", "Nhựa", "Đen, Trắng"],
            "unit_type": ["title", "attribute", "attribute"],
            "value_class": ["", "text", "text"],
            "category": ["home", "home", "home"],
            "key_zh": ["", "材质", "Màu sắc"],
            "key_vi": ["", "Chất liệu", "Màu sắc"],
            "key_class": ["", "text", "text"],
        })
        applied = apply_length_prior_v3(units, self.prior).units

        title = applied.iloc[0]
        self.assertEqual(title["length_prior_version"], 3)
        self.assertEqual(title["length_prior_smoothing"], "hierarchical")
        self.assertTrue(bool(title["length_prior_soft_qc"]))
        self.assertIn(
            title["length_prior_scope"],
            {"title_bucket_hierarchical", "title_hierarchical"},
        )
        self.assertLessEqual(
            title["zh_value_chars_p05"],
            title["expected_zh_value_chars"],
        )
        self.assertLessEqual(
            title["expected_zh_value_chars"],
            title["zh_value_chars_p95"],
        )

        material = applied.iloc[1]
        self.assertEqual(material["length_prior_grain"], "attribute_value")
        self.assertEqual(material["length_prior_smoothing"], "none")
        self.assertEqual(material["key_len_source"], "actual")

    def test_evaluate_v3_returns_metrics(self) -> None:
        pairs = self.pairs.copy()
        pairs["split"] = "test"
        metrics = evaluate_length_prior_v3(
            pairs,
            self.prior,
            split="test",
        )
        self.assertEqual(metrics["version"], 3)
        self.assertGreater(metrics["n"], 0)
        self.assertIn("overall", metrics)
        self.assertIn("by_grain", metrics)


if __name__ == "__main__":
    unittest.main()
