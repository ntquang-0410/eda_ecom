# -*- coding: utf-8 -*-
"""Test length prior 1688 human → Tiki VI (không cần model)."""

from __future__ import annotations

import unittest

import pandas as pd

from cado_core.length_prior import (
    LengthPriorConfig,
    apply_length_prior,
    extract_human_length_pairs,
    fit_length_prior,
    length_bucket,
    length_qc,
    measure_text,
)


def human_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "category": ["home", "beauty", "home"],
        "title_vi": ["Hộp nhựa 1L", "Kem dưỡng da", "Bình nước"],
        "title_zh": ["塑料盒1L", "护肤霜", "水瓶"],
        "description_vi": [
            "Chất liệu: Nhựa; Màu sắc: Đen",
            "Thương hiệu: ABC; Dung tích: 50ml",
            "Chất liệu: Kim loại; Màu sắc: Bạc",
        ],
        "description_zh": [
            "材质: 塑料; 颜色: 黑",
            "品牌: ABC; 容量: 50ml",
            "材质: 金属; 颜色: 银",
        ],
    })


class TextMeasureTest(unittest.TestCase):
    def test_measure_text_and_bucket(self) -> None:
        chars, words = measure_text("  Hộp   nhựa 1L ")
        self.assertEqual(chars, 9)
        self.assertEqual(words, 3)

        self.assertEqual(measure_text(""), (0, 0))
        self.assertEqual(length_bucket(0), "0")
        self.assertEqual(length_bucket(5), "0-10")
        self.assertEqual(length_bucket(15), "11-20")
        self.assertEqual(length_bucket(75), "71-80")
        self.assertEqual(length_bucket(500), "301+")


class ExtractTest(unittest.TestCase):
    def test_extract_title_description_and_aligned_attributes(self) -> None:
        pairs = extract_human_length_pairs(human_frame())

        self.assertEqual(len(pairs), 12)  # 3 title + 3 description + 6 attribute
        self.assertEqual(
            pairs["unit_type"].value_counts().to_dict(),
            {"attribute": 6, "title": 3, "description": 3},
        )
        self.assertTrue((pairs["ratio_vi_zh"] > 0).all())
        self.assertTrue((pairs["zh_chars"] > 0).all())
        self.assertTrue((pairs["vi_char_bucket"] != "0").all())

        material = pairs[
            (pairs["unit_type"] == "attribute")
            & (pairs["text_vi"] == "Chất liệu: Nhựa")
        ].iloc[0]
        self.assertEqual(material["text_zh"], "材质: 塑料")
        self.assertEqual(material["category"], "home")

    def test_mismatched_description_parts_do_not_create_attributes(self) -> None:
        frame = human_frame()
        frame.loc[0, "description_zh"] = "材质: 塑料"
        pairs = extract_human_length_pairs(frame)
        attributes = pairs[pairs["unit_type"] == "attribute"]
        self.assertEqual(len(attributes), 4)  # row 0 bị bỏ, còn 2 hàng × 2


class FitApplyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = extract_human_length_pairs(human_frame())
        self.config = LengthPriorConfig(
            min_group_samples=1,
            min_category_samples=1,
        )
        self.prior = fit_length_prior(self.pairs, config=self.config)

    def test_fit_scopes_and_stats(self) -> None:
        self.assertEqual(self.prior.stats["n_pairs"], 12)
        self.assertEqual(self.prior.stats["pair_counts"]["title"], 3)
        self.assertEqual(self.prior.stats["pair_counts"]["attribute"], 6)
        self.assertIn("overall", self.prior.stats)
        self.assertIn("title", self.prior.stats["by_unit_type"])
        self.assertGreater(self.prior.stats["overall"]["ratio_vi_zh_p50"], 0)

        scopes = set(self.prior.table["scope"])
        self.assertEqual(
            scopes,
            {
                "global",
                "unit_type",
                "unit_type+bucket",
                "unit_type+category",
                "unit_type+category+bucket",
            },
        )
        for column in (
            "ratio_vi_zh_p05",
            "ratio_vi_zh_p50",
            "ratio_vi_zh_p95",
            "zh_per_vi_word_p50",
        ):
            self.assertIn(column, self.prior.table.columns)

    def test_apply_adds_expected_range_and_fallback(self) -> None:
        units = pd.DataFrame({
            "product_id": ["a", "b", "c", "d"],
            "text_vi": ["Hộp nhựa 1L", "Chất liệu: Nhựa", "unknown text", ""],
            "unit_type": ["title", "attribute", "other", "title"],
            "category": ["home", "home", "mystery", "home"],
        })
        result = apply_length_prior(units, self.prior, config=self.config)
        applied = result.units

        self.assertEqual(len(applied), 4)
        self.assertEqual(result.stats["n_units"], 4)
        self.assertTrue((applied["length_prior_source"] == "1688_human").all())
        self.assertTrue((applied["expected_zh_chars"] >= 0).all())
        self.assertTrue((applied["vi_chars"] >= 0).all())
        self.assertTrue(
            (applied["zh_chars_p05"] <= applied["expected_zh_chars"]).all()
        )
        self.assertTrue(
            (applied["expected_zh_chars"] <= applied["zh_chars_p95"]).all()
        )

        title = applied.iloc[0]
        self.assertGreater(title["expected_zh_chars"], 0)
        self.assertEqual(title["length_prior_scope"], "unit_type+category+bucket")

        unknown_type = applied.iloc[2]
        self.assertEqual(unknown_type["length_prior_scope"], "global")
        self.assertGreater(unknown_type["length_prior_n"], 0)

        empty = applied.iloc[3]
        self.assertEqual(empty["expected_zh_chars"], 0)
        self.assertEqual(empty["zh_chars_p05"], 0)
        self.assertEqual(empty["zh_chars_p95"], 0)

    def test_small_bucket_is_preferred_over_unconditioned_category(self) -> None:
        config = LengthPriorConfig(
            min_group_samples=20,
            min_category_samples=100,
        )
        prior = fit_length_prior(self.pairs, config=config)
        units = pd.DataFrame({
            "text_vi": ["Hộp nhựa 1L"],
            "unit_type": ["title"],
            "category": ["home"],
        })
        applied = apply_length_prior(units, prior, config=config).units.iloc[0]

        self.assertEqual(applied["length_prior_scope"], "unit_type+bucket")
        self.assertLess(int(applied["length_prior_n"]), 20)

    def test_length_qc_flags(self) -> None:
        units = pd.DataFrame({
            "text_vi": ["Hộp nhựa 1L"],
            "unit_type": ["title"],
            "category": ["home"],
        })
        row = apply_length_prior(
            units, self.prior, config=self.config
        ).units.iloc[0].to_dict()

        too_short = length_qc("短", row)
        self.assertIn("zh_too_short", too_short["length_qc_flags"])
        self.assertEqual(too_short["length_fit"], "short")

        too_long = length_qc("这是一个非常长的中文描述", row)
        self.assertIn("zh_too_long", too_long["length_qc_flags"])
        self.assertEqual(too_long["length_fit"], "long")

        ok = length_qc("塑料盒1L", row)
        self.assertEqual(ok["length_qc_flags"], [])
        self.assertEqual(ok["length_fit"], "ok")

    def test_empty_units_keep_prior_columns(self) -> None:
        units = pd.DataFrame({"text_vi": [], "unit_type": [], "category": []})
        result = apply_length_prior(units, self.prior, config=self.config)
        self.assertEqual(len(result.units), 0)
        self.assertIn("expected_zh_chars", result.units.columns)
        self.assertEqual(result.stats["n_units"], 0)


if __name__ == "__main__":
    unittest.main()
