# -*- coding: utf-8 -*-
"""Test length prior v2: value-only, value_class, CJK, split-safe."""

from __future__ import annotations

import unittest

import pandas as pd

from cado_core.length_prior_v2 import (
    LengthPriorV2Config,
    apply_length_prior_v2,
    build_key_alignment_map,
    evaluate_length_prior_v2,
    extract_human_value_pairs,
    fit_length_prior_v2,
    measure_zh,
    split_for_product,
    split_value_list,
)


def human_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "category": ["home", "beauty", "home"],
        "title_vi": ["Hộp nhựa 1L", "Kem dưỡng da", "Bình nước"],
        "title_zh": ["塑料盒1L", "护肤霜", "水瓶"],
        "description_vi": [
            "Chất liệu: Nhựa; Màu sắc: Đen, Trắng",
            "Thương hiệu: ABC; Dung tích: 50ml",
            "Chất liệu: Kim loại; Màu sắc: Đen,Trắng,Xanh",
        ],
        "description_zh": [
            "材质: 塑料; 颜色: 黑, 白",
            "品牌: ABC; 容量: 50ml",
            "材质: 金属; 颜色: 黑,白,蓝",
        ],
    })


class SplitAndTextTest(unittest.TestCase):
    def test_split_value_list_keeps_decimal_and_thousand_commas(self) -> None:
        self.assertEqual(
            split_value_list("Đen, Trắng"),
            ["Đen", "Trắng"],
        )
        self.assertEqual(split_value_list("1,5 kg"), ["1,5 kg"])
        self.assertEqual(split_value_list("1,000"), ["1,000"])
        self.assertEqual(
            split_value_list("A、B；C"),
            ["A", "B", "C"],
        )

    def test_split_for_product_is_stable(self) -> None:
        first = split_for_product("123", seed=42)
        second = split_for_product("123", seed=42)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "dev", "test"})
        # Khác seed có thể khác split; chỉ cần hợp lệ.
        self.assertIn(split_for_product("123", seed=7), {"train", "dev", "test"})

    def test_measure_zh(self) -> None:
        measured = measure_zh("塑料盒1L")
        self.assertEqual(measured["zh_chars"], 5)
        self.assertEqual(measured["zh_cjk_chars"], 3)
        self.assertEqual(measured["zh_non_cjk_chars"], 2)
        self.assertEqual(measured["zh_digits"], 1)
        self.assertEqual(measured["zh_latin_tokens"], 1)


class ExtractV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LengthPriorV2Config(
            min_group_samples=1,
            min_category_samples=1,
            min_value_class_samples=1,
            min_value_class_category_samples=1,
        )
        self.result = extract_human_value_pairs(
            human_frame(),
            config=self.config,
        )

    def test_extract_value_grain_and_audit(self) -> None:
        pairs = self.result.pairs
        stats = self.result.stats
        audit = self.result.split_audit

        self.assertEqual(stats["n_title_pairs"], 3)
        self.assertEqual(stats["n_value_pairs"], 9)
        self.assertEqual(stats["n_group_pairs"], 2)
        self.assertEqual(stats["n_attribute_groups"], 6)
        self.assertEqual(
            stats["split_status_counts"],
            {"single": 4, "multi_matched": 2},
        )

        material = pairs[
            (pairs["grain"] == "attribute_value")
            & (pairs["value_vi"] == "Nhựa")
        ].iloc[0]
        self.assertEqual(material["text_vi"], "Nhựa")
        self.assertNotIn("Chất liệu", material["text_vi"])
        self.assertEqual(material["key_zh"], "材质")
        self.assertGreater(material["key_zh_chars"], 0)
        self.assertEqual(material["alignment_method"], "description_order")

        color = pairs[
            (pairs["grain"] == "attribute_value")
            & (pairs["value_vi"] == "Trắng")
            & (pairs["product_id"] == "p1")
        ]
        self.assertEqual(len(color), 1)
        self.assertEqual(color.iloc[0]["n_tokens_vi"], 2)

        self.assertEqual(len(audit), 6)
        self.assertTrue((audit["n_values_added"] >= 0).all())

    def test_key_alignment_map_and_fallback(self) -> None:
        base = extract_human_value_pairs(human_frame(), config=self.config)
        key_map, pair_stats, map_stats = build_key_alignment_map(
            base.split_audit,
            min_freq=1,
            min_share=0.6,
        )
        self.assertIn("材质", key_map)
        self.assertEqual(key_map["材质"], "chất liệu")
        self.assertGreater(map_stats["n_mapped_keys"], 0)

        frame = pd.DataFrame({
            "product_id": ["p1"],
            "category": ["home"],
            "title_vi": ["Hộp nhựa 1L"],
            "title_zh": ["塑料盒1L"],
            "description_vi": "Chất liệu: Nhựa; Màu sắc: Đen",
            "description_zh": "材质: 塑料; 颜色: 黑",
            # Raw VI attributes cố ý đảo thứ tự để test key fallback.
            "attributes_vi": [[
                {"name": "Màu sắc", "values": ["Đen"]},
                {"name": "Chất liệu", "values": ["Nhựa"]},
            ]],
        })
        aligned = extract_human_value_pairs(
            frame,
            config=self.config,
            key_map={"材质": "chất liệu", "颜色": "màu sắc"},
            key_pair_stats=pair_stats,
            min_key_pair_freq=20,
            align_key_fallback=True,
        )
        methods = set(aligned.split_audit["alignment_method"])
        self.assertEqual(methods, {"key_fallback"})
        material = aligned.pairs[
            (aligned.pairs["grain"] == "attribute_value")
            & (aligned.pairs["key_zh"] == "材质")
        ].iloc[0]
        self.assertEqual(material["value_vi"], "Nhựa")
        self.assertEqual(material["alignment_method"], "key_fallback")

    def test_mismatch_stays_group_and_is_audited(self) -> None:
        frame = human_frame()
        frame.loc[0, "description_zh"] = "材质: 塑料; 颜色: 黑"
        result = extract_human_value_pairs(frame, config=self.config)

        mismatch = result.split_audit[
            (result.split_audit["product_id"] == "p1")
            & (result.split_audit["key_vi"] == "Màu sắc")
        ]
        self.assertEqual(len(mismatch), 1)
        self.assertEqual(mismatch.iloc[0]["split_status"], "mismatch")
        self.assertEqual(int(mismatch.iloc[0]["n_values_added"]), 0)

        color_values = result.pairs[
            (result.pairs["product_id"] == "p1")
            & (result.pairs["grain"] == "attribute_value")
            & (result.pairs["key_vi"] == "Màu sắc")
        ]
        self.assertTrue(color_values.empty)
        color_group = result.pairs[
            (result.pairs["product_id"] == "p1")
            & (result.pairs["grain"] == "attribute_group")
            & (result.pairs["key_vi"] == "Màu sắc")
        ]
        self.assertEqual(len(color_group), 1)


class FitApplyV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LengthPriorV2Config(
            min_group_samples=1,
            min_category_samples=1,
            min_value_class_samples=1,
            min_value_class_category_samples=1,
            fit_split="all",
        )
        extracted = extract_human_value_pairs(human_frame(), config=self.config)
        self.pairs = extracted.pairs
        self.prior = fit_length_prior_v2(self.pairs, config=self.config)

    def test_fit_is_value_only_and_has_value_class(self) -> None:
        stats = self.prior.stats
        self.assertEqual(stats["target"], "value_only")
        self.assertEqual(stats["n_pairs"], 12)  # 3 title + 9 value
        self.assertEqual(stats["grain_counts"]["title"], 3)
        self.assertEqual(stats["grain_counts"]["attribute_value"], 9)
        self.assertEqual(stats["value_class_counts"]["text"], 7)
        self.assertEqual(stats["value_class_counts"]["admin"], 1)
        self.assertEqual(stats["value_class_counts"]["numeric"], 1)
        self.assertIn("text", stats["by_value_class"])
        self.assertIn("key_chars", stats)

        scopes = set(self.prior.table["scope"])
        self.assertIn("global", scopes)
        self.assertIn("value_class", scopes)
        self.assertIn("value_class+bucket", scopes)
        self.assertIn("unit_type+bucket", scopes)

    def test_fit_split_excludes_non_train_pairs(self) -> None:
        pairs = self.pairs.copy()
        primary_mask = pairs["grain"].isin(("title", "attribute_value"))
        primary_indices = pairs.index[primary_mask].tolist()
        pairs.loc[primary_indices[0], "split"] = "test"
        pairs.loc[primary_indices[1:], "split"] = "train"
        prior = fit_length_prior_v2(
            pairs,
            config=LengthPriorV2Config(
                fit_split="train",
                min_group_samples=1,
                min_category_samples=1,
                min_value_class_samples=1,
                min_value_class_category_samples=1,
            ),
        )
        self.assertEqual(prior.stats["fit_split"], "train")
        self.assertEqual(prior.stats["n_pairs"], len(primary_indices) - 1)
        self.assertEqual(
            prior.stats["split_counts"],
            {"train": len(primary_indices) - 1},
        )

    def test_apply_value_only_key_correction_and_cjk(self) -> None:
        units = pd.DataFrame({
            "text_vi": [
                "Hộp nhựa 1L",
                "Nhựa",
                "Đen",
                "ABC",
                "50ml",
                "unknown",
            ],
            "value_vi": ["", "Nhựa", "Đen", "ABC", "50ml", ""],
            "unit_type": [
                "title",
                "attribute",
                "attribute",
                "attribute",
                "attribute",
                "other",
            ],
            "value_class": ["", "text", "text", "admin", "numeric", ""],
            "category": ["home", "home", "home", "beauty", "beauty", "unknown"],
            "key_zh": ["", "材质", "", "", "", ""],
            "key_vi": ["", "Chất liệu", "Màu sắc", "Thương hiệu", "Dung tích", ""],
            "key_class": ["", "text", "text", "admin", "numeric", ""],
        })
        applied = apply_length_prior_v2(
            units,
            self.prior,
            config=self.config,
        ).units

        self.assertIn("expected_zh_value_chars", applied.columns)
        self.assertTrue((applied["expected_zh_value_chars"] > 0).all())
        self.assertTrue(
            (
                applied["zh_value_chars_p05"]
                <= applied["expected_zh_value_chars"]
            ).all()
        )
        self.assertTrue(
            (
                applied["expected_zh_value_chars"]
                <= applied["zh_value_chars_p95"]
            ).all()
        )
        self.assertTrue(
            (
                applied["expected_zh_value_chars"]
                + applied["expected_zh_key_chars"]
                <= applied["expected_zh_text_chars"]
            ).all()
        )

        title = applied.iloc[0]
        self.assertEqual(title["length_prior_grain"], "title")
        self.assertTrue(bool(title["length_prior_soft_qc"]))
        self.assertEqual(int(title["expected_zh_key_chars"]), 0)
        self.assertEqual(title["key_len_source"], "title")

        material = applied.iloc[1]
        self.assertEqual(material["length_prior_grain"], "attribute_value")
        self.assertEqual(material["key_len_source"], "actual")
        self.assertEqual(int(material["expected_zh_key_chars"]), 2)
        self.assertFalse(bool(material["length_prior_soft_qc"]))

        admin = applied.iloc[3]
        self.assertTrue(bool(admin["length_prior_soft_qc"]))
        numeric = applied.iloc[4]
        self.assertTrue(bool(numeric["length_prior_soft_qc"]))

        color = applied.iloc[2]
        self.assertEqual(color["key_len_source"], "estimated_key_class")
        self.assertGreater(int(color["expected_zh_key_chars"]), 0)
        self.assertGreaterEqual(int(color["expected_zh_cjk_chars"]), 0)

        unknown = applied.iloc[5]
        self.assertEqual(unknown["length_prior_scope"], "global")

    def test_evaluate_returns_value_metrics(self) -> None:
        pairs = self.pairs.copy()
        pairs["split"] = "test"
        metrics = evaluate_length_prior_v2(
            pairs,
            self.prior,
            split="test",
            config=self.config,
        )
        self.assertEqual(metrics["n"], 12)
        self.assertIn("overall", metrics)
        self.assertIn("by_grain", metrics)
        self.assertIn("by_value_class", metrics)
        self.assertGreaterEqual(
            metrics["overall"]["coverage_value_p05_p95"], 0.0
        )
        self.assertLessEqual(
            metrics["overall"]["coverage_value_p05_p95"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
