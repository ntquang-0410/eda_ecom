# -*- coding: utf-8 -*-
"""Test Qwen key-map review logic (không gọi API)."""

from __future__ import annotations

import unittest

import pandas as pd

from cado_core.key_map_review import (
    _extract_json,
    aggregate_reviews,
    apply_reviews,
    build_review_input,
    evaluate_silver,
    validate_key_zh,
)


def units_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "key_norm": ["chất liệu", "chất liệu", "màu sắc", "màu sắc", "kích cỡ"],
        "key_vi": ["Chất liệu", "Chất liệu", "Màu sắc", "Màu sắc", "Kích cỡ"],
        "key_class": ["text", "text", "text", "text", "numeric"],
        "value_vi": ["Nhựa", "Kim loại", "Đen", "Trắng", "20 cm"],
        "category": ["home", "home", "home", "home", "home"],
    })


def key_map_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "key_norm": ["chất liệu", "màu sắc", "kích cỡ"],
        "key_vi": ["Chất liệu", "Màu sắc", "Kích cỡ"],
        "key_class": ["text", "text", "numeric"],
        "n_units": [2, 2, 1],
        "n_products": [2, 2, 1],
        "n_train_eligible": [2, 2, 0],
        "key_zh": ["材质", "颜色", ""],
        "key_map_status": ["auto_1688", "auto_1688", "pending_review"],
        "source": ["alignment_1688", "alignment_1688", "pending"],
        "auto_key_zh": ["材质", "颜色", ""],
        "auto_freq": [10, 10, 0],
        "auto_share": [0.99, 0.99, 0.0],
    })


class ValidationTest(unittest.TestCase):
    def test_validate_and_json(self) -> None:
        self.assertEqual(validate_key_zh("材质"), ("材质", ""))
        self.assertEqual(validate_key_zh("材 质"), ("材质", ""))
        self.assertEqual(validate_key_zh("material")[1], "non_cjk")
        self.assertEqual(validate_key_zh("材质!")[1], "invalid_chars")
        self.assertEqual(validate_key_zh("")[1], "empty")
        parsed = _extract_json('```json\n{"key_zh": "材质"}\n```')
        self.assertEqual(parsed["key_zh"], "材质")


class ReviewAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.review_input = build_review_input(
            units_frame(),
            key_map_frame().iloc[[2]].reset_index(drop=True),
            alignment_map_path="/tmp/opencode/nonexistent_alignment.csv",
        )

    def test_build_input_samples(self) -> None:
        self.assertEqual(len(self.review_input), 1)
        row = self.review_input.iloc[0]
        self.assertIn("20 cm", row["sample_values_json"])

    def test_aggregate_and_apply(self) -> None:
        votes = pd.DataFrame({
            "key_norm": ["kích cỡ"] * 3,
            "vote": [0, 1, 2],
            "model": ["qwen3-max"] * 3,
            "key_zh": ["尺寸", "尺寸", "尺寸"],
            "alternatives": ["[]"] * 3,
            "confidence": [0.9, 0.8, 0.85],
            "validation_status": ["valid"] * 3,
            "raw_output": ["{}"] * 3,
            "error": [""] * 3,
        })
        review = aggregate_reviews(votes, self.review_input)
        self.assertEqual(review.iloc[0]["qwen_status"], "qwen_high")
        self.assertEqual(review.iloc[0]["qwen_key_zh"], "尺寸")

        key_map = key_map_frame()
        applied = apply_reviews(key_map, review, statuses=("qwen_high",))
        row = applied[applied["key_norm"] == "kích cỡ"].iloc[0]
        self.assertEqual(row["key_zh"], "尺寸")
        self.assertEqual(row["key_map_status"], "qwen_high")

    def test_silver_eval(self) -> None:
        silver = self.review_input.copy()
        silver["gold_key_zh"] = ["尺寸"]
        votes = pd.DataFrame({
            "key_norm": ["kích cỡ"] * 3,
            "vote": [0, 1, 2],
            "model": ["qwen3-max"] * 3,
            "key_zh": ["尺寸", "尺寸", "尺寸"],
            "alternatives": ["[]"] * 3,
            "confidence": [0.9, 0.9, 0.9],
            "validation_status": ["valid"] * 3,
            "raw_output": ["{}"] * 3,
            "error": [""] * 3,
        })
        metrics = evaluate_silver(silver, votes)
        self.assertEqual(metrics["exact"], 1.0)
        self.assertEqual(metrics["n"], 1)


if __name__ == "__main__":
    unittest.main()
