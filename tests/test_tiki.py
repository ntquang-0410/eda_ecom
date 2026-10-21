# -*- coding: utf-8 -*-
"""Test cho EDA Tiki (không cần model)."""

from __future__ import annotations

import unittest

import pandas as pd

from cado_core.text import extract_latin_words
from cado_core.tiki import (
    clean_text,
    extract_protected_spans,
    parse_attributes,
    run_tiki_eda,
    split_sentences,
)

BOILERPLATE = (
    " Giá sản phẩm trên Tiki đã bao gồm thuế theo luật hiện hành. "
    "Bên cạnh đó, tuỳ vào loại sản phẩm, hình thức và địa chỉ giao hàng "
    "mà có thể phát sinh thêm chi phí khác như phí vận chuyển."
)


class TextTest(unittest.TestCase):
    def test_latin_tokenizer_keeps_vietnamese_extended_chars(self) -> None:
        self.assertEqual(extract_latin_words("những"), ["những"])
        self.assertEqual(extract_latin_words("được"), ["được"])
        self.assertEqual(extract_latin_words("10×20cm"), ["cm"])

    def test_clean_removes_boilerplate_and_hashtag(self) -> None:
        text = f"Áo thun nam cao cấp #sale {BOILERPLATE}"
        cleaned, info = clean_text(text)
        self.assertNotIn("Giá sản phẩm", cleaned)
        self.assertNotIn("#sale", cleaned)
        self.assertGreater(info["boilerplate_chars"], 0)
        self.assertEqual(info["n_hashtags"], 1)

    def test_split_sentences_keeps_decimal_and_abbrev(self) -> None:
        sentences = split_sentences(
            "TP.HCM có giá 1.5 triệu. Sản phẩm tốt! Bạn nên mua?"
        )
        self.assertEqual(len(sentences), 3)
        self.assertIn("TP.HCM", sentences[0])
        self.assertIn("1.5", sentences[0])

    def test_parse_attributes_from_description(self) -> None:
        attributes = parse_attributes(
            None, "Thương hiệu: LocknLock; Xuất xứ: Việt Nam"
        )
        self.assertEqual(len(attributes), 2)
        self.assertEqual(attributes[0]["values"], ["LocknLock"])

    def test_protected_spans_keep_numbers_models(self) -> None:
        spans = extract_protected_spans(
            "Hộp LocknLock LBC501 dung tích 750ml, giá 199.000đ",
            brand_tokens={"locknlock"},
            whitelist=set(),
        )
        joined = " | ".join(spans)
        self.assertIn("750ml", joined)
        self.assertIn("LBC501", joined)
        self.assertIn("LocknLock", joined)


class PipelineTest(unittest.TestCase):
    def test_run_tiki_eda_two_rows(self) -> None:
        df = pd.DataFrame({
            "product_id": ["1", "2"],
            "title_vi": [
                "Hộp cơm LocknLock 750ml",
                "SP",
            ],
            "description_vi": [
                "Thương hiệu: LocknLock; Xuất xứ: Việt Nam",
                "Thương hiệu: OEM",
            ],
            "description_prose_vi": [
                "Hộp cơm LocknLock 750ml bằng nhựa an toàn. "
                "Sản phẩm phù hợp cho bữa ăn văn phòng." + BOILERPLATE,
                "",
            ],
            "brand": ["LocknLock", "OEM"],
            "price": [199000, 0],
            "sales": [10, None],
        })
        result = run_tiki_eda(df)
        out = result.df
        self.assertEqual(len(out), 2)
        self.assertEqual(out["product_id"].tolist(), ["1", "2"])
        self.assertNotIn("Giá sản phẩm", out.loc[0, "bt_prose"])
        self.assertGreaterEqual(out.loc[0, "n_sentences"], 1)
        self.assertEqual(out.loc[0, "overall_route"], "BT_TITLE+PROSE")
        self.assertEqual(out.loc[1, "overall_route"], "BT_ATTR")
        self.assertTrue(bool(out.loc[0, "bt_ready"]))
        self.assertFalse(bool(out.loc[1, "bt_ready"]))
        self.assertIn("boilerplate_removed", out.loc[0, "eda_flags"])
        self.assertIn("price_outlier", out.loc[1, "eda_flags"])


if __name__ == "__main__":
    unittest.main()
