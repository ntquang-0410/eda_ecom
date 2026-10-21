"""Smoke test cho các bug đã sửa trong cado_core.

Chạy::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from cado_core.cado import CadoConfig, run_cado
from cado_core.dictionary import Cedict, Whitelist, lemmatize
from cado_core.rules import classify_occurrence
from cado_core.text import remove_words, strip_cjk


class TextBoundaryTest(unittest.TestCase):
    def test_cjk_is_a_valid_boundary(self) -> None:
        self.assertEqual(remove_words("双肩包bags男", {"bags"}), "双肩包男")

    def test_latin_compound_is_not_split(self) -> None:
        self.assertEqual(remove_words("handbags", {"bags"}), "handbags")

    def test_punctuation_and_digits_are_boundaries(self) -> None:
        self.assertEqual(remove_words("bags, 2024", {"bags"}), ", 2024")

    def test_strip_cjk_preserves_newline(self) -> None:
        self.assertEqual(strip_cjk("a 中\nb"), "a\nb")


class DictionaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cedict = Cedict.load()

    def test_lemmatize(self) -> None:
        self.assertIn("bag", lemmatize("bags"))
        self.assertIn("shoe", lemmatize("shoes"))

    def test_simplified_headwords(self) -> None:
        self.assertEqual(
            self.cedict.overlap("schoolbag", "书包").status, "overlap"
        )
        # Dữ liệu 1688 là giản thể; không kỳ vọng khớp phồn thể.
        self.assertEqual(
            self.cedict.overlap("schoolbag", "書包").status, "no_overlap"
        )

    def test_disable_loanwords_falls_through_to_cedict(self) -> None:
        whitelist = Whitelist.load()
        action, reason = classify_occurrence(
            "bags",
            "外贸包包bags",
            whitelist,
            set(),
            self.cedict,
            max_tier=4,
            use_loanwords=False,
        )
        self.assertEqual((action, reason), ("remove", "cedict:包包"))


class PipelineTest(unittest.TestCase):
    def test_run_cado_adds_cleaned_column(self) -> None:
        df = pd.DataFrame({
            "product_id": ["1"],
            "title_zh": ["外贸包包bags"],
            "title_vi": ["Túi xách nữ"],
            "description_vi": [""],
        })
        result = run_cado(df, CadoConfig(columns=("title_zh",), max_tier=4))
        self.assertEqual(result.df.loc[0, "title_zh_cleaned"], "外贸包包")
        self.assertEqual(result.stats["remove"], 1)


if __name__ == "__main__":
    unittest.main()
