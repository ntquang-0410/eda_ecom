#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quét ký tự có thể gây nhiễu kiến thức truyền cho model.

Nhóm quét:
  control        : C0/C1 (trừ \\n \\t)
  zero_width     : ZWSP/ZWNJ/ZWJ/BOM/soft hyphen...
  bidi           : điều khiển hướng chữ
  variation      : variation selectors
  special_space  : NBSP, thin/hair/ideographic space
  fullwidth_alnum: chữ/số full-width (０-９Ａ-Ｚ)
  fullwidth_punct: dấu câu full-width
  html           : tag hoặc entity
  mojibake       : dấu hiệu lỗi encoding
  pua            : Private Use Area
  emoji          : emoji/pictograph
  template       : {var}, %s, ${...}
  repeated_punct : !!!, 。。。
  long_run       : 7+ ký tự giống nhau
  url            : http/www
  escape_literal : "\\n", "\\uXXXX" còn nguyên
  symbols        : tiền tệ/toán tử
  cjk_ext        : CJK extension (hiếm)
  mixed_script   : token vừa Latin vừa Hán

Output: JSON + Markdown + ví dụ.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "control": r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
    "zero_width": r"[\u200b-\u200f\u2028-\u202f\u2060-\u2064\ufeff\u00ad\u034f\u061c\u180e]",
    "bidi": r"[\u202a-\u202e\u2066-\u2069]",
    "variation": r"[\ufe00-\ufe0f\U000E0100-\U000E01EF]",
    "special_space": r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]",
    "fullwidth_alnum": r"[\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]",
    "fullwidth_punct": r"[\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]",
    "html": r"</?[a-zA-Z][^>\n]{0,80}>|&(amp|lt|gt|quot|apos|nbsp|copy|reg|trade|hellip|mdash|ndash|ldquo|rdquo|lsquo|rsquo|times|divide|deg|middot|bull|laquo|raquo|#\d+|#x[0-9a-fA-F]+);",
    "mojibake": r"(Ã[\u0080-\u00bf]|Ä[\u0080-\u00bf]|áº|á»|â€|Æ[\u0080-\u00bf])",
    "pua": r"[\ue000-\uf8ff]",
    "emoji": r"[\U0001F300-\U0001FAFF\u2600-\u27bf\ufe0f]",
    "template": r"\{[A-Za-z_][^}\n]{0,30}\}|%[sd]|\$\{[^}\n]+\}",
    "repeated_punct": r"([!?。，,;:])\1{2,}",
    "long_run": r"(.)\1{6,}",
    "url": r"https?://|www\.",
    "escape_literal": r"\\n|\\t|\\u[0-9a-fA-F]{4}",
    "symbols": r"[¥$€£₫+\-=×÷±]",
    "cjk_ext": r"[\U00020000-\U0002FA1F]",
}
COMPILED = {name: re.compile(pattern) for name, pattern in CATEGORIES.items()}
MIXED = re.compile(r"[A-Za-z][\u4e00-\u9fff]|[\u4e00-\u9fff][A-Za-z]")
BRACKETS = {"(": ")", "[": "]", "（": "）", "【": "】", "《": "》", "「": "」", "『": "』"}


def scan_series(series: pd.Series, label: str) -> dict:
    text = series.fillna("").astype(str)
    result = {"label": label, "rows": int(len(text)), "categories": {}}
    for name, pattern in COMPILED.items():
        hit = text.str.contains(pattern, regex=True, na=False)
        count = int(text.str.count(pattern).sum())
        if count:
            examples = text[hit].head(3).tolist()
            result["categories"][name] = {
                "rows": int(hit.sum()),
                "chars": count,
                "examples": [e[:90] for e in examples],
            }
    mixed = text.str.contains(MIXED, regex=True, na=False)
    if mixed.any():
        result["categories"]["mixed_script"] = {
            "rows": int(mixed.sum()),
            "examples": [e[:90] for e in text[mixed].head(3).tolist()],
        }
    # ngoặc lệch
    unbalanced = 0
    examples = []
    for value in text:
        counts = {op: value.count(op) for op in BRACKETS}
        for op, cl in BRACKETS.items():
            if counts[op] != value.count(cl):
                unbalanced += 1
                if len(examples) < 3:
                    examples.append(value[:90])
                break
    if unbalanced:
        result["categories"]["unbalanced_bracket"] = {"rows": int(unbalanced), "examples": examples}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cleaned/char_audit.json")
    ap.add_argument("--report", default="reports/CHAR_AUDIT.md")
    args = ap.parse_args()

    report = {}

    b = pd.read_parquet("data/cleaned/1688/bilingual_1688_clean.parquet")
    for column in ("title_vi_clean", "title_zh_clean", "description_vi_clean", "description_zh_clean"):
        report[f"1688.{column}"] = scan_series(b[column], column)

    # attribute values
    a = pd.read_parquet("data/cleaned/1688/attributes_aligned.parquet")
    vi_values, zh_values = [], []
    for row in a.itertuples():
        attrs = row.attributes_aligned
        if attrs is None:
            continue
        if isinstance(attrs, np.ndarray):
            attrs = list(attrs)
        for attr in attrs:
            vi_values.extend(map(str, attr["values_vi"]))
            zh_values.extend(map(str, attr["values_zh"]))
    report["1688.attr_vi"] = scan_series(pd.Series(vi_values), "attr_vi")
    report["1688.attr_zh"] = scan_series(pd.Series(zh_values), "attr_zh")

    for model in ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"):
        t = pd.read_parquet(f"data/cleaned/bt_tiki/{model}/units.parquet")
        report[f"tiki.{model}.source"] = scan_series(t["source_text_bt"], "source")
        report[f"tiki.{model}.target"] = scan_series(t["target_text_bt"], "target")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Audit ký tự nhiễu", ""]
    for key, value in report.items():
        if not value["categories"]:
            continue
        lines.append(f"## {key}  ({value['rows']:,} dòng)")
        lines.append("")
        lines.append("| nhóm | rows | chars | ví dụ |")
        lines.append("|---|---:|---:|---|")
        for name, stats in sorted(value["categories"].items(), key=lambda x: -x[1]["rows"]):
            example = stats["examples"][0].replace("|", "\\|") if stats.get("examples") else ""
            chars = stats.get("chars", "-")
            lines.append(f"| {name} | {stats['rows']:,} | {chars} | {example} |")
        lines.append("")
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", args.out, "and", args.report)
    for key, value in report.items():
        if value["categories"]:
            top = sorted(value["categories"].items(), key=lambda x: -x[1]["rows"])[:4]
            print(f"{key:32} " + ", ".join(f"{n}:{s['rows']}" for n, s in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
