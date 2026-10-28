#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vòng 2: cứu JSON hỏng + chuẩn bị input cho các dòng vẫn chưa dịch.

Bước 1 — recovery (không cần GPU):
  Với các dòng json_ok=False, raw_output thường vẫn chứa nội dung ngay sau
  `"translation": "` nhưng thiếu dấu đóng do chạm max_new. Cắt phần đó ra,
  unescape, rồi kiểm tra có chữ Hán.

Bước 2 — input vòng 2:
  Các dòng còn lại (json_ok nhưng vẫn copy tiếng Việt) được đưa vào kernel
  repair v2, kèm `prev_output` để model biết bản trước bị loại.

Output: <out>/recovered.parquet + <out>/repair_input_v2.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(
    r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")


def is_good(text: str) -> bool:
    """Dùng cho recovery: phải có chữ Hán và không còn dấu tiếng Việt."""
    text = (text or "").strip()
    if not text:
        return False
    if not CJK_RE.search(text):
        return False
    return not VI_RE.search(text)


def looks_degenerate(text: str) -> bool:
    """Phát hiện lặp vô hạn kiểu '豆腐 & 豆腐 & ...' hoặc '15 15 15 ...'."""
    tokens = [t for t in re.split(r"[\s,&;|/]+", text) if t]
    if len(tokens) >= 12:
        values, counts = {}, {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        if max(counts.values()) / len(tokens) > 0.5:
            return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 40:
        for size in (4, 6, 8):
            if len(compact) >= size * 5:
                most = max(compact[i:i + size] for i in range(0, len(compact) - size + 1))
        # đơn giản: tỉ lệ ký tự duy nhất
        if len(set(compact)) / len(compact) < 0.12:
            return True
    return False


def recover(raw: str) -> str:
    """Lấy phần nội dung sau khóa translation, chịu được JSON bị cắt."""
    text = str(raw or "")
    marker = text.find('"translation"')
    if marker < 0:
        return ""
    colon = text.find(":", marker)
    if colon < 0:
        return ""
    start = text.find('"', colon)
    if start < 0:
        return ""
    body = text[start + 1:]
    # ưu tiên cắt tại dấu đóng JSON; nếu không có thì lấy hết
    end = body.rfind('"}')
    if end > 0:
        body = body[:end]
    else:
        body = body.rstrip().rstrip("}")
    body = body.rstrip().rstrip('"')
    body = body.replace("\\n", " ").replace('\\"', '"').replace("\\t", " ")
    body = body.strip()
    # bỏ artifact JSON còn sót: ", }, ), ...
    while body and body[-1] in '"},':
        body = body[:-1].rstrip()
    while body.endswith(")") and body.count("(") < body.count(")"):
        body = body[:-1].rstrip()
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-root", required=True, help="thư mục chứa tiki_bt_<model>_full.parquet")
    ap.add_argument("--repair-root", required=True, help="repair_outputs của vòng 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bt_root = Path(args.bt_root)
    repair_root = Path(args.repair_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    recovered_rows, v2_rows, stats = [], [], {}
    for model in MODELS:
        full = pd.read_parquet(bt_root / f"tiki_bt_{model}_full.parquet")
        keys = set(full.loc[full["qc_status"] == "still_untranslated", "cache_key"].astype(str))
        rep = pd.read_parquet(repair_root / model / "repair.parquet")
        rep = rep[rep["cache_key"].astype(str).isin(keys)].copy()
        value_class = (
            full.drop_duplicates("cache_key")
            .set_index(full.drop_duplicates("cache_key")["cache_key"].astype(str))["value_class"]
        )
        rep["value_class"] = rep["cache_key"].astype(str).map(value_class).fillna("text")

        rec_n = 0
        for row in rep.itertuples():
            if bool(row.json_ok):
                continue
            text = recover(row.raw_output)
            if is_good(text) and not looks_degenerate(text):
                recovered_rows.append({
                    "model_key": model,
                    "cache_key": str(row.cache_key),
                    "target_text_bt": text,
                    "origin": "recovered_json",
                })
                rec_n += 1

        # vòng 2 chạy lại toàn bộ dòng còn hỏng; recovery chỉ là fallback sau đó
        todo = rep.copy()
        for row in todo.itertuples():
            v2_rows.append({
                "model_key": model,
                "cache_key": str(row.cache_key),
                "source_text_bt": row.source_text_bt,
                "key_vi": row.key_vi,
                "key_zh_effective": row.key_zh_effective,
                "value_class": row.value_class,
                "prev_output": str(row.target_text_bt or ""),
            })
        stats[model] = {
            "still_untranslated": int(len(rep)),
            "recovered_json": rec_n,
            "to_repair_v2": int(len(todo)),
        }
        print(model, stats[model])

    pd.DataFrame(recovered_rows).to_parquet(out_dir / "recovered.parquet", index=False)
    v2 = pd.DataFrame(v2_rows)
    v2.to_parquet(out_dir / "repair_input_v2.parquet", index=False)
    (out_dir / "repair_prepare_v2.json").write_text(
        json.dumps({"models": stats, "recovered": len(recovered_rows),
                    "to_repair": len(v2)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "Tiki BT Repair Input V2",
        "id": "ntquang0410/tiki-bt-repair-input-v2",
        "licenses": [{"name": "CC0-1.0"}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recovered": len(recovered_rows), "to_repair": len(v2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
