#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA + lọc heuristic + làm sạch cơ bản cho 4 tập BT Tiki (unit-level).

Input : data/cleaned/bt_tiki/tiki_bt_<model>_units.parquet  (45.709 unit/model)
Output: data/cleaned/bt_tiki_clean/
          tiki_bt_<model>_clean.parquet   (đã lọc, còn cờ)
          eda_report.json
        reports/tiki_bt_eda.md
        reports/figures/*.png

Triết lý: chỉ loại bỏ lỗi rõ ràng; các dấu hiệu nghi ngờ gắn cờ để quyết định sau.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
MODEL_SHORT = {m: m.replace("qwen3-", "") for m in MODELS}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_RE = re.compile(
    r"[ăâêôơưđ]|[áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d+")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-+.]{2,}")
TOKEN_RE = re.compile(r"[^\s,;&|/]+")

RATIO_LO, RATIO_HI = 0.15, 2.5


def compact_len(text: str) -> int:
    return len(SPACE_RE.sub("", str(text or "")))


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(str(text or "")))


def digits(text: str) -> set[str]:
    normalised = re.sub(r"(?<=\d)[.,](?=\d)", "", str(text or ""))
    return set(DIGIT_RE.findall(normalised))


def is_degenerate(text: str) -> bool:
    text = str(text or "")
    tokens = [t for t in TOKEN_RE.findall(text) if t]
    if len(tokens) >= 10:
        counts = Counter(tokens)
        if max(counts.values()) / len(tokens) > 0.5:
            return True
    compact = SPACE_RE.sub("", text)
    if len(compact) >= 40 and len(set(compact)) / len(compact) < 0.12:
        return True
    if len(compact) >= 32:
        for size in (6, 8):
            chunks = Counter(compact[i:i + size] for i in range(0, len(compact) - size + 1))
            if chunks and max(chunks.values()) >= 4:
                return True
    return False


def brand_retention(source: str, target: str) -> float | None:
    tokens = [t for t in LATIN_TOKEN_RE.findall(str(source or ""))]
    if not tokens:
        return None
    low = str(target or "").lower()
    hit = sum(1 for t in tokens if t.lower() in low)
    return hit / len(tokens)


def clean_text(text: str) -> str:
    value = CTRL_RE.sub(" ", str(text or ""))
    value = SPACE_RE.sub(" ", value).strip()
    return value


def build_frame(bt_root: Path) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        df = pd.read_parquet(bt_root / f"tiki_bt_{model}_units.parquet")
        df["model"] = MODEL_SHORT[model]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_clean"] = df["source_text_bt"].map(clean_text)
    df["target_clean"] = df["target_text_bt"].map(clean_text)
    df["src_len"] = df["source_clean"].map(compact_len)
    df["tgt_len"] = df["target_clean"].map(compact_len)
    df["ratio"] = np.where(df["src_len"] > 0, df["tgt_len"] / df["src_len"].replace(0, 1), np.nan)
    df["cjk"] = df["target_clean"].map(cjk_count)
    df["cjk_ratio"] = np.where(df["tgt_len"] > 0, df["cjk"] / df["tgt_len"].replace(0, 1), 0.0)
    df["has_vi"] = df["target_clean"].str.contains(VI_RE)
    df["src_has_vi"] = df["source_clean"].str.contains(VI_RE)
    df["brand_keep"] = [
        brand_retention(s, t) for s, t in zip(df["source_clean"], df["target_clean"])
    ]
    df["degenerate"] = df["target_clean"].map(is_degenerate)

    df["f_empty"] = df["target_clean"].eq("")
    # không có chữ Hán & source là tiếng Việt: tách 2 mức
    no_cjk = (~df["f_empty"]) & (df["cjk"] == 0) & df["src_has_vi"]
    df["f_untranslated"] = no_cjk & df["has_vi"]        # target vẫn tiếng Việt -> loại
    df["f_latin_only"] = no_cjk & (~df["has_vi"])       # target Latin thuần -> chỉ gắn cờ
    df["f_partial_vi"] = (~df["f_empty"]) & (df["cjk"] > 0) & df["has_vi"]
    df["f_degenerate"] = (~df["f_empty"]) & df["degenerate"]
    df["f_too_short"] = (~df["f_empty"]) & (df["src_len"] >= 10) & (df["tgt_len"] <= 2)
    # model dịch nhầm key thay vì value
    key_zh = df["key_zh_effective"].fillna("").map(lambda x: SPACE_RE.sub("", str(x)).lower())
    target_norm = df["target_clean"].map(lambda x: SPACE_RE.sub("", str(x)).lower())
    df["f_equals_key"] = (key_zh.ne("")) & (key_zh == target_norm)
    df["f_ratio_out"] = (
        (~df["f_empty"]) & (df["src_len"] >= 6)
        & ((df["ratio"] < RATIO_LO) | (df["ratio"] > RATIO_HI))
    )
    df["f_number_mismatch"] = [
        bool(digits(s) - digits(t)) for s, t in zip(df["source_clean"], df["target_clean"])
    ]
    df["f_identical"] = (
        df["source_clean"].str.lower() == df["target_clean"].str.lower()
    )
    df["hard_drop"] = (
        df["f_empty"] | df["f_untranslated"] | df["f_degenerate"]
        | df["f_too_short"] | df["f_equals_key"]
    )
    df["dup_count"] = (
        df.groupby(["model", "source_clean", "target_clean"])["pair_id"]
        .transform("size").astype(int)
    )
    df["flag_any"] = (
        df["hard_drop"] | df["f_partial_vi"] | df["f_ratio_out"]
        | df["f_number_mismatch"] | df["f_identical"] | df["f_latin_only"]
    )
    return df


def summarise(df: pd.DataFrame) -> dict:
    report: dict[str, dict] = {}
    flags = ["hard_drop", "f_empty", "f_untranslated", "f_latin_only", "f_equals_key",
             "f_partial_vi", "f_degenerate", "f_too_short", "f_ratio_out",
             "f_number_mismatch", "f_identical", "flag_any"]
    for model, part in df.groupby("model"):
        q = part["ratio"].dropna()
        report[model] = {
            "units": int(len(part)),
            "split": part["split"].value_counts().to_dict(),
            "unit_type": part["unit_type"].value_counts().to_dict(),
            "src_len": {
                "p50": float(part["src_len"].quantile(0.5)),
                "p95": float(part["src_len"].quantile(0.95)),
                "max": int(part["src_len"].max()),
            },
            "tgt_len": {
                "p50": float(part["tgt_len"].quantile(0.5)),
                "p95": float(part["tgt_len"].quantile(0.95)),
                "max": int(part["tgt_len"].max()),
            },
            "ratio": {
                "p05": float(q.quantile(0.05)),
                "p50": float(q.quantile(0.5)),
                "p95": float(q.quantile(0.95)),
            },
            "cjk_ratio_mean": float(part["cjk_ratio"].mean()),
            "brand_keep_mean": float(part["brand_keep"].dropna().mean())
            if part["brand_keep"].notna().any() else None,
            "flags": {f: int(part[f].sum()) for f in flags},
            "flag_pct": {f: round(part[f].mean() * 100, 2) for f in flags},
            "qc_status": part["qc_status"].value_counts().to_dict(),
            "dup_pair_source_target": int(part.duplicated(["source_clean", "target_clean"]).sum()),
            "source_multi_target": int((part.groupby("source_clean")["target_clean"].nunique() > 1).sum()),
            "unique_source": int(part["source_clean"].nunique()),
        }
    return report


def cross_model(df: pd.DataFrame) -> dict:
    wide = df.pivot_table(index="pair_id", columns="model", values="target_clean", aggfunc="first")
    wide = wide.dropna(how="any")
    models = list(wide.columns)
    agreement = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            same = (wide[a] == wide[b])
            agreement[f"{a}|{b}"] = round(float(same.mean()) * 100, 2)
    cjk = df.assign(has_cjk=df["cjk"] > 0).pivot_table(
        index="pair_id", columns="model", values="has_cjk", aggfunc="first").dropna(how="any")
    cjk_agree = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            cjk_agree[f"{a}|{b}"] = round(float((cjk[a] == cjk[b]).mean()) * 100, 2)
    return {
        "pairs_compared": int(len(wide)),
        "exact_agreement_pct": agreement,
        "cjk_presence_agreement_pct": cjk_agree,
        "all_four_identical_pct": round(
            float((wide.nunique(axis=1) == 1).mean()) * 100, 2),
    }


def plots(df: pd.DataFrame, report: dict, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = [MODEL_SHORT[m] for m in MODELS]
    paths = []

    # 1. cờ theo model
    flag_names = ["hard_drop", "f_untranslated", "f_equals_key", "f_too_short",
                  "f_partial_vi", "f_latin_only", "f_ratio_out",
                  "f_number_mismatch", "f_identical"]
    matrix = np.array([[report[m]["flag_pct"][f] for f in flag_names] for m in models])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(matrix, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(flag_names)))
    ax.set_xticklabels([f.replace("f_", "") for f in flag_names], rotation=25, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="% units")
    ax.set_title("Heuristic flag rate by model")
    fig.tight_layout()
    p = fig_dir / "flags_heatmap.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # 2. phân bố length ratio
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in models:
        q = df.loc[df["model"] == m, "ratio"].dropna()
        q = q[(q > 0) & (q < 3)]
        ax.hist(q, bins=80, histtype="step", label=m, linewidth=1.4)
    ax.set_xlabel("target chars / source chars")
    ax.set_ylabel("units")
    ax.set_title("Length ratio distribution")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "ratio_hist.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # 3. QC status
    statuses = ["ok", "repaired", "repaired_v2", "recovered_json", "still_untranslated"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = np.zeros(len(models))
    for status in statuses:
        vals = np.array([report[m]["qc_status"].get(status, 0) for m in models])
        ax.bar(models, vals, bottom=bottom, label=status)
        bottom += vals
    ax.set_ylabel("units")
    ax.set_title("QC status by model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "qc_status.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    # 4. length theo unit_type
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = [df.loc[(df["model"] == "8b") & (df["unit_type"] == t), "tgt_len"].clip(upper=400)
            for t in ("title", "attribute")]
    ax.boxplot(data, labels=["title", "attribute"], showfliers=False)
    ax.set_ylabel("target chars (clipped 400)")
    ax.set_title("Target length by unit type (8B)")
    fig.tight_layout()
    p = fig_dir / "length_by_type.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))
    return paths


def markdown(report: dict, cross: dict, df: pd.DataFrame) -> str:
    models = [MODEL_SHORT[m] for m in MODELS]
    lines = ["# EDA — Tập song ngữ Tiki (BT unit-level)", ""]
    lines += ["## 1. Quy mô", "",
              "| model | units | train | dev | test | unique source | source→nhiều target | dup cặp |",
              "|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = report[m]
        lines.append(
            f"| {m} | {r['units']:,} | {r['split'].get('train', 0):,} | "
            f"{r['split'].get('dev', 0):,} | {r['split'].get('test', 0):,} | "
            f"{r['unique_source']:,} | {r['source_multi_target']:,} | {r['dup_pair_source_target']:,} |"
        )
    lines += ["", "## 2. Độ dài & tỉ lệ", "",
              "| model | src p50 | src p95 | tgt p50 | tgt p95 | ratio p05 | ratio p50 | ratio p95 |",
              "|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = report[m]
        lines.append(
            f"| {m} | {r['src_len']['p50']:.0f} | {r['src_len']['p95']:.0f} | "
            f"{r['tgt_len']['p50']:.0f} | {r['tgt_len']['p95']:.0f} | "
            f"{r['ratio']['p05']:.2f} | {r['ratio']['p50']:.2f} | {r['ratio']['p95']:.2f} |"
        )
    lines += ["", "## 3. Cờ heuristic (% units)", "",
              "| model | hard drop | untranslated | equals key | too short | partial VI | latin only | ratio out | number mismatch | identical |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        p = report[m]["flag_pct"]
        lines.append(
            f"| {m} | {p['hard_drop']:.2f}% | {p['f_untranslated']:.2f}% | "
            f"{p['f_equals_key']:.2f}% | {p['f_too_short']:.2f}% | "
            f"{p['f_partial_vi']:.2f}% | {p['f_latin_only']:.2f}% | "
            f"{p['f_ratio_out']:.2f}% | {p['f_number_mismatch']:.2f}% | {p['f_identical']:.2f}% |"
        )
    lines += ["", "## 4. Đồng thuận giữa model", "",
              f"- cặp so sánh: {cross['pairs_compared']:,}",
              f"- cả 4 model giống hệt: **{cross['all_four_identical_pct']:.2f}%**", ""]
    lines += ["| cặp | target giống hệt | cùng có/không chữ Hán |", "|---|---|---|"]
    for key, value in cross["exact_agreement_pct"].items():
        lines.append(f"| {key} | {value:.2f}% | {cross['cjk_presence_agreement_pct'][key]:.2f}% |")
    lines += ["", "## 5. Sau làm sạch", "",
              "| model | giữ | loại | % giữ | test đủ | test dùng được | cặp unique (train+dev) | trùng lặp |",
              "|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = report[m]
        lines.append(
            f"| {m} | {r['kept']:,} | {r['dropped']:,} | {r['kept'] / r['units'] * 100:.2f}% | "
            f"{r['test_rows']:,} | {r['test_eligible']:,} | {r['unique_pairs']:,} | {r['duplicate_rows']:,} |"
        )
    lines += ["", "## 6. Hình", "",
              "![flags](figures/flags_heatmap.png)", "",
              "![ratio](figures/ratio_hist.png)", "",
              "![qc](figures/qc_status.png)", "",
              "![type](figures/length_by_type.png)", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-root", default="data/cleaned/bt_tiki")
    ap.add_argument("--out", default="data/cleaned/bt_tiki_clean")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    raw = build_frame(Path(args.bt_root))
    df = add_features(raw)
    report = summarise(df)
    cross = cross_model(df)
    figures = plots(df, report, report_dir / "figures")

    keep_cols = [
        "pair_id", "product_id", "split", "unit_type", "key_vi", "key_zh",
        "source_clean", "target_clean", "model", "qc_status", "repair_pass",
        "src_len", "tgt_len", "ratio", "cjk_ratio", "brand_keep",
        "key_zh_effective", "hard_drop", "f_untranslated", "f_latin_only",
        "f_equals_key", "f_partial_vi", "f_degenerate", "f_too_short",
        "f_ratio_out", "f_number_mismatch", "f_identical", "flag_any", "dup_count",
    ]
    for model, part in df.groupby("model"):
        # giữ TOÀN BỘ test để 4 model được đánh giá trên cùng một tập
        keep_mask = (~part["hard_drop"]) | (part["split"] == "test")
        clean = part.loc[keep_mask, keep_cols].copy()
        clean["train_eligible"] = ~clean["hard_drop"]
        clean.to_parquet(out_dir / f"tiki_bt_{model}_clean.parquet", index=False)

        train_part = clean[clean["train_eligible"] & (clean["split"] != "test")]
        dedup = train_part.drop_duplicates(["source_clean", "target_clean"], keep="first")
        dedup.to_parquet(out_dir / f"tiki_bt_{model}_clean_dedup.parquet", index=False)

        report[model]["kept"] = int(len(clean))
        report[model]["train_eligible"] = int(clean["train_eligible"].sum())
        report[model]["test_rows"] = int((clean["split"] == "test").sum())
        report[model]["test_eligible"] = int(
            ((clean["split"] == "test") & clean["train_eligible"]).sum())
        report[model]["dropped"] = int(part["hard_drop"].sum())
        report[model]["unique_pairs"] = int(len(dedup))
        report[model]["duplicate_rows"] = int(len(train_part) - len(dedup))

    payload = {"models": report, "cross_model": cross, "figures": figures,
               "thresholds": {"ratio_lo": RATIO_LO, "ratio_hi": RATIO_HI}}
    (out_dir / "eda_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "tiki_bt_eda.md").write_text(markdown(report, cross, df), encoding="utf-8")

    print(markdown(report, cross, df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
