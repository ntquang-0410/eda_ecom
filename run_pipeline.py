#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI thống nhất cho pipeline Trung–Việt.

Ví dụ::

    # Làm sạch từ tiếng Anh trong title_zh + title_vi/description_vi
    python run_pipeline.py cado

    # Chạy CADO đầy đủ T0–T5 (cần tải bge-m3 ~2.2GB)
    python run_pipeline.py cado --tier5

    # Làm sạch cả description_zh (chậm hơn)
    python run_pipeline.py cado --columns title_zh,description_zh

    # Chấm điểm bitext (cần bge-m3)
    python run_pipeline.py bitext

    # Chỉ preprocess, không cần model
    python run_pipeline.py bitext --preprocess-only

    # Chạy cả hai
    python run_pipeline.py all

    # EDA Tiki (đơn ngữ VI) trước back-translation
    python run_pipeline.py tiki-eda

    # Phase 1: profile title + description key-value
    python run_pipeline.py tiki-profile

    # Phase 1: build product-level + pair-level BT units (không gọi MT)
    python run_pipeline.py tiki-units

    # Phase 1.5: học length prior từ 1688 human, gắn khoảng độ dài ZH cho Tiki
    python run_pipeline.py tiki-length-prior

    # Phase A v2: value-only + value_class + CJK + split-safe
    python run_pipeline.py tiki-length-prior-v2

    # Phase B/C v3: alignment fallback + hierarchical title smoothing
    python run_pipeline.py tiki-length-prior-v3

    # Plan A: key map + routing + BT input/cache, chưa gọi MT
    python run_pipeline.py tiki-bt-prepare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cado_core.bitext import BitextConfig, run_bitext
from cado_core.cado import CadoConfig, run_cado
from cado_core.cado_vi import ViCleanConfig, run_cado_vi
from cado_core.config import (
    BITEXT_READY_PARQUET,
    CADO_CSV_DIR,
    CADO_PARQUET,
    DESC_VI,
    DESC_VI_CLEANED,
    DESC_ZH,
    DESC_ZH_CLEANED,
    LENGTH_PRIOR_CSV,
    LENGTH_PRIOR_JSON,
    LENGTH_PRIOR_V2_ALL_JSON,
    LENGTH_PRIOR_V2_CSV,
    LENGTH_PRIOR_V2_HOLDOUT_JSON,
    LENGTH_PRIOR_V2_SPLIT_MISMATCH_CSV,
    LENGTH_PRIOR_V2_SPLIT_SAMPLE_CSV,
    LENGTH_PRIOR_V2_SPLIT_SUMMARY_CSV,
    LENGTH_PRIOR_V2_TRAIN_CSV,
    LENGTH_PRIOR_V2_TRAIN_JSON,
    RAW_PARQUET,
    TIKI_BT_UNITS_PARQUET,
    TIKI_BT_UNITS_PLAN_A_PARQUET,
    TIKI_INPUT_PARQUET,
    TIKI_KEY_INVENTORY_CSV,
    TIKI_KEY_MAP_CSV,
    TIKI_LENGTH_PRIOR_STATS_JSON,
    TIKI_LENGTH_PRIOR_V2_STATS_JSON,
    TIKI_PARSE_ERRORS_CSV,
    TIKI_PRODUCTS_PARQUET,
    TIKI_PROFILE_JSON,
    TIKI_UNITS_LENGTH_PARQUET,
    TIKI_UNITS_LENGTH_V2_PARQUET,
    TIKI_UNITS_LENGTH_V3_PARQUET,
    TIKI_UNITS_STATS_JSON,
    TITLE_VI,
    TITLE_VI_CLEANED,
    TITLE_ZH,
    TITLE_ZH_CLEANED,
    resolve_path,
)
from cado_core.length_prior import (
    LengthPriorConfig,
    apply_length_prior,
    extract_human_length_pairs,
    fit_length_prior,
    write_prior_table,
)
from cado_core.length_prior_v2 import (
    LengthPriorV2Config,
    apply_length_prior_v2,
    evaluate_length_prior_v2,
    extract_human_value_pairs,
    fit_length_prior_v2,
)
from cado_core.rules import aggregate_word_report
from cado_core.tiki import run_tiki_eda
from cado_core.tiki_units import build_tiki_units, profile_tiki


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def log(message: str) -> None:
    print(message, flush=True)


def _parse_columns(value: str) -> tuple[str, ...]:
    columns = tuple(part.strip() for part in value.split(",") if part.strip())
    if not columns:
        raise argparse.ArgumentTypeError("Danh sách cột rỗng")
    return columns


def _pair_for(column: str) -> tuple[str, str] | None:
    mapping = {
        TITLE_ZH: TITLE_ZH_CLEANED,
        DESC_ZH: DESC_ZH_CLEANED,
        TITLE_VI: TITLE_VI_CLEANED,
        DESC_VI: DESC_VI_CLEANED,
    }
    cleaned = mapping.get(column)
    return (column, cleaned) if cleaned else None


def _validate_product_ids(df: pd.DataFrame) -> None:
    ids = df["product_id"].astype(str).str.strip()
    if (ids == "").any():
        raise ValueError("Có product_id rỗng; cần xử lý trước.")
    duplicated = int(ids.duplicated().sum())
    if duplicated:
        raise ValueError(f"product_id bị trùng {duplicated:,} dòng; cần xử lý trước.")


def _changed_mask(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for source, cleaned in pairs:
        if source in df.columns and cleaned in df.columns:
            mask |= df[cleaned].astype(str) != df[source].astype(str)
    return mask


def _print_changed_samples(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    limit: int = 8,
) -> None:
    pairs = [(s, c) for s, c in pairs if s in df.columns and c in df.columns]
    if not pairs:
        return
    sample = df[_changed_mask(df, pairs)].head(limit)
    if sample.empty:
        log("  (không có dòng nào thay đổi)")
        return

    for _, row in sample.iterrows():
        log(f"\n  ── {row.get('product_id', '?')} ──")
        for source, cleaned in pairs:
            before_full = str(row.get(source, ""))
            after_full = str(row.get(cleaned, ""))
            if before_full != after_full:
                log(f"    {source:>22}: {before_full[:110]}")
                log(f"    {'→ ' + cleaned:>22}: {after_full[:110]}")


def _write_cado_reports(result, output_dir: Path, write_occurrences: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result.loanword_report.empty:
        path = output_dir / "cado_tier3_loanword_report.csv"
        result.loanword_report.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"  💾 {path} ({len(result.loanword_report):,} dòng)")

    if not result.occurrences.empty:
        if write_occurrences:
            path = output_dir / "cado_occurrences.csv"
            result.occurrences.to_csv(path, index=False, encoding="utf-8-sig")
            log(f"  💾 {path} ({len(result.occurrences):,} dòng)")

        report = aggregate_word_report(result.occurrences.to_dict("records"))
        path = output_dir / "cado_word_report.csv"
        pd.DataFrame(report).to_csv(path, index=False, encoding="utf-8-sig")
        log(f"  💾 {path} ({len(report):,} từ)")

    if not result.t5_report.empty:
        path = output_dir / "cado_tier5_similarity.csv"
        result.t5_report.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"  💾 {path} ({len(result.t5_report):,} cặp)")


def _write_changed_rows(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    path: Path,
) -> None:
    pairs = [(s, c) for s, c in pairs if s in df.columns and c in df.columns]
    if not pairs:
        return

    subset = df[_changed_mask(df, pairs)].copy()
    keep = [
        column for column in (
            "product_id", "category",
            "title_zh", "title_zh_cleaned",
            "description_zh", "description_zh_cleaned",
            "title_vi", "title_vi_cleaned", "description_vi", "desc_vi_cleaned",
        )
        if column in subset.columns
    ]
    subset[keep].to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  💾 {path} ({len(subset):,} dòng)")


# ═══════════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════════

def command_cado(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input, RAW_PARQUET)
    output_path = resolve_path(args.output, CADO_PARQUET)

    log("=" * 68)
    log("🧹 CADO — LÀM SẠCH TỪ TIẾNG ANH")
    log("=" * 68)
    log(f"📂 Input : {input_path}")
    log(f"📂 Output: {output_path}")

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
        log(f"⚠️  Giới hạn {args.limit:,} dòng để test")

    cado_config = CadoConfig(
        columns=_parse_columns(args.columns),
        loanword_column=args.loanword_column,
        max_tier=5 if args.tier5 else args.max_tier,
        use_loanwords=not args.no_loanwords,
        tier5_progress_bar=not args.no_progress_bar,
        tier5_device=args.device,
        progress=log,
    )
    result = run_cado(df, cado_config)
    out = result.df
    write_reports = not args.limit
    if args.limit:
        log("⚠️  Bỏ qua report CSV vì đang chạy --limit (tránh ghi đè report full)")

    pairs: list[tuple[str, str]] = [
        pair for pair in (_pair_for(column) for column in cado_config.columns)
        if pair is not None
    ]

    # ── CADO-VI ──
    if not args.no_vi:
        log("\n🇻🇳 CADO-VI — làm sạch từ tiếng Anh trong văn bản Việt")
        vi_result = run_cado_vi(out, ViCleanConfig(progress=log))
        out = vi_result.df
        log(
            "  Sản phẩm đổi title_vi_cleaned: "
            f"{vi_result.stats['changed_by_column'].get(TITLE_VI, 0):,}"
        )
        log(
            "  Sản phẩm đổi desc_vi_cleaned : "
            f"{vi_result.stats['changed_by_column'].get(DESC_VI, 0):,}"
        )
        if write_reports and not vi_result.occurrences.empty:
            path = CADO_CSV_DIR / "cado_vi_occurrences.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            vi_result.occurrences.to_csv(path, index=False, encoding="utf-8-sig")
            log(f"  💾 {path} ({len(vi_result.occurrences):,} dòng)")
        for column in (TITLE_VI, DESC_VI):
            pair = _pair_for(column)
            if pair is not None:
                pairs.append(pair)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    log(f"\n✅ Đã lưu {output_path} ({len(out):,} dòng, {len(out.columns)} cột)")

    log("\n📋 Mẫu thay đổi:")
    _print_changed_samples(out, [p for p in pairs if p is not None])

    if write_reports:
        _write_cado_reports(result, CADO_CSV_DIR, write_occurrences=not args.no_occurrences)
        _write_changed_rows(out, [p for p in pairs if p is not None],
                            CADO_CSV_DIR / "cado_changed_rows.csv")


def _run_bitext_command(
    *,
    input_path: Path,
    output_path: Path | None,
    ready_path: Path | None,
    label_mode: str,
    limit: int | None,
    skip_cosine: bool,
    save_ready: bool,
    entity_signals: bool,
    entity_embedding: bool,
    preprocess_only: bool,
) -> None:
    log("=" * 68)
    log("🔍 BITEXT — PREPROCESS + CHẤM ĐIỂM")
    log("=" * 68)
    log(f"📂 Input: {input_path}")

    df = pd.read_parquet(input_path)

    defaults = BitextConfig()
    config = BitextConfig(
        input_path=input_path,
        output_parquet=output_path or defaults.output_parquet,
        ready_parquet=ready_path or defaults.ready_parquet,
        label_mode=label_mode,
        limit=limit,
        save_ready=save_ready,
        entity_signals=entity_signals,
        entity_embedding=entity_embedding,
        progress=log,
    )
    run_bitext(
        df,
        config,
        skip_cosine=skip_cosine,
        preprocess_only=preprocess_only,
    )


def command_bitext(args: argparse.Namespace) -> None:
    _run_bitext_command(
        input_path=resolve_path(args.input, CADO_PARQUET),
        output_path=resolve_path(args.output, None) if args.output else None,
        ready_path=resolve_path(args.ready_output, None) if args.ready_output else None,
        label_mode=args.label_mode,
        limit=args.limit,
        skip_cosine=args.skip_cosine,
        save_ready=args.save_ready,
        entity_signals=args.entity_signals,
        entity_embedding=args.entity_embedding,
        preprocess_only=args.preprocess_only,
    )


def command_all(args: argparse.Namespace) -> None:
    command_cado(args)
    if args.no_bitext:
        log("\n⏩ Bỏ qua bitext (--no-bitext)")
        return

    bitext_input = resolve_path(args.bitext_input, None) if args.bitext_input \
        else resolve_path(args.output, CADO_PARQUET)
    _run_bitext_command(
        input_path=bitext_input,
        output_path=resolve_path(args.bitext_output, None) if args.bitext_output else None,
        ready_path=resolve_path(args.bitext_ready_output, None) if args.bitext_ready_output else None,
        label_mode=args.bitext_label_mode,
        limit=args.limit,
        skip_cosine=args.bitext_skip_cosine,
        save_ready=args.bitext_save_ready,
        entity_signals=args.bitext_entity_signals,
        entity_embedding=args.bitext_entity_embedding,
        preprocess_only=args.bitext_preprocess_only,
    )


def command_tiki_eda(args: argparse.Namespace) -> None:
    input_path = resolve_path(
        args.input, PROJECT_ROOT / "data/cleaned/tiki_vi_cado.parquet"
    )
    output_path = resolve_path(
        args.output, PROJECT_ROOT / "data/cleaned/tiki_eda.parquet"
    )

    log("=" * 68)
    log("📊 TIKI EDA — CHUẨN BỊ BACK-TRANSLATION")
    log("=" * 68)
    log(f"📂 Input : {input_path}")
    log(f"📂 Output: {output_path}")

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
        log(f"⚠️  Giới hạn {args.limit:,} dòng để test")
    if df["product_id"].duplicated().any():
        duplicated = int(df["product_id"].duplicated().sum())
        raise ValueError(f"product_id bị trùng {duplicated:,} dòng.")

    result = run_tiki_eda(df, progress=log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_parquet(output_path, index=False, compression="snappy")

    stats = result.stats
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log(f"\n✅ Đã lưu {output_path} ({size_mb:.1f} MB)")
    log(f"   {stats['rows']:,} dòng × {stats['columns']:,} cột")
    log(f"   BT ready (title/prose): {stats['bt_ready']:,}")
    for route, count in sorted(
        stats["overall_routes"].items(), key=lambda item: -item[1]
    ):
        log(f"   {route:<18} {count:>7,}")


def command_tiki_profile(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input, TIKI_INPUT_PARQUET)
    profile_path = resolve_path(args.profile_output, TIKI_PROFILE_JSON)
    inventory_path = resolve_path(args.key_inventory_output, TIKI_KEY_INVENTORY_CSV)
    errors_path = resolve_path(args.parse_errors_output, TIKI_PARSE_ERRORS_CSV)

    log("=" * 68)
    log("📊 TIKI PHASE 1 — PROFILE TITLE + DESCRIPTION KV")
    log("=" * 68)
    log(f"📂 Input          : {input_path}")
    log(f"📂 Profile        : {profile_path}")
    log(f"📂 Key inventory  : {inventory_path}")
    log(f"📂 Parse errors   : {errors_path}")

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
        log(f"⚠️  Giới hạn {args.limit:,} dòng để test")
    _validate_product_ids(df)

    result = profile_tiki(df, progress=log)
    stats = dict(result.stats)
    stats["input"] = str(input_path)

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    result.key_inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    result.parse_errors.to_csv(errors_path, index=False, encoding="utf-8-sig")

    title = stats["title"]
    attributes = stats["attributes"]
    description = stats["description"]
    log(f"\n✅ Đã lưu {profile_path}")
    log(f"   {stats['rows']:,} dòng")
    log(f"   Title 3–40 từ: {title['n_3_40']:,}")
    log(f"   Description lỗi: {description['n_malformed_parts']:,} phần")
    log(f"   Attribute pairs: {attributes['n_pairs']:,}")
    log(f"   Unique keys: {stats['key_inventory_rows']:,}")
    log(f"   💾 {inventory_path}")
    log(f"   💾 {errors_path}")


def command_tiki_units(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input, TIKI_INPUT_PARQUET)
    products_path = resolve_path(args.products_output, TIKI_PRODUCTS_PARQUET)
    units_path = resolve_path(args.units_output, TIKI_BT_UNITS_PARQUET)
    stats_path = resolve_path(args.stats_output, TIKI_UNITS_STATS_JSON)
    key_map_path = resolve_path(args.key_map, TIKI_KEY_MAP_CSV)

    log("=" * 68)
    log("🧱 TIKI PHASE 1 — BUILD BT UNITS")
    log("=" * 68)
    log(f"📂 Input        : {input_path}")
    log(f"📂 Products     : {products_path}")
    log(f"📂 Units        : {units_path}")
    log(f"📂 Stats        : {stats_path}")
    log(f"📂 Key map      : {key_map_path}")

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
        log(f"⚠️  Giới hạn {args.limit:,} dòng để test")
    _validate_product_ids(df)

    result = build_tiki_units(df, key_map_path=key_map_path, progress=log)
    stats = dict(result.stats)
    stats["input"] = str(input_path)
    stats["products_output"] = str(products_path)
    stats["units_output"] = str(units_path)
    stats["key_map_path"] = str(key_map_path)

    products_path.parent.mkdir(parents=True, exist_ok=True)
    result.products.to_parquet(products_path, index=False, compression="snappy")
    units_path.parent.mkdir(parents=True, exist_ok=True)
    result.units.to_parquet(units_path, index=False, compression="snappy")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    log(f"\n✅ Đã lưu {products_path} ({len(result.products):,} dòng)")
    log(f"✅ Đã lưu {units_path} ({len(result.units):,} dòng)")
    log(f"   Title units     : {stats['title_units']:,}")
    log(f"   Attribute units : {stats['attribute_units']:,}")
    log(f"   Train-eligible  : {stats['train_eligible_units']:,}")
    log(f"   Pending key map : {stats['pending_key_map']:,} units")
    log(f"   💾 {stats_path}")


def command_tiki_length_prior(args: argparse.Namespace) -> None:
    bilingual_path = resolve_path(args.bilingual, CADO_PARQUET)
    units_input_path = resolve_path(args.tiki_units, TIKI_BT_UNITS_PARQUET)
    prior_path = resolve_path(args.prior_output, LENGTH_PRIOR_JSON)
    prior_csv_path = resolve_path(args.prior_csv, LENGTH_PRIOR_CSV)
    units_output_path = resolve_path(args.units_output, TIKI_UNITS_LENGTH_PARQUET)
    stats_path = resolve_path(args.stats_output, TIKI_LENGTH_PRIOR_STATS_JSON)

    log("=" * 68)
    log("📏 TIKI PHASE 1.5 — LENGTH PRIOR 1688 → TIKI")
    log("=" * 68)
    log(f"📂 1688 human   : {bilingual_path}")
    log(f"📂 Tiki units   : {units_input_path}")
    log(f"📂 Prior JSON   : {prior_path}")
    log(f"📂 Prior CSV    : {prior_csv_path}")
    log(f"📂 Units output : {units_output_path}")
    log(f"📂 Apply stats  : {stats_path}")

    bilingual = pd.read_parquet(bilingual_path)
    if args.fit_limit:
        bilingual = bilingual.head(args.fit_limit).copy()
        log(f"⚠️  Giới hạn fit {args.fit_limit:,} dòng 1688")

    progress = None if args.quiet else log
    pairs = extract_human_length_pairs(bilingual, progress=progress)
    config = LengthPriorConfig(
        min_group_samples=args.min_group_samples,
        min_category_samples=args.min_category_samples,
    )
    prior = fit_length_prior(pairs, config=config, progress=progress)

    prior_payload = dict(prior.stats)
    prior_payload["bilingual_input"] = str(bilingual_path)
    prior_payload["groups"] = prior.table.to_dict("records")
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    with prior_path.open("w", encoding="utf-8") as handle:
        json.dump(prior_payload, handle, ensure_ascii=False, indent=2)
    write_prior_table(prior.table, prior_csv_path)

    units = pd.read_parquet(units_input_path)
    if args.limit:
        units = units.head(args.limit).copy()
        log(f"⚠️  Giới hạn apply {args.limit:,} Tiki unit")
    applied = apply_length_prior(units, prior, config=config, progress=progress)

    apply_stats = dict(applied.stats)
    apply_stats.update({
        "bilingual_input": str(bilingual_path),
        "tiki_units_input": str(units_input_path),
        "units_output": str(units_output_path),
        "prior_json": str(prior_path),
        "prior_csv": str(prior_csv_path),
    })
    units_output_path.parent.mkdir(parents=True, exist_ok=True)
    applied.units.to_parquet(units_output_path, index=False, compression="snappy")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(apply_stats, handle, ensure_ascii=False, indent=2)

    overall = prior.stats["overall"]
    by_unit = prior.stats["by_unit_type"]
    log(f"\n✅ Đã lưu {prior_path}")
    log(f"   Human pairs     : {prior.stats['n_pairs']:,}")
    for unit_type, count in prior.stats["pair_counts"].items():
        log(f"   {unit_type:<12} {count:>9,}")
    log(f"   Ratio VI/ZH p50 : {overall['ratio_vi_zh_p50']:.3f}")
    for unit_type in ("title", "attribute", "description"):
        if unit_type in by_unit:
            log(
                f"   {unit_type:<12} n={by_unit[unit_type]['n']:>7,} "
                f"ratio p50={by_unit[unit_type]['ratio_vi_zh_p50']:.3f}"
            )
    log(f"✅ Đã lưu {units_output_path} ({apply_stats['n_units']:,} dòng)")
    log(f"   Expected ZH p50 : {apply_stats['expected_zh_chars']['p50']}")
    log(f"   Expected ZH p05 : {apply_stats['expected_zh_chars']['p05']}")
    log(f"   Expected ZH p95 : {apply_stats['expected_zh_chars']['p95']}")
    log("   Scope counts:")
    for scope, count in apply_stats["scope_counts"].items():
        log(f"     {scope:<28} {count:>9,}")
    log(f"   💾 {prior_csv_path} ({len(prior.table):,} group)")
    log(f"   💾 {stats_path}")


def _write_prior_v2_json(
    prior,
    pairs_result,
    path: Path,
    *,
    bilingual_path: Path,
) -> None:
    payload = dict(prior.stats)
    payload["bilingual_input"] = str(bilingual_path)
    payload["split_audit"] = pairs_result.stats
    payload["groups"] = prior.table.to_dict("records")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_split_audit_v2(
    audit: pd.DataFrame,
    *,
    mismatch_path: Path,
    sample_path: Path,
    summary_path: Path,
    sample_size: int,
    seed: int,
) -> dict:
    summary = (
        audit.groupby("split_status", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    total = int(summary["n"].sum()) if not summary.empty else 0
    summary["pct"] = (summary["n"] / total if total else 0.0).round(4)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    mismatch = audit[audit["split_status"] == "mismatch"].copy()
    mismatch_path.parent.mkdir(parents=True, exist_ok=True)
    mismatch.to_csv(mismatch_path, index=False, encoding="utf-8-sig")

    sample = (
        audit.sample(n=sample_size, random_state=seed)
        if len(audit) > sample_size
        else audit
    )
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    return {
        "summary_path": str(summary_path),
        "mismatch_path": str(mismatch_path),
        "sample_path": str(sample_path),
        "n_mismatch": int(len(mismatch)),
        "n_sample": int(len(sample)),
        "summary": [
            {
                "split_status": str(row["split_status"]),
                "n": int(row["n"]),
                "pct": float(row["pct"]),
            }
            for row in summary.to_dict("records")
        ],
    }


def _v2_config(args: argparse.Namespace, fit_split: str) -> LengthPriorV2Config:
    return LengthPriorV2Config(
        min_group_samples=args.min_group_samples,
        min_category_samples=args.min_category_samples,
        min_value_class_samples=args.min_value_class_samples,
        min_value_class_category_samples=args.min_value_class_category_samples,
        split_seed=args.split_seed,
        fit_split=fit_split,
    )


def command_tiki_length_prior_v2(args: argparse.Namespace) -> None:
    bilingual_path = resolve_path(args.bilingual, CADO_PARQUET)
    tiki_units_path = resolve_path(args.tiki_units, TIKI_BT_UNITS_PARQUET)
    prior_train_path = resolve_path(args.prior_train_output, LENGTH_PRIOR_V2_TRAIN_JSON)
    prior_all_path = resolve_path(args.prior_all_output, LENGTH_PRIOR_V2_ALL_JSON)
    holdout_path = resolve_path(args.holdout_output, LENGTH_PRIOR_V2_HOLDOUT_JSON)
    prior_csv_path = resolve_path(args.prior_csv, LENGTH_PRIOR_V2_CSV)
    prior_train_csv_path = resolve_path(args.prior_train_csv, LENGTH_PRIOR_V2_TRAIN_CSV)
    mismatch_path = resolve_path(args.split_mismatch_output, LENGTH_PRIOR_V2_SPLIT_MISMATCH_CSV)
    sample_path = resolve_path(args.split_sample_output, LENGTH_PRIOR_V2_SPLIT_SAMPLE_CSV)
    split_summary_path = resolve_path(args.split_summary_output, LENGTH_PRIOR_V2_SPLIT_SUMMARY_CSV)
    units_output_path = resolve_path(args.units_output, TIKI_UNITS_LENGTH_V2_PARQUET)
    stats_path = resolve_path(args.stats_output, TIKI_LENGTH_PRIOR_V2_STATS_JSON)

    log("=" * 68)
    log("📏 PHASE A — LENGTH PRIOR V2 (VALUE-ONLY / CLASS / CJK / SPLIT-SAFE)")
    log("=" * 68)
    log(f"📂 1688 human   : {bilingual_path}")
    log(f"📂 Tiki units   : {tiki_units_path}")
    log(f"📂 Prior train  : {prior_train_path}")
    log(f"📂 Prior all    : {prior_all_path}")
    log(f"📂 Holdout      : {holdout_path}")
    log(f"📂 Units v2     : {units_output_path}")
    log(f"📂 Apply stats  : {stats_path}")

    bilingual = pd.read_parquet(bilingual_path)
    if args.fit_limit:
        bilingual = bilingual.head(args.fit_limit).copy()
        log(f"⚠️  Giới hạn fit {args.fit_limit:,} dòng 1688")

    base_config = _v2_config(args, "all")
    progress = None if args.quiet else log
    extracted = extract_human_value_pairs(
        bilingual,
        config=base_config,
        progress=progress,
    )
    pairs = extracted.pairs
    log(
        f"\n✅ Value pairs: {extracted.stats['n_value_pairs']:,} "
        f"+ title {extracted.stats['n_title_pairs']:,} "
        f"(group audit {extracted.stats['n_group_pairs']:,})"
    )
    log(f"   Split status: {extracted.stats['split_status_counts']}")
    log(f"   Split counts: {extracted.stats['split_counts']}")

    train_config = _v2_config(args, "train")
    all_config = _v2_config(args, "all")
    train_prior = fit_length_prior_v2(
        pairs, config=train_config, progress=progress
    )
    all_prior = fit_length_prior_v2(
        pairs, config=all_config, progress=progress
    )

    _write_prior_v2_json(
        train_prior,
        extracted,
        prior_train_path,
        bilingual_path=bilingual_path,
    )
    _write_prior_v2_json(
        all_prior,
        extracted,
        prior_all_path,
        bilingual_path=bilingual_path,
    )
    prior_train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    train_prior.table.to_csv(
        prior_train_csv_path, index=False, encoding="utf-8-sig"
    )
    prior_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_prior.table.to_csv(prior_csv_path, index=False, encoding="utf-8-sig")

    split_audit_info = _write_split_audit_v2(
        extracted.split_audit,
        mismatch_path=mismatch_path,
        sample_path=sample_path,
        summary_path=split_summary_path,
        sample_size=args.sample_size,
        seed=args.split_seed,
    )

    holdout = {
        "fit_split": "train",
        "train_prior": str(prior_train_path),
        "splits": {},
    }
    for split in ("dev", "test"):
        holdout["splits"][split] = evaluate_length_prior_v2(
            pairs,
            train_prior,
            split=split,
            config=train_config,
        )
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with holdout_path.open("w", encoding="utf-8") as handle:
        json.dump(holdout, handle, ensure_ascii=False, indent=2)

    units = pd.read_parquet(tiki_units_path)
    if args.limit:
        units = units.head(args.limit).copy()
        log(f"⚠️  Giới hạn apply {args.limit:,} Tiki unit")
    applied = apply_length_prior_v2(
        units,
        all_prior,
        config=all_config,
        progress=progress,
    )
    apply_stats = dict(applied.stats)
    apply_stats.update({
        "bilingual_input": str(bilingual_path),
        "tiki_units_input": str(tiki_units_path),
        "units_output": str(units_output_path),
        "prior_used": str(prior_all_path),
        "prior_train": str(prior_train_path),
        "holdout_metrics": str(holdout_path),
        "split_audit": split_audit_info,
    })
    units_output_path.parent.mkdir(parents=True, exist_ok=True)
    applied.units.to_parquet(
        units_output_path, index=False, compression="snappy"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(apply_stats, handle, ensure_ascii=False, indent=2)

    log("\n" + "=" * 68)
    log("✅ PHASE A HOÀN TẤT")
    log("=" * 68)
    log(f"Train prior pairs : {train_prior.stats['n_pairs']:,}")
    log(f"All prior pairs   : {all_prior.stats['n_pairs']:,}")
    log(f"Prior groups all  : {all_prior.stats['n_groups']:,}")
    for split in ("dev", "test"):
        metrics = holdout["splits"][split]
        log(
            f"Holdout {split:<4}    : n={metrics.get('n', 0):,} "
            f"coverage={metrics.get('overall', {}).get('coverage_value_p05_p95', 0):.3f} "
            f"MAE={metrics.get('overall', {}).get('mae_value_chars', 0):.3f}"
        )
    log(f"Tiki units        : {apply_stats['n_units']:,}")
    log(f"Expected value p50: {apply_stats['expected_zh_value_chars']['p50']}")
    log(f"Expected text p50 : {apply_stats['expected_zh_text_chars']['p50']}")
    log(f"Scopes            : {apply_stats['scope_counts']}")
    log(f"Confidence        : {apply_stats['confidence_counts']}")
    log(f"\n💾 {prior_train_path}")
    log(f"💾 {prior_all_path}")
    log(f"💾 {holdout_path}")
    log(f"💾 {prior_csv_path}")
    log(f"💾 {prior_train_csv_path}")
    log(f"💾 {split_summary_path}")
    log(f"💾 {mismatch_path}")
    log(f"💾 {sample_path}")
    log(f"💾 {units_output_path}")
    log(f"💾 {stats_path}")


def command_tiki_length_prior_v3(args: argparse.Namespace) -> None:
    from tiki_mining.tiki_length_prior_v3 import run_from_args

    run_from_args(args)


def command_tiki_bt_prepare(args: argparse.Namespace) -> None:
    from tiki_mining.tiki_bt_prepare import run_from_args

    run_from_args(args)


def command_stats(args: argparse.Namespace) -> None:
    from cado_core.dictionary import Cedict, Whitelist

    log("=" * 68)
    log("📊 THỐNG KÊ DỮ LIỆU")
    log("=" * 68)

    for path in (
        RAW_PARQUET,
        CADO_PARQUET,
        BITEXT_READY_PARQUET,
        BitextConfig().output_parquet,
        TIKI_PRODUCTS_PARQUET,
        TIKI_BT_UNITS_PARQUET,
        TIKI_UNITS_LENGTH_PARQUET,
        TIKI_UNITS_LENGTH_V2_PARQUET,
        TIKI_UNITS_LENGTH_V3_PARQUET,
        TIKI_BT_UNITS_PLAN_A_PARQUET,
    ):
        if path.exists():
            frame = pd.read_parquet(path)
            log(f"\n{path}")
            log(f"  shape: {frame.shape[0]:,} × {frame.shape[1]}")
            if "final_label" in frame.columns:
                log(f"  labels: {frame['final_label'].value_counts().to_dict()}")
            for source, cleaned in (
                (TITLE_ZH, TITLE_ZH_CLEANED),
                (DESC_ZH, DESC_ZH_CLEANED),
                (TITLE_VI, TITLE_VI_CLEANED),
                (DESC_VI, DESC_VI_CLEANED),
            ):
                if source in frame.columns and cleaned in frame.columns:
                    changed = (
                        frame[cleaned].astype(str) != frame[source].astype(str)
                    ).sum()
                    log(f"  {cleaned} changed: {changed:,}")
        else:
            log(f"\n{path} — chưa tồn tại")

    whitelist = Whitelist.load()
    cedict = Cedict.load()
    log(f"\nWhitelist: {len(whitelist):,} từ unique — {whitelist.categories()}")
    log(f"CC-CEDICT: {cedict.entries:,} entries, {cedict.english_keys:,} English keys")


# ═══════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════

def _add_cado_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", help="Parquet đầu vào (mặc định data/raw/bilingual_zh_vi.parquet)")
    parser.add_argument("--output", help="Parquet đầu ra (mặc định data/cleaned/bilingual_zh_vi_cado.parquet)")
    parser.add_argument("--columns", default="title_zh",
                        help="Cột ZH cần làm sạch, phân tách bởi dấu phẩy "
                             "(mặc định title_zh; thêm description_zh nếu muốn)")
    parser.add_argument("--loanword-column", default="title_zh",
                        help="Cột dùng để phát hiện từ mượn (mặc định title_zh)")
    parser.add_argument("--max-tier", type=int, default=4, choices=range(1, 6),
                        help="Dừng ở tầng nào (1..5, mặc định 4)")
    parser.add_argument("--tier5", action="store_true", help="Bật T5 bge-m3 (tải model ~2.2GB)")
    parser.add_argument("--no-loanwords", action="store_true", help="Tắt T3 auto loanword")
    parser.add_argument("--no-vi", action="store_true", help="Không chạy CADO-VI")
    parser.add_argument("--no-occurrences", action="store_true",
                        help="Không lưu CSV từng occurrence")
    parser.add_argument("--device", default="cpu", help="Thiết bị cho T5 (cpu/cuda/mps)")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")


def _add_bitext_args(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    def name(value: str) -> str:
        return f"--{prefix}{value}"

    dest = prefix.replace("-", "_")
    parser.add_argument(name("input"), dest=f"{dest}input",
                        help="Parquet đầu vào (mặc định CADO output)")
    parser.add_argument(name("output"), dest=f"{dest}output",
                        help="Parquet scored đầu ra")
    parser.add_argument(name("ready-output"), dest=f"{dest}ready_output",
                        help="Parquet preprocessed đầu ra")
    parser.add_argument(name("preprocess-only"), dest=f"{dest}preprocess_only",
                        action="store_true", help="Chỉ preprocess, không cần model")
    parser.add_argument(name("skip-cosine"), dest=f"{dest}skip_cosine",
                        action="store_true", help="Dùng cột cosine_bge_m3 có sẵn")
    parser.add_argument(name("label-mode"), dest=f"{dest}label_mode",
                        choices=("cosine", "hybrid"), default="cosine",
                        help="cosine = chỉ ngưỡng cosine (mặc định); "
                             "hybrid = cosine + hybrid_score")
    parser.add_argument(name("save-ready"), dest=f"{dest}save_ready",
                        action="store_true", help="Lưu thêm file preprocessed")
    parser.add_argument(name("entity-signals"), dest=f"{dest}entity_signals",
                        action="store_true", help="Tính 7 entity signals")
    parser.add_argument(name("entity-embedding"), dest=f"{dest}entity_embedding",
                        action="store_true", help="Dùng MiniLM cho S4b (chậm hơn)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline song ngữ Trung–Việt: CADO + Bitext Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cado = subparsers.add_parser("cado", help="Làm sạch từ tiếng Anh trong ZH/VI")
    _add_cado_args(cado)
    cado.set_defaults(func=command_cado)

    bitext = subparsers.add_parser("bitext", help="Preprocess + chấm điểm cặp song ngữ")
    _add_bitext_args(bitext)
    bitext.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")
    bitext.set_defaults(func=command_bitext)

    all_parser = subparsers.add_parser("all", help="Chạy CADO rồi Bitext")
    _add_cado_args(all_parser)
    _add_bitext_args(all_parser, prefix="bitext-")
    all_parser.add_argument("--no-bitext", action="store_true", help="Chỉ chạy CADO")
    all_parser.set_defaults(func=command_all)

    stats = subparsers.add_parser("stats", help="In thống kê nhanh các artifact")
    stats.set_defaults(func=command_stats)

    tiki_eda = subparsers.add_parser(
        "tiki-eda",
        help="EDA toàn bộ Tiki, xuất parquet product-level sẵn sàng BT",
    )
    tiki_eda.add_argument(
        "--input",
        help="Parquet Tiki đầu vào (mặc định data/cleaned/tiki_vi_cado.parquet)",
    )
    tiki_eda.add_argument(
        "--output",
        help="Parquet EDA đầu ra (mặc định data/cleaned/tiki_eda.parquet)",
    )
    tiki_eda.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")
    tiki_eda.set_defaults(func=command_tiki_eda)

    tiki_profile = subparsers.add_parser(
        "tiki-profile",
        help="Phase 1: profile title + description key-value, xuất JSON/CSV",
    )
    tiki_profile.add_argument(
        "--input",
        help="Parquet Tiki đầu vào (mặc định data/cleaned/tiki_vi_cado.parquet)",
    )
    tiki_profile.add_argument(
        "--profile-output",
        help="JSON profile đầu ra (mặc định data/cleaned/tiki_profile.json)",
    )
    tiki_profile.add_argument(
        "--key-inventory-output",
        help="CSV key inventory (mặc định csv/tiki/tiki_key_inventory.csv)",
    )
    tiki_profile.add_argument(
        "--parse-errors-output",
        help="CSV phần description lỗi parse (mặc định csv/tiki/tiki_parse_errors.csv)",
    )
    tiki_profile.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")
    tiki_profile.set_defaults(func=command_tiki_profile)

    tiki_units = subparsers.add_parser(
        "tiki-units",
        help="Phase 1: tạo product-level + pair-level BT units, không gọi MT",
    )
    tiki_units.add_argument(
        "--input",
        help="Parquet Tiki đầu vào (mặc định data/cleaned/tiki_vi_cado.parquet)",
    )
    tiki_units.add_argument(
        "--products-output",
        help="Parquet product-level (mặc định data/cleaned/tiki_products.parquet)",
    )
    tiki_units.add_argument(
        "--units-output",
        help="Parquet pair-level (mặc định data/cleaned/tiki_bt_units.parquet)",
    )
    tiki_units.add_argument(
        "--stats-output",
        help="JSON stats (mặc định data/cleaned/tiki_units_stats.json)",
    )
    tiki_units.add_argument(
        "--key-map",
        help="Glossary VI→ZH nếu có (mặc định csv/tiki/tiki_key_map.csv)",
    )
    tiki_units.add_argument("--limit", type=int, help="Chỉ xử lý N dòng đầu (để test)")
    tiki_units.set_defaults(func=command_tiki_units)

    length_prior = subparsers.add_parser(
        "tiki-length-prior",
        help="Phase 1.5: học length prior từ 1688 human, gắn khoảng độ dài ZH cho Tiki",
    )
    length_prior.add_argument(
        "--bilingual",
        help="Parquet 1688 human ZH–VI (mặc định data/cleaned/bilingual_zh_vi_cado.parquet)",
    )
    length_prior.add_argument(
        "--tiki-units",
        help="Parquet Tiki units (mặc định data/cleaned/tiki_bt_units.parquet)",
    )
    length_prior.add_argument(
        "--prior-output",
        help="JSON prior (mặc định data/cleaned/length_prior_1688.json)",
    )
    length_prior.add_argument(
        "--prior-csv",
        help="CSV bảng prior (mặc định csv/length_prior/length_prior_by_bucket.csv)",
    )
    length_prior.add_argument(
        "--units-output",
        help="Parquet Tiki units đã gắn prior (mặc định data/cleaned/tiki_bt_units_length.parquet)",
    )
    length_prior.add_argument(
        "--stats-output",
        help="JSON thống kê apply (mặc định data/cleaned/tiki_length_prior_stats.json)",
    )
    length_prior.add_argument(
        "--min-group-samples",
        type=int,
        default=20,
        help="Số cặp tối thiểu cho group unit_type+bucket (mặc định 20)",
    )
    length_prior.add_argument(
        "--min-category-samples",
        type=int,
        default=100,
        help="Số cặp tối thiểu cho group có category (mặc định 100)",
    )
    length_prior.add_argument("--fit-limit", type=int, help="Chỉ dùng N dòng 1688 đầu (để test)")
    length_prior.add_argument("--limit", type=int, help="Chỉ áp dụng N Tiki unit đầu (để test)")
    length_prior.add_argument("--quiet", action="store_true", help="Không in tiến độ")
    length_prior.set_defaults(func=command_tiki_length_prior)

    length_prior_v2 = subparsers.add_parser(
        "tiki-length-prior-v2",
        help="Phase A v2: value-only + value_class + CJK + split-safe",
    )
    length_prior_v2.add_argument(
        "--bilingual",
        help="Parquet 1688 human ZH–VI (mặc định data/cleaned/bilingual_zh_vi_cado.parquet)",
    )
    length_prior_v2.add_argument(
        "--tiki-units",
        help="Parquet Tiki units (mặc định data/cleaned/tiki_bt_units.parquet)",
    )
    length_prior_v2.add_argument(
        "--prior-train-output",
        help="JSON prior train (mặc định data/cleaned/length_prior_1688_v2_train.json)",
    )
    length_prior_v2.add_argument(
        "--prior-all-output",
        help="JSON prior all (mặc định data/cleaned/length_prior_1688_v2_all.json)",
    )
    length_prior_v2.add_argument(
        "--holdout-output",
        help="JSON metrics dev/test (mặc định data/cleaned/length_prior_1688_v2_holdout_metrics.json)",
    )
    length_prior_v2.add_argument(
        "--prior-csv",
        help="CSV prior all (mặc định csv/length_prior_v2/prior_value_by_class_bucket.csv)",
    )
    length_prior_v2.add_argument(
        "--prior-train-csv",
        help="CSV prior train (mặc định csv/length_prior_v2/prior_value_by_class_bucket_train.csv)",
    )
    length_prior_v2.add_argument(
        "--split-mismatch-output",
        help="CSV các group split mismatch",
    )
    length_prior_v2.add_argument(
        "--split-sample-output",
        help="CSV sample audit tách value",
    )
    length_prior_v2.add_argument(
        "--split-summary-output",
        help="CSV summary split status",
    )
    length_prior_v2.add_argument(
        "--units-output",
        help="Parquet Tiki units v2 (mặc định data/cleaned/tiki_bt_units_length_v2.parquet)",
    )
    length_prior_v2.add_argument(
        "--stats-output",
        help="JSON stats apply v2 (mặc định data/cleaned/tiki_length_prior_v2_stats.json)",
    )
    length_prior_v2.add_argument("--min-group-samples", type=int, default=20)
    length_prior_v2.add_argument("--min-category-samples", type=int, default=100)
    length_prior_v2.add_argument("--min-value-class-samples", type=int, default=50)
    length_prior_v2.add_argument(
        "--min-value-class-category-samples", type=int, default=100
    )
    length_prior_v2.add_argument("--split-seed", type=int, default=42)
    length_prior_v2.add_argument("--sample-size", type=int, default=2000)
    length_prior_v2.add_argument(
        "--fit-limit", type=int, help="Chỉ dùng N dòng 1688 đầu (để test)"
    )
    length_prior_v2.add_argument(
        "--limit", type=int, help="Chỉ áp dụng N Tiki unit đầu (để test)"
    )
    length_prior_v2.add_argument(
        "--quiet", action="store_true", help="Không in tiến độ"
    )
    length_prior_v2.set_defaults(func=command_tiki_length_prior_v2)

    length_prior_v3 = subparsers.add_parser(
        "tiki-length-prior-v3",
        help="Phase B/C: alignment fallback + hierarchical title smoothing",
    )
    length_prior_v3.add_argument("--bilingual")
    length_prior_v3.add_argument("--tiki-units")
    length_prior_v3.add_argument("--prior-train-output")
    length_prior_v3.add_argument("--prior-all-output")
    length_prior_v3.add_argument("--holdout-output")
    length_prior_v3.add_argument("--prior-csv")
    length_prior_v3.add_argument("--prior-train-csv")
    length_prior_v3.add_argument("--alignment-map-output")
    length_prior_v3.add_argument("--alignment-sample-output")
    length_prior_v3.add_argument("--alignment-summary-output")
    length_prior_v3.add_argument("--phase-eval-output")
    length_prior_v3.add_argument("--phase-eval-md-output")
    length_prior_v3.add_argument("--units-output")
    length_prior_v3.add_argument("--stats-output")
    length_prior_v3.add_argument("--min-group-samples", type=int, default=20)
    length_prior_v3.add_argument("--min-category-samples", type=int, default=100)
    length_prior_v3.add_argument("--min-value-class-samples", type=int, default=50)
    length_prior_v3.add_argument(
        "--min-value-class-category-samples", type=int, default=100
    )
    length_prior_v3.add_argument("--alignment-min-freq", type=int, default=20)
    length_prior_v3.add_argument("--alignment-min-share", type=float, default=0.60)
    length_prior_v3.add_argument("--alignment-sample-size", type=int, default=500)
    length_prior_v3.add_argument("--title-smoothing-k", type=int, default=20)
    length_prior_v3.add_argument(
        "--title-interval-inflation", type=float, default=0.5
    )
    length_prior_v3.add_argument("--split-seed", type=int, default=42)
    length_prior_v3.add_argument("--fit-limit", type=int)
    length_prior_v3.add_argument("--limit", type=int)
    length_prior_v3.add_argument("--no-key-fallback", action="store_true")
    length_prior_v3.add_argument("--quiet", action="store_true")
    length_prior_v3.set_defaults(func=command_tiki_length_prior_v3)

    bt_prepare = subparsers.add_parser(
        "tiki-bt-prepare",
        help="Plan A: key map + routing + BT input/cache, chưa gọi MT",
    )
    bt_prepare.add_argument("--units")
    bt_prepare.add_argument("--products")
    bt_prepare.add_argument("--alignment-map")
    bt_prepare.add_argument("--key-map")
    bt_prepare.add_argument("--key-map-review")
    bt_prepare.add_argument("--units-output")
    bt_prepare.add_argument("--input-output")
    bt_prepare.add_argument("--cache-output")
    bt_prepare.add_argument("--product-state-output")
    bt_prepare.add_argument("--routing-output")
    bt_prepare.add_argument("--sample-output")
    bt_prepare.add_argument("--stats-output")
    bt_prepare.add_argument("--min-key-freq", type=int, default=20)
    bt_prepare.add_argument("--min-key-share", type=float, default=0.60)
    bt_prepare.add_argument("--separator-chars", type=int, default=2)
    bt_prepare.add_argument("--limit", type=int)
    bt_prepare.add_argument("--quiet", action="store_true")
    bt_prepare.set_defaults(func=command_tiki_bt_prepare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
