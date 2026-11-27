#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn đánh giá canonical: index cố định 831 dòng + artefact per-row + manifest.

Quy ước:
- Test set paired cố định: 1688 = 231, Tiki = 627 (data/cleaned/eval_paired_858.parquet).
- Mọi run join theo eval_id; hyp rỗng ⇒ điểm 0 (không lọc dòng).
- Artefact per-row: eval_id, dataset, stratum, hyp, ref, comet_da, comet_qe, bleu, chrf, nonempty.
- Manifest mỗi run: commit_hash, config_hash, seed, token_budget, adapter_path...
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data/cleaned/eval_paired_858.parquet"
INDEX_META = ROOT / "data/cleaned/eval_paired_858.json"
CANON = ROOT / ".artifacts/canonical"
SCHEMA = ["eval_id", "dataset", "stratum", "hyp", "ref",
          "comet_da", "comet_qe", "bleu", "chrf", "nonempty"]


def index_hash(ev: pd.DataFrame) -> str:
    ids = "\n".join(sorted(ev.eval_id.astype(str)))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()


def load_index(create: bool = True) -> pd.DataFrame:
    if not INDEX_PATH.exists():
        if not create:
            raise FileNotFoundError(INDEX_PATH)
        src = pd.read_parquet(ROOT / "data/cleaned/eval_F_v2.parquet")
        ev = src[["eval_id", "dataset", "pair_type", "product_id", "source",
                  "reference", "max_new"]].copy()
        ev = ev.rename(columns={"pair_type": "stratum"})
        ev["eval_id"] = ev.eval_id.astype(str)
        assert len(ev) == 858, len(ev)
        assert ev.dataset.value_counts().to_dict() == {"tiki": 627, "1688": 231}
        ev = ev.sort_values(["dataset", "eval_id"]).reset_index(drop=True)
        ev.to_parquet(INDEX_PATH, index=False)
        INDEX_META.write_text(json.dumps({
            "n": 858, "by_dataset": {"1688": 231, "tiki": 627},
            "by_stratum": {f"{d}/{t}": int(n) for (d, t), n in
                           ev.groupby(["dataset", "stratum"]).size().items()},
            "eval_ids_sha256": index_hash(ev),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    ev = pd.read_parquet(INDEX_PATH)
    ev["eval_id"] = ev.eval_id.astype(str)
    meta = json.loads(INDEX_META.read_text(encoding="utf-8"))
    assert index_hash(ev) == meta["eval_ids_sha256"], "index hash mismatch"
    return ev


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def sent_metrics(hyp: str, ref: str) -> tuple[float, float]:
    if not str(hyp).strip() or not str(ref).strip():
        return 0.0, 0.0
    b = sacrebleu.sentence_bleu(str(hyp), [str(ref)], tokenize="zh").score
    c = sacrebleu.sentence_chrf(str(hyp), [str(ref)]).score
    return float(b), float(c)


def build_artefact(cell: str, seed: int, hyp: pd.DataFrame,
                   da: pd.DataFrame | None, qe: pd.DataFrame | None,
                   manifest: dict) -> tuple[pd.DataFrame, dict]:
    """Ghép hyp + score vào index cố định, zero-fill hyp rỗng, ghi parquet + manifest."""
    ev = load_index()
    h = hyp.copy()
    h["eval_id"] = h.eval_id.astype(str)
    h = h.drop_duplicates("eval_id")
    h = h.set_index("eval_id")
    da_map = dict(zip(da.eval_id.astype(str), da.score)) if da is not None else {}
    qe_map = dict(zip(qe.eval_id.astype(str), qe.score)) if qe is not None else {}

    rows = []
    for r in ev.itertuples():
        hyp_txt = str(h.hypothesis.get(r.eval_id, "") or "")
        nonempty = bool(hyp_txt.strip())
        is_1688 = r.dataset == "1688"
        d = da_map.get(r.eval_id) if is_1688 else None
        q = None if is_1688 else qe_map.get(r.eval_id)
        if not nonempty:
            if is_1688:
                d = 0.0
            else:
                q = 0.0
        if is_1688:
            bleu, chrf = sent_metrics(hyp_txt, r.reference)
        else:
            bleu = chrf = float("nan")
        rows.append({"eval_id": r.eval_id, "dataset": r.dataset, "stratum": r.stratum,
                     "hyp": hyp_txt, "ref": str(r.reference) if is_1688 else "",
                     "comet_da": d if d is not None else float("nan"),
                     "comet_qe": q if q is not None else float("nan"),
                     "bleu": bleu, "chrf": chrf, "nonempty": nonempty})
    df = pd.DataFrame(rows)[SCHEMA]

    out = CANON / cell
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"seed{seed}.parquet"
    df.to_parquet(dst, index=False)

    cov = {ds: float(df[(df.dataset == ds) & df.nonempty].shape[0] / max(1, (df.dataset == ds).sum()))
           for ds in ("1688", "tiki")}
    man = dict(manifest)
    man.update({
        "cell": cell, "seed": seed, "eval_index": str(INDEX_PATH.relative_to(ROOT)),
        "eval_index_hash": json.loads(INDEX_META.read_text(encoding="utf-8"))["eval_ids_sha256"],
        "n_rows": int(len(df)), "coverage": cov,
        "missing_scores": {ds: int(df[(df.dataset == ds) & df.comet_da.isna()].shape[0]) if ds == "1688"
                           else int(df[(df.dataset == ds) & df.comet_qe.isna()].shape[0])
                           for ds in ("1688", "tiki")},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    (out / f"seed{seed}.manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return df, man


def summarise(df: pd.DataFrame, ds: str) -> dict:
    sub = df[df.dataset == ds]
    score_col = "comet_da" if ds == "1688" else "comet_qe"
    s = sub[score_col]
    return {"n": int(len(sub)), "coverage": float(sub.nonempty.mean()),
            "mean": float(s.mean()), "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0}
