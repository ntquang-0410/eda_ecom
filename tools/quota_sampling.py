#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quota-based sampling cho weight survey A/B.

Mỗi nhóm g có target token T_g = s_g * budget.
Trong từng nhóm, sample ĐỀU các dòng (có hoàn lại) đến khi đủ T_g.
Không dùng xác suất p_i ∝ t_i (cách cũ gây lệch bình phương độ dài).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def natural_shares(train: pd.DataFrame) -> dict[str, float]:
    mass = train.groupby("group")["eff_tokens"].sum()
    total = float(mass.sum())
    return {str(g): float(v) / total for g, v in mass.items()}


def quota_indices(train: pd.DataFrame, config: dict | None, seed: int,
                  budget: int) -> tuple[list[int], dict[str, float]]:
    rng = np.random.default_rng(seed)
    groups = train["group"].astype(str).to_numpy()
    eff = train["eff_tokens"].to_numpy(dtype=float)
    weights = (train["sample_weight"].to_numpy(dtype=float)
               if "sample_weight" in train.columns else None)
    shares = natural_shares(train) if config is None else {
        g: float(config.get(g, 0.0)) for g in set(groups)
    }
    chosen: list[int] = []
    actual: dict[str, float] = {}
    for group in sorted(set(groups)):
        target = float(shares.get(group, 0.0)) * budget
        if target <= 0:
            actual[group] = 0.0
            continue
        pool = np.flatnonzero(groups == group)
        if len(pool) == 0:
            actual[group] = 0.0
            continue
        total = 0.0
        if weights is not None:
            w = np.clip(weights[pool], 1e-6, None)
            p = w / w.sum()
        else:
            p = None
        while total < target:
            idx = int(rng.choice(pool, p=p)) if p is not None else int(pool[rng.integers(0, len(pool))])
            chosen.append(idx)
            total += float(eff[idx])
        actual[group] = total
    # xáo trộn để group không bị xếp liên tiếp
    order = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order]
    return chosen, {g: t / max(1.0, sum(actual.values())) for g, t in actual.items()}


def balance_plan(train: pd.DataFrame, config: dict | None, seed: int,
                 budget: int, steps: int) -> tuple[list[list[int]], dict[str, float]]:
    """Chia stream quota thành đúng `steps` step, mỗi step ~budget/steps token."""
    stream, actual = quota_indices(train, config, seed, budget)
    target = budget / steps
    plan: list[list[int]] = []
    cur: list[int] = []
    cur_tok = 0.0
    for idx in stream:
        cur.append(idx)
        cur_tok += float(train["eff_tokens"].iloc[idx])
        if cur_tok >= target and len(plan) < steps - 1:
            plan.append(cur)
            cur, cur_tok = [], 0.0
    if cur:
        plan.append(cur)
    # nếu thiếu step do overshoot, chia nhỏ step cuối
    while len(plan) < steps:
        last = plan.pop()
        if len(last) <= 1:
            plan.extend([last, last])
            continue
        mid = len(last) // 2
        plan.extend([last[:mid], last[mid:]])
    return plan[:steps], actual


def old_probabilities(train: pd.DataFrame, config: dict | None) -> np.ndarray:
    """Cách cũ p_i ∝ w_g * t_i (dùng để so sánh, KHÔNG dùng để train)."""
    eff = train["eff_tokens"].to_numpy(dtype=float)
    groups = train["group"].astype(str).to_numpy()
    mass = train.groupby("group")["eff_tokens"].sum()
    if config is None:
        shares = natural_shares(train)
    else:
        shares = {g: float(config.get(g, 0.0)) for g in set(groups)}
    w = {g: shares.get(g, 0.0) / max(1.0, float(mass.get(g, 0.0))) for g in set(groups)}
    raw = np.array([w[g] for g in groups]) * eff
    return raw / raw.sum()


def old_sample(train: pd.DataFrame, config: dict | None, seed: int,
               budget: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    p = old_probabilities(train, config)
    eff = train["eff_tokens"].to_numpy(dtype=float)
    groups = train["group"].astype(str).to_numpy()
    totals = {}
    total = 0.0
    while total < budget:
        idx = int(rng.choice(len(train), p=p))
        total += float(eff[idx])
        g = groups[idx]
        totals[g] = totals.get(g, 0.0) + float(eff[idx])
    return {g: v / max(1.0, total) for g, v in totals.items()}
