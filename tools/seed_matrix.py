#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bảng seed × cell: đã có gì, còn thiếu gì để đạt ≥5 seed, chi phí GPU ước tính."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / ".artifacts/canonical"
TARGET_MAIN = [7, 42, 123, 2024, 2025]
TARGET_CONTROL = [7, 42, 123]

# cell -> (nhãn, giờ GPU/run ước tính, ghi chú)
CELLS = {
    "qwen8b_a": ("Qwen3-8B A-natural", 7.0, "500 step A100 + infer"),
    "qwen8b_b": ("Qwen3-8B B-balanced", 7.0, "500 step A100 + infer"),
    "qwen8b_a_balanced": ("Qwen3-8B A-balanced", 7.0, "thiếu score từng dòng"),
    "qwen8b_b_natural": ("Qwen3-8B B-natural", 7.0, "artefact tái lập 24/11"),
    "qwen4b_a": ("Qwen3-4B A-natural", 5.0, "500 step + infer"),
    "qwen4b_b": ("Qwen3-4B B-balanced", 4.5, "500 step + infer"),
    "qwen4b_a_balanced": ("Qwen3-4B A-balanced", 5.0, ""),
    "qwen4b_b_natural": ("Qwen3-4B B-natural", 4.5, ""),
    "ecomgpt_a": ("EcomGPT-7B A-natural", 2.2, "recipe token encoded"),
    "ecomgpt_b": ("EcomGPT-7B B-balanced", 2.2, ""),
    "ecomgpt_a_balanced": ("EcomGPT-7B A-balanced", 2.2, ""),
    "moses_a": ("Moses A", 0.5, "CPU"),
    "moses_b": ("Moses B", 0.5, "CPU"),
    "onmt_a": ("OpenNMT A", 1.0, ""),
    "onmt_b": ("OpenNMT B-natural", 1.0, ""),
    "nllb_a": ("NLLB-600M A", 7.0, "424 phút + infer"),
    "nllb_b": ("NLLB-600M B", 5.0, "282 phút + infer"),
}
CONTROL = {"qwen8b_a_balanced", "qwen4b_a_balanced", "qwen4b_b_natural",
           "ecomgpt_a_balanced", "nllb_a", "nllb_b", "onmt_b"}


def main() -> None:
    have = {}
    for d in CANON.glob("*"):
        seeds = sorted(int(p.stem.replace("seed", "")) for p in d.glob("seed*.parquet"))
        if seeds:
            have[d.name] = seeds

    rows = []
    for cell, (label, hours, note) in CELLS.items():
        seeds = have.get(cell, [])
        target = TARGET_CONTROL if cell in CONTROL else TARGET_MAIN
        miss = [s for s in target if s not in seeds]
        rows.append({"cell": cell, "label": label, "role": "control" if cell in CONTROL else "main",
                     "seeds_have": ",".join(map(str, seeds)) or "—", "n_have": len(seeds),
                     "seeds_missing": ",".join(map(str, miss)) or "—", "n_missing": len(miss),
                     "hours_missing": round(len(miss) * hours, 1), "note": note})
    df = pd.DataFrame(rows).sort_values(["role", "cell"])

    total = float(df.hours_missing.sum())
    main_miss = int(df[df.role == "main"].n_missing.sum())
    md = ["# Seed matrix — policy ≥5 seed/cell", "",
          f"Seed đích: cell chính {TARGET_MAIN}; control {TARGET_CONTROL}. "
          "Nguồn: `.artifacts/canonical/*/seed*.parquet`.",
          f"Ước tính còn thiếu: **{df.n_missing.sum()} run ≈ {total:.0f} giờ GPU**, "
          f"trong đó {main_miss} run thuộc cell chính (chưa tính thời gian chờ/quota).", "",
          "| Cell | Vai trò | Seed đã có | # | Seed thiếu | # | Giờ GPU còn thiếu | Ghi chú |",
          "|---|---|---|---:|---|---:|---:|---|"]
    for r in df.itertuples():
        md.append(f"| {r.label} | {r.role} | {r.seeds_have} | {r.n_have} | {r.seeds_missing} | "
                  f"{r.n_missing} | {r.hours_missing:.1f} | {r.note} |")
    md += ["", "## Thứ tự đề xuất", "",
           "1. **4 cell Qwen 4B** (A-natural, B-balanced) lên 5 seed — đây là nơi có tuyên bố so sánh A/B. "
           "Cần thêm 5 run ≈ 24 h.",
           "2. **Qwen 8B** thêm 4 seed mỗi cell ≈ 56 h — nặng nhất, chạy song song 2 account Kaggle sau khi quota reset.",
           "3. **EcomGPT** thêm 4 seed mỗi cell ≈ 18 h (nhanh, nhưng phải sửa ngân sách token trước — xem MATRIX §11.3).",
           "4. **NLLB** thêm 3 seed mỗi cell ≈ 36 h (control, có thể chạy sau).",
           "5. **Moses/OpenNMT** thêm 4 seed ≈ 12 h (CPU/rẻ, chạy cuối).",
           "",
           "## Lưu ý", "",
           "- Chi phí trên là giờ GPU thuần; thực tế cần thêm ~30% cho preemption, tải model, inference.",
           "- Kaggle: 2 account × 30 h GPU/tuần (reset thứ Bảy). Colab A100: phiên 12 h, hay bị thu hồi.",
           "- Với 5 seed, báo cáo mean ± std qua seed; paired bootstrap CI chỉ phủ variance eval, "
           "phải ghi rõ khi so sánh giữa cell.",
           "- Cell Qwen3-8B A-balanced hiện **không có score từng dòng** → phải chấm lại trước khi tính seed mean."]
    out = ROOT / "reports/SEED_MATRIX.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"missing runs={df.n_missing.sum()} hours={total:.1f}")
    print(df[["label", "seeds_have", "n_missing", "hours_missing"]].to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
