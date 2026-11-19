# -*- coding: utf-8 -*-
"""3.5 Duong hoc OpenNMT tu train.log (khong train).

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p1_opennmt_hoc.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GOC = Path(__file__).resolve().parents[2]
R = GOC / "data" / "processed" / "ft_run"
RA = R / "phan_bien_v2"
HINH = GOC / "docs" / "phan_bien_vong2" / "hinh"

RE_STEP = re.compile(r"Step (\d+)/\s*\d+; acc:\s*([\d.]+); ppl:\s*([\d.]+)")
RE_VAL_PPL = re.compile(r"Validation perplexity:\s*([\d.]+)")
RE_VAL_ACC = re.compile(r"Validation accuracy:\s*([\d.]+)")
RE_CKPT = re.compile(r"model_step_(\d+)\.pt")


def doc_log(hg):
    p = R / "kaggle_onmt_that" / "onmt" / f"{hg}_train" / "train.log"
    if not p.exists():
        return None
    L = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    train, val, best = [], [], None
    i = 0
    while i < len(L):
        m = RE_STEP.search(L[i])
        if m:
            train.append({"step": int(m.group(1)), "acc": float(m.group(2)), "ppl": float(m.group(3))})
        if "Best model found at step" in L[i]:
            try:
                best = int(re.search(r"step (\d+)", L[i]).group(1))
            except Exception:
                pass
        i += 1
    # validation: ppl va acc nam hai dong lien tiep, checkpoint model_step_X.pt ngay sau
    last_ppl = None
    for j, x in enumerate(L):
        mp = RE_VAL_PPL.search(x)
        if mp:
            last_ppl = float(mp.group(1))
            continue
        ma = RE_VAL_ACC.search(x)
        if ma and last_ppl is not None:
            step = None
            for y in L[j:j + 8]:
                mc = RE_CKPT.search(y)
                if mc:
                    step = int(mc.group(1))
                    break
            if step is not None:
                val.append({"step": step, "ppl": last_ppl, "acc": float(ma.group(1))})
            last_ppl = None
    return {"huong": hg, "train": train, "val": val, "best_step": best,
            "so_dong_log": len(L), "tep": str(p.relative_to(GOC))}


def ve(d):
    hg = d["huong"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    t = d["train"]
    v = d["val"]
    if t:
        a1.plot([x["step"] for x in t], [x["ppl"] for x in t], label="train", color="#1f77b4")
        a2.plot([x["step"] for x in t], [x["acc"] for x in t], label="train", color="#1f77b4")
    if v:
        a1.plot([x["step"] for x in v], [x["ppl"] for x in v], "o-", label="dev", color="#d62728")
        a2.plot([x["step"] for x in v], [x["acc"] for x in v], "o-", label="dev", color="#d62728")
    if d.get("best_step"):
        for a in (a1, a2):
            a.axvline(d["best_step"], ls="--", lw=1, color="#555")
    a1.set_yscale("log")
    a1.set_title(f"{hg}: perplexity (log)")
    a2.set_title(f"{hg}: accuracy")
    for a in (a1, a2):
        a.set_xlabel("step")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    HINH.mkdir(parents=True, exist_ok=True)
    p = HINH / f"opennmt_{hg.replace('-', '')}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    RA.mkdir(parents=True, exist_ok=True)
    ra = {}
    for hg in ("zh-vi", "vi-zh"):
        d = doc_log(hg)
        if d is None:
            ra[hg] = {"loi": "khong co train.log"}
            continue
        d["hinh"] = str(ve(d).relative_to(GOC)) if d["val"] or d["train"] else None
        ra[hg] = d
        print(hg, "train", len(d["train"]), "dev", len(d["val"]), "best", d["best_step"], "hinh", d["hinh"])
    (RA / "opennmt_hoc.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
