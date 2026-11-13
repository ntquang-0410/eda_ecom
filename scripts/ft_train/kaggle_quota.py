# -*- coding: utf-8 -*-
"""Quota Survey A tren Kaggle - ban sua loi 10/11/2026.

Ba loi da sua so voi ban screen cu:
1. DO DAI: loc theo DO DAI THAT (dung tokenizer + dung khuon chat) TRUOC khi
   lay mau, roi huan luyen dung tap do. Khong con chuyen "ke hoach cat 960 /
   luc chay bo mau": assert so mau bi bo = 0.
2. NGAN SACH: moi buoc dung ~8.000 token DICH; di qua danh sach dung mot lan,
   khong cycle. So buoc = tong token dich / 8.000 (tinh tu du lieu).
3. LOSS: chuan hoa theo token dich cua ca buoc (tong loss / tong token dich),
   khong chia theo so mau.

s_g tinh tren token DICH. Ngach tinh theo token dich. Tong ngan sach tinh theo
compute (~4M token ca prompt) de so sanh cong bang; tong token dich tuong ung
duoc tinh tu du lieu.

Khong doc test. Khong sua du lieu.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import kaggle_train as K

VAO = Path("/kaggle/input")
RA = Path("/kaggle/working")

CHI_DAN_TITLE = "Translate this product title from {SRC} to {TGT}. Output only the translation.\n\n{NGUON}"
CHI_DAN_CUM = (
    "Translate this product's attribute list from {SRC} to {TGT}. The attributes are separated by \"; \" "
    "and each is written as \"name: value\". Keep the same number and order of attributes, separated by \"; \", "
    "and output only the translation.\nProduct title (context only, do not translate): {TITLE}\n\n{NGUON}"
)
CHI_DAN_ATTR = "Translate this attribute from {SRC} to {TGT}. Product title (context only, do not translate): {TITLE}\n\n{NGUON}"

TEN_NGU = {"zh->vi": ("Chinese", "Vietnamese"), "vi->zh": ("Vietnamese", "Chinese")}
HUONG = {"1688": "zh->vi", "tiki": "vi->zh"}

# Khuôn lời dẫn kiểu Alpaca cho EcomGPT (theo cấu hình của Quang: có mục "### Input"); chi_dan = phần chỉ dẫn, nguon = văn bản nguồn.
ALPACA = ("Below is an instruction that describes a task, paired with an input that provides further context. "
          "Write a response that appropriately completes the request.\n\n### Instruction:\n{chi_dan}\n\n### Input:\n{nguon}\n\n### Response:")


def tao_khuon(tok, kieu="qwen"):
    """Trả {"lam": hàm mẫu -> chuỗi lời dẫn, "duoi": chuỗi nối sau bản dịch khi huấn luyện}.

    qwen: khuôn chat Qwen3 (tắt suy luận), bản dịch kết thúc bằng <|im_end|> và xuống dòng, đúng như mọi lần chạy từ trước.
    ecomgpt: khuôn Alpaca, bản dịch kết thúc bằng </s> (eos của tokenizer BLOOM)."""
    if kieu == "qwen":
        return {"lam": lambda m: tok.apply_chat_template(m["messages"][:1], tokenize=False,
                                                           add_generation_prompt=True, enable_thinking=False),
                "duoi": K.NHAC_EOS + "\n"}
    if kieu == "ecomgpt":
        if not tok.eos_token:
            raise RuntimeError("TOKENIZER_KHONG_CO_EOS")
        return {"lam": lambda m: ALPACA.format(chi_dan=m["chi_dan"], nguon=m["nguon"]), "duoi": tok.eos_token}
    raise RuntimeError(f"KHUON_LA:{kieu}")

CONFIGS = {
    "cum": {
        "natural": None,
        "balanced": {"1688_title": 25, "1688_description": 25, "tiki_title": 25, "tiki_description": 25},
        "title_heavy": {"1688_title": 40, "1688_description": 10, "tiki_title": 40, "tiki_description": 10},
        "desc_heavy": {"1688_title": 2, "1688_description": 55, "tiki_title": 3, "tiki_description": 40},
        "tiki_heavy": {"1688_title": 10, "1688_description": 15, "tiki_title": 35, "tiki_description": 40},
    },
    "pair": {
        "natural": None,
        "balanced": {"1688_title": 25, "1688_attribute": 25, "tiki_title": 25, "tiki_attribute": 25},
        "title_heavy": {"1688_title": 40, "1688_attribute": 20, "tiki_title": 25, "tiki_attribute": 15},
        "attr_heavy": {"1688_title": 10, "1688_attribute": 60, "tiki_title": 5, "tiki_attribute": 25},
        "tiki_heavy": {"1688_title": 10, "1688_attribute": 20, "tiki_title": 30, "tiki_attribute": 40},
    },
}
THU_TU = {
    "cum": ["1688_title", "1688_description", "tiki_title", "tiki_description"],
    "pair": ["1688_title", "1688_attribute", "tiki_title", "tiki_attribute"],
}


def tim_file(ten):
    ds = [p for p in VAO.rglob(ten) if p.is_file()]
    if not ds:
        raise RuntimeError(f"khong_thay:{ten}")
    return ds[0]


def doc_ref(tap, flt=0):
    return pd.read_csv(tim_file(f"train_ref_{tap}_flt{flt}.csv"), encoding="utf-8-sig")


def doc_eval(tap):
    return pd.read_csv(tim_file(f"eval_{tap}_sample.csv"), encoding="utf-8-sig")


def doc_pool(tap):
    return K.doc_mau(tap)


def map_pool(recs):
    return {(str(r["product_id"]), r["loai"]): r for r in recs}


def build_mau(ref, idx, pool_by, tap):
    """Dung messages tu train_ref/eval_ref + pool. Tra list mau cho xay_mau."""
    ra = []
    tach = {}
    idx = list(idx)
    for dem, j in enumerate(idx, 1):
        if dem % 50000 == 0:
            print(f"  build_mau {tap}: {dem}/{len(idx)}", flush=True)
        r = ref.iloc[int(j)]
        pid = str(r["product_id"])
        nhom = r["group"]
        loai_ref = r["loai"]  # title | description | attribute
        src = HUONG[nhom.split("_")[0]]
        s, t = TEN_NGU[src]
        title_rec = pool_by.get((pid, "title"))
        rec = pool_by.get((pid, "title" if loai_ref == "title" else "mo_ta"))
        if rec is None or title_rec is None:
            continue
        if loai_ref == "title":
            nguon, dich = rec["nguon"], rec["dich"]
            user = CHI_DAN_TITLE.format(SRC=s, TGT=t, NGUON=nguon)
            chi_dan = CHI_DAN_TITLE.format(SRC=s, TGT=t, NGUON="").rstrip()
            loai = "title"
        elif loai_ref == "description":
            nguon, dich = rec["nguon"], rec["dich"]
            user = CHI_DAN_CUM.format(SRC=s, TGT=t, TITLE=title_rec["nguon"], NGUON=nguon)
            chi_dan = CHI_DAN_CUM.format(SRC=s, TGT=t, TITLE=title_rec["nguon"], NGUON="").rstrip()
            loai = "mo_ta"
        else:  # attribute line
            if pid not in tach:
                tach[pid] = (str(rec["nguon"]).split("\n"), str(rec["dich"]).split("\n"))
            ls, ld = tach[pid]
            dong = int(r["dong"])
            if dong < 0 or dong >= len(ls) or dong >= len(ld):
                continue
            nguon, dich = ls[dong], ld[dong]
            user = CHI_DAN_ATTR.format(SRC=s, TGT=t, TITLE=title_rec["nguon"], NGUON=nguon)
            chi_dan = CHI_DAN_ATTR.format(SRC=s, TGT=t, TITLE=title_rec["nguon"], NGUON="").rstrip()
            loai = "mo_ta"
        ra.append({
            "chi_dan": chi_dan,
            "id": f"{pid}_{loai_ref}_{r['dong']}_{int(j)}",
            "product_id": pid,
            "huong": src,
            "loai": loai,
            "group": nhom,
            "category": rec.get("category", "?"),
            "fold5": int(r["fold5"]),
            "nguon": nguon,
            "dich": dich,
            "nhanh": "cum" if tap == "cum" else "pair",
            "e": float(r["e"]),
            "tgt": float(r["tgt"]),
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": dich}],
        })
    return ra


def do_dai(mau, tok, lo=1000, khuon=None):
    """Dien n_p (prompt), n_t (dich), n_chat = n_p + n_t, tokenize theo lo cho nhanh. khuon=None: khuon Qwen (mac dinh, dung de LAY MAU)."""
    khuon = khuon or tao_khuon(tok, "qwen")
    for i in range(0, len(mau), lo):
        if i and (i // lo) % 50 == 0:
            print(f"  do_dai: {i}/{len(mau)}", flush=True)
        k = mau[i:i + lo]
        ps = [khuon["lam"](m) for m in k]
        as_ = [m["messages"][1]["content"] + khuon["duoi"] for m in k]
        ip = tok(ps, add_special_tokens=False)["input_ids"]
        ia = tok(as_, add_special_tokens=False)["input_ids"]
        for m, a, b in zip(k, ip, ia):
            m["n_p"], m["n_t"] = len(a), len(b)
            m["n_chat"] = len(a) + len(b)


def lay_nhom_t(rng, pos, nt, T):
    """Boc deu co hoan lai trong nhom cho toi khi du T token dich (vuot o mau cuoi)."""
    if T <= 0 or len(pos) == 0:
        return np.array([], dtype=np.int64)
    mean = float(nt[pos].mean())
    out, tot = [], 0.0
    while tot < T:
        k = max(10, int((T - tot) / mean * 1.1) + 10)
        pick = pos[rng.choice(len(pos), size=k, replace=True)]
        out.append(pick)
        tot += float(nt[pick].sum())
    out = np.concatenate(out)
    cs = np.cumsum(nt[out])
    cut = int(np.searchsorted(cs, T, side="left")) + 1
    return out[:cut]


def ngach(mau, tap, cap, ngan_sach_compute, seed=42):
    """Tra dict config -> {idx, s_g}; ngach tinh tren token DICH."""
    giu = np.array([i for i, m in enumerate(mau) if m["n_chat"] <= cap], dtype=np.int64)
    if len(giu) == 0:
        raise RuntimeError(f"KHONG_CON_MAU:{tap}:cap{cap}")
    grp = np.array([mau[i]["group"] for i in giu])
    nt = np.zeros(len(mau), dtype=np.float64)      # chi so theo vi tri goc trong mau
    nc = np.zeros(len(mau), dtype=np.float64)
    for i in giu:
        nt[i] = mau[i]["n_t"]
        nc[i] = mau[i]["n_chat"]
    r = float(nc[giu].sum() / max(nt[giu].sum(), 1.0))          # compute tren moi token dich
    ngan_sach_t = int(ngan_sach_compute / max(r, 1e-9))
    M = {g: float(nt[giu][grp == g].sum()) for g in THU_TU[tap] if (grp == g).any()}
    tong = sum(M.values())
    natural = {g: M[g] / tong for g in M}

    ra = {}
    for cname, cfg in CONFIGS[tap].items():
        s = natural if cfg is None else {g: cfg[g] / 100.0 for g in THU_TU[tap]}
        if abs(sum(s.values()) - 1) > 1e-9:
            raise RuntimeError(f"S_G_KHONG_TONG_1:{tap}:{cname}")
        rng = np.random.default_rng(seed)
        parts = []
        for g in THU_TU[tap]:
            pos = giu[grp == g]
            if len(pos) == 0:
                if s.get(g, 0) > 0:
                    raise RuntimeError(f"THIEU_NHOM:{tap}:{cname}:{g}")
                continue
            parts.append(lay_nhom_t(rng, pos, nt, s[g] * ngan_sach_t))
        sel = np.concatenate([p for p in parts if len(p)])
        rng.shuffle(sel)
        ra[cname] = {"idx": sel, "s_g": s}

    return ra, {
        "tap": tap, "cap": cap, "n_tat_ca": len(mau), "n_giu": int(len(giu)), "n_bo": int(len(mau) - len(giu)),
        "dai_nhat_tat_ca": int(max(m["n_chat"] for m in mau)),
        "dai_nhat_giu": int(max(m["n_chat"] for m in (mau[i] for i in giu))),
        "compute_tren_token_dich": round(r, 4),
        "ngan_sach_token_dich": int(ngan_sach_t), "natural": natural,
        "token_dich_toan_bo": int(nt[giu].sum()),
    }


def gom_buoc_dich(lo, budget):
    """Gom micro-batch thanh buoc sao cho moi buoc ~budget token DICH."""
    buoc, cur, tot = [], [], 0
    for k in lo:
        t = sum(m["n_tgt"] for m in k)
        if cur and tot + t > budget:
            buoc.append(cur)
            cur, tot = [], 0
        cur.append(k)
        tot += t
        if tot >= budget:
            buoc.append(cur)
            cur, tot = [], 0
    if cur:
        buoc.append(cur)
    return buoc


def chuan_bi(tap, tok, cfg):
    cap = int(cfg["max_len"])
    ref = doc_ref(tap, int(cfg.get("flt", 0)))
    pool_by = map_pool(doc_pool(tap))
    mau = build_mau(ref, range(len(ref)), pool_by, tap)
    print(f"[{tap}] dung {len(mau)} mau, dang tokenize do dai that...", flush=True)
    do_dai(mau, tok)
    _, kiem = ngach(mau, tap, cap, float(cfg["ngan_sach_compute"]), int(cfg["seed"]))
    print(f"[{tap}] kiem: {json.dumps(kiem, ensure_ascii=False)}", flush=True)
    if kiem["dai_nhat_giu"] > cap:
        raise RuntimeError(f"VUOT_CAP_SAU_LOC:{tap}:{kiem['dai_nhat_giu']}>{cap}")
    return ref, mau, kiem


W_TITLE_ZHVI = 0.10      # trọng số CỐ ĐỊNH cho title trong metric chính (zh->vi), theo quyết định đã chốt


def chi_so_chinh(tap, th, eval_mau=None):
    """Metric chinh: chrF zh->vi voi trong so CO DINH 10% title / 90% mo ta.

    Ban cu lay trong so bang tong token dich cua mau eval; voi pair, moi dong
    attribute chi 2-3 token nen trong so title tut thanh ~67%, sai han so voi
    ti le trien khai ~8-10%. Nay dung hang so 0,10.
    """
    nhom = th.get("theo_nhom") or {}
    t = (nhom.get("zh->vi/title") or {}).get("chrf_corpus")
    m = (nhom.get("zh->vi/mo_ta") or {}).get("chrf_corpus")
    if t is None or m is None:
        return None
    return W_TITLE_ZHVI * float(t) + (1 - W_TITLE_ZHVI) * float(m)


def guardrail(th):
    """Điểm riêng từng nhánh zh->vi để kiểm điều kiện không tụt quá 1 điểm."""
    nhom = th.get("theo_nhom") or {}
    return {k: (nhom[k]["chrf_corpus"] if k in nhom else None)
            for k in ("zh->vi/title", "zh->vi/mo_ta")}


def ma_hoa(mau_sel, tok, cap, lo=500, khuon=None):
    """Mã hoá theo lô, đúng khuôn chat như kaggle_train.xay_mau nhưng nhanh hơn.

    Trả list {id, ids, labels, fold5, group, n_tgt}; lỗi nếu mẫu vượt cap. khuon=None: khuôn Qwen.
    """
    khuon = khuon or tao_khuon(tok, "qwen")
    ra = []
    for i in range(0, len(mau_sel), lo):
        k = mau_sel[i:i + lo]
        ps = [khuon["lam"](m) for m in k]
        as_ = [m["messages"][1]["content"] + khuon["duoi"] for m in k]
        ip = tok(ps, add_special_tokens=False)["input_ids"]
        ia = tok(as_, add_special_tokens=False)["input_ids"]
        for m, a, b in zip(k, ip, ia):
            if len(a) + len(b) > cap:
                raise RuntimeError(f"VUOT_CAP_KHI_MA_HOA:{m['id']}:{len(a)+len(b)}>{cap}")
            ra.append({"id": m["id"], "ids": a + b, "labels": [-100] * len(a) + b,
                       "fold5": m["fold5"], "group": m["group"], "n_tgt": len(b)})
    return ra


def do_dem(ma, tap):
    """Ti le token dich thuc te theo nhom."""
    nhom = {g: 0 for g in THU_TU[tap]}
    for m in ma:
        nhom[m["group"]] = nhom.get(m["group"], 0) + m["n_tgt"]
    tong = sum(nhom.values()) or 1
    return {g: nhom.get(g, 0) / tong for g in THU_TU[tap]}, tong


def cham_mot(model, tok, eval_mau, tap, thiet_bi, zs_cache, khuon=None):
    import cham_diem as D
    lam = khuon["lam"] if khuon else None
    du = K.sinh(model, tok, eval_mau, thiet_bi, bs=32, ngan_sach=32768, lam_prompt=lam)
    if tap not in zs_cache:      # zero-shot giống nhau cho mọi cấu hình cùng bộ -> chỉ sinh 1 lần
        with model.disable_adapter():
            zs_cache[tap] = K.sinh(model, tok, eval_mau, thiet_bi, bs=32, ngan_sach=32768, lam_prompt=lam)
    tm, th = D.cham(pd.DataFrame(eval_mau), pd.DataFrame(du), nhanh=tap)
    _, th_zs = D.cham(pd.DataFrame(eval_mau), pd.DataFrame(zs_cache[tap]), nhanh=tap)
    return th, th_zs, du


def chay_job(job, cfg, tok, thu_mh, cache, ng, zs_cache):
    import torch
    from peft import LoraConfig, get_peft_model
    tap, config = job["tap"], job["config"]
    seed = int(job.get("seed", cfg["seed"]))
    torch.manual_seed(seed)
    random.seed(seed)
    cap = int(cfg["max_len"])
    mh = cfg.get("mo_hinh") or {}
    kieu = mh.get("khuon", "qwen")
    khuon = tao_khuon(tok, kieu)
    ref, mau, kiem = cache[tap]
    mau_sel = [mau[int(i)] for i in ng[config]["idx"]]
    ma = ma_hoa(mau_sel, tok, cap, khuon=khuon)
    ti_le, tong_dich = do_dem(ma, tap)
    lech = {g: abs(ti_le.get(g, 0) - ng[config]["s_g"].get(g, 0)) for g in THU_TU[tap]}
    # Danh sach mau lay theo token Qwen; voi mo hinh khac tokenizer, ti le token dich theo token cua chinh mo hinh do lech nhe so voi s_g
    # nen chi bao lech (khong dung job); voi Qwen van dung neu lech > 0,01 nhu truoc.
    if kieu == "qwen" and max(lech.values()) > 0.01:
        raise RuntimeError(f"LECH_NGACH:{config}:{lech}")
    print(f"[{tap}/{config}/s{seed}] khuon={kieu} n={len(ma)} token_dich={tong_dich} lech_max={max(lech.values()):.4f}", flush=True)

    thiet_bi = "cuda:0"
    torch.cuda.reset_peak_memory_stats()
    goc = K.nap_goc(thu_mh, attn=mh.get("attn", "sdpa"))
    if mh.get("gc"):
        goc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        goc.enable_input_require_grads()
    model = get_peft_model(goc, LoraConfig(r=cfg["rank"], lora_alpha=2 * cfg["rank"], lora_dropout=0.05,
                                           target_modules="all-linear", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    tham_so = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tham_so, lr=cfg["lr"], weight_decay=0.0)
    lo = K.chia_lo(ma, cfg["micro_bs"], seed, cfg["ngan_sach_token"])
    buoc = gom_buoc_dich(lo, int(cfg["so_token_moi_buoc"]))
    model.train()
    t0 = time.time()
    tok_dem = 0
    nhat_ky = []
    for i, cac_lo in enumerate(buoc):
        N = sum(m["n_tgt"] for k in cac_lo for m in k)
        if N <= 0:
            raise RuntimeError(f"BUOC_KHONG_CO_TOKEN:{i}")
        tong = 0.0
        for k in cac_lo:
            n_k = sum(m["n_tgt"] for m in k)
            ids, lab, mask = K.lo_tensor(k, tok, thiet_bi)
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            (out.loss * (n_k / N)).backward()
            tong += out.loss.item() * (n_k / N)
            tok_dem += int(mask.sum().item())
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * K.lich_lr(i, len(buoc))
        torch.nn.utils.clip_grad_norm_(tham_so, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i % 50 == 0 or i == len(buoc) - 1:
            nhat_ky.append({"buoc": i, "loss": round(tong, 4), "giay": round(time.time() - t0, 1)})
            print(nhat_ky[-1], flush=True)
        if cfg.get("dung_sau_buoc") and i + 1 >= int(cfg["dung_sau_buoc"]):     # chi cho chay khoi: dung som, vi du 8 buoc
            nhat_ky.append({"buoc": i, "loss": round(tong, 4), "giay": round(time.time() - t0, 1), "dung_som": True})
            print("DUNG SOM (chay khoi)", nhat_ky[-1], flush=True)
            break
    giay = time.time() - t0
    bo_nho_gb = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)

    eval_ref = doc_eval(tap)
    pool_by = map_pool(doc_pool(tap))
    eval_mau = build_mau(eval_ref, range(len(eval_ref)), pool_by, tap)
    if cfg.get("so_eval") and int(cfg["so_eval"]) < len(eval_mau):
        rng = random.Random(seed)
        eval_mau = rng.sample(eval_mau, int(cfg["so_eval"]))
    th, th_zs, du = cham_mot(model, tok, eval_mau, tap, thiet_bi, zs_cache, khuon)
    n_eval = len(du)
    out_dir = RA / "quota" / f"{tap}_{config}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    (out_dir / "sinh.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in du), encoding="utf-8")
    del model, opt
    torch.cuda.empty_cache()
    return {
        "tap": tap, "config": config, "seed": seed, "s_g": ng[config]["s_g"], "realized": ti_le,
        "mo_hinh": mh.get("ten", "qwen3-4b"), "khuon": kieu, "attn": mh.get("attn", "sdpa"), "gc": bool(mh.get("gc")),
        "bo_nho_dinh_gb": bo_nho_gb, "buoc_ke_hoach": len(buoc), "dung_som": bool(cfg.get("dung_sau_buoc")),
        "lech_max": max(lech.values()), "n_mau": len(ma), "bo_max_len": 0,
        "so_buoc": (nhat_ky[-1]["buoc"] + 1 if cfg.get("dung_sau_buoc") else len(buoc)), "token_dich": int(tong_dich),
        "ngan_sach_token_moi_buoc": int(cfg["so_token_moi_buoc"]),
        "ngan_sach_token": cfg["ngan_sach_token"], "micro_bs": cfg["micro_bs"],
        "rank": cfg["rank"], "lr": cfg["lr"], "max_len": cap,
        "token_compute": tok_dem, "giay": round(giay, 1), "token_moi_giay": round(tok_dem / max(giay, 1e-6), 1),
        "loss": nhat_ky[-1]["loss"] if nhat_ky else None, "nhat_ky": nhat_ky,
        "eval_n": n_eval, "chrf": th.get("diem_chinh"), "chrf_zs": th_zs.get("diem_chinh"),
        "diem_chinh": chi_so_chinh(tap, th, eval_mau),
        "guardrail_zhvi": guardrail(th),
        "theo_huong": th.get("theo_huong"), "theo_huong_zs": th_zs.get("theo_huong"),
        "theo_nhom": th.get("theo_nhom"), "theo_nhom_zs": th_zs.get("theo_nhom"),
        "adapter": str(out_dir),
    }


def do_dem_mau(mau_sel, tap):
    """Ti le token dich theo nhom, tinh truc tiep tu n_t (khong can ma hoa)."""
    nhom = {g: 0 for g in THU_TU[tap]}
    for m in mau_sel:
        nhom[m["group"]] = nhom.get(m["group"], 0) + m["n_t"]
    tong = sum(nhom.values()) or 1
    return {g: nhom.get(g, 0) / tong for g in THU_TU[tap]}, tong


def thong_ke_dai(mau_sel, tok, cap, khuon, tap, lo=500):
    """Ma hoa TOAN BO mau da chon bang tokenizer + khuon cua chinh mo hinh (khong tao model): do dai, so mau vuot cap, ti le token dich theo nhom,
    ti le mau ket thuc dung ma EOS, va vi du loi dan that de nguoi doc kiem bang mat."""
    dai, nhom_t, eos_dung, n_eos = [], {g: 0 for g in THU_TU[tap]}, 0, 0
    eos_id = tok.convert_tokens_to_ids(khuon["duoi"].strip()) if khuon["duoi"].strip() else None
    vi_du = {}
    for i in range(0, len(mau_sel), lo):
        k = mau_sel[i:i + lo]
        ps = [khuon["lam"](m) for m in k]
        as_ = [m["messages"][1]["content"] + khuon["duoi"] for m in k]
        ip = tok(ps, add_special_tokens=False)["input_ids"]
        ia = tok(as_, add_special_tokens=False)["input_ids"]
        for m, p, a, b in zip(k, ps, ip, ia):
            dai.append(len(a) + len(b))
            nhom_t[m["group"]] = nhom_t.get(m["group"], 0) + len(b)
            if eos_id is not None and eos_id != tok.unk_token_id:
                n_eos += 1
                eos_dung += int(eos_id in b[-2:])
            if m["loai"] not in vi_du:
                vi_du[m["loai"]] = {"loi_dan": p[:700], "dich": m["messages"][1]["content"][:120], "n_p": len(a), "n_t": len(b)}
    tong = sum(nhom_t.values()) or 1
    return {"n": len(dai), "n_vuot_cap": int(sum(1 for x in dai if x > cap)), "dai_nhat": int(max(dai) if dai else 0),
            "dai_tb": round(sum(dai) / max(len(dai), 1), 1), "token_dich": int(tong),
            "ti_le_token_dich_theo_nhom": {g: round(nhom_t.get(g, 0) / tong, 4) for g in THU_TU[tap]},
            "ti_le_mau_ket_thuc_dung_eos": (round(eos_dung / n_eos, 4) if n_eos else None), "vi_du": vi_du}


def kiem_job(job, cfg, tok, cache, ng):
    """Dry run: kiem ngach + kiem mau hoa toan bo bang tokenizer/khuon cua mo hinh, khong tao model."""
    tap, config = job["tap"], job["config"]
    cap = int(cfg["max_len"])
    mh = cfg.get("mo_hinh") or {}
    kieu = mh.get("khuon", "qwen")
    ref, mau, kiem = cache[tap]
    sel = ng[config]["idx"]
    mau_sel = [mau[int(i)] for i in sel]
    ti_le, tong_dich = do_dem_mau(mau_sel, tap)
    s = ng[config]["s_g"]
    lech = {g: abs(ti_le.get(g, 0) - s.get(g, 0)) for g in THU_TU[tap]}
    # Luu danh sach id mau da chon (theo thu tu huan luyen) de cac mo hinh khac (OpenNMT, Moses) dung dung cung tap mau "khop ngan sach 4M token"
    (RA / f"chon_{tap}_{config}_s{int(job.get('seed', cfg['seed']))}.json").write_text(
        json.dumps([m["id"] for m in mau_sel]), encoding="utf-8")

    # ma hoa toan bo mau chon (khong raise) de biet truoc co mau nao vuot cap trong token cua mo hinh nay
    tk = thong_ke_dai(mau_sel, tok, cap, tao_khuon(tok, kieu), tap)
    dai_thu = tk["dai_nhat"]
    bo_thu = tk["n_vuot_cap"]

    compute = sum(m["n_p"] + m["n_t"] for m in mau_sel)
    buoc_uoc = int(math.ceil(tong_dich / max(int(cfg["so_token_moi_buoc"]), 1)))
    ra = {
        "tap": tap, "config": config, "seed": int(job.get("seed", cfg["seed"])),
        "s_g": s, "realized": ti_le,
        "lech_max": max(lech.values()), "n_mau": len(mau_sel), "bo_max_len": int(bo_thu),
        "so_buoc_uoc": buoc_uoc, "token_dich": int(tong_dich), "token_compute": int(compute),
        "ngan_sach_token_dich": kiem["ngan_sach_token_dich"],
        "ngan_sach_token_moi_buoc": int(cfg["so_token_moi_buoc"]),
        "max_len": cap, "dai_nhat_mau_ma_hoa": int(dai_thu), "mo_hinh": mh.get("ten", "qwen3-4b"), "khuon": kieu,
        "thong_ke_mo_hinh": tk,
        "dai_nhat_mau_chon": max(m["n_chat"] for m in mau_sel),
        "kiem": kiem,
    }
    if kieu == "qwen" and max(lech.values()) > 0.01:
        ra["loi"] = f"LECH_NGACH:{lech}"
    if bo_thu or dai_thu > cap:
        ra["loi"] = f"MA_HOA_THU_LOI:bo={bo_thu}:dai={dai_thu}"
    print(f"[dry {tap}/{config}] n={len(mau_sel)} dich={tong_dich} buoc~{buoc_uoc} "
          f"compute={compute} lech_max={max(lech.values()):.4f} bo_thu={bo_thu} dai_thu={dai_thu}", flush=True)
    return ra


def tim_thu_muc_mo_hinh(ten_thu_muc=None):
    """Thu muc chua REVISION.txt cua mo hinh. ten_thu_muc=None: nhu truoc (REVISION.txt dau tien); co ten: chi lay thu muc ten do, vi co the gan nhieu mo hinh."""
    if not ten_thu_muc:
        return K.tim_mot("REVISION.txt").parent
    ds = sorted(p for p in VAO.rglob(f"{ten_thu_muc}/REVISION.txt") if p.is_file())
    if not ds:
        raise RuntimeError(f"khong_thay_mo_hinh:{ten_thu_muc}")
    return ds[0].parent


def main(cfg):
    import torch
    import transformers
    from transformers import AutoTokenizer
    K.cai_thu_vien()
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    dry = cfg.get("che_do") == "quota_dry"
    mh = cfg.get("mo_hinh") or {}
    bc = {"moi_truong": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
          "che_do": cfg.get("che_do"), "mo_hinh": mh, "viec": []}
    thu_mh = tim_thu_muc_mo_hinh(mh.get("ten_thu_muc"))
    print("thu muc mo hinh:", thu_mh, flush=True)
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # Danh sach mau LAY theo do dai token cua Qwen (cung mot danh sach cho moi mo hinh); mo hinh khac chi doi tokenizer luc huan luyen
    tok_mau = tok
    if mh.get("tok_lay_mau_thu_muc"):
        tok_mau = AutoTokenizer.from_pretrained(str(tim_thu_muc_mo_hinh(mh["tok_lay_mau_thu_muc"])))
        print("tokenizer lay mau:", mh["tok_lay_mau_thu_muc"], flush=True)

    cache = {}
    zs_cache = {}
    ng_cache = {}
    for tap in sorted({j["tap"] for j in cfg["viec"]}):
        cache[tap] = chuan_bi(tap, tok_mau, cfg)
        bc.setdefault("kiem_tung_tap", {})[tap] = cache[tap][2]
        (RA / ("quota_dry.json" if dry else "quota_report.json")).write_text(
            json.dumps(bc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    for job in cfg["viec"]:
        try:
            tap, seed = job["tap"], int(job.get("seed", cfg["seed"]))
            if (tap, seed) not in ng_cache:
                ng_cache[(tap, seed)], _ = ngach(cache[tap][1], tap, int(cfg["max_len"]),
                                                 float(cfg["ngan_sach_compute"]), seed)
            ng = ng_cache[(tap, seed)]
            bc["viec"].append(kiem_job(job, cfg, tok, cache, ng) if dry
                              else chay_job(job, cfg, tok, thu_mh, cache, ng, zs_cache))
        except Exception as e:
            import traceback
            bc["viec"].append({"tap": job["tap"], "config": job["config"], "seed": job.get("seed", cfg["seed"]),
                               "loi": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-3000:]})
            print("LOI", job, e, flush=True)
        (RA / ("quota_dry.json" if dry else "quota_report.json")).write_text(
            json.dumps(bc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("XONG", len(bc["viec"]), flush=True)
    return bc


if __name__ == "__main__":
    raise SystemExit("Nhung vao notebook, goi main(CAU_HINH)")
