# -*- coding: utf-8 -*-
"""Dòng thứ sáu của bảng năm mô hình (14/11/2026, Nhật Anh duyệt, Prompt 979): Qwen3-4B công thức cuối HỌC FOLD 1-4 (cum_f0_p11, pair_f0_p11; không phải adapter demo,
vì adapter demo đã học fold 0) dịch ĐÚNG bộ eval 1.200 mẫu của bảng năm mô hình (eval_cum_sample.csv, eval_pair_sample.csv, dựng bằng kaggle_quota.build_mau).

Khác bảng năm mô hình ở lời dẫn: hai adapter này học lời dẫn dung_mau.py (mô tả nguyên khối kèm title ngữ cảnh), không phải lời dẫn của vòng sàng lọc, nên mỗi mẫu dùng
messages[0] của mẫu pool thật (đúng khuôn lúc huấn luyện).
- Cụm: mỗi hàng eval (title, hoặc mô tả nguyên khối) là một lượt gọi, như bảng cụm.
- Pair: title một lượt gọi; mô tả dịch CẢ KHỐI một lượt gọi theo từng sản phẩm (đúng cách dùng khi demo), rồi tách theo dòng và lấy đúng dòng `dong` của hàng eval để chấm
  cùng đơn vị từng dòng với bảng pair. Khối ra không đủ số dòng nguồn thì mọi dòng của sản phẩm đó coi là rỗng (ghi số lượng), không đoán.
Không đọc test. Chạy trên GPU nào Kaggle cấp (T4 được chấp nhận); ghi tên GPU và kiểu số vào báo cáo.
"""
import gc
import json
import random
import time
import traceback
from pathlib import Path

import pandas as pd

import kaggle_train as K
import kaggle_quota as Q
import kaggle_cheo as KC

RA = Path("/kaggle/working")
ADAPTER = {"cum": "cum_f0_p11", "pair": "pair_f0_p11"}


# ---------------------------------------------------------------- hàm thuần (có test)
def tach_khoi(hyp, n_nguon):
    """Tách khối pair ra từng dòng; trả None khi số dòng không khớp số dòng nguồn (không đoán dòng)."""
    dong = str(hyp).strip("\n").split("\n")
    return dong if len(dong) == n_nguon else None


def dong_cua_id(id_mau):
    """id eval = f"{pid}_{loai_ref}_{dong}_{j}" (kaggle_quota.build_mau); lấy `dong`."""
    return int(id_mau.rsplit("_", 2)[1])


def dung_gen(tap, mau, pool_by):
    """Trả (gen, hang): gen = các lượt gọi thật sự (mẫu có messages theo khuôn dung_mau của pool), hang = (id mẫu eval, id lượt gọi, chỉ số dòng hoặc None, số dòng nguồn)."""
    gen, hang, da_co = [], [], set()
    for m in mau:
        pid = m["product_id"]
        if m["loai"] == "title":
            rec = pool_by[(pid, "title")]
            gen.append({**m, "messages": rec["messages"]})
            hang.append((m["id"], m["id"], None, None))
        elif tap == "cum":
            rec = pool_by[(pid, "mo_ta")]
            gen.append({**m, "messages": rec["messages"]})
            hang.append((m["id"], m["id"], None, None))
        else:       # pair, mô tả: một lượt gọi cho cả khối của sản phẩm
            rec = pool_by[(pid, "mo_ta")]
            gid = f"{pid}_khoi"
            if gid not in da_co:
                da_co.add(gid)
                gen.append({"id": gid, "product_id": pid, "huong": m["huong"], "loai": "mo_ta", "nguon": rec["nguon"], "dich": rec["dich"],
                            "messages": rec["messages"]})
            hang.append((m["id"], gid, dong_cua_id(m["id"]), len(str(rec["nguon"]).split("\n"))))
    return gen, hang


def ghep(sinh_gen, hang):
    """Từ bản dịch của các lượt gọi ra bản dịch từng hàng eval; trả (du, so_khoi_lech_dong)."""
    theo = {r["id"]: r for r in sinh_gen}
    du, lech, cache = [], set(), {}
    for id_eval, gid, dong, n in hang:
        r = theo[gid]
        if dong is None:
            du.append({"id": id_eval, "hyp_tho": r["hyp_tho"], "hyp": r["hyp"]})
            continue
        if gid not in cache:
            cache[gid] = tach_khoi(r["hyp"], n)
            if cache[gid] is None:
                lech.add(gid)
        dd = cache[gid]
        h = dd[dong] if dd is not None and 0 <= dong < len(dd) else ""
        du.append({"id": id_eval, "hyp_tho": h, "hyp": h})
    return du, len(lech)


# ---------------------------------------------------------------- chạy trên Kaggle
def nap(thu_mh, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    kieu = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    return AutoModelForCausalLM.from_pretrained(str(thu_mh), dtype=kieu, device_map={"": 0}, attn_implementation="sdpa")


def chay_tap(tap, cfg, tok, thu_mh, bc):
    import torch
    import cham_diem as D
    from peft import PeftModel
    thiet_bi = "cuda:0"
    ref = Q.doc_eval(tap)
    pool_by = Q.map_pool(Q.doc_pool(tap))
    mau = Q.build_mau(ref, range(len(ref)), pool_by, tap)
    if cfg.get("so_toi_da"):                        # chỉ khi chạy khói
        mau = random.Random(42).sample(mau, min(cfg["so_toi_da"], len(mau)))
    gen, hang = dung_gen(tap, mau, pool_by)
    ad = KC.tim_adapter(ADAPTER[tap])
    print(f"[{tap}] eval {len(mau)} hàng, {len(gen)} lượt gọi, adapter {ad}", flush=True)
    t0 = time.time()
    goc = nap(thu_mh, cfg.get("dtype", "bf16"))
    goc.config.use_cache = False
    model = PeftModel.from_pretrained(goc, str(ad))
    giay_nap = time.time() - t0
    t1 = time.time()
    sinh_gen = K.sinh(model, tok, gen, thiet_bi, bs=cfg.get("bs", 32), ngan_sach=cfg.get("ngan_sach", 12000))
    giay_sinh = time.time() - t1
    du, so_lech = ghep(sinh_gen, hang)
    ra = RA / "dong6"
    ra.mkdir(exist_ok=True)
    KC.ghi_dong_json(ra / f"eval_items_{tap}.jsonl", [{k: m[k] for k in ("id", "product_id", "huong", "loai", "group", "nguon", "dich", "category", "fold5", "nhanh")} for m in mau])
    KC.ghi_dong_json(ra / f"sinh_{ADAPTER[tap]}_{tap}.jsonl", du)
    KC.ghi_dong_json(ra / f"sinh_khoi_{ADAPTER[tap]}_{tap}.jsonl", [{"id": r["id"], "hyp": r["hyp"]} for r in sinh_gen])
    tm, th = D.cham(pd.DataFrame(mau), pd.DataFrame(du), nhanh=tap)
    ket = {"tap": tap, "adapter": ADAPTER[tap], "duong_dan": str(ad), "n_hang_eval": len(mau), "n_luot_goi": len(gen), "giay_nap": round(giay_nap, 1),
           "giay_sinh": round(giay_sinh, 1), "giay_moi_luot": round(giay_sinh / max(1, len(gen)), 3), "so_khoi_lech_dong": so_lech,
           "ti_le_rong": float((pd.DataFrame(du)["hyp"].astype(str).str.strip() == "").mean()), "chrf": D._sach(th.get("diem_chinh")),
           "theo_huong": D._sach(th.get("theo_huong")), "theo_nhom": D._sach(th.get("theo_nhom")), "sinh_lap": int(tm.sinh_lap.sum())}
    bc["viec"].append(ket)
    print(f"[{tap}] xong: chrF {ket['chrf']} lech dòng {so_lech} rỗng {ket['ti_le_rong']:.4f} {ket['giay_moi_luot']} giây/lượt", flush=True)
    del model, goc
    gc.collect()
    torch.cuda.empty_cache()


def main(cfg):
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer
    K.cai_thu_vien()
    K.kiem_khoa_test()
    torch.manual_seed(42)
    random.seed(42)
    bc = {"cau_hinh": cfg, "viec": [], "moi_truong": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
          "gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "dtype": cfg.get("dtype", "bf16")}}
    print(bc["moi_truong"], flush=True)
    thu_mh = K.tim_mot("REVISION.txt").parent
    bc["mo_hinh"] = thu_mh.joinpath("REVISION.txt").read_text().split()
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    t0 = time.time()
    for tap in cfg.get("tap", ["cum", "pair"]):
        if time.time() - t0 > cfg.get("gio_toi_da", 3.0) * 3600:
            bc["viec"].append({"tap": tap, "bo_qua": "het_gio"})
            continue
        try:
            chay_tap(tap, cfg, tok, thu_mh, bc)
        except Exception as e:
            bc["viec"].append({"tap": tap, "loi": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]})
            print("LOI", tap, e, flush=True)
        (RA / "bao_cao_dong6.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    bc["dat"] = all("loi" not in v and "bo_qua" not in v for v in bc["viec"])
    bc["tong_giay"] = round(time.time() - t0, 1)
    (RA / "bao_cao_dong6.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("XONG dong6", bc["dat"], bc["tong_giay"], flush=True)
    return bc
