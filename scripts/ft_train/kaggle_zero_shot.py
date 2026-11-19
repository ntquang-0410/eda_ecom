# -*- coding: utf-8 -*-
"""Zero-shot Qwen3-4B (KHÔNG adapter) trên đúng bộ eval 1.200 mẫu của bảng năm mô hình (eval_cum_sample.csv, eval_pair_sample.csv), cùng cách dựng mẫu, cùng lời dẫn
messages[0] của pool, cùng cách sinh và chấm với dòng thứ sáu (kaggle_dong6). Khác dòng thứ sáu đúng một điểm: không nạp adapter.

Mục đích (phản biện vòng 2, mục 1.2(d), việc 2 đã duyệt ngày 18/11/2026): OpenCode báo thiếu bản dịch zero-shot đầy đủ trên eval fold 0, nên không lập được
chênh lệch kép "Qwen có lợi thế sẵn chưa fine-tune không". Tệp ra: sinh_zero_shot_{cum,pair}.jsonl (từng hàng eval), sinh_khoi_zero_shot_{tap}.jsonl (từng lượt gọi),
eval_items_{tap}.jsonl, bao_cao_zero_shot.json. Không đọc test. Nếu mô hình trả lời kèm lời thừa (do chưa học khuôn), giữ nguyên bản sinh, không làm sạch.
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
import kaggle_dong6 as D6

RA = Path("/kaggle/working")


def chay_tap(tap, cfg, tok, thu_mh, bc):
    import torch
    import cham_diem as D
    thiet_bi = "cuda:0"
    ref = Q.doc_eval(tap)
    pool_by = Q.map_pool(Q.doc_pool(tap))
    mau = Q.build_mau(ref, range(len(ref)), pool_by, tap)
    if cfg.get("so_toi_da"):                        # chỉ khi chạy khói
        mau = random.Random(42).sample(mau, min(cfg["so_toi_da"], len(mau)))
    gen, hang = D6.dung_gen(tap, mau, pool_by)
    print(f"[{tap}] zero-shot, eval {len(mau)} hàng, {len(gen)} lượt gọi", flush=True)
    t0 = time.time()
    model = D6.nap(thu_mh, cfg.get("dtype", "bf16"))
    model.config.use_cache = False
    giay_nap = time.time() - t0
    t1 = time.time()
    sinh_gen = K.sinh(model, tok, gen, thiet_bi, bs=cfg.get("bs", 32), ngan_sach=cfg.get("ngan_sach", 12000))
    giay_sinh = time.time() - t1
    du, so_lech = D6.ghep(sinh_gen, hang)
    ra = RA / "zero_shot"
    ra.mkdir(exist_ok=True)
    KC.ghi_dong_json(ra / f"eval_items_{tap}.jsonl", [{k: m[k] for k in ("id", "product_id", "huong", "loai", "group", "nguon", "dich", "category", "fold5", "nhanh")} for m in mau])
    KC.ghi_dong_json(ra / f"sinh_zero_shot_{tap}.jsonl", du)
    KC.ghi_dong_json(ra / f"sinh_khoi_zero_shot_{tap}.jsonl", [{"id": r["id"], "hyp": r["hyp"]} for r in sinh_gen])
    tm, th = D.cham(pd.DataFrame(mau), pd.DataFrame(du), nhanh=tap)
    ket = {"tap": tap, "adapter": None, "n_hang_eval": len(mau), "n_luot_goi": len(gen), "giay_nap": round(giay_nap, 1),
           "giay_sinh": round(giay_sinh, 1), "giay_moi_luot": round(giay_sinh / max(1, len(gen)), 3), "so_khoi_lech_dong": so_lech,
           "ti_le_rong": float((pd.DataFrame(du)["hyp"].astype(str).str.strip() == "").mean()), "chrf": D._sach(th.get("diem_chinh")),
           "theo_huong": D._sach(th.get("theo_huong")), "theo_nhom": D._sach(th.get("theo_nhom")), "sinh_lap": int(tm.sinh_lap.sum())}
    bc["viec"].append(ket)
    print(f"[{tap}] zero-shot xong: chrF {ket['chrf']} lech dòng {so_lech} rỗng {ket['ti_le_rong']:.4f} {ket['giay_moi_luot']} giây/lượt", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main(cfg):
    import torch
    import transformers
    from transformers import AutoTokenizer
    K.kiem_khoa_test()
    torch.manual_seed(42)
    random.seed(42)
    bc = {"cau_hinh": cfg, "viec": [], "moi_truong": {"torch": torch.__version__, "transformers": transformers.__version__,
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
        try:
            chay_tap(tap, cfg, tok, thu_mh, bc)
        except Exception as e:
            bc["viec"].append({"tap": tap, "loi": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]})
            print("LOI", tap, e, flush=True)
        (RA / "bao_cao_zero_shot.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    bc["dat"] = all("loi" not in v for v in bc["viec"])
    bc["tong_giay"] = round(time.time() - t0, 1)
    (RA / "bao_cao_zero_shot.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("XONG zero-shot", bc["dat"], bc["tong_giay"], flush=True)
    return bc
