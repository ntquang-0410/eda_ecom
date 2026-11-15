# -*- coding: utf-8 -*-
"""Giai đoạn 0 của plan demo (14/11/2026): đo bản nén 4-bit chạy trên laptop (Ollama: Qwen3-4B Q4_K_M + LoRA GGUF) so với bản fp16 đã đo trên Kaggle T4.

Dùng adapter pair_f0_p11 (học fold 1-4, chưa thấy fold 0) trên phần zh->vi của bộ eval fold 0 bảng pair; lời dẫn là messages[0] của mẫu pool (đúng khuôn huấn luyện),
title một lượt gọi, mô tả dịch cả khối theo sản phẩm rồi tách dòng (kaggle_dong6.dung_gen/ghep). Bản fp16 để so: kaggle_dong6/dong6/sinh_khoi_pair_f0_p11_pair.jsonl.
Không đọc test khóa.

Chạy (venv EDA):  python -X utf8 demo/do_luong_tu.py sinh --mo-hinh nlp-do-pair-f0 [--so-sp 150]
                  python -X utf8 demo/do_luong_tu.py cham --mo-hinh nlp-do-pair-f0
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
FT = GOC / "data" / "processed" / "ft_run"
DONG6 = FT / "kaggle_dong6" / "dong6"
RA = FT / "demo_do_luong_tu"
OLLAMA = "http://127.0.0.1:8770/completion"        # llama-server (llama.cpp)
KHUON = "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"     # khuôn chat Qwen3 khi tắt suy luận (kiểm bằng kiem_khuon.py)


def doc_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def goi_ollama(mo_hinh, loi_dan, tran, num_ctx=4096):
    """Một lượt sinh greedy qua llama-server (/completion, lời dẫn thô nên không phụ thuộc khuôn chat của máy chủ). Trả (văn bản, thống kê).
    Ollama 0.40 đã bỏ nạp LoRA nên dùng thẳng llama-server của llama.cpp (nền Q4_K_M + --lora); tên hàm giữ lại cho gọn. mo_hinh chỉ là nhãn tên file kết quả."""
    body = {"prompt": KHUON.format(u=loi_dan), "n_predict": int(tran), "temperature": 0, "top_k": 1, "seed": 42, "repeat_penalty": 1.0,
            "cache_prompt": False, "stop": ["<|im_end|>", "<|endoftext|>"]}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode("utf-8"))
    t = d.get("timings") or {}
    return d.get("content", ""), {"eval_count": t.get("predicted_n"), "eval_duration": (t.get("predicted_ms") or 0) * 1e6, "prompt_eval_count": t.get("prompt_n"),
                                  "done_reason": "length" if d.get("stop_type") == "limit" else d.get("stop_type")}


def chon_mau(so_sp):
    """Mẫu eval zh->vi của so_sp sản phẩm (seed 42), kèm lượt gọi (gen) và ánh xạ hàng (hang)."""
    import kaggle_dong6 as K6
    import kaggle_quota as Q
    mau = [m for m in doc_jsonl(DONG6 / "eval_items_pair.jsonl") if m["huong"] == "zh->vi"]
    pids = sorted({m["product_id"] for m in mau})
    chon = set(random.Random(42).sample(pids, min(so_sp, len(pids))))
    mau = [m for m in mau if m["product_id"] in chon]
    can = {m["product_id"] for m in mau}
    pool_by = Q.map_pool(r for r in doc_jsonl(FT / "mau" / "pool_pair.jsonl") if str(r["product_id"]) in can)
    gen, hang = K6.dung_gen("pair", mau, pool_by)
    return mau, gen, hang


def sinh(a):
    mau, gen, hang = chon_mau(a.so_sp)
    RA.mkdir(parents=True, exist_ok=True)
    ra = RA / f"sinh_{a.mo_hinh.replace(':', '_')}.jsonl"
    da = {r["id"] for r in doc_jsonl(ra)} if ra.exists() else set()
    print(f"{len(mau)} hàng eval, {len(gen)} lượt gọi, đã có {len(da)}", flush=True)
    t0 = time.time()
    with open(ra, "a", encoding="utf-8", newline="\n") as f:
        for i, g in enumerate(gen, 1):
            if g["id"] in da:
                continue
            tran = min(1500, 2 * len(str(g["dich"])) + 64)      # bản fp16 chặn theo 2 x token đích + 64; ở đây theo ký tự (rộng hơn), greedy vẫn dừng ở <|im_end|>
            t1 = time.time()
            hyp, tk = goi_ollama(a.mo_hinh, g["messages"][0]["content"], tran)
            f.write(json.dumps({"id": g["id"], "loai": g["loai"], "hyp": hyp.strip("\n"), "giay": round(time.time() - t1, 3),
                                "so_dong_nguon": len(str(g["nguon"]).split("\n")), **tk}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(gen)}  {time.time() - t0:.0f}s", flush=True)
    print("XONG sinh", ra, flush=True)


def _diem(mau, du):
    import pandas as pd
    import cham_bleu as B
    import cham_diem as D
    tm, th = D.cham(pd.DataFrame(mau), pd.DataFrame(du), nhanh="pair")
    bt = B.diem_bleu_ter(tm)
    nh = D._sach(th.get("theo_nhom")) or {}
    return tm, {"chrf_theo_nhom": nh, "chrf_diem_chinh": D._sach(th.get("diem_chinh")), "bleu_ter": {k: v for k, v in bt.items() if v},
                "ti_le_rong": float((pd.DataFrame(du)["hyp"].astype(str).str.strip() == "").mean())}


def cham(a):
    import statistics as st
    import kaggle_dong6 as K6
    mau, gen, hang = chon_mau(a.so_sp)
    may = doc_jsonl(RA / f"sinh_{a.mo_hinh.replace(':', '_')}.jsonl")
    ids = {g["id"] for g in gen}
    assert {r["id"] for r in may} == ids, f"thiếu lượt gọi: {len(ids - {r['id'] for r in may})}"
    fp = [r for r in doc_jsonl(DONG6 / "sinh_khoi_pair_f0_p11_pair.jsonl") if r["id"] in ids]
    assert len(fp) == len(ids), "bản fp16 thiếu lượt gọi"
    ket = {"mo_hinh": a.mo_hinh, "so_san_pham": len({m["product_id"] for m in mau}), "so_hang_eval": len(mau), "so_luot_goi": len(gen)}
    for ten, sg in (("may_4bit", may), ("kaggle_fp16", fp)):
        du, lech = K6.ghep([{"id": r["id"], "hyp_tho": r["hyp"], "hyp": r["hyp"]} for r in sg], hang)
        _, d = _diem(mau, du)
        d["so_khoi_lech_dong"], d["so_khoi"] = lech, sum(1 for g in gen if g["id"].endswith("_khoi"))
        ket[ten] = d
    fpm = {r["id"]: r["hyp"].strip() for r in fp}
    ket["ti_le_trung_khop_voi_fp16"] = sum(1 for r in may if r["hyp"].strip() == fpm[r["id"]]) / len(may)
    tl, kh = [r for r in may if r["loai"] == "title"], [r for r in may if r["loai"] != "title"]
    tps = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in may if r.get("eval_count") and r.get("eval_duration")]
    ket["toc_do"] = {"token_moi_giay_trung_vi": round(st.median(tps), 1) if tps else None,
                     "giay_title_trung_vi": round(st.median(r["giay"] for r in tl), 2), "giay_title_p90": round(sorted(r["giay"] for r in tl)[int(0.9 * len(tl))], 2),
                     "giay_khoi_trung_vi": round(st.median(r["giay"] for r in kh), 2), "giay_khoi_p90": round(sorted(r["giay"] for r in kh)[int(0.9 * len(kh))], 2),
                     "so_dong_khoi_trung_vi": st.median(r["so_dong_nguon"] for r in kh),
                     "giay_moi_dong_khoi": round(sum(r["giay"] for r in kh) / sum(r["so_dong_nguon"] for r in kh), 3),
                     "het_tran": sum(1 for r in may if r.get("done_reason") == "length")}
    (RA / f"ket_qua_{a.mo_hinh.replace(':', '_')}.json").write_text(json.dumps(ket, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ket, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("viec", choices=["sinh", "cham"])
    ap.add_argument("--mo-hinh", required=True)
    ap.add_argument("--so-sp", type=int, default=150)
    a = ap.parse_args()
    {"sinh": sinh, "cham": cham}[a.viec](a)
