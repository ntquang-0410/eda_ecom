# -*- coding: utf-8 -*-
"""Chấm C0 `cum_f0_p11` / `pair_f0_p11` trên dev.

- Cụm: dùng đúng 1.200 mẫu `eval_cum_sample.csv` như screen -> so trực tiếp với cum natural 74,06.
- Pair: dùng đúng bộ eval pair của screen (1.198 mẫu) -> so trực tiếp với 71,51; lưu ý C0 train
  trên title/mo_ta, còn eval có attribute line, nên đây là cùng thước đo chứ chưa phải B3 line-level.
- Không đọc test.
"""
import json
import random
import time
from pathlib import Path

import pandas as pd

import kaggle_train as K
import kaggle_quota as Q

VAO = Path("/kaggle/input")
RA = Path("/kaggle/working")


def _co_adapter(d):
    """Thư mục có adapter thật (config + trọng số), không chỉ config rỗng."""
    if not (d / "adapter_config.json").exists():
        return False
    return (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()


def _liet_ke(ten, sau=3):
    """Gợi ý đường dẫn khi không tìm thấy adapter (ghi vào report để debug)."""
    goi_y = []
    for p in VAO.glob("*"):
        if p.is_dir():
            goi_y.append(str(p))
            if sau > 1:
                for q in p.rglob(ten):
                    goi_y.append(str(q))
                    if sau > 2:
                        goi_y += [str(x) for x in list(q.rglob("adapter_config.json"))[:5]]
    return goi_y[:40]


def tim_adapter(ten):
    """Tìm thư mục adapter của `ten` ở bất kỳ độ sâu nào dưới /kaggle/input.

    Kho `tl-luu` có cấu trúc `tl/<ten>/ep2/adapter_*`, nên phải tìm thư mục
    chứa adapter_config.json nằm trong cây con của `<ten>`, không phải `<ten>`.
    """
    ung = []
    for p in VAO.rglob(ten):
        if not p.is_dir():
            continue
        if _co_adapter(p):
            ung.append(p)
        ung += [x.parent for x in p.rglob("adapter_config.json") if _co_adapter(x.parent)]
    ung = sorted(set(ung), key=lambda d: (d.name != "ep2", "p11" not in str(d), len(str(d))))
    if not ung:
        raise RuntimeError(f"khong_thay_adapter:{ten} | goi_y={_liet_ke(ten)}")
    return ung[0]


def eval_cum():
    ref = Q.doc_eval("cum")
    pool_by = Q.map_pool(Q.doc_pool("cum"))
    mau = Q.build_mau(ref, range(len(ref)), pool_by, "cum")
    return mau, "cum_block_1200"


def eval_pair():
    """Đúng bộ eval pair của screen (eval_pair_sample.csv + build_mau), 1.198 mẫu.

    Dùng cùng bộ với screen để C0 so được trực tiếp với 71,51, dù C0 train trên
    title/mo_ta còn eval có thêm attribute line; đây là so trên cùng thước đo.
    """
    ref = Q.doc_eval("pair")
    pool_by = Q.map_pool(Q.doc_pool("pair"))
    mau = Q.build_mau(ref, range(len(ref)), pool_by, "pair")
    return mau, "pair_eval_1198"


def cham_c0(job, tok, thu_mh):
    import torch
    import cham_diem as D
    from peft import PeftModel
    thiet_bi = "cuda:0"
    tap, ten = job["tap"], job["c0"]
    mau, kieu = eval_cum() if tap == "cum" else eval_pair()
    ad = tim_adapter(ten)
    print(f"[{tap}/{ten}] eval={kieu} n={len(mau)} adapter={ad}", flush=True)
    t0 = time.time()
    goc = K.nap_goc(thu_mh)
    model = PeftModel.from_pretrained(goc, str(ad))
    du = K.sinh(model, tok, mau, thiet_bi, bs=32, ngan_sach=32768)
    tm, th = D.cham(pd.DataFrame(mau), pd.DataFrame(du), nhanh=tap)
    giay = time.time() - t0
    del model, goc
    torch.cuda.empty_cache()
    return {
        "tap": tap, "c0": ten, "kieu_eval": kieu, "n_eval": len(mau),
        "chrf": th.get("diem_chinh"), "theo_huong": th.get("theo_huong"),
        "theo_nhom": th.get("theo_nhom"), "giay": round(giay, 1), "adapter": str(ad),
        "ti_le_cat_loi_dan": float(tm.da_cat_loi_dan.mean()),
    }


def main(cfg):
    import torch
    from transformers import AutoTokenizer
    K.cai_thu_vien()
    torch.manual_seed(42)
    random.seed(42)
    thu_mh = K.tim_mot("REVISION.txt").parent
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bc = {"moi_truong": {"gpu": torch.cuda.get_device_name(0)}, "viec": []}
    for job in cfg.get("jobs", [{"tap": "cum", "c0": "cum_f0_p11"}, {"tap": "pair", "c0": "pair_f0_p11"}]):
        try:
            bc["viec"].append(cham_c0(job, tok, thu_mh))
        except Exception as e:
            import traceback
            bc["viec"].append({"tap": job["tap"], "c0": job["c0"], "loi": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-3000:]})
            print("LOI", job, e, flush=True)
        (RA / "c0_report.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("XONG C0", len(bc["viec"]), flush=True)
    return bc


if __name__ == "__main__":
    raise SystemExit("Nhúng vào notebook, gọi main(CAU_HINH)")
