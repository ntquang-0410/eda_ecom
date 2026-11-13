# -*- coding: utf-8 -*-
"""Thử chéo định dạng pair và cụm (12/11/2026, Nhật Anh duyệt, Prompt 956). Chạy trên Kaggle, tắt Internet, KHÔNG có dữ liệu test.

Câu hỏi: dịch mô tả sản phẩm zh->vi, đưa nguồn theo dạng pair (mỗi thuộc tính một dòng) hay dạng cụm (nối bằng "; "), và dùng adapter nào?
Mỗi sản phẩm của tập kiểm (fold 0, chỉ mô tả zh->vi) được dựng ĐÚNG MỘT nội dung nguồn ở cả hai dạng, rồi sinh bản dịch cho đủ 3 x 2 tổ hợp:
  adapter in {cum_f0_p11, pair_f0_p11, zero_shot (không adapter)} x dạng đầu vào in {cum, pair}.
Hai adapter fold 0 được huấn luyện lại ở tài khoản 2 (chay_cheo_huan_luyen.py), chưa từng học mẫu của fold 0; không dùng adapter bản cuối vì nó đã học cả mẫu kiểm.
Chỉ SINH và lưu bản dịch; chấm điểm làm trên máy (cham_cheo.py), không tốn GPU.

Cấu hình (cfg): so_toi_da (số sản phẩm, mặc định 600), seed (mặc định 43 = seed 42 + 1 như chon_mau_sinh), gio_toi_da.
Ra: cheo/eval_items.jsonl (nguồn, tham chiếu, lời dẫn từng dạng), cheo/sinh_{adapter}_{dang}.jsonl, cheo/cheo_trang_thai.json, bao_cao_cheo.json.
"""
import gc
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import kaggle_train as K

VAO = Path("/kaggle/input")
RA = Path("/kaggle/working")
HUONG = "zh->vi"
TACH = {"cum": "; ", "pair": "\n"}
DANG = ("pair", "cum")
ADAPTER = ("zero_shot", "cum_f0_p11", "pair_f0_p11")

# Chép đúng khuôn lời dẫn mô tả của dung_mau.py (nơi mọi mẫu huấn luyện được dựng). Phép thử dưới đây (kiem_loi_dan) bắt buộc lời dẫn dựng lại
# khớp từng chữ với messages[0] của mẫu pool thật, nên nếu khuôn lệch thì chạy dừng ngay, không sinh bản dịch sai khuôn.
_MO_TA_DAU = "Translate this product's attribute list from {SRC} to {TGT}. "
_MO_TA_CUOI = "\nProduct title (context only, do not translate): {TITLE}\n\n{NGUON}"
CHI_DAN_MO_TA = {
    "pair": _MO_TA_DAU + 'Each line is one attribute written as "name: value". '
            "Keep the same number and order of lines, and output only the translation." + _MO_TA_CUOI,
    "cum": _MO_TA_DAU + 'The attributes are separated by "; " and each is written as "name: value". '
           'Keep the same number and order of attributes, separated by "; ", and output only the translation.' + _MO_TA_CUOI,
}


def loi_dan(dang, nguon, title):
    return CHI_DAN_MO_TA[dang].format(SRC="Chinese", TGT="Vietnamese", TITLE=title, NGUON=nguon)


def chon_eval(pool_cum, pool_pair, so_toi_da=600, seed=43, so_co_dinh=250):
    """Sản phẩm kiểm: fold 0, mô tả nguyên (phan == 0), zh->vi, có ở cả hai nhánh, và nội dung NGUỒN hai nhánh giống hệt (nguồn cụm == các dòng nguồn pair nối bằng "; ").
    Nhóm "cố định" = đúng các mẫu mô tả zh->vi mà mọi lần chạy huấn luyện đã sinh và chấm (chon_mau_sinh, 250 mỗi nhóm, seed 43), để đối chiếu lại điểm cũ.
    Phần còn lại lấy ngẫu nhiên (seed cố định) tới so_toi_da sản phẩm. Trả (danh sách mục, thống kê)."""
    cum = {m["id"]: m for m in pool_cum}
    pair = {m["id"]: m for m in pool_pair}
    ung = []
    bo = {"khong_phai_zh_vi_fold0_nguyen": 0, "thieu_nhanh_hoac_title": 0, "nguon_hai_nhanh_khac_nhau": 0}
    for id_ in sorted(cum):
        mc = cum[id_]
        if mc["loai"] != "mo_ta" or mc["fold5"] != 0 or mc["phan"] != 0 or mc["huong"] != HUONG:
            bo["khong_phai_zh_vi_fold0_nguyen"] += 1
            continue
        mp, tit = pair.get(id_), cum.get(f"{mc['product_id']}_title")
        if mp is None or tit is None:
            bo["thieu_nhanh_hoac_title"] += 1
            continue
        dong = mp["nguon"].split(TACH["pair"])
        if TACH["cum"].join(dong) != mc["nguon"] or len(mp["dich"].split(TACH["pair"])) != len(dong):
            bo["nguon_hai_nhanh_khac_nhau"] += 1
            continue
        ung.append({"id": id_, "product_id": mc["product_id"], "category": mc.get("category", ""), "title": tit["nguon"], "so_dong": len(dong),
                    "nguon": {"pair": mp["nguon"], "cum": mc["nguon"]}, "ref": {"pair": mp["dich"], "cum": mc["dich"]},
                    "loi_dan_goc": {"pair": mp["messages"][0]["content"], "cum": mc["messages"][0]["content"]}})
    cd = {m["id"] for m in K.lay_phan_tang([m for m in pool_cum if m["fold5"] == 0 and m["phan"] == 0], so_co_dinh, seed)
          if m["huong"] == HUONG and m["loai"] == "mo_ta"}
    for x in ung:
        x["co_dinh"] = x["id"] in cd
    co_dinh = [x for x in ung if x["co_dinh"]]
    con = [x for x in ung if not x["co_dinh"]]
    random.Random(seed).shuffle(con)
    ra = (co_dinh + con)[:so_toi_da]
    ra.sort(key=lambda x: x["id"])
    tk = {"so_ung_vien": len(ung), "so_chon": len(ra), "so_co_dinh_trong_chon": sum(x["co_dinh"] for x in ra), "so_co_dinh_goc": len(cd),
          "so_co_dinh_bi_loai_do_nguon_khac": len(cd) - len(co_dinh), "bo": bo}
    return ra, tk


def kiem_loi_dan(items):
    """Lời dẫn dựng lại phải khớp từng chữ với lời dẫn đã dùng để huấn luyện (messages[0] của mẫu pool), ở cả hai dạng."""
    sai = [(x["id"], d) for x in items for d in DANG if loi_dan(d, x["nguon"][d], x["title"]) != x["loi_dan_goc"][d]]
    if sai:
        raise RuntimeError(f"LOI_DAN_LECH_KHUON:{len(sai)}:{sai[:3]}")


def sha256(duong_dan):
    h = hashlib.sha256()
    with open(duong_dan, "rb") as f:
        for k in iter(lambda: f.read(2 ** 20), b""):
            h.update(k)
    return h.hexdigest()


def tim_adapter(ten):
    ds = sorted(p for p in VAO.rglob(f"{ten}/ep2/adapter_model.safetensors") if p.is_file())
    if len(ds) != 1:
        raise RuntimeError(f"adapter_{ten}: thay {len(ds)}: {[str(p) for p in ds]}")
    cf = json.loads((ds[0].parent / "adapter_config.json").read_text(encoding="utf-8"))
    if cf.get("r") != 64:
        raise RuntimeError(f"adapter_{ten}: rank {cf.get('r')} != 64")
    return ds[0].parent


def doc_dong_json(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def ghi_dong_json(p, rows):
    tam = Path(p).with_suffix(".tmp")
    tam.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    tam.replace(p)


def chay(cfg, bc):
    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoTokenizer

    thiet_bi = "cuda:0"
    t_bat_dau = time.time()
    het_giay = cfg.get("gio_toi_da", 2.0) * 3600
    thu_ra = RA / "cheo"
    thu_ra.mkdir(parents=True, exist_ok=True)
    tep = thu_ra / "cheo_trang_thai.json"
    st = {"xong": False, "da_sinh": {}, "cau_hinh": {k: v for k, v in cfg.items() if not k.startswith("_")}}

    def ghi():
        tam = tep.with_suffix(".tmp")
        tam.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tam.replace(tep)

    bc["moi_truong"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)}
    thu_mh = K.tim_mot("REVISION.txt").parent
    bc["mo_hinh"] = thu_mh.joinpath("REVISION.txt").read_text().split()
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    pool_cum, pool_pair = K.doc_mau("cum"), K.doc_mau("pair")
    items, tk = chon_eval(pool_cum, pool_pair, int(cfg.get("so_toi_da", 600)), int(cfg.get("seed", 43)))
    if len(items) < 50:
        raise RuntimeError(f"QUA_IT_MAU:{len(items)}:{tk}")
    kiem_loi_dan(items)                      # lời dẫn dựng lại khớp từng chữ với mẫu huấn luyện thật
    st["mau"] = tk
    ghi_dong_json(thu_ra / "eval_items.jsonl", [{k: v for k, v in x.items() if k != "loi_dan_goc"} for x in items])
    print("mau kiem:", json.dumps(tk, ensure_ascii=False), flush=True)
    ghi()
    duong = {t: tim_adapter(t) for t in ADAPTER if t != "zero_shot"}
    st["adapter"] = {t: {"duong_dan": str(p), "sha256": sha256(p / "adapter_model.safetensors")} for t, p in duong.items()}
    ghi()

    goc = K.nap_goc(thu_mh)
    for ten in ADAPTER:
        con = [d for d in DANG if not (thu_ra / f"sinh_{ten}_{d}.jsonl").exists()]
        if not con:
            continue
        if time.time() - t_bat_dau > het_giay:
            st["dung_het_gio"] = ten
            ghi()
            return
        m = goc if ten == "zero_shot" else PeftModel.from_pretrained(goc, str(duong[ten]))
        for d in con:
            mau = [{"id": x["id"], "messages": [{"role": "user", "content": loi_dan(d, x["nguon"][d], x["title"])}], "nguon": x["nguon"][d], "dich": x["ref"][d]} for x in items]
            t0 = time.time()
            du = K.sinh(m, tok, mau, thiet_bi, tran_theo="nguon")
            giay = round((time.time() - t0) / len(du), 3)
            ghi_dong_json(thu_ra / f"sinh_{ten}_{d}.jsonl", [{"id": r["id"], "hyp": r["hyp"]} for r in du])
            st["da_sinh"][f"{ten}/{d}"] = {"so_mau": len(du), "giay_moi_mau": giay}
            ghi()
            print(f"{ten}/{d} XONG {len(du)} mau, {giay} giay/mau, gio da chay {round((time.time() - t_bat_dau) / 3600, 2)}", flush=True)
        if ten != "zero_shot":
            goc = m.unload()
            del m
        gc.collect()
        torch.cuda.empty_cache()
    st["xong"] = all(f"{t}/{d}" in st["da_sinh"] or (thu_ra / f"sinh_{t}_{d}.jsonl").exists() for t in ADAPTER for d in DANG)
    ghi()


def main(cfg):
    import os
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    bc = {"cau_hinh": cfg, "dat": False}
    t = time.time()
    try:
        K.kiem_khoa_test()
        bc["khoa_test"] = "khong_co_file_test_trong_input"
        K.cai_thu_vien()
        chay(cfg, bc)
        bc["dat"] = True
    except Exception:
        bc["loi"] = traceback.format_exc()
        print(bc["loi"], flush=True)
    bc["tong_giay"] = round(time.time() - t, 1)
    (RA / "bao_cao_cheo.json").write_text(json.dumps(bc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("KET QUA:", json.dumps({k: v for k, v in bc.items() if k != "cau_hinh"}, ensure_ascii=False, default=str)[:2000], flush=True)
    if "loi" in bc:
        raise RuntimeError("loi, xem bao_cao_cheo.json")
