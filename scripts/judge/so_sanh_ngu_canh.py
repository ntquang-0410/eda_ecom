"""So hai cách chọn cho bộ CỤM trên CÙNG các cặp: CÓ ngữ cảnh sản phẩm (loại cum_ctx) và KHÔNG ngữ cảnh (loại cum).

  python -I -X utf8 so_sanh_ngu_canh.py --ctx cumctx_40 --khong cum_40k [--in-mau 8]

Chỉ đọc file kết quả đã có (không gọi API). Mỗi mục cum_ctx ghi sẵn `id_khong_ngu_canh` (mục không ngữ cảnh ứng với nó).
Trả lời các câu hỏi: ngữ cảnh làm judge quyết được nhiều hơn không, có đảo ngược lựa chọn không (và đảo theo hướng nào), tốn thêm bao
nhiêu token và thời gian, thiên vị vị trí có đổi không. Không có 'chân lý' ở đây: ca đảo ngược cần người đọc."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so_cap_du_lieu as DL  # noqa: E402

KQ = ("gemma", "qwen", "hoa_that", "khong_nhat_quan")


def doc(duong):
    cuoi = {}
    for dong in open(duong, encoding="utf-8"):
        d = json.loads(dong)
        cuoi[d["item_id"]] = d
    return {k: v for k, v in cuoi.items() if v.get("trang_thai") == "ok"}


def vi_tri_1(rows):
    luot = [l for d in rows for l in d["luot"] if l["vi_tri_chon"] in (1, 2)]
    return round(sum(l["vi_tri_chon"] == 1 for l in luot) / max(len(luot), 1), 3)


def so_sanh(ctx, khong, tong_muc_ctx=None):
    """ctx: {item_id: bản ghi cum_ctx}; khong: {item_id: bản ghi cum}. Chỉ so các mục có cả hai."""
    cap = [(c, khong[c["id_khong_ngu_canh"]]) for c in ctx.values() if c.get("id_khong_ngu_canh") in khong]
    r = {"so_muc_ctx_ok": len(ctx), "so_cap_so_duoc": len(cap)}
    if not cap:
        return r
    ma_tran = Counter((k["ket_qua"], c["ket_qua"]) for c, k in cap)
    r["ma_tran_khong_ngu_canh_sang_ngu_canh"] = {f"{a}->{b}": ma_tran.get((a, b), 0) for a in KQ for b in KQ if ma_tran.get((a, b), 0)}
    quyet = ("gemma", "qwen")
    ca_hai = [(c, k) for c, k in cap if c["ket_qua"] in quyet and k["ket_qua"] in quyet]
    r["ti_le_co_nguoi_thang"] = {"khong_ngu_canh": round(sum(k["ket_qua"] in quyet for _, k in cap) / len(cap), 3),
                                 "ngu_canh": round(sum(c["ket_qua"] in quyet for c, _ in cap) / len(cap), 3)}
    r["chua_quyet_nhung_nay_quyet_duoc"] = sum(c["ket_qua"] in quyet and k["ket_qua"] not in quyet for c, k in cap)
    r["quyet_duoc_nhung_nay_khong_quyet"] = sum(k["ket_qua"] in quyet and c["ket_qua"] not in quyet for c, k in cap)
    r["ca_hai_deu_quyet"] = {"so_muc": len(ca_hai), "dao_nguoc": sum(c["ket_qua"] != k["ket_qua"] for c, k in ca_hai),
                             "dao_sang_gemma": sum(c["ket_qua"] == "gemma" and k["ket_qua"] == "qwen" for c, k in ca_hai),
                             "dao_sang_qwen": sum(c["ket_qua"] == "qwen" and k["ket_qua"] == "gemma" for c, k in ca_hai)}
    r["ti_le_gemma_trong_so_co_nguoi_thang"] = {
        "khong_ngu_canh": round(sum(k["ket_qua"] == "gemma" for _, k in cap) / max(sum(k["ket_qua"] in quyet for _, k in cap), 1), 3),
        "ngu_canh": round(sum(c["ket_qua"] == "gemma" for c, _ in cap) / max(sum(c["ket_qua"] in quyet for c, _ in cap), 1), 3)}
    r["thien_vi_vi_tri_chon_vi_tri_1"] = {"khong_ngu_canh": vi_tri_1([k for _, k in cap]), "ngu_canh": vi_tri_1([c for c, _ in cap])}
    tv_c = sum(c["tokens_vao"] for c, _ in cap) / len(cap)
    tv_k = sum(k["tokens_vao"] for _, k in cap) / len(cap)
    tr_c = sum(c["tokens_ra"] for c, _ in cap) / len(cap)
    tr_k = sum(k["tokens_ra"] for _, k in cap) / len(cap)
    gi_c = sum(c.get("giay_goi") or 0 for c, _ in cap) / len(cap)
    gi_k = sum(k.get("giay_goi") or 0 for _, k in cap) / len(cap)
    r["moi_muc_trung_binh"] = {"tokens_vao": {"khong_ngu_canh": round(tv_k), "ngu_canh": round(tv_c), "ti_so": round(tv_c / tv_k, 2)},
                               "tokens_ra": {"khong_ngu_canh": round(tr_k), "ngu_canh": round(tr_c)},
                               "giay_goi_hai_luot": {"khong_ngu_canh": round(gi_k, 2), "ngu_canh": round(gi_c, 2)}}
    if tong_muc_ctx:
        r["uoc_tinh_toan_bo_cum_ngu_canh"] = {"so_muc": tong_muc_ctx, "so_loi_goi": 2 * tong_muc_ctx,
                                              "tokens_vao_trieu": round(tv_c * tong_muc_ctx / 1e6, 1),
                                              "tokens_ra_trieu": round(tr_c * tong_muc_ctx / 1e6, 1),
                                              "ghi_chu": "ngoại suy tuyến tính từ mẫu nhỏ; độ dài ngữ cảnh thay đổi theo sản phẩm"}
    return r


def so_sanh_lap(a, b):
    """Hai lần chạy KHÔNG ngữ cảnh của cùng các mục (cùng lời nhắc, cùng cấu hình): đo độ nhiễu của chính judge, để tách ra khỏi tác
    động của ngữ cảnh. a, b: {item_id: bản ghi}. Chỉ so các mục có ở cả hai."""
    chung = [i for i in a if i in b]
    r = {"so_muc_chung": len(chung)}
    if not chung:
        return r
    ma_tran = Counter((a[i]["ket_qua"], b[i]["ket_qua"]) for i in chung)
    r["ma_tran_lan1_sang_lan2"] = {f"{x}->{y}": ma_tran.get((x, y), 0) for x in KQ for y in KQ if ma_tran.get((x, y), 0)}
    quyet = ("gemma", "qwen")
    r["giong_het_ket_qua"] = round(sum(a[i]["ket_qua"] == b[i]["ket_qua"] for i in chung) / len(chung), 3)
    r["lan1_co_nguoi_thang"] = sum(a[i]["ket_qua"] in quyet for i in chung)
    r["trong_do_lan2_van_cung_nguoi_thang"] = sum(a[i]["ket_qua"] in quyet and a[i]["ket_qua"] == b[i]["ket_qua"] for i in chung)
    r["lan1_chua_quyet_lan2_quyet_duoc"] = sum(a[i]["ket_qua"] not in quyet and b[i]["ket_qua"] in quyet for i in chung)
    r["lan1_quyet_lan2_chua_quyet"] = sum(a[i]["ket_qua"] in quyet and b[i]["ket_qua"] not in quyet for i in chung)
    r["dao_nguoc_gemma_qwen"] = sum({a[i]["ket_qua"], b[i]["ket_qua"]} == set(quyet) for i in chung)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctx", required=True, help="tên đợt cum_ctx (ket_qua_<tên>.jsonl)")
    p.add_argument("--khong", required=True, help="tên đợt cum không ngữ cảnh ứng với nó")
    p.add_argument("--khong2", help="lần chạy KHÔNG ngữ cảnh thứ hai của cùng các mục: in thêm độ nhiễu của judge (lần 1 so lần 2) "
                                    "và so ngữ cảnh với lần 2")
    p.add_argument("--in-mau", type=int, default=0, help="in tối đa N ca đảo ngược / mới quyết được để người đọc")
    p.add_argument("--ra", default=None)
    a = p.parse_args(argv)
    ra = Path(a.ra) if a.ra else DL.RA
    ctx, khong = doc(ra / f"ket_qua_{a.ctx}.jsonl"), doc(ra / f"ket_qua_{a.khong}.jsonl")
    tong = len(pd.read_parquet(DL.RA / "muc_cum_ctx.parquet", columns=["item_id"]))
    r = so_sanh(ctx, khong, tong)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    ghi = {"ctx": a.ctx, "khong": a.khong, "so_sanh": r}
    if a.khong2:
        khong2 = doc(ra / f"ket_qua_{a.khong2}.jsonl")
        print("\n== ĐỘ NHIỄU CỦA JUDGE: cùng mục, cùng lời nhắc KHÔNG ngữ cảnh, hai lần chạy ==")
        lap = so_sanh_lap(khong, khong2)
        print(json.dumps(lap, ensure_ascii=False, indent=1))
        r2 = so_sanh(ctx, khong2, tong)
        r2 = {k: r2[k] for k in r2 if k not in ("moi_muc_trung_binh", "uoc_tinh_toan_bo_cum_ngu_canh")}
        print("\n== NGỮ CẢNH so với lần chạy KHÔNG ngữ cảnh THỨ HAI ==")
        print(json.dumps(r2, ensure_ascii=False, indent=1))
        ghi.update({"khong2": a.khong2, "do_nhieu_lan1_so_lan2": lap, "so_sanh_voi_lan2": r2})
    # ghi ra file để Claude đọc thẳng, khỏi dán từ terminal (kèm các ca đảo ngược để người đọc)
    ca = []
    muc = pd.read_parquet(DL.RA / "muc_cum_ctx.parquet", columns=["item_id", "nguon", "cand_gemma", "cand_qwen"]).set_index("item_id")
    for c in ctx.values():
        k = khong.get(c.get("id_khong_ngu_canh"))
        if k is None or c["ket_qua"] == k["ket_qua"] or c["item_id"] not in muc.index:
            continue
        m = muc.loc[c["item_id"]]
        ca.append({"item_id": c["item_id"], "khong_ngu_canh": k["ket_qua"], "ngu_canh": c["ket_qua"], "nguon": m.nguon,
                   "gemma": m.cand_gemma, "qwen": m.cand_qwen,
                   "ly_do_khong_ngu_canh": [l["ly_do"] for l in k["luot"]], "ly_do_ngu_canh": [l["ly_do"] for l in c["luot"]]})
    ghi["ca_khac_nhau"] = ca
    ten_ra = f"so_sanh_{a.ctx}_{a.khong}{'_' + a.khong2 if a.khong2 else ''}.json"
    (ra / ten_ra).write_text(json.dumps(ghi, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n(đã ghi {ra / ten_ra})")
    if a.in_mau and r.get("so_cap_so_duoc"):
        muc = pd.read_parquet(DL.RA / "muc_cum_ctx.parquet", columns=["item_id", "nguon", "cand_gemma", "cand_qwen"]).set_index("item_id")
        n = 0
        for c in ctx.values():
            k = khong.get(c.get("id_khong_ngu_canh"))
            if k is None or c["ket_qua"] == k["ket_qua"]:
                continue
            m = muc.loc[c["item_id"]]
            print(f"\n== {c['item_id']}: không ngữ cảnh = {k['ket_qua']} | có ngữ cảnh = {c['ket_qua']}")
            print("  nguồn:", m.nguon, "\n  Gemma:", m.cand_gemma, "\n  Qwen: ", m.cand_qwen)
            for nhan, d in (("không NC", k), ("có NC", c)):
                for l in d["luot"]:
                    print(f"  [{nhan} {l['thu_tu']}] {l['chon']} ({l['tieu_chi']}): {l['ly_do']}")
            n += 1
            if n >= a.in_mau:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
