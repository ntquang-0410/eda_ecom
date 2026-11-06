# Bước 3 của bộ 3 (laptop): đọc kết quả dịch cả cụm tải về từ Kaggle, tách mỗi cụm về từng cặp (theo pair_ids của cụm), kiểm luật từng
# cặp bằng chính luật của bộ 2 (song_ngu_2.luat_kiem.kiem_mot), rồi ghép với bản dịch đơn vị của bộ 2 (bản đã lên HF) trên cùng các cặp
# để so. Không chọn bản, không xuất bộ dữ liệu: bước này chỉ phục vụ quyết định "cả cụm có hơn rõ không" ở mẫu 200 sản phẩm.
import json
from pathlib import Path

import pandas as pd

from song_ngu_2.luat_kiem import kiem_mot
from song_ngu_2.van_ban import chuan_hoa_zh

from . import cau_hinh as C

MO_HINH = {"A": "api_tuy_chon", "B": "qwen3_14b"}      # A = Gemma 4 qua API, B = Qwen3-14B (như bộ 2)


def doc_lo(thu_muc, mo_hinh):
    """Gộp mọi lo_*.parquet của một mô hình trong các thư mục con tên mo_hinh (có thể nhiều phiên bản chạy nối tiếp); cụm trùng giữ
    bản cuối."""
    files = sorted(Path(thu_muc).rglob(f"{mo_hinh}/lo_*.parquet"))
    if not files:
        return pd.DataFrame(columns=["ma_cum", "ket_qua", "trang_thai"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("ma_cum", keep="last")


def trai_phang(CUM, KQ, hau_to):
    """Tách mỗi cụm về từng cặp. Cụm chưa dịch hoặc sai cấu trúc: tên và giá trị Trung để None, ghi trạng thái."""
    kq = KQ.set_index("ma_cum")
    rows = []
    for r in CUM.itertuples(index=False):
        muc = json.loads(r.muc_json)
        pids = json.loads(r.pair_ids)
        dich, tt = None, "chua_dich"
        if r.ma_cum in kq.index:
            tt = kq.at[r.ma_cum, "trang_thai"]
            if tt == "ok":
                dich = {m["i"]: m for m in json.loads(kq.at[r.ma_cum, "ket_qua"])}
        for m, pid in zip(muc, pids):
            d = dich.get(m["i"]) if dich else None
            rows.append({"pair_id": pid, "product_id": r.product_id, "category": r.category, "ma_cum": r.ma_cum, "i": m["i"],
                         "name_vi": m["n"], "gia_tri_vi": m["v"], f"name_zh_{hau_to}": d["n"] if d else None,
                         f"gia_tri_zh_{hau_to}": d["v"] if d else None, f"trang_thai_{hau_to}": tt})
    return pd.DataFrame(rows)


def kiem_cap(P, hau_to):
    """Luật bộ 2 cho tên và giá trị của một bản dịch cả cụm. Thêm cột truot_ten_X, truot_gt_X (danh sách mã trượt; ['thieu'] khi chưa có)."""
    for cot_vi, cot_zh, loai, ra in (("name_vi", f"name_zh_{hau_to}", "ten", f"truot_ten_{hau_to}"),
                                     ("gia_tri_vi", f"gia_tri_zh_{hau_to}", "gia_tri", f"truot_gt_{hau_to}")):
        P[ra] = [kiem_mot(v, v, z, None, loai)[0] if isinstance(z, str) else ["thieu"] for v, z in zip(P[cot_vi], P[cot_zh])]
    return P


def giong(a, b):
    return isinstance(a, str) and isinstance(b, str) and chuan_hoa_zh(a) == chuan_hoa_zh(b)


def ghep_bo_2(P, file_bo_2=C.FILE_BO_2_DAY_DU):
    """Ghép bản dịch đơn vị của bộ 2 (bản cuối sau từ điển và chọn bản; cặp bị cất ở bộ 2 thì để None)."""
    B2 = pd.read_parquet(file_bo_2, columns=["pair_id", "name_zh", "gia_tri_zh", "gia_tri_zh_A", "gia_tri_zh_B", "cach_chon",
                                             "do_tin_cay"])
    B2 = B2.rename(columns={"name_zh": "name_zh_don_vi", "gia_tri_zh": "gia_tri_zh_don_vi", "gia_tri_zh_A": "gia_tri_zh_don_vi_A",
                            "gia_tri_zh_B": "gia_tri_zh_don_vi_B", "cach_chon": "cach_chon_bo_2", "do_tin_cay": "do_tin_cay_bo_2"})
    return P.merge(B2, on="pair_id", how="left")


def lap_bang(CUM, thu_muc_kaggle, file_bo_2=C.FILE_BO_2_DAY_DU):
    """Bảng một dòng một cặp cho các cụm trong CUM: bản cả cụm A, B (kèm luật) và bản đơn vị bộ 2."""
    P = None
    for k, m in MO_HINH.items():
        Pk = kiem_cap(trai_phang(CUM, doc_lo(thu_muc_kaggle, m), k), k)
        P = Pk if P is None else P.merge(Pk[["pair_id"] + [c for c in Pk.columns if c.endswith(f"_{k}")]], on="pair_id")
    P = ghep_bo_2(P, file_bo_2)
    for k in MO_HINH:
        P[f"gt_giong_don_vi_{k}"] = [giong(a, b) for a, b in zip(P[f"gia_tri_zh_{k}"], P.gia_tri_zh_don_vi)]
        P[f"ten_giong_don_vi_{k}"] = [giong(a, b) for a, b in zip(P[f"name_zh_{k}"], P.name_zh_don_vi)]
    P["gt_A_giong_B"] = [giong(a, b) for a, b in zip(P.gia_tri_zh_A, P.gia_tri_zh_B)]
    return P


def tom_tat(CUM, P):
    """Số liệu so sánh chính (theo cụm và theo cặp)."""
    out = {"cum": int(len(CUM)), "san_pham": int(CUM.product_id.nunique()), "cap": int(len(P))}
    for k in MO_HINH:
        tt = P.drop_duplicates("ma_cum")[f"trang_thai_{k}"].value_counts().to_dict()
        co = P[f"gia_tri_zh_{k}"].notna()
        out[k] = {"cum_theo_trang_thai": tt,
                  "cap_co_ban_dich": int(co.sum()),
                  "cap_truot_luat_gia_tri": int(sum(bool(x) and x != ["thieu"] for x in P[f"truot_gt_{k}"])),
                  "cap_truot_luat_ten": int(sum(bool(x) and x != ["thieu"] for x in P[f"truot_ten_{k}"])),
                  "gia_tri_giong_ban_don_vi": int(P[f"gt_giong_don_vi_{k}"].sum()),
                  "ten_giong_ban_don_vi": int(P[f"ten_giong_don_vi_{k}"].sum())}
    out["gia_tri_A_giong_B"] = int(P.gt_A_giong_B.sum())
    out["cap_khong_co_o_bo_2"] = int(P.gia_tri_zh_don_vi.isna().sum())
    return out
