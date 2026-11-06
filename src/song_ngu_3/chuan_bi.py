# Bước 1 của bộ 3 (laptop): gom các cặp thuộc tính của bộ 2 theo từng sản phẩm, đúng thứ tự trên trang (vi_tri), thành "cụm" để dịch một
# lần. Lọc riêng tư TRƯỚC khi đưa vào mô hình: bỏ cặp bộ 2 đã cất riêng ở bước định tuyến (tên, địa chỉ đơn vị chịu trách nhiệm, liên hệ)
# và cặp bộ lọc rieng_tu bắt được (tên shop, địa chỉ đường phố). Cặp bị cất ở bộ 2 vì KHÔNG CHỌN ĐƯỢC BẢN DỊCH vẫn đưa vào (không riêng tư).
# Sản phẩm dài thì chia phần (cau_hinh.MAX_KY_TU_PHAN). Rút mẫu thử 200 sản phẩm chia đều theo danh mục, hạt giống cố định.
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd

from song_ngu_2.rieng_tu import rieng_tu

from . import cau_hinh as C


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def chia_phan(muc, max_ky_tu=C.MAX_KY_TU_PHAN):
    """muc: danh sách (pair_id, ten, gia_tri) theo thứ tự trên trang. Trả về danh sách phần, mỗi phần là danh sách liên tiếp có tổng
    ký tự (tên + giá trị) không quá max_ky_tu; một mục một mình đã quá thì thành một phần riêng. Không đổi thứ tự, không bỏ mục."""
    phan, cur, n = [], [], 0
    for m in muc:
        d = len(m[1]) + len(m[2])
        if cur and n + d > max_ky_tu:
            phan.append(cur)
            cur, n = [], 0
        cur.append(m)
        n += d
    if cur:
        phan.append(cur)
    return phan


def ten_khac(phan, k, max_ky_tu=C.MAX_KY_TU_TEN_KHAC):
    """Tên các thuộc tính ở những phần khác của cùng sản phẩm (ngữ cảnh cho phần k), bỏ trùng, cắt ở max_ky_tu."""
    if len(phan) == 1:
        return ""
    ten, thay = [], set()
    for j, p in enumerate(phan):
        if j != k:
            for _, t, _ in p:
                if t not in thay:
                    thay.add(t)
                    ten.append(t)
    s = "; ".join(ten)
    return s if len(s) <= max_ky_tu else s[:max_ky_tu].rsplit("; ", 1)[0] + "; ..."


def rut_mau(sp, n=None, hat=C.HAT_GIONG):
    """sp: DataFrame một dòng một sản phẩm (product_id, category). Chia đều n (mặc định cau_hinh.SO_SAN_PHAM_THU) theo danh mục (danh mục
    xếp theo tên; phần dư cho các danh mục đầu), trong mỗi danh mục rút ngẫu nhiên với hạt giống cố định. Trả về tập product_id."""
    n = n or C.SO_SAN_PHAM_THU
    dm = sorted(sp.category.unique())
    co_so, du = divmod(n, len(dm))
    rng = np.random.RandomState(hat)
    chon = []
    for k, c in enumerate(dm):
        ids = sorted(sp.loc[sp.category == c, "product_id"])
        chon += list(rng.choice(ids, size=min(len(ids), co_so + (1 if k < du else 0)), replace=False))
    return set(chon)


def lap_cum(B):
    """B: bảng cặp đã lọc. Trả về DataFrame một dòng một phần (cụm) cần dịch."""
    rows = []
    for (pid, cat), g in B.sort_values(["product_id", "vi_tri"]).groupby(["product_id", "category"], sort=True):
        muc = list(zip(g.pair_id, g.name_vi, g.gia_tri_vi))
        phan = chia_phan(muc)
        for k, p in enumerate(phan):
            rows.append({"ma_cum": f"{pid}#{k}", "product_id": pid, "category": cat, "phan": k, "so_phan": len(phan),
                         "muc_json": json.dumps([{"i": i, "n": t, "v": v} for i, (_, t, v) in enumerate(p, 1)], ensure_ascii=False),
                         "pair_ids": json.dumps([int(x) for x, _, _ in p]), "ten_khac": ten_khac(phan, k),
                         "so_muc": len(p), "so_ky_tu": sum(len(t) + len(v) for _, t, v in p)})
    return pd.DataFrame(rows)


def chay(thu_muc_ra=C.THU_MUC_RA, file_bang_cap=C.FILE_BANG_CAP, file_dich=None, file_dich_cum=None, so_mau=None, in_ra=print):
    t0 = time.time()
    thu_muc_ra.mkdir(parents=True, exist_ok=True)
    B0 = pd.read_parquet(file_bang_cap)
    B = B0[B0.trang_thai_vao == "dung"]
    ly = [rieng_tu(n, v) for n, v in zip(B.name_vi, B.gia_tri_vi)]
    bo_rieng = int(sum(x is not None for x in ly))
    B = B[[x is None for x in ly]]
    CUM = lap_cum(B)
    sp = CUM.drop_duplicates("product_id")[["product_id", "category"]]
    mau = rut_mau(sp, so_mau)
    CUM["mau_thu"] = CUM.product_id.isin(mau)
    vao = thu_muc_ra / "vao_kaggle"
    vao.mkdir(parents=True, exist_ok=True)
    CUM.to_parquet(vao / "cum_can_dich.parquet", index=False)
    (vao / "cau_hinh_kaggle.json").write_text(json.dumps({"huong": "vi_zh", "loai": "cum", "phien_ban_cau_lenh": C.PHIEN_BAN_CAU_LENH},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    goc = C.ROOT / "src"
    for f, ten in ((file_dich or goc / "song_ngu_1" / "kaggle" / "dich.py", "dich.py"),
                   (file_dich_cum or goc / "song_ngu_3" / "kaggle" / "dich_cum.py", "dich_cum.py")):
        shutil.copyfile(f, vao / ten)
    m = CUM[CUM.mau_thu]
    the = {"buoc": "song_ngu_3.chuan_bi", "ngay_chay": time.strftime("%Y-%m-%d %H:%M:%S"),
           "file_bang_cap": file_bang_cap.name, "sha256_bang_cap": sha(file_bang_cap),
           "cap_nguon": int(len(B0)), "cap_bo_vi_van_loai": int((B0.trang_thai_vao != "dung").sum()), "cap_bo_vi_rieng_tu": bo_rieng,
           "cap_dua_vao": int(len(B)), "san_pham": int(CUM.product_id.nunique()), "cum": int(len(CUM)),
           "san_pham_nhieu_phan": int((CUM.drop_duplicates("product_id").so_phan > 1).sum()), "max_ky_tu_phan": C.MAX_KY_TU_PHAN,
           "mau_thu": {"san_pham": int(m.product_id.nunique()), "cum": int(len(m)), "cap": int(m.so_muc.sum()),
                       "theo_danh_muc": m.drop_duplicates("product_id").category.value_counts().sort_index().to_dict(),
                       "hat_giong": C.HAT_GIONG},
           "phien_ban_cau_lenh": C.PHIEN_BAN_CAU_LENH}
    (thu_muc_ra / "the_chuan_bi.json").write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    in_ra(f"Cặp đưa vào {the['cap_dua_vao']:,} (bỏ {the['cap_bo_vi_van_loai']:,} vì định tuyến cất riêng, {bo_rieng} vì lọc riêng tư); "
          f"{the['san_pham']:,} sản phẩm, {the['cum']:,} cụm ({the['san_pham_nhieu_phan']} sản phẩm chia nhiều phần)")
    in_ra(f"Mẫu thử: {the['mau_thu']['san_pham']} sản phẩm, {the['mau_thu']['cum']} cụm, {the['mau_thu']['cap']:,} cặp; "
          f"theo danh mục {the['mau_thu']['theo_danh_muc']} ({time.time() - t0:.0f} giây)")
    return CUM, the
