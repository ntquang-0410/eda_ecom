# Bước 1-3 (chạy trên laptop, CPU): nạp file cặp, chọn cặp và bỏ hẳn bản Việt cũ, phân loại, gộp trùng.
# Ra: bang_cap.parquet (mọi cặp, kèm cách xử lý) và vao_kaggle/don_vi_can_dich.parquet (đơn vị gửi mô hình).
import hashlib
import json
import re
import shutil
import time
from collections import Counter

import pandas as pd

from . import cau_hinh as C
from .dinh_tuyen import dinh_tuyen, doc_tu_chung, van_ban_gui
from .van_ban import co_han

PII = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)0\d{2,3}-\d{7,8}(?!\d)|[\w.+-]+@[\w-]+\.[\w.]+|https?://|www\.", re.I)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest().upper()


def ma_cua(ly_do):
    return [m for m in (ly_do or "").split(";") if m]


def phan_loai_vao(ly_do):
    """'dung' (làm đầu vào), 'van_loai' hoặc 'cho_xem'. Mã lạ thì báo lỗi để người xem, không tự đoán."""
    ma = set(ma_cua(ly_do))
    la = ma - C.MA_VAN_LOAI - C.MA_NHAN_LAI - C.MA_CHO_XEM
    if la:
        raise SystemExit(f"Gặp mã loại chưa có trong cau_hinh.py: {sorted(la)}. Dừng, chờ người quyết định.")
    if ma & C.MA_VAN_LOAI:
        return "van_loai"
    if ma & C.MA_CHO_XEM:
        return "cho_xem"
    return "dung"


def mode(s):
    c = Counter(s)
    m = max(c.values())
    return sorted(k for k, v in c.items() if v == m)[0]


def chay(file_cap=C.FILE_CAP, file_san_pham=C.FILE_SAN_PHAM, thu_muc_ra=C.THU_MUC_RA, in_ra=print):
    t0 = time.time()
    thu_muc_ra.mkdir(parents=True, exist_ok=True)
    vao_kaggle = thu_muc_ra / "vao_kaggle"
    vao_kaggle.mkdir(parents=True, exist_ok=True)

    # Bước 1: nạp, ghi dấu vân tay, kiểm lược đồ
    sha = sha256(file_cap)
    P = pd.read_parquet(file_cap)
    thieu = [c for c in C.COT_BAT_BUOC if c not in P.columns]
    if thieu:
        raise SystemExit(f"File cặp thiếu cột {thieu}: dừng.")
    if not P.pair_id.is_unique:
        raise SystemExit("pair_id bị trùng: dừng.")
    for c in ["name_zh", "gia_tri_zh", "nhan", "ly_do"]:
        P[c] = P[c].fillna("").astype(str)
    in_ra(f"Bước 1: {len(P):,} cặp, SHA-256 {sha[:16]}...")

    # Bước 2: bỏ hẳn các cột Việt cũ, lọc theo mã loại, quét lại thông tin cá nhân phía Trung
    B = P[["pair_id", "product_id", "category", "vi_tri", "name_zh", "gia_tri_zh", "vai_tro", "nhan", "ly_do"]].copy()
    B = B.rename(columns={"ly_do": "ly_do_goc"})
    if file_san_pham is not None and file_san_pham.exists():
        sp = pd.read_parquet(file_san_pham, columns=["product_id", "nhom_trung"])
        B = B.merge(sp, on="product_id", how="left")
    else:
        B["nhom_trung"] = pd.NA
    B["trang_thai_vao"] = B.ly_do_goc.map(phan_loai_vao)
    B["pii_sot"] = (B.name_zh + " " + B.gia_tri_zh).map(lambda s: bool(PII.search(s)))
    B.loc[B.pii_sot & B.trang_thai_vao.eq("dung"), "trang_thai_vao"] = "van_loai"
    dem_ma = Counter(m for ld in P.ly_do for m in ma_cua(ld))
    in_ra(f"Bước 2: dùng {int((B.trang_thai_vao == 'dung').sum()):,} cặp; vẫn loại {int((B.trang_thai_vao == 'van_loai').sum()):,}; "
          f"cất riêng chờ xem {int((B.trang_thai_vao == 'cho_xem').sum()):,}; thông tin cá nhân còn sót {int(B.pii_sot.sum()):,}")

    # Bước 3: phân loại (quy tắc hay mô hình) và gộp trùng
    tu_chung = doc_tu_chung(C.FILE_TU_CHUNG)
    kq = [dinh_tuyen(n, z, nh, tu_chung) for n, z, nh in zip(B.name_zh, B.gia_tri_zh, B.nhan)]
    B["dinh_tuyen"] = [k[0] for k in kq]
    B["gia_tri_vi_quy_tac"] = [k[1] for k in kq]
    dung = B.trang_thai_vao.eq("dung")

    # đơn vị giá trị: (tên, cả chuỗi giá trị) của các cặp cần mô hình
    gt = B[dung & B.dinh_tuyen.eq("mo_hinh")]
    dv_gt = (gt.groupby(["name_zh", "gia_tri_zh"]).agg(category=("category", mode), so_cap=("pair_id", "size"))
             .reset_index().sort_values(["name_zh", "gia_tri_zh"]).reset_index(drop=True))
    dv_gt["ma_don_vi"] = [f"G{i:06d}" for i in range(len(dv_gt))]
    dv_gt["loai"] = "gia_tri"
    dv_gt["van_ban_zh"] = dv_gt.gia_tri_zh
    # đơn vị tên thuộc tính: tên có chữ Hán của các cặp được dùng (cả cặp đi theo quy tắc cũng cần tên Việt)
    tn = B[dung & B.name_zh.map(co_han)]
    dv_tn = (tn.groupby("name_zh").agg(category=("category", mode), so_cap=("pair_id", "size"))
             .reset_index().sort_values("name_zh").reset_index(drop=True))
    dv_tn["ma_don_vi"] = [f"T{i:05d}" for i in range(len(dv_tn))]
    dv_tn["loai"] = "ten"
    dv_tn["gia_tri_zh"] = ""
    dv_tn["van_ban_zh"] = dv_tn.name_zh
    DV = pd.concat([dv_tn, dv_gt], ignore_index=True)
    DV["van_ban_gui"] = DV.van_ban_zh.map(van_ban_gui)
    DV = DV[["ma_don_vi", "loai", "name_zh", "gia_tri_zh", "van_ban_zh", "van_ban_gui", "category", "so_cap"]]

    ma_gt = dict(zip(zip(dv_gt.name_zh, dv_gt.gia_tri_zh), dv_gt.ma_don_vi))
    ma_tn = dict(zip(dv_tn.name_zh, dv_tn.ma_don_vi))
    B["ma_don_vi_gia_tri"] = [ma_gt.get((n, z)) if d == "mo_hinh" and t == "dung" else None
                              for n, z, d, t in zip(B.name_zh, B.gia_tri_zh, B.dinh_tuyen, B.trang_thai_vao)]
    B["ma_don_vi_ten"] = [ma_tn.get(n) if t == "dung" else None for n, t in zip(B.name_zh, B.trang_thai_vao)]

    # đối chiếu phần đi theo quy tắc với bản v1 (cùng quy tắc nên phải gần như trùng): chỉ để kiểm mã, không dùng bản Việt cũ
    doi_chieu = {}
    if "gia_tri_vi" in P.columns:
        cu = P.set_index("pair_id").gia_tri_vi
        for loai in ["ma_that", "ten_hang"]:
            m = dung & B.dinh_tuyen.eq(loai)
            if m.any():
                doi_chieu[loai] = round(float((B.loc[m, "gia_tri_vi_quy_tac"].values == cu.loc[B.loc[m, "pair_id"]].values).mean()), 4)

    B.to_parquet(thu_muc_ra / "bang_cap.parquet", index=False)
    DV.to_parquet(vao_kaggle / "don_vi_can_dich.parquet", index=False)
    for f in ["dich.py", "cham_kiwi.py"]:
        shutil.copy2(C.THU_MUC_MA_KAGGLE / f, vao_kaggle / f)
    (vao_kaggle / "cau_hinh_kaggle.json").write_text(json.dumps({"mo_hinh": C.MO_HINH}, ensure_ascii=False, indent=1), encoding="utf-8")

    the = {
        "buoc": "chuan_bi (1-3)", "ngay_chay": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_cap": str(file_cap.name), "sha256_file_cap": sha, "so_cap": int(len(P)),
        "ma_loai_ban_dau": dict(dem_ma),
        "trang_thai_vao": B.trang_thai_vao.value_counts().to_dict(),
        "dinh_tuyen_cap_dung": B.loc[dung, "dinh_tuyen"].value_counts().to_dict(),
        "don_vi": {"ten": int(len(dv_tn)), "gia_tri": int(len(dv_gt)),
                   "ky_tu_trung_gia_tri": int(dv_gt.gia_tri_zh.str.len().sum())},
        "khop_voi_ban_v1_phan_quy_tac": doi_chieu,
        "giay_to_kem": ["don_vi_can_dich.parquet", "dich.py", "cham_kiwi.py", "cau_hinh_kaggle.json"],
    }
    (thu_muc_ra / "the_chuan_bi.json").write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    in_ra(f"Bước 3: phân loại { {k: int(v) for k, v in B.loc[dung, 'dinh_tuyen'].value_counts().items()} }")
    in_ra(f"        đơn vị gửi mô hình: {len(dv_tn):,} tên thuộc tính + {len(dv_gt):,} giá trị "
          f"({dv_gt.gia_tri_zh.str.len().sum() / 1e6:.2f} triệu ký tự Trung)")
    in_ra(f"        phần quy tắc khớp bản v1: {doi_chieu}")
    in_ra(f"Xong sau {time.time() - t0:.0f} giây. Thư mục tải lên Kaggle: {vao_kaggle}")
    return B, DV, the
