# Bước 1-3 của bộ 2 (laptop, CPU): nạp bản CADO của Tiki, tách thuộc tính thành cặp (tên, giá trị), phân loại, gộp trùng.
# Ra: bang_cap.parquet (mọi cặp, kèm cách xử lý) và vao_kaggle/don_vi_can_dich.parquet (đơn vị gửi mô hình, cột nguồn là tiếng Việt).
import hashlib
import json
import re
import shutil
import time
import unicodedata
from collections import Counter

import pandas as pd

from . import cau_hinh as C
from .dinh_tuyen import dinh_tuyen


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest().lower()


def mode(s):
    c = Counter(s)
    m = max(c.values())
    return sorted(k for k, v in c.items() if v == m)[0]


def nfc(s):
    """NFC và gọn khoảng trắng: chữ Việt có dấu viết dạng tổ hợp khác nhau sẽ làm mô hình và CometKiwi thấy hai chuỗi khác nhau."""
    return " ".join(unicodedata.normalize("NFC", str(s)).split())


# Làm sạch giống bước EDA của nhóm (cột bt_attributes_text): bỏ emoji và hashtag, bỏ gạch đầu dòng "-" ở hai đầu giá trị, bỏ dấu ":" cuối tên.
# Thử trên cả bộ: ghép lại các cặp sau làm sạch khớp bt_attributes_text 13.141/13.200 sản phẩm (59 sản phẩm lệch vài dấu câu như "...", "_", "***").
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")      # ký hiệu cảm xúc, dingbat, bộ chọn biến thể, ZWJ
HASHTAG = re.compile(r"(?<!\w)#[^\s#]+")


def lam_gia_tri(v):
    v = EMOJI.sub(" ", str(v))
    v = HASHTAG.sub(" ", v)
    v = " ".join(unicodedata.normalize("NFC", v).split())
    return re.sub(r"^-+\s*|\s*-+$", "", v).strip()


def lam_ten(n):
    n = " ".join(unicodedata.normalize("NFC", EMOJI.sub(" ", str(n))).split())
    return n.rstrip(":").strip()


def tach_cap(P):
    """Mỗi sản phẩm có attributes_vi là danh sách {name, values[]}: mỗi phần tử thành một cặp; nhiều giá trị nối bằng ', ' (như
    bt_attributes_text của bản EDA). Giá trị sau làm sạch rỗng (toàn emoji/hashtag) thì bỏ cả cặp, giống bước EDA. Giữ split và row_hash của nguồn."""
    rows = []
    for pid, cat, at, sp, rh in zip(P.product_id, P.category, P.attributes_vi, P.split, P.row_hash):
        for k, a in enumerate(at if at is not None else []):
            vs = [t for t in (lam_gia_tri(x) for x in a["values"]) if t]
            ten = lam_ten(a["name"])
            if vs and ten:
                rows.append((pid, cat, k, ten, C.DAU_NGAN_GIA_TRI.join(vs), len(vs), sp, rh))
    B = pd.DataFrame(rows, columns=["product_id", "category", "vi_tri", "name_vi", "gia_tri_vi", "so_gia_tri", "split_nguon", "row_hash"])
    B.insert(0, "pair_id", range(len(B)))
    return B


def chay(file_nguon=C.FILE_NGUON, thu_muc_ra=C.THU_MUC_RA, in_ra=print):
    t0 = time.time()
    thu_muc_ra.mkdir(parents=True, exist_ok=True)
    vao_kaggle = thu_muc_ra / "vao_kaggle"
    vao_kaggle.mkdir(parents=True, exist_ok=True)

    # Bước 1: nạp, ghi dấu vân tay, kiểm lược đồ
    sha = sha256(file_nguon)
    if sha != C.SHA256_FILE_NGUON:
        raise SystemExit(f"SHA256 của {file_nguon.name} là {sha}, khác bản đã khảo sát ({C.SHA256_FILE_NGUON}): dừng, chờ người xem.")
    P = pd.read_parquet(file_nguon, columns=C.COT_BAT_BUOC)
    if not P.product_id.is_unique:
        raise SystemExit("product_id bị trùng: dừng.")
    in_ra(f"Bước 1: {len(P):,} sản phẩm, SHA-256 {sha[:16]}...")

    # Bước 2: tách cặp, chuẩn hoá chữ, bỏ cặp rỗng
    B = tach_cap(P)
    rong = (B.name_vi == "") | (B.gia_tri_vi == "")
    n_goc = int(sum(len(a) for a in P.attributes_vi if a is not None))
    in_ra(f"Bước 2: {len(B):,} cặp thuộc tính từ {B.product_id.nunique():,} sản phẩm (nguồn có {n_goc:,} cặp; "
          f"{n_goc - len(B):,} cặp bỏ vì giá trị toàn emoji/hashtag); tên hoặc giá trị rỗng sau chuẩn hoá: {int(rong.sum())}")
    B["trang_thai_vao"] = "dung"
    B.loc[rong, "trang_thai_vao"] = "van_loai"

    # Bước 3: định tuyến và gộp trùng
    kq = [dinh_tuyen(n, v) for n, v in zip(B.name_vi, B.gia_tri_vi)]
    B["dinh_tuyen"] = [k[0] for k in kq]
    B["ly_do_van_loai"] = [k[1] for k in kq]
    B.loc[rong, ["dinh_tuyen", "ly_do_van_loai"]] = ["van_loai", "rong"]
    B.loc[B.dinh_tuyen.eq("van_loai"), "trang_thai_vao"] = "van_loai"
    dung = B.trang_thai_vao.eq("dung")
    quy_tac = B.dinh_tuyen.isin(["chep_nguyen", "ten_hang", "ma_model", "do_luong", "co_chu"])
    B["gia_tri_zh_quy_tac"] = B.gia_tri_vi.where(quy_tac & dung, None)       # các route này chép nguyên giá trị

    gt = B[dung & B.dinh_tuyen.eq("mo_hinh")]
    dv_gt = (gt.groupby(["name_vi", "gia_tri_vi"]).agg(category=("category", mode), so_cap=("pair_id", "size"))
             .reset_index().sort_values(["name_vi", "gia_tri_vi"]).reset_index(drop=True))
    dv_gt["ma_don_vi"] = [f"G{i:06d}" for i in range(len(dv_gt))]
    dv_gt["loai"] = "gia_tri"
    dv_gt["name_nguon"] = dv_gt.name_vi
    dv_gt["gia_tri_nguon"] = dv_gt.gia_tri_vi
    dv_gt["van_ban_nguon"] = dv_gt.gia_tri_vi
    tn = B[dung]
    dv_tn = (tn.groupby("name_vi").agg(category=("category", mode), so_cap=("pair_id", "size"))
             .reset_index().sort_values("name_vi").reset_index(drop=True))
    dv_tn["ma_don_vi"] = [f"T{i:05d}" for i in range(len(dv_tn))]
    dv_tn["loai"] = "ten"
    dv_tn["name_nguon"] = dv_tn.name_vi
    dv_tn["gia_tri_nguon"] = ""
    dv_tn["van_ban_nguon"] = dv_tn.name_vi
    DV = pd.concat([dv_tn, dv_gt], ignore_index=True)
    DV["van_ban_gui"] = DV.van_ban_nguon
    DV = DV[["ma_don_vi", "loai", "name_nguon", "gia_tri_nguon", "van_ban_nguon", "van_ban_gui", "category", "so_cap"]]

    ma_gt = dict(zip(zip(dv_gt.name_vi, dv_gt.gia_tri_vi), dv_gt.ma_don_vi))
    ma_tn = dict(zip(dv_tn.name_vi, dv_tn.ma_don_vi))
    B["ma_don_vi_gia_tri"] = [ma_gt.get((n, v)) if d == "mo_hinh" and t == "dung" else None
                              for n, v, d, t in zip(B.name_vi, B.gia_tri_vi, B.dinh_tuyen, B.trang_thai_vao)]
    B["ma_don_vi_ten"] = [ma_tn.get(n) if t == "dung" else None for n, t in zip(B.name_vi, B.trang_thai_vao)]

    B.to_parquet(thu_muc_ra / "bang_cap.parquet", index=False)
    DV.to_parquet(vao_kaggle / "don_vi_can_dich.parquet", index=False)
    for f in ["dich.py", "cham_kiwi.py"]:
        shutil.copy2(C.THU_MUC_MA_KAGGLE / f, vao_kaggle / f)
    (vao_kaggle / "cau_hinh_kaggle.json").write_text(json.dumps({"mo_hinh": C.MO_HINH, "huong": C.HUONG}, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    the = {
        "buoc": "song_ngu_2.chuan_bi (1-3)", "ngay_chay": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_nguon": file_nguon.name, "sha256_file_nguon": sha, "so_san_pham": int(len(P)), "so_cap": int(len(B)),
        "so_cap_goc_truoc_lam_sach": n_goc, "dau_ngan_gia_tri": C.DAU_NGAN_GIA_TRI,
        "split_nguon_san_pham": P.split.value_counts().to_dict(),
        "trang_thai_vao": B.trang_thai_vao.value_counts().to_dict(),
        "ly_do_van_loai": B.loc[~dung, "ly_do_van_loai"].value_counts().to_dict(),
        "dinh_tuyen_cap_dung": B.loc[dung, "dinh_tuyen"].value_counts().to_dict(),
        "don_vi": {"ten": int(len(dv_tn)), "gia_tri": int(len(dv_gt)), "ky_tu_viet_gia_tri": int(dv_gt.gia_tri_vi.str.len().sum())},
        "giay_to_kem": ["don_vi_can_dich.parquet", "dich.py", "cham_kiwi.py", "cau_hinh_kaggle.json"],
    }
    (thu_muc_ra / "the_chuan_bi.json").write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    in_ra(f"Bước 3: phân loại { {k: int(v) for k, v in B.loc[dung, 'dinh_tuyen'].value_counts().items()} }; cất riêng "
          f"{ {k: int(v) for k, v in B.loc[~dung, 'ly_do_van_loai'].value_counts().items()} }")
    in_ra(f"        đơn vị gửi mô hình: {len(dv_tn):,} tên thuộc tính + {len(dv_gt):,} giá trị ({dv_gt.gia_tri_vi.str.len().sum() / 1e6:.2f} triệu ký tự Việt)")
    in_ra(f"Xong sau {time.time() - t0:.0f} giây. Thư mục tải lên Kaggle: {vao_kaggle}")
    return B, DV, the
