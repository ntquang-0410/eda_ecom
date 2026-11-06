"""Tổng hợp thử tỉ lệ title trong loss (bước 2, fold 0): áp cổng và viết báo cáo. Chạy trên máy sau khi tải kết quả Kaggle về.

Chạy:  python -I -X utf8 scripts/ft_train/tong_hop_ti_le.py --trang-thai <tl_trang_thai.json> --ra <thu_muc>
Ghi <thu_muc>/bao_cao_ti_le.md và bao_cao_ti_le.json, in tóm tắt. Chỉ kiểm mốc: thêm --chi-kiem-moc (không cần trạng thái).

Mốc (~50% title) được chấm lại bằng cùng hàm từ bản dịch đã có; mỗi mức thử (~11%, 75%) được chấm lại từ <ten_viec>/sinh_ep2.jsonl
nằm cạnh file trạng thái, rồi so theo cặp với mốc bằng bootstrap_chenh trên đúng 1.000 mẫu cố định (không tự chọn lại).
Công cụ chỉ báo, không quyết định chạy thêm GPU. Báo cáo chỉ đọc số, không kết luận mô hình tốt hay ổn.
KHÔNG đọc test_khoa, split.parquet, ds_test.txt: chỉ đọc pool_*.jsonl và các bản dịch của fold 0."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tong_hop_b78 import _lay, _so, vuot  # noqa: E402

# ---------------------------------------------------------------- ngưỡng [MẶC ĐỊNH]
NGUONG_LECH = 1.0            # một trong title, mô tả, zh->vi tụt quá mức này (điểm chrF) so với mốc thì coi là học lệch
DUNG_SAI_MOC = 1e-6          # tự chấm mốc phải khớp số đã biết trong sai số này
SO_MOI_NHOM, SO_MAU, FOLD, SEED = 250, 1000, 0, 42
MOC_KY_VONG = {"cum": 75.76131971214022, "pair": 75.25867040638212}   # chrf_macro_huong của mốc

QUA, KHONG_QUA, CHUA_DU = "qua", "khong_qua", "chua_du_du_lieu"
KHOA_LECH = ("title", "mo_ta", "zh_vi")
TEN_LECH = {"title": "title", "mo_ta": "mô tả", "zh_vi": "zh->vi"}
NHANH = ("cum", "pair")
MUC = ("p11", "p75")
NHAN_MUC = {"moc": "mốc ~50%", "p11": "~11%", "p75": "75%"}
VIEC_MUC = {"cum": {"p11": "cum_f0_p11", "p75": "cum_f0_p75"}, "pair": {"p11": "pair_f0_p11", "p75": "pair_f0_p75"}}
NHOM_LOSS = ("zh->vi|title", "zh->vi|mo_ta", "vi->zh|title", "vi->zh|mo_ta")

_REPO = Path(__file__).resolve().parents[2]
_FT = _REPO / "data" / "processed" / "ft_run"
DUONG_DAN_THAT = {
    "moc_cum": _FT / "kaggle_b6" / "ket_qua_cuoi" / "b6" / "lr2e-4_r64" / "sinh_ep2.jsonl",
    "moc_pair": _FT / "kaggle_b78" / "phien_1_20261105_181337" / "b78" / "pair_f0" / "sinh_ep2.jsonl",
    "moc_trang_thai": _FT / "kaggle_b78" / "phien_1_20261105_181337" / "b78" / "b78_trang_thai.json",
    "pool_cum": _FT / "mau" / "pool_cum.jsonl",
    "pool_pair": _FT / "mau" / "pool_pair.jsonl",
}
COT_DA_THAY = ["loai", "category", "nguon", "nhanh", "fold5"]


# ---------------------------------------------------------------- hàm thuần
def khong_hoc_lech(moc, muc):
    """moc, muc: {"title", "mo_ta", "zh_vi"} (chrF corpus; title, mo_ta là trung bình hai hướng, zh_vi là theo hướng).
    Mỗi chỉ số tụt quá NGUONG_LECH so với mốc thì không đạt; đúng bằng ngưỡng thì đạt. Có chỉ số vượt ngưỡng thì dat=False dù còn thiếu số khác;
    không vượt mà thiếu số thì dat=None (không coi là đạt). Trả {"dat", "ly_do", "chenh"} với chenh = muc trừ moc."""
    chenh, ly_do, thieu, vuot_nguong = {}, [], [], False
    for k in KHOA_LECH:
        a, b = _so(_lay(moc, k)), _so(_lay(muc, k))
        if a is None or b is None:
            chenh[k] = None
            thieu.append(TEN_LECH[k])
            continue
        chenh[k] = b - a
        if vuot(a - b, NGUONG_LECH):
            vuot_nguong = True
            ly_do.append(f"{TEN_LECH[k]} tụt {a - b:.2f} điểm chrF so với mốc, quá ngưỡng {NGUONG_LECH} [MẶC ĐỊNH]")
    if vuot_nguong:
        dat = False
    elif thieu:
        dat = None
        ly_do.append("thiếu số: " + ", ".join(thieu))
    else:
        dat = True
        ly_do.append(f"title, mô tả, zh->vi đều không tụt quá {NGUONG_LECH} điểm so với mốc")
    return {"dat": dat, "ly_do": ly_do, "chenh": chenh}


def qua_cong_fold0(so_cap, lech):
    """Qua khi cận dưới khoảng 95% của chênh (mức trừ mốc) > 0 và lech["dat"] là True. Có bằng chứng không qua thì "khong_qua"
    (dù còn thiếu phần khác); không thì thiếu số là "chua_du_du_lieu". Trả {"trang_thai", "ly_do"}."""
    thap, dat = _so(_lay(so_cap, "thap")), _lay(lech, "dat")
    ly_do, khong = [], False
    if thap is not None and not thap > 0:
        khong = True
        ly_do.append(f"cận dưới khoảng 95% của chênh là {thap:+.2f}, không dương: khoảng chứa 0 hoặc nằm dưới 0, chưa hơn mốc rõ ràng")
    if dat is False:
        khong = True
        ly_do += ["học lệch: " + x for x in _lay(lech, "ly_do") or []] or ["học lệch"]
    if khong:
        return {"trang_thai": KHONG_QUA, "ly_do": ly_do}
    if thap is None:
        ly_do.append("thiếu khoảng bootstrap của chênh so với mốc")
    if dat is not True:
        ly_do.append("chưa kết luận được có học lệch hay không (thiếu số)")
    if ly_do:
        return {"trang_thai": CHUA_DU, "ly_do": ly_do}
    return {"trang_thai": QUA, "ly_do": [f"cận dưới khoảng 95% của chênh là {thap:+.2f} > 0 và không học lệch"]}


def chon_ung_vien(cac_muc):
    """cac_muc: danh sách {"ten", "trang_thai", "chenh"} (hoặc dict ten -> {...}). Chỉ chọn trong các mức "qua", lấy chênh lớn nhất
    (bằng nhau thì theo tên). Không có mức nào qua thì giữ mức hiện tại. Có mức "chua_du_du_lieu" (hoặc "qua" mà thiếu chênh)
    thì chưa kết luận được, không chọn bừa. Trả {"ung_vien": tên|None, "trang_thai": "chon"|"giu_muc_hien_tai"|"chua_ket_luan", "ly_do"}."""
    if isinstance(cac_muc, dict):
        cac_muc = [{"ten": k, **v} for k, v in cac_muc.items() if isinstance(v, dict)]
    ds = [m for m in (cac_muc or []) if isinstance(m, dict)]
    chua = [str(m.get("ten")) for m in ds if m.get("trang_thai") == CHUA_DU
            or (m.get("trang_thai") == QUA and _so(m.get("chenh")) is None)]
    if chua:
        return {"ung_vien": None, "trang_thai": "chua_ket_luan",
                "ly_do": ["chưa kết luận được vì chưa đủ dữ liệu ở: " + ", ".join(chua) + "; không chọn bừa"]}
    qua = sorted((m for m in ds if m.get("trang_thai") == QUA), key=lambda m: (-_so(m["chenh"]), str(m.get("ten"))))
    if not qua:
        return {"ung_vien": None, "trang_thai": "giu_muc_hien_tai", "ly_do": ["không có mức nào qua cổng, giữ mức hiện tại"]}
    ly_do = [f"mức {qua[0].get('ten')} qua cổng với chênh chrF {_so(qua[0]['chenh']):+.2f}, lớn nhất trong {len(qua)} mức qua cổng"]
    if len(qua) > 1 and _so(qua[1]["chenh"]) == _so(qua[0]["chenh"]):
        ly_do.append("có mức khác bằng chênh, chọn theo thứ tự tên")
    return {"ung_vien": qua[0].get("ten"), "trang_thai": "chon", "ly_do": ly_do}


def ket_luan_kfold(chenh_fold, lech):
    """chenh_fold: chênh chrF từng fold. Đổi tỉ lệ khi đủ 3 fold, cả 3 > 0, trung bình > độ lệch chuẩn mẫu (n-1) và lech["dat"] là True.
    Chưa đủ 3 fold (hoặc thiếu số, hoặc lech["dat"] chưa rõ) thì doi_ti_le = None. Trả {"doi_ti_le": bool|None, "ly_do": [...]}."""
    x = [_so(v) for v in chenh_fold] if isinstance(chenh_fold, (list, tuple)) else []
    if len(x) != 3 or None in x:
        return {"doi_ti_le": None, "ly_do": [f"chưa đủ 3 fold có số (hiện có {sum(v is not None for v in x)} trên {len(x)})"]}
    tb, sd = statistics.fmean(x), statistics.stdev(x)
    ly_do, doi = [], True
    if not all(v > 0 for v in x):
        doi = False
        ly_do.append("không phải cả 3 fold đều dương: " + ", ".join(f"{v:+.2f}" for v in x))
    if not tb > sd:
        doi = False
        ly_do.append(f"trung bình {tb:.2f} không lớn hơn độ lệch chuẩn mẫu {sd:.2f}")
    dat = _lay(lech, "dat")
    if dat is False:
        doi = False
        ly_do.append("học lệch")
    elif dat is not True and doi:
        return {"doi_ti_le": None, "ly_do": ["các điều kiện về chênh đều thỏa nhưng chưa kết luận được có học lệch hay không (thiếu số)"]}
    if doi:
        ly_do.append(f"cả 3 fold dương, trung bình {tb:.2f} lớn hơn độ lệch chuẩn mẫu {sd:.2f}, không học lệch")
    return {"doi_ti_le": doi, "ly_do": ly_do}


def _tb2(a, b):
    a, b = _so(a), _so(b)
    return None if a is None or b is None else (a + b) / 2


def gom_so(cham):
    """Rút các số cần so từ khối "cham" (định dạng trạng thái Kaggle): title, mo_ta (trung bình chrF corpus hai hướng), zh_vi, vi_zh, chrF chung,
    dòng mới theo hai hướng, sinh lặp. Thiếu thì None. cham None thì trả None."""
    if not isinstance(cham, dict):
        return None
    nh = lambda k: _so(_lay(cham, "theo_nhom", k, "chrf_corpus"))  # noqa: E731
    moi = lambda h: {"n": _so(_lay(cham, "dong_da_thay_moi", h, "moi", "n_dong")),  # noqa: E731
                     "chrf": _so(_lay(cham, "dong_da_thay_moi", h, "moi", "chrf_corpus"))}
    return {"chrf": _so(_lay(cham, "chrf", "chrf_macro_huong")),
            "title": _tb2(nh("zh->vi/title"), nh("vi->zh/title")), "mo_ta": _tb2(nh("zh->vi/mo_ta"), nh("vi->zh/mo_ta")),
            "zh_vi": _so(_lay(cham, "theo_huong", "zh->vi", "chrf_corpus")), "vi_zh": _so(_lay(cham, "theo_huong", "vi->zh", "chrf_corpus")),
            "moi_zh_vi": moi("zh->vi"), "moi_vi_zh": moi("vi->zh"),
            "sinh_lap": _so(_lay(cham, "sinh_lap")), "so_mau": _so(_lay(cham, "so_mau"))}


# ---------------------------------------------------------------- đọc, chấm (cần pandas, sacrebleu, chỉ nạp khi dùng)
def _doc_jsonl(duong_dan):
    ra = []
    for dong in Path(duong_dan).read_text(encoding="utf-8-sig").splitlines():
        if dong.strip():
            ra.append(json.loads(dong))
    return ra


def _doc_json(duong_dan):
    """Không ném lỗi: trả (đối tượng hoặc None, lỗi hoặc None)."""
    try:
        return json.loads(Path(duong_dan).read_text(encoding="utf-8-sig")), None
    except (OSError, ValueError) as e:
        return None, f"không đọc được {Path(duong_dan).name}: {e}"


def _thu(ham, *args):
    try:
        return ham(*args), None
    except Exception as e:  # noqa: BLE001  báo cáo không được ném lỗi vì một file hỏng
        return None, f"{type(e).__name__}: {e}"


def _nap_cong_cu():
    import pandas as pd
    import cham_diem as D
    import kaggle_train as K
    return K, D, pd


def chuan_bi_mau(cong_cu, nhanh, pool_cum, pool_nhanh, so_moi_nhom=SO_MOI_NHOM, so_mau=SO_MAU, fold=FOLD):
    """Mẫu chấm cố định của một fold (chon_mau_sinh, seed 42, mặc định fold 0) và tập dòng đã thấy. Không đúng so_mau thì lỗi, không tự chọn lại."""
    K, D, pd = cong_cu
    mau, thieu = K.chon_mau_sinh(pool_cum, pool_nhanh, fold, so_moi_nhom, SEED)
    if thieu or len(mau) != so_mau:
        raise RuntimeError(f"mẫu chấm nhánh {nhanh} có {len(mau)} mẫu (cần đúng {so_mau}), thiếu {thieu} id ở nhánh này")
    tap = D.xay_tap_da_thay(pd.DataFrame([{c: m.get(c) for c in COT_DA_THAY} for m in pool_nhanh]), fold, nhanh)
    return {"mau": mau, "tap": tap, "ids": [str(m["id"]) for m in mau]}


def cham_ban_dich(cong_cu, nhanh, bo_mau, duong_dan):
    """Chấm một file bản dịch. Trả {"cham": khối kiểu trạng thái Kaggle, "chrf_muc": chrF từng mục theo thứ tự id của mẫu}. Thiếu bản dịch của mẫu thì lỗi."""
    K, D, pd = cong_cu
    ids = bo_mau["ids"]
    co = set(ids)
    du = [r for r in _doc_jsonl(duong_dan) if isinstance(r, dict) and str(r.get("id")) in co]
    thieu = co - {str(r["id"]) for r in du}
    if thieu:
        raise RuntimeError(f"thiếu {len(thieu)}/{len(ids)} bản dịch của mẫu chấm trong {Path(duong_dan).name}")
    df, th = D.cham(pd.DataFrame(bo_mau["mau"]), pd.DataFrame(du), nhanh=nhanh, tap_da_thay=bo_mau["tap"])
    chrf = df.set_index("id")["chrf"].reindex(ids).tolist()
    cham = {"so_mau": len(ids), "sinh_lap": int(df["sinh_lap"].sum()), "chrf": th["diem_chinh"], "theo_huong": th["theo_huong"],
            "theo_nhom": th["theo_nhom"], "dong_da_thay_moi": th.get("dong_da_thay_moi")}
    return {"cham": cham, "chrf_muc": [float(x) for x in chrf]}


def so_cap_voi_moc(cong_cu, bo_mau, chrf_muc, chrf_moc):
    """Chênh (mức trừ mốc) theo bootstrap_chenh, a là mức, b là mốc, cùng thứ tự id của mẫu, nhóm "huong|loai"."""
    K = cong_cu[0]
    if not all(isinstance(v, float) and math.isfinite(v) for v in list(chrf_muc) + list(chrf_moc)):
        raise RuntimeError("có chrF từng mục không hữu hạn, không bootstrap được")
    nhom = [f"{m['huong']}|{m['loai']}" for m in bo_mau["mau"]]
    return K.bootstrap_chenh(list(chrf_muc), list(chrf_moc), nhom)


# ---------------------------------------------------------------- đọc trạng thái
def doc_trang_thai(duong_dan):
    """Không ném lỗi: file hỏng hoặc phần sai kiểu thì thành None. Trả {"xong", "bat_thuong", "viec": {4 việc: dict|None}, "loi_doc", "thu_muc"}."""
    goc, loi = _doc_json(duong_dan)
    if not isinstance(goc, dict):
        loi = loi or "file trạng thái không phải JSON đối tượng"
        goc = {}
    viec = goc.get("viec")
    viec = viec if isinstance(viec, dict) else {}
    bt = goc.get("bat_thuong")
    if bt is not None and not isinstance(bt, str):
        bt = json.dumps(bt, ensure_ascii=False)
    ten = [t for d in VIEC_MUC.values() for t in d.values()]
    return {"xong": goc.get("xong") is True, "bat_thuong": bt or None, "loi_doc": loi,
            "viec": {t: viec[t] if isinstance(viec.get(t), dict) else None for t in ten}, "thu_muc": Path(duong_dan).resolve().parent}


def _loss2(viec):
    return {k: _so(_lay(viec, "loss", "2", k)) for k in NHOM_LOSS}


def _thong_tin(viec):
    return {"p_muc_tieu": _so(_lay(viec, "p_muc_tieu")), "p_thuc": _so(_lay(viec, "p_thuc")), "w_title": _so(_lay(viec, "w_title")),
            "lr": _so(_lay(viec, "lr")), "epoch_xong": _so(_lay(viec, "epoch_xong")), "hong": bool(_lay(viec, "hong")),
            "diem_e2": _so(_lay(viec, "diem", "2")), "loss": _loss2(viec)}


# ---------------------------------------------------------------- xử lý từng nhánh
class _BoiCanh:
    """Nạp công cụ và pool một lần, lỗi thì ghi lại để từng nhánh báo chưa đủ dữ liệu."""

    def __init__(self, dd):
        self.dd = dd
        self.cong_cu, self.loi_cong_cu = _thu(_nap_cong_cu)
        self._pool = {}

    def pool(self, ten):
        if ten not in self._pool:
            self._pool[ten] = _thu(_doc_jsonl, self.dd[f"pool_{ten}"])
        return self._pool[ten]


def _tu_cham_moc(bc, nhanh, so_moi_nhom, so_mau, ky_vong):
    """Chuẩn bị mẫu và tự chấm mốc của một nhánh. Trả dict có bo_mau, cham, chrf_muc, so, ky_vong, lech, khop, loi (danh sách)."""
    r = {"bo_mau": None, "cham": None, "chrf_muc": None, "so": None, "ky_vong": ky_vong.get(nhanh), "chrf_tu_cham": None,
         "lech_ky_vong": None, "khop": None, "loi": []}
    if bc.cong_cu is None:
        r["loi"].append(f"không nạp được công cụ chấm (pandas, sacrebleu, cham_diem): {bc.loi_cong_cu}")
        return r
    (pc, lc), (pn, ln) = bc.pool("cum"), bc.pool(nhanh)
    if lc or ln:
        r["loi"].append("không đọc được pool: " + "; ".join(x for x in (lc, ln) if x))
        return r
    bo, loi = _thu(chuan_bi_mau, bc.cong_cu, nhanh, pc, pn, so_moi_nhom, so_mau)
    if loi:
        r["loi"].append(loi)
        return r
    r["bo_mau"] = bo
    kq, loi = _thu(cham_ban_dich, bc.cong_cu, nhanh, bo, bc.dd[f"moc_{nhanh}"])
    if loi:
        r["loi"].append("chấm mốc lỗi: " + loi)
        return r
    r.update(cham=kq["cham"], chrf_muc=kq["chrf_muc"], so=gom_so(kq["cham"]))
    r["chrf_tu_cham"] = r["so"]["chrf"]
    if r["ky_vong"] is not None and r["chrf_tu_cham"] is not None:
        r["lech_ky_vong"] = abs(r["chrf_tu_cham"] - r["ky_vong"])
        r["khop"] = r["lech_ky_vong"] <= DUNG_SAI_MOC
    return r


def _xu_ly_muc(bc, nhanh, ma, tt, moc, loi_chung, moc_tt):
    ten_viec = VIEC_MUC[nhanh][ma]
    viec = tt["viec"][ten_viec]
    r = {"ten": NHAN_MUC[ma], "ma": ma, "ten_viec": ten_viec, "co_viec": viec is not None, **(_thong_tin(viec) if viec else {}),
         "so": None, "nguon_so": None, "so_cap": None, "lech": None, "cong": None, "chenh": None, "canh_bao": [], "loi": []}
    chan = []
    if tt["bat_thuong"]:
        chan.append(f"Kaggle báo bất thường: {tt['bat_thuong']}")
    if viec is None:
        chan.append(f"việc {ten_viec} không có trong trạng thái")
    elif r["hong"]:
        chan.append(f"việc {ten_viec} bị đánh dấu hỏng")
    elif r["epoch_xong"] != 2:
        chan.append(f"việc {ten_viec} mới xong epoch {_f(r['epoch_xong'], 0)}, cần 2")
    chrf_muc = None
    if viec is not None and not r["hong"] and r["epoch_xong"] == 2:
        if moc["bo_mau"] is None or bc.cong_cu is None:
            r["loi"].append("không chấm lại được vì thiếu mẫu hoặc công cụ chấm")
        else:
            kq, loi = _thu(cham_ban_dich, bc.cong_cu, nhanh, moc["bo_mau"], tt["thu_muc"] / ten_viec / "sinh_ep2.jsonl")
            if loi:
                r["loi"].append("chấm lại bản dịch lỗi: " + loi)
            else:
                r["so"], r["nguon_so"], chrf_muc = gom_so(kq["cham"]), "chấm lại trên máy", kq["chrf_muc"]
        so_tt = gom_so(_lay(viec, "cham"))
        if r["so"] is None and so_tt is not None:
            r["so"], r["nguon_so"] = so_tt, "ghi sẵn trong trạng thái (chưa chấm lại)"
        elif r["so"] is not None and so_tt is not None and so_tt["chrf"] is not None and r["so"]["chrf"] is not None \
                and abs(so_tt["chrf"] - r["so"]["chrf"]) > DUNG_SAI_MOC:
            r["canh_bao"].append(f"chrF chấm lại {r['so']['chrf']:.6f} khác số Kaggle ghi {so_tt['chrf']:.6f}")
    r["loi"] = list(r["loi"])
    chan += r["loi"]
    chan += loi_chung
    moc_so = moc["so"]
    if moc_so is not None and r["so"] is not None:
        r["lech"] = khong_hoc_lech(moc_so, r["so"])
        if chrf_muc is not None and moc["chrf_muc"] is not None:
            sc, loi = _thu(so_cap_voi_moc, bc.cong_cu, moc["bo_mau"], chrf_muc, moc["chrf_muc"])
            r["so_cap"] = sc
            if loi:
                chan.append("bootstrap lỗi: " + loi)
    if chan:
        r["cong"] = {"trang_thai": CHUA_DU, "ly_do": chan}
    else:
        r["cong"] = qua_cong_fold0(r["so_cap"], r["lech"])
    r["chenh"] = _so(_lay(r["so_cap"], "chenh"))
    return r


def xu_ly_nhanh(bc, nhanh, tt, moc_tt, so_moi_nhom, so_mau, ky_vong):
    moc = _tu_cham_moc(bc, nhanh, so_moi_nhom, so_mau, ky_vong)
    v_moc = _lay(moc_tt, "viec", f"{nhanh}_f0")
    moc.update(ten=NHAN_MUC["moc"], loss=_loss2(v_moc), diem_e2=_so(_lay(v_moc, "diem", "2")))
    loi_chung = [f"mốc: {x}" for x in moc["loi"]]
    if moc["khop"] is False:
        loi_chung.append(f"tự chấm mốc {nhanh} ra {moc['chrf_tu_cham']!r}, kỳ vọng {moc['ky_vong']!r}, lệch {moc['lech_ky_vong']:.3g} "
                         f"(quá {DUNG_SAI_MOC}): không kết luận cổng cho bản này")
    elif moc["khop"] is None and not moc["loi"]:
        loi_chung.append("không so được với số mốc kỳ vọng")
    muc = {ma: _xu_ly_muc(bc, nhanh, ma, tt, moc, loi_chung, moc_tt) for ma in MUC}
    ung_vien = chon_ung_vien([{"ten": m["ten"], "trang_thai": m["cong"]["trang_thai"], "chenh": m["chenh"]} for m in muc.values()])
    kfold = {ma: ket_luan_kfold([m["chenh"]], m["lech"]) for ma, m in muc.items()}
    return {"nhanh": nhanh, "moc": {k: v for k, v in moc.items() if k not in ("bo_mau", "cham", "chrf_muc")}, "muc": muc,
            "ung_vien": ung_vien, "kfold": kfold, "canh_bao_moc": loi_chung}


def tinh_tat_ca(tt, dd, so_moi_nhom=SO_MOI_NHOM, so_mau=SO_MAU, ky_vong=None):
    bc = _BoiCanh(dd)
    moc_tt, loi = _doc_json(dd["moc_trang_thai"])
    moc_tt = moc_tt if isinstance(moc_tt, dict) else None
    loi_chung = [loi] if loi else []
    if moc_tt is None and not loi:
        loi_chung.append("trạng thái mốc không phải JSON đối tượng")
    return {"loi_chung": loi_chung, "nhanh": {n: xu_ly_nhanh(bc, n, tt, moc_tt, so_moi_nhom, so_mau, ky_vong or MOC_KY_VONG) for n in NHANH}}


def kiem_moc(dd, so_moi_nhom=SO_MOI_NHOM, so_mau=SO_MAU, ky_vong=None):
    """Chỉ tự chấm mốc từng nhánh, trả {nhanh: {chrf, ky_vong, lech, khop, loi}}. Không cần trạng thái tl."""
    bc, ky_vong, ra = _BoiCanh(dd), ky_vong or MOC_KY_VONG, {}
    for n in NHANH:
        m = _tu_cham_moc(bc, n, so_moi_nhom, so_mau, ky_vong)
        ra[n] = {k: m[k] for k in ("chrf_tu_cham", "ky_vong", "lech_ky_vong", "khop", "loi")}
    return ra


# ---------------------------------------------------------------- báo cáo
def _f(x, n=2):
    x = _so(x)
    return "thiếu" if x is None else f"{x:.{n}f}"


def _pc(x):
    x = _so(x)
    return "thiếu" if x is None else f"{x * 100:.2f}"


def _dong_bang(cells):
    return "| " + " | ".join(cells) + " |"


def _cap(so_cap):
    c, a, b = (_so(_lay(so_cap, k)) for k in ("chenh", "thap", "cao"))
    if None in (c, a, b):
        return "thiếu"
    return f"{c:+.2f} [{a:+.2f}, {b:+.2f}]"


def _moi(m):
    if not isinstance(m, dict) or m.get("chrf") is None:
        return "thiếu"
    return f"{m['chrf']:.2f} ({_f(m.get('n'), 0)} dòng)"


def _sinh_lap(so):
    sl, n = _so(_lay(so, "sinh_lap")), _so(_lay(so, "so_mau"))
    return "thiếu" if sl is None or not n else f"{sl:.0f}/{n:.0f} ({sl / n * 100:.2f}%)"


def _hang_chrf(nhan, p, w, so, chenh):
    if so is None:
        return _dong_bang([nhan, p, w, "chưa có số"] + [""] * 8)
    return _dong_bang([nhan, p, w, _f(so["chrf"]), chenh, _f(so["title"]), _f(so["mo_ta"]), _f(so["zh_vi"]), _f(so["vi_zh"]),
                       _moi(so["moi_zh_vi"]), _moi(so["moi_vi_zh"]), _sinh_lap(so)])


def _bang_nhanh(n):
    moc = n["moc"]
    L = [_dong_bang(["Mức", "% title thực tế", "w_title", "chrF chung", "Chênh so với mốc [khoảng 95%]", "Title", "Mô tả", "zh→vi", "vi→zh",
                     "Mô tả dòng mới zh→vi", "Mô tả dòng mới vi→zh", "Sinh lặp"]), "|" + "---|" * 12]
    L.append(_hang_chrf("mốc ~50%", "~50 (danh nghĩa)", "không có", moc["so"], "0 (mốc)"))
    for ma in MUC:
        m = n["muc"][ma]
        L.append(_hang_chrf(m["ten"], _pc(m.get("p_thuc")), _f(m.get("w_title"), 3), m["so"], _cap(m["so_cap"])))
    L += ["", "Loss epoch 2 theo bốn nhóm (hướng, loại), và điểm macro:", "",
          _dong_bang(["Mức", "zh→vi title", "zh→vi mô tả", "vi→zh title", "vi→zh mô tả", "Điểm macro epoch 2", "Epoch xong"]), "|" + "---|" * 7]
    L.append(_dong_bang(["mốc ~50%"] + [_f(moc["loss"].get(k), 4) for k in NHOM_LOSS] + [_f(moc["diem_e2"], 4), "2 (mốc)"]))
    for ma in MUC:
        m = n["muc"][ma]
        L.append(_dong_bang([m["ten"]] + [_f(_lay(m, "loss", k), 4) for k in NHOM_LOSS] + [_f(m.get("diem_e2"), 4), _f(m.get("epoch_xong"), 0)]))
    return L


def _khoi_cong(n):
    L = []
    for ma in MUC:
        m = n["muc"][ma]
        L.append(f"**{m['ten']}: {m['cong']['trang_thai']}**" + (f" (số lấy từ: {m['nguon_so']})" if m["nguon_so"] else ""))
        L += [f"- {x}" for x in m["cong"]["ly_do"]]
        L += [f"- cảnh báo: {x}" for x in m["canh_bao"]]
        if m["lech"] is not None:
            L.append("- chênh chrF (mức trừ mốc): " + ", ".join(f"{TEN_LECH[k]} {_f(m['lech']['chenh'].get(k))}" for k in KHOA_LECH))
        L.append("")
    u = n["ung_vien"]
    if u["trang_thai"] == "chon":
        L.append(f"**Ứng viên: {u['ung_vien']}** (chỉ là báo kết quả cổng, không quyết định chạy thêm GPU)")
    elif u["trang_thai"] == "giu_muc_hien_tai":
        L.append("**Giữ mức hiện tại (~50%)**")
    else:
        L.append("**Chưa kết luận được ứng viên**")
    L += [f"- {x}" for x in u["ly_do"]]
    kf = [f"{m['ten']}: {'chưa kết luận' if k['doi_ti_le'] is None else ('đổi tỉ lệ' if k['doi_ti_le'] else 'không đổi')}"
          for m, k in zip(n["muc"].values(), n["kfold"].values())]
    L.append("- k-fold (mới có fold 0, cần đủ 3 fold mới đổi tỉ lệ): " + "; ".join(kf))
    return L + [""]


def viet_bao_cao(tt, kq):
    L = ["# Báo cáo thử tỉ lệ title trong loss (bước 2, fold 0)", "",
         "Mốc là mức ~50% đã có kết quả. Hai mức thử trên fold 0, mỗi mức huấn luyện 2 epoch rồi sinh 1.000 bản dịch; số mốc và số từng mức "
         "được chấm lại bằng cùng hàm, so theo cặp trên cùng 1.000 mẫu.", "",
         f"Kaggle báo xong: {'có' if tt['xong'] else 'chưa'}."]
    if tt["loi_doc"]:
        L.append(f"CẢNH BÁO: {tt['loi_doc']}.")
    if tt["bat_thuong"]:
        L.append(f"BẤT THƯỜNG do Kaggle báo: {tt['bat_thuong']}. Không kết luận cổng nào.")
    for x in kq["loi_chung"]:
        L.append(f"CẢNH BÁO: {x}.")
    for ten, n in kq["nhanh"].items():
        m = n["moc"]
        if m["khop"] is False:
            L += ["", f"## CẢNH BÁO LỚN: tự chấm mốc {ten} KHÔNG KHỚP",
                  f"Ra {m['chrf_tu_cham']!r}, kỳ vọng {m['ky_vong']!r}, lệch {m['lech_ky_vong']:.3g}, quá {DUNG_SAI_MOC}. Không kết luận cổng cho bản {ten}."]
    L += ["", "## Cách đọc", "",
          "- Khoảng tin cậy 95% của chênh chứa 0 nghĩa là hòa, không phải hơn.",
          f"- Ngưỡng [MẶC ĐỊNH]: một trong title, mô tả, zh→vi tụt quá {NGUONG_LECH} điểm chrF so với mốc thì coi là học lệch (đúng bằng ngưỡng thì đạt). "
          "Qua cổng fold 0: cận dưới khoảng 95% của chênh chrF chung > 0 và không học lệch.",
          "- Title, mô tả là trung bình chrF corpus hai hướng; chrF chung là macro theo hướng.",
          "- Đây là phần đọc số, không phải đánh giá của người; việc có chạy thêm GPU hay không do người quyết.", ""]
    for i, (ten, n) in enumerate(kq["nhanh"].items(), 1):
        m = n["moc"]
        L += [f"## {i}. Bản {ten}", "",
              f"Tự chấm mốc: {_f(m['chrf_tu_cham'], 10)}, kỳ vọng {_f(m['ky_vong'], 10)}, lệch {_f(m['lech_ky_vong'], 3)}: "
              f"{'khớp' if m['khop'] else ('KHÔNG KHỚP' if m['khop'] is False else 'chưa so được')}.", ""]
        L += _bang_nhanh(n) + ["", f"### Cổng và ứng viên, bản {ten}", ""] + _khoi_cong(n)
    return "\n".join(L)


def tom_tat(kq):
    dong = ["Tóm tắt thử tỉ lệ title:"]
    for ten, n in kq["nhanh"].items():
        m = n["moc"]
        dong.append(f"  {ten}: mốc chrF {_f(m['chrf_tu_cham'], 6)} ({'khớp' if m['khop'] else ('KHÔNG KHỚP' if m['khop'] is False else 'chưa so được')})")
        for ma in MUC:
            x = n["muc"][ma]
            dong.append(f"    {x['ten']}: cổng {x['cong']['trang_thai']}, chênh {_cap(x['so_cap'])}")
        dong.append(f"    ứng viên: {n['ung_vien']['ung_vien'] or n['ung_vien']['trang_thai']}")
    return "\n".join(dong)


def _sach(o):
    if isinstance(o, dict):
        return {str(k): _sach(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_sach(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def main(argv=None, *, duong_dan=None, so_moi_nhom=SO_MOI_NHOM, so_mau=SO_MAU, moc_ky_vong=None):
    """duong_dan: dict ghi đè DUONG_DAN_THAT (moc_cum, moc_pair, moc_trang_thai, pool_cum, pool_pair), để test không cần dữ liệu thật.
    so_moi_nhom, so_mau, moc_ky_vong cũng chỉ để test. Không ném lỗi vì dữ liệu thiếu hoặc hỏng; trả 0 (--chi-kiem-moc: 0 khi cả hai mốc khớp, 1 nếu không)."""
    ap = argparse.ArgumentParser(description="Tổng hợp thử tỉ lệ title trong loss từ trạng thái Kaggle")
    ap.add_argument("--trang-thai", help="file tl_trang_thai.json")
    ap.add_argument("--ra", help="thư mục ghi bao_cao_ti_le.md và bao_cao_ti_le.json")
    ap.add_argument("--chi-kiem-moc", action="store_true", help="chỉ tự chấm mốc cum và pair rồi so với số đã biết")
    args = ap.parse_args(argv)
    dd = {**DUONG_DAN_THAT, **(duong_dan or {})}
    ky_vong = moc_ky_vong or MOC_KY_VONG
    if args.chi_kiem_moc:
        ra = kiem_moc(dd, so_moi_nhom, so_mau, ky_vong)
        for n, r in ra.items():
            print(f"mốc {n}: tự chấm {r['chrf_tu_cham']!r}, kỳ vọng {r['ky_vong']!r}, lệch {r['lech_ky_vong']!r}, khớp {r['khop']}, lỗi {r['loi']}")
        return 0 if all(r["khop"] for r in ra.values()) else 1
    if not args.trang_thai or not args.ra:
        ap.error("cần --trang-thai và --ra (hoặc chỉ --chi-kiem-moc)")
    tt = doc_trang_thai(args.trang_thai)
    kq = tinh_tat_ca(tt, dd, so_moi_nhom, so_mau, ky_vong)
    ra = Path(args.ra)
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "bao_cao_ti_le.md").write_text(viet_bao_cao(tt, kq), encoding="utf-8")
    goi = {"xong": tt["xong"], "bat_thuong": tt["bat_thuong"], "loi_doc": tt["loi_doc"], **kq,
           "nguong_mac_dinh": {"lech": NGUONG_LECH, "dung_sai_moc": DUNG_SAI_MOC}}
    (ra / "bao_cao_ti_le.json").write_text(json.dumps(_sach(goi), ensure_ascii=False, indent=2), encoding="utf-8")
    print(tom_tat(kq))
    print(f"Đã ghi {ra / 'bao_cao_ti_le.md'} và {ra / 'bao_cao_ti_le.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
