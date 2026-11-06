"""Tổng hợp B7, B8: đọc file trạng thái JSON do Kaggle sinh ra (b78_trang_thai.json), tính các chốt chặn C1, C2, C3,
túi lỗi category và kết luận LoRA attention, ghi bao_cao_b78.md và bao_cao_b78.json. Chỉ dùng thư viện chuẩn, chạy trên máy.

Chạy:  python -I -X utf8 scripts/ft_train/tong_hop_b78.py --trang-thai <b78_trang_thai.json> --ra <thu_muc>
Mỗi chốt chặn có ba trạng thái: "bat", "khong_bat", "chua_du_du_lieu". Thiếu dữ liệu không bao giờ được tính là đạt.
Báo cáo chỉ nêu số và chốt chặn, không kết luận mô hình tốt hay ổn: quyết định thuộc về người đọc."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------- ngưỡng [MẶC ĐỊNH]
NGUONG_CHRF = 2.0            # điểm chrF: C1 (pair kém cum), C2 (lệch giữa fold, f1 f2 kém f0), C3 (lệch cum và pair, theo hướng)
NGUONG_SINH_LAP = 0.01       # C1: tỉ lệ sinh_lap/so_mau của pair vượt cum quá mức này
NGUONG_LOSS_TUONG_DOI = 0.05  # C3: loss cả fold lệch quá 5% so với cum
HE_SO_CATEGORY = 1.25        # túi lỗi: loss category / loss nhóm hướng+loại
TOI_THIEU_TOKEN = 2000       # túi lỗi: bỏ category có ít token hơn mức này
EPS = 1e-9                   # chống sai số làm tròn ở biên: đúng bằng ngưỡng thì không bật

BAT, KHONG_BAT, CHUA_DU = "bat", "khong_bat", "chua_du_du_lieu"
TEN_VIEC = ("cum_f0", "pair_f0", "cum_f1", "cum_f2")
HANH_DONG_C1 = "chạy thêm cấu hình thứ hai của B6 (lr1e-4_r64) trên pair, ~1,3 giờ"
HANH_DONG_C2_LECH = "chạy thêm fold 3, 4 của cum"
HANH_DONG_C2_F0 = "chạy cấu hình thứ hai trên fold 1 để đối chứng"
HANH_DONG_C3 = "chạy pair fold 1, 2"
GHI_CHU_ATTN = ("So ở epoch 1, chỉ trên 1 fold, chưa chạy tới epoch 2. "
                "Quy tắc plan: LoRA attention chỉ được dùng tiếp nếu chrF không kém LoRA mọi lớp cùng lr, rank (cận trên khoảng bootstrap 95% >= 0).")
GHI_CHU_ZERO_SHOT = ('Tiêu chí ổn định "độ lệch giữa fold nhỏ hơn chênh fine-tune so với zero-shot": '
                     "CHỜ số zero-shot, chưa đánh giá được.")


# ---------------------------------------------------------------- tiện ích
def _lay(d, *khoa):
    """Đi sâu vào dict lồng nhau; thiếu khóa hoặc gặp kiểu khác dict thì trả None."""
    for k in khoa:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _so(x):
    """Chỉ nhận số thật (không phải bool, NaN, vô cực); còn lại None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _dict_hoac_none(x):
    return x if isinstance(x, dict) else None


def vuot(x, nguong):
    """x > nguong, có dung sai để đúng bằng ngưỡng thì không vượt."""
    return x > nguong + EPS


def chrf_macro(viec):
    return _so(_lay(viec, "cham", "chrf", "chrf_macro_huong"))


def chrf_huong(viec, huong):
    return _so(_lay(viec, "cham", "theo_huong", huong, "chrf_corpus"))


def ti_le_sinh_lap(viec):
    sl, n = _so(_lay(viec, "cham", "sinh_lap")), _so(_lay(viec, "cham", "so_mau"))
    if sl is None or n is None or n <= 0:
        return None
    return sl / n


def loss_fold(viec):
    return _so(_lay(viec, "loss_ca_fold", "diem_macro"))


def _kq(trang_thai, ly_do, hanh_dong, thieu, **them):
    return {"trang_thai": trang_thai, "ly_do": ly_do, "hanh_dong": hanh_dong, "thieu": thieu, **them}


def _chot(ly_do, hanh_dong, thieu, **them):
    """Có bằng chứng vượt ngưỡng thì bật (dù còn thiếu phần khác); không thì thiếu là chưa đủ dữ liệu, không bao giờ là đạt."""
    if ly_do:
        return _kq(BAT, ly_do, hanh_dong, thieu, **them)
    return _kq(CHUA_DU if thieu else KHONG_BAT, [], None, thieu, **them)


# ---------------------------------------------------------------- chốt chặn
def kiem_c1(cum_f0, pair_f0):
    """Bật nếu chrF macro pair thấp hơn cum quá NGUONG_CHRF, hoặc tỉ lệ sinh lặp pair vượt cum quá NGUONG_SINH_LAP."""
    ly_do, thieu = [], []
    cc, cp = chrf_macro(cum_f0), chrf_macro(pair_f0)
    if cc is None or cp is None:
        thieu.append("chrF macro cum_f0 hoặc pair_f0")
    elif vuot(cc - cp, NGUONG_CHRF):
        ly_do.append(f"chrF macro pair {cp:.2f} thấp hơn cum {cc:.2f} là {cc - cp:.2f} điểm (ngưỡng {NGUONG_CHRF})")
    lc, lp = ti_le_sinh_lap(cum_f0), ti_le_sinh_lap(pair_f0)
    if lc is None or lp is None:
        thieu.append("tỉ lệ sinh lặp cum_f0 hoặc pair_f0")
    elif vuot(lp - lc, NGUONG_SINH_LAP):
        ly_do.append(f"sinh lặp pair {lp:.4f} cao hơn cum {lc:.4f} là {lp - lc:.4f} (ngưỡng {NGUONG_SINH_LAP})")
    return _chot(ly_do, HANH_DONG_C1, thieu)


def kiem_c2(f0, f1, f2):
    """Bật nếu max trừ min chrF macro của các fold cum vượt NGUONG_CHRF, hoặc trung bình f1, f2 kém f0 quá NGUONG_CHRF.
    Lý do thứ hai báo riêng (B6 chọn cấu hình theo fold 0 nên có thể là may rủi). Trả kèm trung bình và độ lệch chuẩn mẫu (n-1)."""
    gia_tri = {"f0": chrf_macro(f0), "f1": chrf_macro(f1), "f2": chrf_macro(f2)}
    co = [v for v in gia_tri.values() if v is not None]
    thieu = [f"chrF macro cum {k}" for k, v in gia_tri.items() if v is None]
    ly_do, hanh_dong = [], []
    ma = []
    if len(co) >= 2 and vuot(max(co) - min(co), NGUONG_CHRF):
        ly_do.append(f"lệch giữa fold: max trừ min = {max(co) - min(co):.2f} điểm (ngưỡng {NGUONG_CHRF})")
        hanh_dong.append(HANH_DONG_C2_LECH)
        ma.append("lech_giua_fold")
    if None not in (gia_tri["f0"], gia_tri["f1"], gia_tri["f2"]):
        tb12 = (gia_tri["f1"] + gia_tri["f2"]) / 2
        if vuot(gia_tri["f0"] - tb12, NGUONG_CHRF):
            ly_do.append(f"trung bình fold 1, 2 là {tb12:.2f}, kém fold 0 ({gia_tri['f0']:.2f}) quá {NGUONG_CHRF} điểm: "
                         "cấu hình B6 có thể được chọn nhờ may rủi trên fold 0")
            hanh_dong.append(HANH_DONG_C2_F0)
            ma.append("f1f2_kem_f0")
    trung_binh = statistics.fmean(co) if co else None
    do_lech = statistics.stdev(co) if len(co) >= 2 else None
    return _chot(ly_do, "; ".join(hanh_dong) if hanh_dong else None, thieu,
                 ma_ly_do=ma, so_fold=len(co), trung_binh=trung_binh, do_lech_chuan=do_lech,
                 chrf_macro=gia_tri)


def kiem_c3(cum_f0, pair_f0):
    """Bật nếu |chênh chrF macro| > NGUONG_CHRF, hoặc loss cả fold lệch quá NGUONG_LOSS_TUONG_DOI so với cum,
    hoặc chrF ở một hướng chênh quá NGUONG_CHRF."""
    ly_do, thieu = [], []
    cc, cp = chrf_macro(cum_f0), chrf_macro(pair_f0)
    if cc is None or cp is None:
        thieu.append("chrF macro cum_f0 hoặc pair_f0")
    elif vuot(abs(cp - cc), NGUONG_CHRF):
        ly_do.append(f"chênh chrF macro pair trừ cum = {cp - cc:+.2f} (ngưỡng {NGUONG_CHRF})")
    lc, lp = loss_fold(cum_f0), loss_fold(pair_f0)
    if lc is None or lp is None or lc <= 0:
        thieu.append("loss cả fold cum_f0 hoặc pair_f0")
    elif vuot(abs(lp - lc) / lc, NGUONG_LOSS_TUONG_DOI):
        ly_do.append(f"loss cả fold pair {lp:.4f} lệch cum {lc:.4f} là {abs(lp - lc) / lc * 100:.1f}% (ngưỡng {NGUONG_LOSS_TUONG_DOI * 100:.0f}%)")
    huong_cum = _lay(cum_f0, "cham", "theo_huong")
    huong_pair = _lay(pair_f0, "cham", "theo_huong")
    if not isinstance(huong_cum, dict) or not isinstance(huong_pair, dict) or not huong_cum:
        thieu.append("chrF theo hướng của cum_f0 hoặc pair_f0")
    else:
        for h in sorted(huong_cum):
            a, b = chrf_huong(cum_f0, h), chrf_huong(pair_f0, h)
            if a is None or b is None:
                thieu.append(f"chrF hướng {h} của cum_f0 hoặc pair_f0")
            elif vuot(abs(b - a), NGUONG_CHRF):
                ly_do.append(f"hướng {h}: chênh chrF pair trừ cum = {b - a:+.2f} (ngưỡng {NGUONG_CHRF})")
    return _chot(ly_do, HANH_DONG_C3, thieu)


def tui_loi_category(cac_viec_cung_nhanh, he_so=HE_SO_CATEGORY, toi_thieu_token=TOI_THIEU_TOKEN):
    """Khóa "huong|loai|category" có loss / loss nhóm "huong|loai" vượt he_so ở MỌI fold có dữ liệu của nhánh.
    Bỏ khóa có so_token dưới toi_thieu_token ở bất kỳ fold nào; khóa vắng ở một fold có dữ liệu thì không báo (không thể nói vượt ở mọi fold).
    Trả danh sách dict xếp giảm dần theo tỉ số trung bình."""
    cac_fold = [v for v in (cac_viec_cung_nhanh or []) if isinstance(_lay(v, "loss_ca_fold", "theo_category"), dict)]
    if not cac_fold:
        return []
    bang = []  # mỗi fold: {khoa: (ti_so, so_token)}
    for v in cac_fold:
        nhom = _lay(v, "loss_ca_fold", "theo_nhom") or {}
        d = {}
        for khoa, o in _lay(v, "loss_ca_fold", "theo_category").items():
            phan = str(khoa).split("|", 2)
            loss, tok = _so(_lay(o, "loss")), _so(_lay(o, "so_token"))
            loss_nhom = _so(nhom.get("|".join(phan[:2]))) if isinstance(nhom, dict) and len(phan) == 3 else None
            if loss is None or tok is None or loss_nhom is None or loss_nhom <= 0:
                continue
            d[khoa] = (loss / loss_nhom, tok)
        bang.append(d)
    ra = []
    for khoa in set().union(*[set(d) for d in bang]):
        if not all(khoa in d for d in bang):
            continue
        ti_so = [d[khoa][0] for d in bang]
        token = [d[khoa][1] for d in bang]
        if min(token) < toi_thieu_token or not all(vuot(t, he_so) for t in ti_so):
            continue
        ra.append({"khoa": khoa, "ti_so_trung_binh": statistics.fmean(ti_so), "ti_so_theo_fold": ti_so,
                   "so_fold": len(ti_so), "so_token_nho_nhat": min(token)})
    ra.sort(key=lambda x: (-x["ti_so_trung_binh"], x["khoa"]))
    return ra


def ket_luan_attn(attn):
    """LoRA attention chỉ được dùng tiếp nếu cận trên khoảng bootstrap của (attn trừ mọi lớp) >= 0: "khong_kem"; cận trên âm: "kem"."""
    so_cap = _lay(attn, "so_cap")
    chenh, thap, cao = (_so(_lay(so_cap, k)) for k in ("chenh", "thap", "cao"))
    ra = {"ghi_chu": GHI_CHU_ATTN, "chenh": chenh, "thap": thap, "cao": cao,
          "chrf_attn": _chrf_cau_hinh(_lay(attn, "lr1e-4_r16_attn")),
          "chrf_moi_lop": _chrf_cau_hinh(_lay(attn, "lr1e-4_r16"))}
    if None in (chenh, thap, cao):
        return {**ra, "ket_luan": CHUA_DU}
    return {**ra, "ket_luan": "khong_kem" if cao >= -EPS else "kem"}


def _chrf_cau_hinh(cham):
    return _so(_lay(cham, "chrf", "chrf_macro_huong"))


# ---------------------------------------------------------------- đọc trạng thái
def doc_trang_thai(duong_dan):
    """Đọc file trạng thái, không ném lỗi: file hỏng hoặc phần nào sai kiểu thì thành None.
    Trả {"xong", "attn", "viec": {cum_f0, pair_f0, cum_f1, cum_f2}, "loi_doc"}."""
    loi, goc = None, None
    try:
        goc = json.loads(Path(duong_dan).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        loi = f"không đọc được file trạng thái: {e}"
    if not isinstance(goc, dict):
        goc = {}
        loi = loi or "file trạng thái không phải JSON đối tượng"
    viec_goc = goc.get("viec")
    viec_goc = viec_goc if isinstance(viec_goc, dict) else {}
    return {"xong": goc.get("xong") is True, "attn": _dict_hoac_none(goc.get("attn")) or {},
            "viec": {t: _dict_hoac_none(viec_goc.get(t)) for t in TEN_VIEC}, "loi_doc": loi}


def tinh_tat_ca(tt):
    v = tt["viec"]
    cum = [v["cum_f0"], v["cum_f1"], v["cum_f2"]]
    return {"c1": kiem_c1(v["cum_f0"], v["pair_f0"]), "c2": kiem_c2(*cum), "c3": kiem_c3(v["cum_f0"], v["pair_f0"]),
            "tui_loi": {"cum": tui_loi_category([x for x in cum if x]), "pair": tui_loi_category([v["pair_f0"]] if v["pair_f0"] else [])},
            "attn": ket_luan_attn(tt["attn"])}


# ---------------------------------------------------------------- báo cáo
def _f(x, n=2):
    x = _so(x)
    return "thiếu" if x is None else f"{x:.{n}f}"


def _ep(d, ep):
    return _f(_lay(d, ep), 4)


def _ten(khoa):
    """Khóa có dấu | hoặc / thành dạng đọc được và không phá bảng markdown."""
    return str(khoa).replace("|", " / ")


def _nhom_chuan(khoa):
    return str(khoa).replace("|", "/")


def _dong_viec(ten, v):
    if v is None:
        return f"| {ten} | chưa có | | | | | | | |"
    sl = ti_le_sinh_lap(v)
    sinh_lap = "thiếu" if sl is None else f"{_f(_lay(v, 'cham', 'sinh_lap'), 0)}/{_f(_lay(v, 'cham', 'so_mau'), 0)} ({sl * 100:.2f}%)"
    return (f"| {ten} | {_f(_lay(v, 'epoch_xong'), 0)} | {_ep(v.get('diem'), '1')} / {_ep(v.get('diem'), '2')} | {_f(loss_fold(v), 4)} "
            f"| {_f(chrf_macro(v))} | {_f(chrf_huong(v, 'zh->vi'))} / {_f(chrf_huong(v, 'vi->zh'))} | {sinh_lap} "
            f"| {_f(_lay(v, 'token_moi_giay', '1'), 0)} / {_f(_lay(v, 'token_moi_giay', '2'), 0)} | {_f(_lay(v, 'cham', 'giay_moi_mau'))} |")


def _bang_nhom(viec):
    dong = ["| Việc | Nhóm hướng / loại | chrF corpus | Loss nhóm | Giữ số đạt | Giữ Latin đạt | Sai chữ Việt | Sinh lặp |", "|---|---|---|---|---|---|---|---|"]
    for ten in TEN_VIEC:
        v = viec[ten]
        if v is None:
            continue
        cham_nhom = _lay(v, "cham", "theo_nhom")
        loss_nhom = _lay(v, "loss_ca_fold", "theo_nhom")
        cham_nhom = cham_nhom if isinstance(cham_nhom, dict) else {}
        loss_nhom = {_nhom_chuan(k): x for k, x in loss_nhom.items()} if isinstance(loss_nhom, dict) else {}
        for nhom in sorted(set(cham_nhom) | set(loss_nhom)):
            c = cham_nhom.get(nhom)
            dong.append(f"| {ten} | {_ten(nhom)} | {_f(_lay(c, 'chrf_corpus'))} | {_f(loss_nhom.get(nhom), 4)} | {_f(_lay(c, 'ti_le_giu_so_dat'), 3)} "
                        f"| {_f(_lay(c, 'ti_le_giu_latin_dat'), 3)} | {_f(_lay(c, 'ti_le_sai_chu_viet'), 3)} | {_f(_lay(c, 'ti_le_sinh_lap'), 3)} |")
    return dong if len(dong) > 2 else ["Chưa có việc nào có dữ liệu nhóm."]


def _khoi_chot(ten, kq):
    dong = [f"**{ten}: {kq['trang_thai']}**"]
    dong += [f"- Lý do: {x}" for x in kq["ly_do"]]
    if kq["hanh_dong"]:
        dong.append(f"- Hành động: {kq['hanh_dong']}")
    if kq["thieu"]:
        dong.append("- Thiếu dữ liệu: " + "; ".join(kq["thieu"]))
    return dong + [""]


def _bang_tui(nhan, ds, co_du_lieu):
    if not co_du_lieu:
        return [f"{nhan}: chưa đủ dữ liệu (chưa có loss theo category).", ""]
    if not ds:
        return [f"{nhan}: không có category nào vượt hệ số {HE_SO_CATEGORY} ở mọi fold có dữ liệu (bỏ category dưới {TOI_THIEU_TOKEN} token).", ""]
    dong = [f"{nhan}:", "", "| Khóa | Tỉ số trung bình | Tỉ số từng fold | Số fold | Token ít nhất |", "|---|---|---|---|---|"]
    dong += [f"| {_ten(x['khoa'])} | {x['ti_so_trung_binh']:.3f} | {', '.join(f'{t:.3f}' for t in x['ti_so_theo_fold'])} | {x['so_fold']} | {x['so_token_nho_nhat']:.0f} |"
             for x in ds]
    return dong + [""]


def _dong_moi(viec):
    dong = ["| Việc | Hướng | Dòng đã thấy: số dòng | chrF | Dòng mới: số dòng | chrF |", "|---|---|---|---|---|---|"]
    for ten in TEN_VIEC:
        d = _lay(viec[ten], "cham", "dong_da_thay_moi")
        if not isinstance(d, dict):
            continue
        for h in sorted(d):
            a, b = _lay(d, h, "da_thay"), _lay(d, h, "moi")
            dong.append(f"| {ten} | {h} | {_f(_lay(a, 'n_dong'), 0)} | {_f(_lay(a, 'chrf_corpus'))} | {_f(_lay(b, 'n_dong'), 0)} | {_f(_lay(b, 'chrf_corpus'))} |")
    return dong if len(dong) > 2 else ["Chưa có dữ liệu tách dòng đã thấy và dòng mới."]


def viet_bao_cao(tt, kq):
    viec, c2, a = tt["viec"], kq["c2"], kq["attn"]
    L = ["# Báo cáo B7, B8", "",
         "Cấu hình chốt lr2e-4, rank 64, 2 epoch. Mục tiêu: xem từng bộ có đủ dữ liệu hợp lệ không và cấu hình có giữ được trên fold khác không, "
         "không phải so hơn thua cum với pair. Báo cáo chỉ nêu số và chốt chặn.", "",
         f"Kaggle báo xong: {'có' if tt['xong'] else 'chưa'}."]
    if tt["loi_doc"]:
        L.append(f"Cảnh báo: {tt['loi_doc']}.")
    L += ["", "## 1. Từng việc", "",
          "Loss 3.000 mẫu và token/giây theo dạng epoch 1 / epoch 2. chrF tính trên epoch được chấm.", "",
          "| Việc | Epoch xong | Loss 3.000 mẫu (e1 / e2) | Loss cả fold | chrF macro | chrF zh->vi / vi->zh | Sinh lặp | Token/giây (e1 / e2) | Giây/mẫu |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [_dong_viec(t, viec[t]) for t in TEN_VIEC]
    L += ["", "## 2. Bốn nhóm hướng / loại", ""] + _bang_nhom(viec)
    L += ["", "## 3. Ổn định giữa fold của cum", "",
          "| Fold 0 | Fold 1 | Fold 2 | Số fold có số | Trung bình | Độ lệch chuẩn (n-1) |", "|---|---|---|---|---|---|",
          f"| {_f(c2['chrf_macro']['f0'])} | {_f(c2['chrf_macro']['f1'])} | {_f(c2['chrf_macro']['f2'])} | {c2['so_fold']} "
          f"| {_f(c2['trung_binh'])} | {_f(c2['do_lech_chuan'])} |", "",
          GHI_CHU_ZERO_SHOT, "",
          "## 4. Chốt chặn", "",
          f"Ngưỡng [MẶC ĐỊNH]: chrF {NGUONG_CHRF} điểm; sinh lặp {NGUONG_SINH_LAP}; loss cả fold lệch {NGUONG_LOSS_TUONG_DOI * 100:.0f}%. "
          "Đúng bằng ngưỡng thì không bật. Thiếu dữ liệu thì \"chua_du_du_lieu\", không tính là đạt.", ""]
    L += _khoi_chot("C1 (pair kém cum hoặc sinh lặp)", kq["c1"]) + _khoi_chot("C2 (cấu hình có giữ được giữa fold cum)", kq["c2"])
    L += _khoi_chot("C3 (cum và pair lệch nhau)", kq["c3"])
    L += ["## 5. Túi lỗi category", "",
          f"Loss category chia loss nhóm hướng / loại, chỉ báo khi vượt {HE_SO_CATEGORY} [MẶC ĐỊNH] ở mọi fold có dữ liệu của nhánh.", ""]
    co = lambda *ten: any(_lay(viec[t], "loss_ca_fold", "theo_category") for t in ten)  # noqa: E731
    L += _bang_tui("Nhánh cum", kq["tui_loi"]["cum"], co("cum_f0", "cum_f1", "cum_f2"))
    L += _bang_tui("Nhánh pair", kq["tui_loi"]["pair"], co("pair_f0"))
    L += ["## 6. LoRA attention so với LoRA mọi lớp (lr1e-4, rank 16)", "",
          f"chrF macro attn: {_f(a['chrf_attn'])}; mọi lớp: {_f(a['chrf_moi_lop'])}; "
          f"chênh attn trừ mọi lớp: {_f(a['chenh'])}, khoảng bootstrap 95% [{_f(a['thap'])}, {_f(a['cao'])}].", "",
          f"**Kết luận: {a['ket_luan']}**", "", a["ghi_chu"], "",
          "## 7. Dòng đã thấy và dòng mới", ""] + _dong_moi(viec)
    L += ["", "## 8. Chưa đánh giá được", "", "- " + GHI_CHU_ZERO_SHOT, ""]
    return "\n".join(L)


def tom_tat(kq):
    c2 = kq["c2"]
    ds = [f"C1={kq['c1']['trang_thai']}", f"C2={c2['trang_thai']}", f"C3={kq['c3']['trang_thai']}", f"attn={kq['attn']['ket_luan']}"]
    dong = ["Tóm tắt B7, B8: " + ", ".join(ds),
            f"  cum chrF macro trung bình {_f(c2['trung_binh'])} ± {_f(c2['do_lech_chuan'])} (số fold {c2['so_fold']})",
            f"  túi lỗi category: cum {len(kq['tui_loi']['cum'])} khóa, pair {len(kq['tui_loi']['pair'])} khóa"]
    for ten in ("c1", "c2", "c3"):
        dong += [f"  {ten.upper()} {x}" for x in kq[ten]["ly_do"]]
        if kq[ten]["hanh_dong"]:
            dong.append(f"  {ten.upper()} hành động: {kq[ten]['hanh_dong']}")
    return "\n".join(dong)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tổng hợp B7, B8 từ file trạng thái Kaggle")
    ap.add_argument("--trang-thai", required=True, help="file b78_trang_thai.json")
    ap.add_argument("--ra", required=True, help="thư mục ghi bao_cao_b78.md và bao_cao_b78.json")
    args = ap.parse_args(argv)
    tt = doc_trang_thai(args.trang_thai)
    kq = tinh_tat_ca(tt)
    ra = Path(args.ra)
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "bao_cao_b78.md").write_text(viet_bao_cao(tt, kq), encoding="utf-8")
    goi = {"xong": tt["xong"], "loi_doc": tt["loi_doc"], **kq,
           "nguong_mac_dinh": {"chrf": NGUONG_CHRF, "sinh_lap": NGUONG_SINH_LAP, "loss_tuong_doi": NGUONG_LOSS_TUONG_DOI,
                               "he_so_category": HE_SO_CATEGORY, "toi_thieu_token": TOI_THIEU_TOKEN},
           "cho_zero_shot": GHI_CHU_ZERO_SHOT}
    (ra / "bao_cao_b78.json").write_text(json.dumps(goi, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tom_tat(kq))
    print(f"Đã ghi {ra / 'bao_cao_b78.md'} và {ra / 'bao_cao_b78.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
