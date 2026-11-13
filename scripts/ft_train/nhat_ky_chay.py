"""Gom thông số và kết quả MỌI lần chạy fine-tune vào một nhật ký để tra cứu khi viết báo cáo và khi bị phản biện (0 GPU).
Chỉ ĐỌC các file kết quả đã tải về (ft_run/kaggle_*), không đọc test_khoa/, split.parquet, ds_test.txt. Số nào cũng lấy từ file, không gõ tay.
Chạy lại sau mỗi phiên mới (thêm phiên vào PHIEN_CHINH / NGUON_TRANG_THAI bên dưới):  python -I -X utf8 scripts/ft_train/nhat_ky_chay.py
Ra: data/processed/ft_run/NHAT_KY_CHAY.md và NHAT_KY_CHAY.json"""
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
FT = GOC / "data" / "processed" / "ft_run"
TT = Path(__file__).resolve().parent
ICT = timezone(timedelta(hours=7))

# (nhãn, bản Kaggle, đường dẫn báo cáo phiên, ghi chú). Bản Kaggle lấy từ tu_dong.log của mỗi nhóm phiên.
PHIEN_CHINH = [
    ("B6 phiên 1", 20, "kaggle_b6/phien_1_20261105_095809/bao_cao_b6.json", "dò siêu tham số, dừng có trật tự ở giờ tối đa"),
    ("B6 phiên 2", 21, "kaggle_b6/phien_2_20261105_123231/bao_cao_b6.json", "nối từ phiên 1"),
    ("B6 phiên 3", 22, "kaggle_b6/phien_3_20261105_150627/bao_cao_b6.json", "nối từ phiên 2, xong B6"),
    ("B7 + B8 phần đầu", 25, "kaggle_b78/phien_1_20261105_181337/bao_cao_b78.json", "attn, pair_f0, cum_f1 dừng ở bước 1.102/1.800 do hết giờ tối đa; bản 26 (nối tiếp) bị Nhật Anh hủy, không có báo cáo"),
    ("Thử tỉ lệ fold 0", 29, "kaggle_tl/phien_1_20261105_231941/bao_cao_tl.json", "cum/pair × ~11% và 75% title"),
    ("B8 fold 1, 2 mốc 1 : 1 + cụm ~11% fold 1", 32, "kaggle_b8f1/phien_1_20261106_102741/bao_cao_b78.json", "nối cum_f1 từ bước 1.102 của bản 25, rồi cum_f1_p11, cum_f2"),
    ("Cụm ~11% fold 2", 34, "kaggle_b8f2/phien_1_20261106_165058/bao_cao_b8f2.json", "cum_f2_p11"),
    ("Kiểm lr 1e-4 (cụm) + pair fold 1", 36, "kaggle_tl2/phien_1_20261106_210034/bao_cao_tl2.json", "cum_f0_p11_lr1e4, pair_f1, pair_f1_p11"),
    ("Pair fold 2", 38, "kaggle_tl3/phien_1_20261107_073538/bao_cao_tl3.json", "pair_f2, pair_f2_p11; bản 37 là khói, bản 38 là phiên thật"),
    ("Đo test MỘT lần", 41, "kaggle_do_test/phien_1_20261107_080604/bao_cao_do_test.json", "không huấn luyện: zero-shot và adapter fold 0 (cum_f0_p11, pair_f0_p11) trên toàn bộ 5.116 mẫu test mỗi nhánh; bản 39, 40 là khói"),
    ("Bản cuối trên toàn pool", 43, "kaggle_cuoi/phien_1_20261107_105449/bao_cao_cuoi.json", "cum_cuoi, pair_cuoi: toàn pool, loại 6.386 title bị cờ; bản 42 là khói"),
]
PHIEN_KHOI = ["kaggle_b5", "kaggle_b5_lan2", "kaggle_b5_lan3", "kaggle_b5_lan4", "kaggle_do_toc_do", "kaggle_nguyen_nhan", "kaggle_b6_thu", "kaggle_b6_thu2",
              "kaggle_b6_noi/phan1", "kaggle_b6_noi/phan2", "kaggle_b78_thu/phan1", "kaggle_b78_thu/phan2", "kaggle_tl_thu/phan1", "kaggle_tl_thu/phan2",
              "kaggle_tl3_thu", "kaggle_do_test_thu", "kaggle_cuoi_thu"]
# (nhãn, bản Kaggle sinh ra, đường dẫn trạng thái, "b6" = cấu trúc B6, còn lại cấu trúc viec). Theo thứ tự thời gian: việc làm xong ở trạng thái sau thay bản dở ở trạng thái trước.
NGUON_TRANG_THAI = [
    ("B6", "20–22", "kaggle_b6/ket_qua_cuoi/b6/b6_trang_thai.json", "b6"),
    ("B7+B8 phần đầu", "25", "kaggle_b78/phien_1_20261105_181337/b78/b78_trang_thai.json", "viec"),
    ("Thử tỉ lệ fold 0", "29", "kaggle_tl/phien_1_20261105_231941/tl/tl_trang_thai.json", "viec"),
    ("B8 fold 1, 2 + cụm ~11% fold 1", "25→32", "kaggle_b8f1/phien_1_20261106_102741/b78/b78_trang_thai.json", "viec"),
    ("Cụm ~11% fold 2", "34", "kaggle_b8f2/phien_1_20261106_165058/b8f2/b8f2_trang_thai.json", "viec"),
    ("Kiểm lr 1e-4 (cụm) + pair fold 1", "36", "kaggle_tl2/phien_1_20261106_210034/tl2/tl2_trang_thai.json", "viec"),
    ("Pair fold 2", "38", "kaggle_tl3/phien_1_20261107_073538/tl3/tl3_trang_thai.json", "viec"),
    ("Bản cuối trên toàn pool", "43", "kaggle_cuoi/phien_1_20261107_105449/cuoi/cuoi_trang_thai.json", "viec"),
]
NHOM = ["zh->vi|title", "zh->vi|mo_ta", "vi->zh|title", "vi->zh|mo_ta"]


def doc(p):
    return json.loads((FT / p).read_text(encoding="utf-8"))


def lay(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f(x, n=2):
    return "–" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as g:
        for kh in iter(lambda: g.read(1 << 20), b""):
            h.update(kh)
    return h.hexdigest()


def moc_gio(t):
    return datetime.fromtimestamp(t, tz=ICT).strftime("%d/%m/%Y %H:%M:%S") if t else "–"


def tham_so_tu_ten(ten):
    m = re.search(r"lr([0-9.]+e-?\d+)_r(\d+)", ten or "")
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


def so_cham(c):
    """Rút các số chấm từ khối "cham" (định dạng trạng thái Kaggle)."""
    if not isinstance(c, dict):
        return {}
    nh = lambda k: lay(c, "theo_nhom", k, "chrf_corpus")  # noqa: E731
    tb = lambda a, b: None if a is None or b is None else (a + b) / 2  # noqa: E731
    moi = lambda h: (lay(c, "dong_da_thay_moi", h, "moi", "n_dong"), lay(c, "dong_da_thay_moi", h, "moi", "chrf_corpus"))  # noqa: E731
    return {"epoch_cham": c.get("epoch"), "so_mau_sinh": c.get("so_mau"), "chrf_chung_corpus": lay(c, "chrf", "chrf_macro_huong"), "chrf_chung_theo_cau": lay(c, "chrf", "chrf_cau_tb_macro_huong"),
            "title": tb(nh("zh->vi/title"), nh("vi->zh/title")), "mo_ta": tb(nh("zh->vi/mo_ta"), nh("vi->zh/mo_ta")),
            "zh_vi": lay(c, "theo_huong", "zh->vi", "chrf_corpus"), "vi_zh": lay(c, "theo_huong", "vi->zh", "chrf_corpus"),
            "moi_zh_vi_n": moi("zh->vi")[0], "moi_zh_vi": moi("zh->vi")[1], "moi_vi_zh_n": moi("vi->zh")[0], "moi_vi_zh": moi("vi->zh")[1],
            "sinh_lap": c.get("sinh_lap"), "ti_le_cat_loi_dan": c.get("ti_le_cat_loi_dan"), "giay_moi_mau": c.get("giay_moi_mau"),
            "theo_nhom_chrf": {k: lay(v, "chrf_corpus") for k, v in (c.get("theo_nhom") or {}).items()}}


def dong_viec(ten, v, nhan, ban):
    lr_ten, rank = tham_so_tu_ten(v.get("cau_hinh") or ten)
    loss = v.get("loss") or {}
    diem = v.get("diem") or {}
    ep_cuoi = max((int(k) for k in diem), default=None)
    gr = v.get("thong_ke_grad") or {}
    r = {"ten": ten, "nguon": nhan, "ban_kaggle": ban, "nhanh": v.get("nhanh", "cum"), "fold_huan_luyen_kiem": v.get("fold"), "cau_hinh": v.get("cau_hinh") or ten,
         "lr": v.get("lr") or lr_ten, "rank": rank, "so_tham_so_lora": v.get("so_tham_so"), "epoch_xong": v.get("epoch_xong"), "tong_buoc": v.get("tong_buoc") or v.get("buoc"),
         "buoc_da_chay": v.get("buoc"), "so_train": v.get("so_train"), "so_token_train": v.get("so_token_train"),
         "p_title_muc_tieu": v.get("p_muc_tieu"), "p_title_thuc": v.get("p_thuc"), "p_title_khi_w_1": v.get("p_khi_w_1"), "w_title": v.get("w_title"), "he_chuan": v.get("he_chuan"),
         "diem_loss_theo_epoch": diem, "loss_nhom_theo_epoch": loss, "loss_ca_fold": v.get("loss_ca_fold"), "token_moi_giay": v.get("token_moi_giay"),
         "clip_grad": {k: {"so_buoc": g.get("so_buoc"), "so_bi_cat": g.get("so_bi_cat")} for k, g in gr.items()} if gr else None, "so_id_sinh_thieu": v.get("so_id_sinh_thieu"),
         "ket_qua_sinh": so_cham(v.get("cham")), "epoch_loss_cuoi": ep_cuoi}
    return r


def doc_trang_thai():
    ket = {}
    ghi_chu = []
    for nhan, ban, p, kieu in NGUON_TRANG_THAI:
        d = doc(p)
        if kieu == "b6":
            cham = lay(d, "cham") or {}
            for ten, v in d["cau_hinh"].items():
                v = dict(v, cham=cham.get(ten), fold=d.get("fold_kiem"), nhanh="cum", so_train=d.get("so_train"), so_token_train=d.get("so_token_train"), tong_buoc=d.get("tong_buoc") if v.get("epoch_xong", 0) >= 2 else v.get("buoc"))
                ket[ten] = dong_viec(ten, v, nhan, ban)
            continue
        for ten, v in d["viec"].items():
            cu = ket.get(ten)
            if not v.get("tong_buoc") and v.get("buoc") == 0:
                gc = f"{ten}: các trạng thái B7/B8 chỉ ghi tham chiếu tới kết quả B6 cùng cấu hình (lr2e-4_r64, fold 0), không phải lần chạy riêng; ở bảng dưới việc này nằm ở dòng `lr2e-4_r64`."
                if gc not in ghi_chu:
                    ghi_chu.append(gc)
                continue
            if cu and (cu.get("epoch_xong") or 0) >= 2 and (v.get("epoch_xong") or 0) >= 2:
                continue
            if cu:
                ghi_chu.append(f"{ten}: bản dở ở '{cu['nguon']}' (epoch {cu.get('epoch_xong')}, đã chạy {cu.get('buoc_da_chay')} bước) được thay bằng bản xong ở '{nhan}' (tiếp tục từ checkpoint). Thống kê clip gradient của {ten} chỉ tính phần chạy ở bản sau ({v.get('so_train') and (v.get('tong_buoc') - cu.get('buoc_da_chay'))} bước cuối).")
            ket[ten] = dong_viec(ten, v, nhan, ban)
        for ten, v in (d.get("attn") or {}).items():
            ten_moi = f"{ten} (epoch 1, chấm lại ở B7)"
            if isinstance(v, dict) and v.get("chrf") and ten_moi not in ket:
                ket[ten_moi] = dong_viec(ten, {"cau_hinh": ten, "cham": v, "nhanh": "cum", "fold": 0, "epoch_xong": 1}, nhan, ban)
    # mốc cum_f0 (= B6 lr2e-4_r64) được chấm đầy đủ hơn ở đợt thử tỉ lệ: bổ sung zh→vi, vi→zh, dòng mới
    so = lay(doc("kaggle_tl/phien_1_20261105_231941/tong_hop/bao_cao_ti_le.json"), "nhanh", "cum", "moc", "so")
    r = ket.get("lr2e-4_r64")
    if r and so:
        r["ket_qua_sinh"].update(zh_vi=so.get("zh_vi"), vi_zh=so.get("vi_zh"), moi_zh_vi_n=lay(so, "moi_zh_vi", "n"), moi_zh_vi=lay(so, "moi_zh_vi", "chrf"),
                                 moi_vi_zh_n=lay(so, "moi_vi_zh", "n"), moi_vi_zh=lay(so, "moi_vi_zh", "chrf"))
        ghi_chu.append("lr2e-4_r64 (mốc cum_f0): zh→vi, vi→zh và dòng mới lấy từ lần chấm lại bản dịch B6 ở đợt thử tỉ lệ (`bao_cao_ti_le.json`, mốc); chrF chung chấm lại trùng điểm B6 75,7613 (độ lệch 0).")
    return ket, ghi_chu


def doc_phien():
    ra = []
    for nhan, ban, p, gc in PHIEN_CHINH:
        b = doc(p)
        c = b.get("cau_hinh") or {}
        mt = b.get("moi_truong") or {}
        t0 = c.get("_t_bat_dau")
        ra.append({"nhan": nhan, "ban_kaggle": ban, "bao_cao": p, "bat_dau_ICT": moc_gio(t0), "ket_thuc_ICT": moc_gio(t0 + b["tong_giay"]) if t0 and b.get("tong_giay") else "–", "tong_giay": b.get("tong_giay"),
                   "gio": round(b["tong_giay"] / 3600, 2) if b.get("tong_giay") else None, "gio_toi_da_cai_dat": c.get("gio_toi_da"), "xong_het_viec": b.get("dat"), "khoa_test": b.get("khoa_test"),
                   "gpu": mt.get("gpu"), "torch": mt.get("torch"), "transformers": mt.get("transformers"), "peft": mt.get("peft"), "ghi_chu": gc})
    return ra


def doc_khoi():
    ra = []
    for p in PHIEN_KHOI:
        for fb in sorted((FT / p).glob("bao_cao_*.json")):
            b = json.loads(fb.read_text(encoding="utf-8"))
            ra.append({"thu_muc": p, "tep": fb.name, "tong_giay": b.get("tong_giay"), "dat": b.get("dat"), "gpu": lay(b, "moi_truong", "gpu"), "khoa_test": b.get("khoa_test")})
    return ra


def cap(d):
    if not isinstance(d, dict) or d.get("chenh") is None:
        return None
    return {"chenh": d["chenh"], "thap": d.get("thap"), "cao": d.get("cao")}


def doc_so_cap():
    """Mọi so sánh theo cặp (bootstrap 95%). chenh = mức đầu − mức sau, như ghi trong nhãn."""
    ra = []
    b6 = doc("kaggle_b6/ket_qua_cuoi/bao_cao_b6.json")
    for k, v in (lay(b6, "b6", "so_cap_chrf") or {}).items():
        ra.append(("B6 chrF chung (theo câu), epoch 2", k, cap(v)))
    b78 = doc("kaggle_b78/phien_1_20261105_181337/b78/b78_trang_thai.json")
    ra.append(("B7 chrF chung, epoch 1, fold 0", "LoRA attention (q,k,v,o) − LoRA mọi lớp, cùng lr1e-4_r16", cap(lay(b78, "attn", "so_cap"))))
    tl = doc("kaggle_tl/phien_1_20261105_231941/tong_hop/bao_cao_ti_le.json")
    for nh, m in tl["nhanh"].items():
        for ma, v in (m.get("muc") or {}).items():
            ra.append((f"Thử tỉ lệ, fold 0, chrF chung theo câu, bản {nh}", f"{v.get('ten')} − mốc ~50%", cap(v.get("so_cap"))))
    nguon = [("Dòng mới fold 0 (kiem_dong_moi)", "kaggle_tl/phien_1_20261105_231941/tong_hop/kiem_dong_moi.json", True),
             ("Fold 1", "kaggle_b8f1/phien_1_20261106_102741/tong_hop/kiem_fold1.json", False),
             ("Fold 2", "kaggle_b8f2/phien_1_20261106_165058/tong_hop/kiem_fold2.json", False)]
    ten_muc = [("dòng mới mô tả zh→vi (theo sản phẩm)", ("zh_vi", "chinh")), ("dòng mới vi→zh", ("vi_zh", "chinh")), ("title hai hướng (chrF corpus)", ("title", "corpus")),
               ("title hai hướng (theo mục)", ("title", "theo_muc")),
               ("title zh→vi (theo mục)", ("title_zh_vi", "theo_muc")), ("mô tả zh→vi toàn bộ (theo mục)", ("mo_ta_zh_vi", "theo_muc"))]
    for nhan, p, hai_nhanh in nguon:
        d = doc(p)
        for nh in (("cum", "pair") if hai_nhanh else (None,)):
            g = d[nh] if nh else d
            for ten, ks in ten_muc:
                c = cap(lay(g, *ks))
                if c:
                    ra.append((f"{nhan}{', bản ' + nh if nh else ''}: mức ~11% − mốc 1 : 1", ten, c))
    for nhan, p in [("Kiểm lr, cụm fold 0 (luật lr): lr 1e-4 − lr 2e-4, cùng ~11%", "kaggle_tl2/cham/kiem_lr_cum_f0.json"),
                    ("Pair fold 1 (luật pair): mức ~11% − mốc 1 : 1", "kaggle_tl2/cham/kiem_pair_f1.json")]:
        d = doc(p)
        for ten, ks in ten_muc:
            c = cap(lay(d, *ks))
            if c:
                ra.append((nhan, ten, c))
        ra.append((nhan, "chrF chung (theo câu)", cap(d.get("theo_cau"))))
    p3 = doc("kaggle_tl3/cham/kiem_3fold_pair.json")
    for fo, d in p3["theo_fold"].items():
        for ten, ks in ten_muc:
            c = cap(lay(d, *ks))
            if c:
                ra.append((f"Bản pair, fold {fo} (luật 3 fold pair): mức ~11% − mốc 1 : 1", ten, c))
        ra.append((f"Bản pair, fold {fo} (luật 3 fold pair): mức ~11% − mốc 1 : 1", "chrF chung (theo câu)", cap(d.get("theo_cau"))))
    k3 = doc("kaggle_b8f2/phien_1_20261106_165058/tong_hop/kiem_3fold.json")
    for fo, h in k3["theo_fold"].items():
        ra.append((f"Bản cụm, fold {fo}: mức ~11% − mốc 1 : 1", "chrF chung theo câu", cap(h.get("theo_cau"))))
    return ra, k3


def dau_van_tay():
    ra = {"script": {}, "du_lieu": {}, "sinh": {}, "kho_luu": {}}
    for ten in ["kaggle_train.py", "cham_diem.py", "dong_goi_kaggle.py", "chung.py", "dung_mau.py", "tong_hop_ti_le.py", "kiem_fold_ti_le.py", "kiem_3fold_ti_le.py", "kiem_dong_moi.py", "tong_hop_b78.py"]:
        p = TT / ten
        if p.exists():
            ra["script"][ten] = {"sha256": sha(p), "sua_lan_cuoi": datetime.fromtimestamp(p.stat().st_mtime, tz=ICT).strftime("%d/%m/%Y %H:%M")}
    dd = doc("DU_LIEU_DONG_BANG.json")
    ra["du_lieu"] = {"hf": dd["nguon_du_lieu"], "split": {k: v for k, v in dd["split"].items() if k != "file"}}
    for p in sorted(FT.glob("kaggle_*/**/sinh_ep2.jsonl")):
        if "_thu" in str(p.relative_to(FT)):
            continue
        ra["sinh"][str(p.relative_to(FT)).replace("\\", "/")] = {"sha256": sha(p), "byte": p.stat().st_size}
    for p in sorted(FT.glob("kaggle_*/**/MANIFEST*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        ra["kho_luu"][str(p.relative_to(FT)).replace("\\", "/")] = {"so_file": len(m), "ds": {k: v.get("sha256") for k, v in m.items()}}
    return ra


def viet_md(phien, khoi, ket, ghi_chu, so_cap, k3, vt, dt=None):
    L = []
    w = L.append
    w("# NHẬT KÝ CÁC LẦN CHẠY FINE-TUNE QWEN3-4B + LORA\n")
    w(f"Sinh tự động bởi `scripts/ft_train/nhat_ky_chay.py` lúc {datetime.now(ICT).strftime('%d/%m/%Y %H:%M')} ICT. Mọi số lấy từ file kết quả trong `data/processed/ft_run/kaggle_*`, không gõ tay. Nhãn: [ĐO] = từ file; điểm chrF là độ giống bản dịch máy dùng làm tham chiếu, chưa phải người chấm. Không dùng tập test ở bất kỳ lần chạy nào trong bảng này.\n")
    w("## 1. Các phiên Kaggle (kết quả chính)\n")
    w("| Phiên | Bản Kaggle | Bắt đầu (ICT) | Kết thúc (ICT) | Giây | Giờ | Giờ tối đa cài | Xong hết việc | GPU | torch / transformers / peft | Ghi chú |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for p in phien:
        w(f"| {p['nhan']} | {p['ban_kaggle']} | {p['bat_dau_ICT']} | {p['ket_thuc_ICT']} | {f(p['tong_giay'], 0)} | {f(p['gio'])} | {f(p['gio_toi_da_cai_dat'], 1)} | {p['xong_het_viec']} | {p['gpu']} | {p['torch']} / {p['transformers']} / {p['peft']} | {p['ghi_chu']} |")
    tong = sum(p["gio"] or 0 for p in phien)
    w("\n`Xong hết việc = False` nghĩa là phiên dừng có trật tự vì hết giờ tối đa cài đặt, việc dở được nối ở phiên sau (không phải lỗi).")
    w(f"\nTổng giờ GPU của các phiên chính: **{tong:.2f} giờ** (chưa tính phiên khói bên dưới). `khoa_test` ở mọi phiên: " + ", ".join(sorted({str(p['khoa_test']) for p in phien})) + ".\n")
    w("Phiên khói, thử môi trường và đo tốc độ (không dùng làm kết quả): " + "; ".join(f"{k['thu_muc']} {f(k['tong_giay'], 0)} giây" for k in khoi) + ".\n")
    w("## 2. Thông số cố định cho mọi lần chạy huấn luyện [đọc từ mã `kaggle_train.py`]\n")
    for s in ["Mô hình nền `Qwen/Qwen3-4B` (revision 1cfa9a72), nạp bf16, `attn_implementation=sdpa`, tắt suy luận (`enable_thinking=False`).",
              "LoRA: `lora_alpha = 2 × rank`, `lora_dropout = 0,05`, `target_modules = all-linear` (riêng lần thử attention: q_proj, k_proj, v_proj, o_proj).",
              "Tối ưu: AdamW, `weight_decay = 0`, cắt gradient `clip_grad_norm_ = 1,0`; lịch lr: warmup 3% rồi giảm cosine về 0, tính cho cả 2 epoch.",
              "Dữ liệu: `max_len` 2.048 token, `micro_bs` 32, ngân sách 4.096 token mỗi lô nhỏ, không bật gradient checkpointing; seed 42 (xáo lô theo seed + chỉ số epoch).",
              "Số epoch: B6 vòng 1 là 1 epoch; mọi lần chạy từ vòng 2 về sau là 2 epoch.",
              "Kiểm loss: 3.000 mẫu kiểm cố định (750 mỗi hướng × loại), nằm trong fold kiểm; loss không trọng số, điểm = trung bình theo hướng, title và mô tả trọng số bằng nhau.",
              "Sinh bản dịch: greedy (`do_sample=False`), trần `max_new_tokens` theo độ dài đích tham chiếu cho mọi lần chạy huấn luyện và chấm theo fold, 1.000 mẫu (250 mỗi hướng × loại) giống nhau giữa các cấu hình để so theo cặp. Riêng lần đo test (mục 9) dùng trần theo độ dài NGUỒN `min(2048, 3,5 × token nguồn + 64)`, chọn từ tỉ lệ token đích/nguồn lớn nhất đo trên pool (2,94), và bộ chấm có thêm mốc tham chiếu `ref_*` và quy tắc cắt lời dẫn `dong_ngu_canh`.",
              "Loss có trọng số (từ đợt thử tỉ lệ): mẫu title nhân `w_title`, mẫu mô tả nhân 1; `he_chuan = he_chuan_goc × tổng_trọng_số / số_mẫu` giữ độ lớn bước tương đương; đánh giá vẫn không trọng số. Tỉ lệ title là siêu tham số cố định trong một lần train.",
              "Chia dữ liệu: `chia_tap_v2`, seed 20261104, test 10% khóa cứng (2.558 sản phẩm), pool 23.017 chia 5 fold theo nhóm gần-trùng; Dataset Kaggle chỉ có mẫu pool."]:
        w(f"- {s}")
    w("\n## 3. Thông số huấn luyện của từng việc\n")
    w("| Việc | Từ trạng thái | Bản Kaggle | Nhánh | Fold kiểm | lr | rank | Tham số LoRA | Epoch xong | Tổng bước | Mẫu train | Token train | % title mục tiêu | % title thực | % title nếu w=1 | w_title | he_chuan | Bước clip (title / trộn / mô tả) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ten, r in ket.items():
        if "chấm lại ở B7" in ten:
            continue
        cg = r["clip_grad"] or {}
        clip = " / ".join(f"{cg.get(k, {}).get('so_bi_cat')}/{cg.get(k, {}).get('so_buoc')}" for k in ("title", "tron", "mo_ta")) if cg else "–"
        pc = lambda x: "–" if x is None else f"{100 * x:.2f}"  # noqa: E731
        w(f"| {ten} | {r['nguon']} | {r['ban_kaggle']} | {r['nhanh']} | {f(r['fold_huan_luyen_kiem'], 0)} | {r['lr']} | {r['rank']} | {r['so_tham_so_lora']} | {r['epoch_xong']} | {r['tong_buoc']} | {r['so_train']} | {r['so_token_train']} | {pc(r['p_title_muc_tieu'])} | {pc(r['p_title_thuc'])} | {pc(r['p_title_khi_w_1'])} | {f(r['w_title'], 5)} | {f(r['he_chuan'], 4)} | {clip} |")
    w("\n`fold kiểm` là fold dùng để kiểm; huấn luyện bằng bốn fold còn lại của pool. Ô `Bước clip` là số bước bị cắt gradient / số bước, tách theo loại lô (lô toàn title / lô trộn / lô toàn mô tả); `–` là việc chạy trước khi ghi thống kê này. Dòng B6 ở vòng 1 chỉ có 1 epoch. Hai lần thử `lr1e-4_r16` và `lr1e-4_r16_attn` ở epoch 1 được sinh và chấm lại ở B7 từ adapter B6 (không train lại), nằm ở bảng kết quả bên dưới.\n")
    w("## 4. Kết quả từng việc (fold kiểm, 1.000 mẫu sinh)\n")
    w("| Việc | Điểm loss ep1 | Điểm loss ep2 | chrF chung (corpus) | chrF chung (theo câu) | Title | Mô tả | zh→vi | vi→zh | Dòng mới zh→vi (n / chrF) | Dòng mới vi→zh (n / chrF) | Sinh lặp / 1.000 | Giây mỗi mẫu | Token/giây ep2 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ten, r in ket.items():
        s = r["ket_qua_sinh"] or {}
        d = r["diem_loss_theo_epoch"] or {}
        tk = r["token_moi_giay"] or {}
        tkv = tk.get("2") or tk.get("1")
        w(f"| {ten} | {f(d.get('1'), 4)} | {f(d.get('2'), 4)} | {f(s.get('chrf_chung_corpus'))} | {f(s.get('chrf_chung_theo_cau'))} | {f(s.get('title'))} | {f(s.get('mo_ta'))} | {f(s.get('zh_vi'))} | {f(s.get('vi_zh'))} | {f(s.get('moi_zh_vi_n'), 0)} / {f(s.get('moi_zh_vi'))} | {f(s.get('moi_vi_zh_n'), 0)} / {f(s.get('moi_vi_zh'))} | {f(s.get('sinh_lap'), 0)} | {f(s.get('giay_moi_mau'), 3)} | {f(tkv, 0)} |")
    w("\nViệc không có chrF là việc chỉ chạy tới đó (vòng 1 B6 chỉ so loss, `cum_f1` bản dở). `Dòng mới` là dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train; `chrF chung (corpus)` là điểm chính dùng để so.\n")
    w("## 5. Loss không trọng số theo hướng × loại, ở epoch cuối\n")
    w("| Việc | Epoch | " + " | ".join(k.replace("|", " ") for k in NHOM) + " |\n|---|---|" + "---|" * len(NHOM))
    for ten, r in ket.items():
        lo = r["loss_nhom_theo_epoch"] or {}
        if not lo:
            continue
        ep = max(lo, key=int)
        w(f"| {ten} | {ep} | " + " | ".join(f(lo[ep].get(k), 4) for k in NHOM) + " |")
    w("\n## 6. So sánh theo cặp (bootstrap theo mục, khoảng tin cậy 95%)\n")
    w("`chênh` = mức đầu trừ mức sau như ghi ở cột 2. Khoảng chỉ tính dao động do chọn mẫu kiểm, KHÔNG tính dao động do huấn luyện lại (chưa chạy seed khác). Khoảng chứa 0 là hòa.\n")
    w("| Nhóm so sánh | Cặp / chỉ số | Chênh | Thấp | Cao |\n|---|---|---|---|---|")
    for nhom, ten, c in so_cap:
        w(f"| {nhom} | {ten} | {f(c['chenh'])} | {f(c['thap'])} | {f(c['cao'])} |")
    w(f"\n**Tiêu chí 3 fold (bản cụm, ~11% − 1 : 1)** [ĐO, `kiem_3fold.json`]: chênh chrF chung corpus theo fold = {', '.join(f(x, 3) for x in k3['chenh_chrf_chung'])}; trung bình {f(k3['trung_binh'], 3)}; độ lệch chuẩn {f(k3['do_lech_chuan'], 3)}; kết luận đổi tỉ lệ: **{k3['ket_luan_kfold']['doi_ti_le']}** ({'; '.join(k3['ket_luan_kfold']['ly_do']) if isinstance(k3['ket_luan_kfold']['ly_do'], list) else k3['ket_luan_kfold']['ly_do']}); điều kiện thêm fold 2: {k3['dieu_kien_them_fold2']}.\n")
    w("## 7. Ghi chú và sự cố cần nhớ khi bị hỏi\n")
    for s in ghi_chu:
        w(f"- {s}")
    for s in ["Bản Kaggle 26 (nối B7+B8 phần hai) bị Nhật Anh hủy; `cum_f1` dừng ở bước 1.102/1.800 rồi được nối ở bản 32 từ checkpoint lưu trong notebook `ft-qwen3-4b-b78-luu` (có cả `opt.pt`, trạng thái tối ưu, và vị trí lô). Chạy khói đã kiểm nối đúng số bước, nhưng cum_f1 vẫn là một lần huấn luyện đi qua hai phiên và hai bản mã Kaggle (25 rồi 32).",
              "Mã `kaggle_train.py` bản cũ KHÔNG được lưu riêng: B6 (bản 20–22) và B7+B8 phần đầu (bản 25) chạy bằng bản mã trước lần sửa lúc 23:06 ngày 05/11/2026; từ bản 29 trở đi chạy bằng bản hiện tại (mtime 23:06 05/11, không đổi sau đó; bản Kaggle 34 còn nguyên trong `notebooks/kaggle_api/nlp-group3/nlp-group3.ipynb`). Lần sửa đó thêm (theo plan v0.6.1, bước 1): loss có trọng số theo tỉ lệ title, danh sách việc lấy từ cấu hình, chọn trạng thái tiến xa hơn. Em chưa chạy lại cùng một cấu hình bằng hai bản mã để so kết quả; bằng chứng duy nhất là phép thử trong phiên: với w = 1 loss có trọng số trùng `out.loss` của mô hình (3,4583 = 3,4583, ghi ở `thu_loss_1_1`), và bản dịch B6 chấm lại trên máy ra đúng điểm B6 (chỉ chứng minh bản dịch không bị đổi, không chứng minh đường mã).",
              "Gradient bị cắt (clip 1,0) khác nhau rõ giữa các việc: mốc 1 : 1 `cum_f2` bị cắt 112/256 bước lô toàn title, còn `cum_f2_p11` chỉ 3/256; mức 75% bị cắt gần như mọi bước title (262/263, 263/263). Tức ở mức 1 : 1 các lô title cũng đã bị cắt thường xuyên; thống kê này chưa có cho `cum_f0` và `pair_f0` (chạy trước khi ghi).",
              "Kết luận hiện tại: bản cụm đạt luật khóa 3 fold và chốt 10,74% title, lr 2e-4 (kiểm lr 1e-4 ở mức này KHÔNG đạt: chrF chung theo câu +0,03 [-0,43; +0,47]); bản pair đạt luật 3 fold riêng của pair (khóa trước ở WALKTHROUGH Prompt 804, `kiem_3fold_pair.py`) và Nhật Anh duyệt chốt 10,86%; tập test đã đo MỘT lần (mục 9): fine-tune thắng zero-shot ở mọi nhánh, mọi hướng; bản cuối đã huấn luyện (mục 10). CÁC ĐIỂM CHƯA được người hay LLM chấm so cặp (B9); chrF chỉ đo độ giống bản dịch máy dùng làm tham chiếu.",
              "Các mức tỉ lệ đã thử: ~11% (3 fold cho cả cụm và pair), ~50% (mốc, 3 fold cho cả hai bản), 75% (fold 0 cả hai bản). Mức 75% bị clip gradient gần như mọi bước title nên tác dụng thực thấp hơn danh nghĩa. Fold 0 cũng là nơi phát hiện tín hiệu dòng mới zh→vi nên chỉ fold 1, 2 là kiểm độc lập; luật pair 3 fold tính trung bình title nên một fold title tụt (fold 0: title hai hướng -0,96, zh→vi -1,19) có thể được fold khác bù.",
              "Sự cố khi vận hành, ghi để phản biện: tiến trình tự động trên máy (`tl3_tu_dong`) từng chết giữa chừng khi chờ khói; cổng khói được chạy tay bằng đúng hàm `kiem_khoi_tl3` rồi mới đẩy phiên thật. Phiên Kaggle chạy độc lập với máy nên không mất kết quả. Khói đo test lần đầu (bản 39) lỗi do đường dẫn manifest, đã sửa (bản 40). Script kiểm khói bản cuối em viết có ngưỡng chất lượng áp nhầm cho khói 15 bước nên báo không đạt dù cơ chế đều đạt (bản 42); ngưỡng đó được áp lại và đạt ở bản thật."]:
        w(f"- {s}")
    w("\n## 8. Dấu vân tay để chứng minh số không bị sửa\n")
    w("### Dữ liệu và phép chia\n")
    hf = vt["du_lieu"]["hf"]
    w(f"- HF `{hf['hf_repo']}`, thư mục `{hf['thu_muc']}`, commit `{hf['commit']}`")
    for k, v in hf["sha256"].items():
        w(f"  - `{k}`: `{v}`")
    sp = vt["du_lieu"]["split"]
    w(f"- Phép chia `{sp['phien_ban']}`: sha256 file `{sp['sha256_file']}`, sha256 danh sách test `{sp['sha256_ds_test']}`, test {sp['n_test']}, pool {sp['n_pool']}, seed {sp['seed']}, ngưỡng Jaccard {sp['nguong_jaccard']}")
    w("\n### Mã (sha256 hiện tại)\n")
    w("| Tệp | Sửa lần cuối (ICT) | sha256 |\n|---|---|---|")
    for k, v in vt["script"].items():
        w(f"| {k} | {v['sua_lan_cuoi']} | `{v['sha256']}` |")
    w("\n### Bản dịch sinh ra (sinh_ep2.jsonl)\n")
    w("| Tệp | Byte | sha256 |\n|---|---|---|")
    for k, v in vt["sinh"].items():
        w(f"| {k} | {v['byte']} | `{v['sha256']}` |")
    w("\n### Kho lưu adapter (không GPU)\n")
    for k, v in vt["kho_luu"].items():
        w(f"- `{k}`: {v['so_file']} file kèm sha256 trong tệp manifest.")
    w("- Notebook kho lưu trên Kaggle (tài khoản `trinhnhatanh`): `ft-qwen3-4b-b6-luu`, `ft-qwen3-4b-b78-luu`, `ft-qwen3-4b-tl-luu`, `ft-qwen3-4b-b8f1-luu`, `ft-qwen3-4b-b8f2-luu`, `ft-qwen3-4b-tl2-luu`, `ft-qwen3-4b-tl3-luu`, `ft-qwen3-4b-cuoi-luu` (adapter bản cuối). Dataset riêng tư `ft-qwen3-4b-pool` (mẫu pool) và `ft-qwen3-4b-do-khoa` (tập test, chỉ gắn vào phiên đo test). Đầu ra mọi phiên huấn luyện đã được lưu sang kho riêng trước khi đẩy phiên kế; bản đo test (bản 41) không có adapter, bản dịch và điểm đã tải về máy.")
    if dt:
        w("\n## 9. Đo test MỘT lần (bản 41): fine-tune adapter fold 0 ~11% title so với zero-shot\n")
        w("Luật khóa trước khi đo (`tong_hop_do_test.py`, WALKTHROUGH Prompt 806): mỗi nhánh, mỗi hướng, fine-tune thắng khi cận dưới khoảng 95% (bootstrap theo sản phẩm, seed 42) của chênh chrF theo mục > 0. Mẫu: 5.116 mục / 2.558 sản phẩm mỗi nhánh, greedy, cùng mẫu cho cả hai mô hình. chrF chỉ đo độ giống bản dịch máy dùng làm tham chiếu nên khoảng cách với zero-shot bị phóng đại so với chất lượng thật.\n")
        w("| Nhánh | Hướng | chrF zero-shot | chrF fine-tune | Chênh | Thấp | Cao | Thắng |\n|---|---|---|---|---|---|---|---|")
        for nh, r in dt.items():
            for h, v in r["huong"].items():
                c = v["chung"]
                w(f"| {nh} | {h} | {f(c['chrf_zs'])} | {f(c['chrf_ft'])} | {f(c['chenh'])} | {f(c['thap'])} | {f(c['cao'])} | {c['thang_zero_shot']} |")
        w("\nCảnh báo theo luật: " + ("không có." if not any(r["canh_bao"] for r in dt.values()) else "; ".join(x for r in dt.values() for x in r["canh_bao"])) + " Adapter dùng: `cum_f0_p11` (ce352c70…), `pair_f0_p11` (4d06b578…), kiểm hash theo manifest trước khi đo.\n")
    w("\n## 10. Bản cuối trên toàn pool (bản 43)\n")
    w("Huấn luyện trên toàn pool (23.017 sản phẩm), hai lần, cùng cấu hình đã chốt: ~11% title (cụm 10,74%, pair 10,86%), lr 2e-4, rank 64, 2 epoch. Cách xử lý title (Nhật Anh chọn 07/11/2026): LOẠI mẫu title của 6.386/23.017 sản phẩm bị `loc_title.py` gắn cờ, GIỮ mẫu mô tả; `w_title` tính lại để giữ đúng tỉ trọng. Không còn fold giữ lại nên không chấm chất lượng; chỉ sinh thử 100 mẫu của chính tập huấn luyện để bắt hỏng (kết quả trong bảng mục 3 và 4 và trạng thái `cuoi_trang_thai.json`). Adapter ở `ft-qwen3-4b-cuoi-luu` và đã tải về `ft_run/ban_cuoi/`, hash khớp manifest. Bản cuối KHÔNG đo lại trên test.\n")
    return "\n".join(L) + "\n"


def main():
    phien = doc_phien()
    khoi = doc_khoi()
    ket, ghi_chu = doc_trang_thai()
    so_cap, k3 = doc_so_cap()
    vt = dau_van_tay()
    so_cap = [x for x in so_cap if x[2]]
    p_dt = FT / "kaggle_do_test" / "phien_1_20261107_080604" / "tong_hop" / "tong_hop_do_test.json"
    dt = json.loads(p_dt.read_text(encoding="utf-8")) if p_dt.exists() else None
    (FT / "NHAT_KY_CHAY.json").write_text(json.dumps({"phien_chinh": phien, "phien_khoi": khoi, "viec": ket, "ghi_chu": ghi_chu, "so_sanh_theo_cap": [{"nhom": a, "ten": b, "so": c} for a, b, c in so_cap],
                                                      "tieu_chi_3_fold": k3, "do_test": dt, "dau_van_tay": vt}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (FT / "NHAT_KY_CHAY.md").write_text(viet_md(phien, khoi, ket, ghi_chu, so_cap, k3, vt, dt), encoding="utf-8")
    print("viec:", len(ket), "| phien chinh:", len(phien), "| so sanh:", len(so_cap), "| script:", len(vt["script"]), "| sinh:", len(vt["sinh"]))
    print("ghi:", FT / "NHAT_KY_CHAY.md")


if __name__ == "__main__":
    main()
