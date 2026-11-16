"""Xuất MỖI phiên Kaggle (tài khoản 1 và 2) đã tải về máy thành MỘT file .md riêng + INDEX.md, để nhóm đọc, phản biện và viết báo cáo (0 GPU).

- Mọi CON SỐ đọc thẳng từ file kết quả trong data/processed/ (ft_run/kaggle_*, song_ngu_*), không gõ tay.
- Phần chữ (mục đích, kết luận, ghi chú) viết tay trong bảng PHIEN bên dưới, lấy từ NHAT_KY_CHAY.md, WALKTHROUGH.md và docs/.
- Không đọc test_khoa/, split.parquet, ds_test.txt.
Chạy lại sau mỗi phiên mới (thêm phiên vào PHIEN):  python -I -X utf8 scripts/ft_train/xuat_ket_qua_kaggle.py
Ra: docs/ket_qua_kaggle/*.md (xóa các .md cũ trong thư mục đó trước khi ghi)."""
import hashlib
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
DP = GOC / "data" / "processed"
FT = DP / "ft_run"
RA = GOC / "docs" / "ket_qua_kaggle"
ICT = timezone(timedelta(hours=7))
DUOI_TEP = {".json", ".md", ".log", ".txt", ".jsonl", ".csv"}
NHOM4 = ["zh->vi/title", "zh->vi/mo_ta", "vi->zh/title", "vi->zh/mo_ta"]
_SHA = {}


# ---------------------------------------------------------------- tiện ích
def doc(p):
    p = DP / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def lay(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f(x, n=2):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "có" if x else "không"
    if isinstance(x, int):
        return f"{x:,}".replace(",", ".")
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def nguyen(x):
    return f(int(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else f(x)


def tg(x):
    """token/giây: số, hoặc dict theo epoch {'1': .., '2': ..} -> 'ep1 / ep2'."""
    if isinstance(x, dict):
        return " / ".join(f"ep{k}: {v:.0f}" for k, v in sorted(x.items()))
    return f(x, 0)


def pt(x, n=2):
    return "–" if x is None else f"{100 * x:.{n}f}%"


def ci(c, n=2):
    """{'chenh','thap','cao'} hoặc [chenh, thap, cao] -> '+x [a; b]'."""
    if isinstance(c, dict):
        c = (c.get("chenh"), c.get("thap"), c.get("cao"))
    if not c or c[0] is None:
        return "–"
    return f"{c[0]:+.{n}f} [{c[1]:+.{n}f}; {c[2]:+.{n}f}]"


def bang(tieu_de, dong):
    ra = ["| " + " | ".join(tieu_de) + " |", "|" + "---|" * len(tieu_de)]
    ra += ["| " + " | ".join(str(o).replace("|", "/") for o in d) + " |" for d in dong]
    return ra


def jtxt(v, n=400):
    return f"`{json.dumps(v, ensure_ascii=False)[:n]}`"


def bang_json(p, bo=()):
    """Bảng trường / giá trị của một file JSON phẳng (dict lồng in dạng JSON)."""
    d = doc(p) or {}
    return bang(["Trường", "Giá trị"], [(k, jtxt(v) if isinstance(v, (dict, list)) else f(v)) for k, v in d.items() if k not in bo])


def sha(p):
    if p not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as g:
            for kh in iter(lambda: g.read(1 << 20), b""):
                h.update(kh)
        _SHA[p] = h.hexdigest()
    return _SHA[p]


def gio(giay):
    if not giay:
        return "–"
    return f"{giay:,.0f} giây ({giay / 3600:.2f} giờ)".replace(",", ".")


def mtime(p):
    p = DP / p
    return datetime.fromtimestamp(p.stat().st_mtime, tz=ICT).strftime("%d/%m/%Y %H:%M") if p.exists() else "–"


def rel(p):
    return str(p.relative_to(DP)).replace("\\", "/")


def bo_dau(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def ten_tep(ph):
    return f"{ph['ma']}_{re.sub(r'[^a-z0-9]+', '_', bo_dau(ph['ten']).lower())[:48].strip('_')}.md"


# ---------------------------------------------------------------- phần dùng chung cho mọi phiên
def phan_thong_tin(ph):
    dong = [("Mã", ph["ma"]), ("Giai đoạn", ph["giai_doan"]), ("Notebook Kaggle", ph.get("notebook", "–")), ("Tài khoản", ph.get("tai_khoan", "–")),
            ("Bản Kaggle", ph.get("ban", "không ghi trong file kết quả")), ("Thời gian", ph.get("thoi_gian", "–")),
            ("Thư mục output trên máy Nhật Anh", ", ".join(f"`data/processed/{t}`" for t in ph["thu_muc"]))]
    ra = ["## Thông tin phiên", ""] + bang(["Mục", "Giá trị"], dong) + [""]
    mt = []
    for p in ph.get("bao_cao", []):
        d = doc(p)
        if not isinstance(d, dict):
            continue
        m = d.get("moi_truong") or {}
        mt.append((f"`{p}`", f(m.get("gpu")), f(m.get("vram_gb"), 1), f(m.get("torch")), f(m.get("transformers")), f(m.get("peft")),
                   gio(d.get("tong_giay") or d.get("giay_tong")), f(d.get("dat")), f(d.get("khoa_test"))))
    if mt:
        ra += ["Môi trường và thời lượng, đọc từ file báo cáo của phiên:", ""]
        ra += bang(["File báo cáo", "GPU", "VRAM (GB)", "torch", "transformers", "peft", "Thời lượng", "Đạt / xong", "Khóa test"], mt) + [""]
    return ra


def phan_tep(ph):
    """Liệt kê file gốc + sha256 (16 ký tự đầu) để đối chiếu số không bị sửa."""
    ra = ["## File gốc trên máy (sha256, 16 ký tự đầu)", ""]
    ds = []
    for t in ph["thu_muc"]:
        if ph.get("tep_chinh"):
            ds += [DP / x for x in ph["tep_chinh"] if x.startswith(t) and (DP / x).exists()]
        elif (DP / t).is_dir():
            ds += sorted(x for x in (DP / t).rglob("*") if x.is_file())
    dong, adapter = [], []
    for p in ds:
        if "__pycache__" in p.parts:
            continue
        if p.name == "adapter_model.safetensors":
            adapter.append(p)
        elif p.suffix in DUOI_TEP and p.stat().st_size < 60 * 2**20:
            dong.append((f"`{rel(p)}`", f(p.stat().st_size), f"`{sha(p)[:16]}`"))
    if len(dong) > 60:
        ra += [f"Có {len(dong)} file văn bản; chỉ liệt kê 60 file đầu theo tên.", ""]
        dong = dong[:60]
    ra += bang(["File", "Byte", "sha256"], dong) if dong else ["(không có file văn bản)"]
    if adapter:
        ra += ["", "Adapter LoRA (trọng số) có trong output đã tải:", ""]
        ra += bang(["File", "Byte", "sha256"], [(f"`{rel(p)}`", f(p.stat().st_size), f"`{sha(p)[:16]}`") for p in adapter])
    return ra + [""]


# ---------------------------------------------------------------- phần riêng theo loại phiên
_NK = {}


def nk():
    if not _NK:
        _NK.update(doc("ft_run/NHAT_KY_CHAY.json"))
    return _NK


def r_4b(ban_kaggle, nguon_viec, nhom_so_cap):
    """Phiên fine-tune 4B ở tài khoản 1: lấy từ NHAT_KY_CHAY.json (sinh bởi nhat_ky_chay.py từ file kết quả)."""
    d = nk()
    ra = ["## Các lần chạy Kaggle trong phiên", ""]
    pc = [p for p in d["phien_chinh"] if str(p["ban_kaggle"]) in ban_kaggle]
    ra += bang(["Nhãn", "Bản", "Bắt đầu (ICT)", "Kết thúc (ICT)", "Giờ", "Giờ tối đa cài", "Xong hết việc", "GPU", "torch / transformers / peft"],
               [(p["nhan"], p["ban_kaggle"], p["bat_dau_ICT"], p["ket_thuc_ICT"], f(p["gio"]), f(p["gio_toi_da_cai_dat"]), f(p["xong_het_viec"]), p["gpu"],
                 f"{p.get('torch')} / {p.get('transformers')} / {p.get('peft')}") for p in pc]) + [""]
    viec = [v for v in d["viec"].values() if v["nguon"] in nguon_viec]
    if viec:
        ra += ["## Cấu hình từng việc huấn luyện", "", "Cấu hình chung (mọi việc): Qwen3-4B revision `1cfa9a72`, bf16, LoRA all-linear, alpha = 2 × rank, dropout 0,05, AdamW, clip 1,0, warmup 3% + cosine, `max_len` 2.048, seed 42.", ""]
        ra += bang(["Việc", "Nhánh", "Fold kiểm", "lr", "rank", "Tham số LoRA", "Epoch xong", "Tổng bước", "Mẫu train", "Token train", "% title thực", "% title nếu w=1", "w_title", "Token/giây"],
                   [(v["ten"], v["nhanh"], f(v["fold_huan_luyen_kiem"]), f(v["lr"], 5), f(v["rank"]), f(v["so_tham_so_lora"]), f(v["epoch_xong"]), f(v["tong_buoc"]), f(v["so_train"]),
                     f(v["so_token_train"]), pt(v["p_title_thuc"]), pt(v["p_title_khi_w_1"]), f(v["w_title"], 5), tg(v["token_moi_giay"])) for v in viec]) + [""]
        ra += ["## Kết quả từng việc (fold kiểm, 1.000 mẫu sinh greedy)", "",
               "`chrF chung` = trung bình đều hai hướng (zh→vi chrF++, vi→zh chrF ký tự), mỗi hướng = trung bình đều title và mô tả. `Dòng mới` = dòng mô tả chưa từng xuất hiện nguyên dòng trong dữ liệu train. Ô `–` là việc chỉ chạy tới bước so loss hoặc không có fold giữ lại.", ""]
        dong = []
        for v in viec:
            k = v["ket_qua_sinh"] or {}
            dl = v.get("diem_loss_theo_epoch") or {}
            dong.append((v["ten"], f(dl.get("1"), 4), f(dl.get("2"), 4), f(k.get("chrf_chung_corpus")), f(k.get("chrf_chung_theo_cau")), f(k.get("title")), f(k.get("mo_ta")),
                         f(k.get("zh_vi")), f(k.get("vi_zh")), f"{nguyen(k.get('moi_zh_vi_n'))} / {f(k.get('moi_zh_vi'))}", f"{nguyen(k.get('moi_vi_zh_n'))} / {f(k.get('moi_vi_zh'))}", f(k.get("sinh_lap"))))
        ra += bang(["Việc", "Loss kiểm ep1", "Loss kiểm ep2", "chrF chung (corpus)", "chrF chung (câu)", "Title", "Mô tả", "zh→vi", "vi→zh", "Dòng mới zh→vi (n / chrF)", "Dòng mới vi→zh (n / chrF)", "Sinh lặp / 1.000"], dong) + [""]
        gr = []
        for v in viec:
            g = v.get("clip_grad") or {}
            if g:
                gr.append((v["ten"], " / ".join(f"{lay(g, k, 'so_bi_cat')}/{lay(g, k, 'so_buoc')}" for k in ("title", "tron", "mo_ta"))))
        if gr:
            ra += ["Số bước bị cắt gradient (clip 1,0) / số bước, theo loại lô (toàn title / trộn / toàn mô tả):", ""] + bang(["Việc", "Title / trộn / mô tả"], gr) + [""]
    sc = [s for s in d["so_sanh_theo_cap"] if any(s["nhom"].startswith(n) for n in nhom_so_cap)]
    if sc:
        ra += ["## So sánh theo cặp (bootstrap, khoảng 95%)", "", "Chênh = mức đầu − mức sau như ghi ở cột 2. Khoảng chứa 0 là hòa. Khoảng chỉ tính dao động do chọn mẫu, không tính dao động khi huấn luyện lại.", ""]
        ra += bang(["Nhóm", "Cặp / chỉ số", "Chênh [95%]"], [(s["nhom"], s["ten"], ci(s["so"])) for s in sc]) + [""]
    ten = {v["ten"] for v in viec}
    gc = [g for g in d["ghi_chu"] if g.split(":")[0].split(" (")[0] in ten]
    if gc:
        ra += ["## Ghi chú tự động từ nhật ký", ""] + [f"- {g}" for g in gc] + [""]
    return ra


def r_3fold():
    t = nk()["tieu_chi_3_fold"]
    ra = ["## Luật 3 fold của bản cụm (khóa trước khi chạy)", ""]
    ra += bang(["Mục", "Giá trị"], [("Chênh chrF chung (corpus), ~11% − 1 : 1, fold 0 / 1 / 2", ", ".join(f"{x:+.3f}" for x in t["chenh_chrf_chung"])), ("Trung bình", f(t["trung_binh"], 3)),
                                     ("Độ lệch chuẩn mẫu", f(t["do_lech_chuan"], 3)), ("Không học lệch (title, mô tả, zh→vi không tụt quá 1,0)", f(lay(t, "khong_hoc_lech_tb", "dat"))),
                                     ("Kết luận đổi tỉ lệ", f(lay(t, "ket_luan_kfold", "doi_ti_le"))), ("Lý do", "; ".join(lay(t, "ket_luan_kfold", "ly_do") or []))])
    return ra + [""]


def r_3fold_pair():
    ra = ["## Luật 3 fold của bản pair (khóa trước ở WALKTHROUGH Prompt 804, `kiem_3fold_pair.py`)", ""]
    md = FT / "kaggle_tl3/cham/kiem_3fold_pair.md"
    if md.exists():
        ra += ["Chép nguyên kết luận do `kiem_3fold_pair.py` sinh ra:", ""] + [x for x in md.read_text(encoding="utf-8").splitlines() if not x.startswith("# ")] + [""]
    return ra


def r_do_test():
    d = doc("ft_run/kaggle_do_test/phien_1_20261107_080604/tong_hop/tong_hop_do_test.json")
    ra = ["## Kết quả: fine-tune (adapter fold 0, ~11% title) so với zero-shot trên TEST KHÓA", "", "chrF theo mục, bootstrap theo sản phẩm (seed 42). Luật: thắng khi cận dưới khoảng 95% của chênh > 0.", ""]
    dong, dl, dk = [], [], []
    for nh in ("cum", "pair"):
        for h, v in d[nh]["huong"].items():
            c = v["chung"]
            dong.append((nh, h, f(c["n_sp"]), f(c["n_muc"]), f(c["chrf_zs"]), f(c["chrf_ft"]), ci(c), f(c["thang_zero_shot"])))
            for lo, x in v["loai"].items():
                dl.append((nh, h, lo, f(x["chrf_zs"]), f(x["chrf_ft"]), ci(x)))
            kr_all = v.get("kiem_rieng") or {}
            ref = kr_all.get("ref") or {}
            for kr, x in kr_all.items():
                if kr != "ref":
                    dk.append((nh, h, kr, f(x.get("zero_shot"), 3), f(x.get("fine_tune"), 3), f(ref.get(f"ref_{kr}"), 3)))
    ra += bang(["Nhánh", "Hướng", "Sản phẩm", "Mục", "chrF zero-shot", "chrF fine-tune", "Chênh [95%]", "Thắng (luật khóa)"], dong) + [""]
    ra += ["Theo loại:", ""] + bang(["Nhánh", "Hướng", "Loại", "Zero-shot", "Fine-tune", "Chênh [95%]"], dl) + [""]
    if dk:
        ra += ["Kiểm riêng (tỉ lệ trên số mục; `da_cat_loi_dan` = bản sinh có lời dẫn bị cắt bỏ; cột Tham chiếu = cùng phép kiểm áp lên chính bản dịch tham chiếu):", ""]
        ra += bang(["Nhánh", "Hướng", "Kiểm", "Zero-shot", "Fine-tune", "Tham chiếu"], dk) + [""]
    cb = [x for nh in ("cum", "pair") for x in d[nh].get("canh_bao", [])]
    ra += [f"Cảnh báo theo luật: {'; '.join(cb) if cb else 'không có'}.", ""]
    return ra


def r_cuoi():
    ra = r_4b(["43"], ["Bản cuối trên toàn pool"], [])
    ra += ["## Manifest adapter bản cuối (đã kiểm hash sau khi tải)", ""]
    ra += bang(["File", "Byte", "sha256 đầy đủ"], [(f"`{k}`", f(v["byte"]), f"`{v['sha256']}`") for k, v in (doc("ft_run/ban_cuoi/MANIFEST_CUOI.json") or {}).items()]) + [""]
    return ra


def dong_quota(v):
    rl = v.get("realized") or {}
    # bản cũ (v47/v48) ghi mỗi nhóm dạng {"e_share", "t_share", "n"}; dùng e_share như bảng docs/BANG_KET_QUA_CUOI.md
    ty = lambda k: (rl.get(k) or {}).get("e_share") if isinstance(rl.get(k), dict) else (rl.get(k) or 0)  # noqa: E731
    p_title = (ty("1688_title") or 0) + (ty("tiki_title") or 0) if rl else None
    nh = lambda k: lay(v, "theo_nhom", k, "chrf_corpus")  # noqa: E731
    th = lambda h: lay(v, "theo_huong", h, "chrf_corpus")  # noqa: E731
    t, m = nh("zh->vi/title") or lay(v, "guardrail_zhvi", "zh->vi/title"), nh("zh->vi/mo_ta") or lay(v, "guardrail_zhvi", "zh->vi/mo_ta")
    # KHÔNG dùng trường diem_chinh của báo cáo: ở v55 nó tính theo tỉ trọng token của eval (cụm 8,33%, pair 67,48% title) và đã được ghi là sai
    # trong docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md; tính lại đúng định nghĩa 0,10 × title + 0,90 × mô tả.
    v = dict(v, diem_chinh=(0.1 * t + 0.9 * m) if t is not None and m is not None else None)
    return (v.get("tap"), v.get("config"), f(v.get("seed")), f(v.get("rank")), f(v.get("max_len")), f(v.get("so_buoc") or v.get("so_buoc_uoc")), f(v.get("token_dich") or v.get("token")),
            f(v.get("bo_max_len")), pt(((v.get("s_g") or {}).get("1688_title") or 0) + ((v.get("s_g") or {}).get("tiki_title") or 0) if v.get("s_g") else None), pt(p_title), f(nh("zh->vi/title") or lay(v, "guardrail_zhvi", "zh->vi/title")), f(nh("zh->vi/mo_ta") or lay(v, "guardrail_zhvi", "zh->vi/mo_ta")),
            f(v.get("diem_chinh")), f(lay(v, "chrf", "chrf_macro_huong")), f(lay(v, "chrf_zs", "chrf_macro_huong")), f(th("zh->vi")), f(th("vi->zh")),
            f(v.get("loss"), 4), f(v.get("token_moi_giay"), 0), f(v.get("bo_nho_dinh_gb"), 1))


TD_QUOTA = ["Bộ", "Cấu hình", "Seed", "rank", "max_len", "Bước", "Token đích", "Mẫu bị bỏ vì dài", "% title đặt", "% title thực", "Title zh→vi", "Mô tả zh→vi", "Chính (0,1/0,9)", "chrF macro",
            "Zero-shot macro", "zh→vi", "vi→zh", "Loss cuối", "Token/giây", "VRAM đỉnh (GB)"]


def r_quota(ds_bao_cao, ghi=None):
    ra = ["## Kết quả từng việc", "",
          "chrF corpus trên eval fold 0 (cụm 1.200 mẫu, pair 1.198 mẫu). `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi, **tính lại** từ hai cột title/mô tả (không lấy trường `diem_chinh` của báo cáo v55 vì trường đó dùng sai trọng số); trống khi báo cáo chưa ghi điểm theo nhóm. `% title thực` = tỉ trọng token đích của title trong mẫu đã lấy. Ô `–` là trường không có trong báo cáo của bản đó.", ""]
    for p in ds_bao_cao:
        d = doc(p)
        if not d:
            continue
        loi = [v for v in d.get("viec", []) if v.get("loi")]
        ok = [v for v in d.get("viec", []) if not v.get("loi")]
        mh = lay(d, "mo_hinh", "ten_thu_muc")
        ra += [f"**`{p}`**" + (f" (mô hình {mh}, khuôn `{lay(d, 'mo_hinh', 'khuon')}`, attention `{lay(d, 'mo_hinh', 'attn')}`, gradient checkpointing {f(lay(d, 'mo_hinh', 'gc'))})" if mh else ""), ""]
        if ok:
            ra += bang(TD_QUOTA, [dong_quota(v) for v in ok]) + [""]
        for v in loi:
            dc = [x for x in str(v.get("traceback") or v.get("loi")).strip().splitlines() if x.strip()][-1:]
            ra += [f"- Việc `{v.get('tap')}/{v.get('config')}` **LỖI**: `{(dc or [''])[0][:300]}`"]
        if loi:
            ra += [""]
        kt = d.get("kiem_tung_tap")
        if kt:
            ra += ["Kiểm dữ liệu từng bộ trước khi lấy mẫu:", ""]
            ra += bang(["Bộ", "Trần token", "Mẫu tất cả", "Mẫu giữ", "Mẫu bỏ", "Dài nhất", "Token đích toàn bộ", "Ngân sách token đích"],
                       [(k, f(x.get("cap")), f(x.get("n_tat_ca")), f(x.get("n_giu")), f(x.get("n_bo")), f(x.get("dai_nhat_giu")), f(x.get("token_dich_toan_bo")), f(x.get("ngan_sach_token_dich")))
                        for k, x in kt.items()]) + [""]
    return ra + ((ghi + [""]) if ghi else [])


def r_check1():
    a = next((v for v in (doc("ft_run/kaggle_quota_step1/quota_report.json") or {}).get("viec", []) if v.get("tap") == "cum" and v.get("config") == "natural" and v.get("seed") == 42), {})
    b = next(iter((doc("ft_run/kaggle_quota_check1_acc2/quota_report.json") or {}).get("viec", [])), {})
    ra = r_quota(["ft_run/kaggle_quota_check1_acc2/quota_report.json"])
    ra += ["## Đối chiếu với C4 (cùng cấu hình, cùng seed)", ""]
    ra += bang(["Chỉ số", "C4, tài khoản 1 (v55)", "C5, tài khoản 2", "Chênh"],
               [(ten, f(x), f(y), f(y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else "–") for ten, x, y in [
                   ("Số bước", a.get("so_buoc"), b.get("so_buoc")), ("Token đích", a.get("token_dich"), b.get("token_dich")),
                   ("chrF macro", lay(a, "chrf", "chrf_macro_huong"), lay(b, "chrf", "chrf_macro_huong")),
                   ("Title zh→vi", lay(a, "theo_nhom", "zh->vi/title", "chrf_corpus"), lay(b, "theo_nhom", "zh->vi/title", "chrf_corpus")),
                   ("Mô tả zh→vi", lay(a, "theo_nhom", "zh->vi/mo_ta", "chrf_corpus"), lay(b, "theo_nhom", "zh->vi/mo_ta", "chrf_corpus")),
                   ("Token/giây", a.get("token_moi_giay"), b.get("token_moi_giay"))]])
    return ra + [""]


def r_c0():
    ra = ["## Kết quả: adapter chính thức chấm trên đúng eval của vòng sàng lọc", ""]
    dong = []
    for p in ("ft_run/kaggle_c0/c0_report_v50_cum.json", "ft_run/kaggle_c0/c0_report_v51_pair.json"):
        for v in (doc(p) or {}).get("viec", []):
            nh = lambda k: lay(v, "theo_nhom", k, "chrf_corpus")  # noqa: E731
            dong.append((f"`{p.split('/')[-1]}`", v.get("tap"), v.get("c0"), v.get("kieu_eval"), f(v.get("n_eval")), f(lay(v, "chrf", "chrf_macro_huong")),
                         f(lay(v, "theo_huong", "zh->vi", "chrf_corpus")), f(lay(v, "theo_huong", "vi->zh", "chrf_corpus")), *(f(nh(k)) for k in NHOM4), f(v.get("giay"), 1), f"`{v.get('adapter')}`"))
    ra += bang(["File", "Bộ", "Adapter", "Kiểu eval", "n", "chrF macro", "zh→vi", "vi→zh", "zh→vi title", "zh→vi mô tả", "vi→zh title", "vi→zh mô tả", "Giây", "Đường dẫn adapter"], dong)
    return ra + ["", "`c0_report.json` trong cùng thư mục là bản sao của `c0_report_v51_pair.json` (cùng sha256).", ""]


def r_5mh(ten_mh, tieu_de="## Điểm trong bảng năm mô hình (tổng hợp trên máy, `tong_hop_5mh.py`)"):
    ra = [tieu_de, "",
          "Eval fold 0. Pair: 1.198 mẫu (từng dòng); cụm: 1.200 mẫu. chrF corpus; `Chính` = 0,10 × title + 0,90 × mô tả hướng zh→vi; `Macro` = trung bình đều hai hướng. "
          "Chênh so với Qwen3-4B (vòng sàng lọc) tính trên chrF câu trung bình, bootstrap theo sản phẩm (2.000 lần, cùng mẫu rút cho mọi mô hình). LLM là trung bình 2 seed.", ""]
    dong, dong_b = [], []
    for tap in ("pair", "cum"):
        kq = doc("ft_run/tong_hop_5mh/ket_qua.json" if tap == "pair" else "ft_run/tong_hop_5mh_cum/ket_qua.json") or {}
        for ten in ten_mh:
            m = kq.get(ten)
            if not m:
                continue
            cl = list((m.get("cac_lan") or {}).values())
            tb = lambda k: sum(lay(x, "chrf_corpus", k) or 0 for x in cl) / max(1, len(cl))  # noqa: E731
            ch = lay(m, "chenh_so_voi_4B", "chinh")
            lap = [sum(x["ti_le_sinh_lap"].values()) / len(x["ti_le_sinh_lap"]) for x in cl if x.get("ti_le_sinh_lap")]
            dong.append((tap, ten, f(len(cl)), f(tb("zh->vi/title")), f(tb("zh->vi/mo_ta")), f(tb("vi->zh/title")), f(tb("vi->zh/mo_ta")), f(m.get("tb_chinh")), f(m.get("tb_macro")),
                         f(lay(m, "chrf_cau_tb", "chinh")), ci(ch) if ch else "mốc", ci(lay(m, "chenh_so_voi_4B", "macro")) if ch else "mốc", pt(sum(lap) / len(lap) if lap else None), f(m.get("lech_chinh_giua_lan"))))
            bt = m.get("bleu_ter") or {}
            if bt:
                dong_b.append((tap, ten, *(f"{f(lay(bt, k, 'sacrebleu'))} / {f(lay(bt, k, 'ter'))}" for k in NHOM4)))
    ra += bang(["Bộ", "Mô hình", "Số lần chạy", "zh→vi title", "zh→vi mô tả", "vi→zh title", "vi→zh mô tả", "Chính (corpus)", "Macro (corpus)", "Chính (câu TB)",
                "Chênh chính (câu) so với 4B [95%]", "Chênh macro (câu) so với 4B [95%]", "Sinh lặp (TB 4 nhóm)", "Lệch chính giữa 2 seed"], dong)
    if dong_b:
        ra += ["", "SacreBLEU / TER (trung bình các lần chạy; sacrebleu 2.6.0, zh→vi `tok:13a`, vi→zh `tok:zh`, TER `tercom`; BLEU kiểu mteval-v13a trùng SacreBLEU):", ""]
        ra += bang(["Bộ", "Mô hình", "zh→vi title", "zh→vi mô tả", "vi→zh title", "vi→zh mô tả"], dong_b)
    return ra + [""]


def r_onmt():
    d = doc("ft_run/kaggle_onmt_that/onmt_report.json") or {}
    ra = ["## Huấn luyện (`kaggle_onmt_that/onmt_report.json`)", "", f"Cấu hình chung: {jtxt(d.get('cau_hinh'))}. Tổng thời gian phiên: {gio(d.get('giay_tong'))}.", ""]
    ra += bang(["Hướng", "Biến thể", "Dòng train", "Token SP train", "Bước", "Bước/epoch", "Warmup", "lr đỉnh", "learning_rate Noam", "Checkpoint trung bình", "Bước đã chạy", "Giây train", "n eval"],
               [(v.get("huong"), v.get("bien_the"), f(v.get("dong_train")), f(v.get("token_sp_train")), f(v.get("steps")), f(v.get("steps_epoch")), f(v.get("warmup")), f(v.get("lr_dinh"), 4),
                 f(v.get("learning_rate_noam"), 4), f(v.get("so_checkpoint")), f(v.get("buoc_da_chay")), f(v.get("giay_train"), 0), f(v.get("n_eval"))) for v in d.get("viec", [])]) + [""]
    tt = (doc("ft_run/kaggle_onmt_that/cham_onmt.json") or {}).get("tom_tat") or {}
    if tt:
        ra += ["Chấm ngay trên Kaggle, bản pair (`cham_onmt.json`, chrF corpus):", ""]
        ra += bang(["zh→vi title", "zh→vi mô tả", "Chính (0,1/0,9)", "vi→zh title", "vi→zh mô tả", "Macro"],
                   [tuple(f(tt.get(k)) for k in ("zh->vi/title", "zh->vi/mo_ta", "chinh_w0.10", "vi->zh/title", "vi->zh/mo_ta", "macro"))]) + [""]
    cu = doc("ft_run/kaggle_onmt_cum/onmt_cum_report.json") or {}
    if cu:
        ra += ["## Dịch eval bản cụm (`kaggle_onmt_cum/onmt_cum_report.json`)", "", "Không huấn luyện thêm: dùng checkpoint trung bình ở trên, dịch từng mảnh thuộc tính rồi nối bằng \"; \".", ""]
        ra += bang(["Hướng", "Số mảnh", "Số mẫu", "Checkpoint"], [(v.get("huong"), f(v.get("n_manh")), f(v.get("n_mau")), f"`{v.get('checkpoint')}`") for v in cu.get("viec", [])])
        ra += ["", f"Thời gian phiên cụm: {gio(cu.get('giay_tong'))}.", ""]
    return ra


def r_moses(ds):
    ra = ["## Huấn luyện (`moses_report.json`)", ""]
    for p in ds:
        d = doc(p) or {}
        ra += [f"**`{p}`**: cấu hình {jtxt(d.get('cau_hinh'))}, tổng {gio(d.get('giay_tong'))}.", ""]
        ra += bang(["Hướng", "Tách chữ Hán", "Biến thể", "Dòng train", "Dòng sau lọc", "MERT", "chrF dev", "n eval", "Bản rỗng", "Mảnh cụm (n mẫu)", "Giây"],
                   [(v.get("huong"), v.get("seg"), v.get("bien_the", "train"), f(v.get("dong_train")), f(v.get("dong_sau_loc")), f(v.get("mert")), f(v.get("dev_chrf")), f(v.get("n_eval")),
                     f(v.get("so_hyp_rong")), f"{f(v.get('n_manh_cum'))} ({f(v.get('n_eval_cum'))})" if v.get("n_manh_cum") else "–", f(v.get("giay"), 0)) for v in d.get("viec", [])]) + [""]
    return ra


def r_cheo_train():
    ra = ["## Kết quả huấn luyện lại hai adapter fold 0 (tài khoản 2)", ""]
    dong = []
    for p, k in (("ft_run/kaggle_cheo_cum/bao_cao_cheo_cum.json", "cum_f0_p11"), ("ft_run/kaggle_cheo_pair/bao_cao_cheo_pair.json", "pair_f0_p11")):
        v = lay(doc(p), "b78", "viec", k) or {}
        c = v.get("cham") or {}
        dl = v.get("diem") or {}
        dong.append((k, v.get("nhanh"), f(v.get("fold")), v.get("cau_hinh"), f(v.get("lr"), 5), f(v.get("epoch_xong")), f(v.get("tong_buoc")), f(v.get("so_train")), f(v.get("so_token_train")),
                     pt(v.get("p_thuc")), f(v.get("w_title"), 5), f(dl.get("1"), 4), f(dl.get("2"), 4), f(lay(c, "chrf", "chrf_macro_huong")), f(lay(c, "chrf", "chrf_cau_tb_macro_huong"))))
    ra += bang(["Adapter", "Nhánh", "Fold kiểm", "Cấu hình", "lr", "Epoch", "Bước", "Mẫu train", "Token train", "% title thực", "w_title", "Loss kiểm ep1", "Loss kiểm ep2", "chrF chung (corpus)", "chrF chung (câu)"], dong)
    sl = [(k, lay(nk(), "viec", k, "ket_qua_sinh", "chrf_chung_corpus"), lay(nk(), "viec", k, "diem_loss_theo_epoch", "2")) for k in ("cum_f0_p11", "pair_f0_p11")]
    ra += ["", "Đối chiếu với lần huấn luyện cùng cấu hình ở tài khoản 1 (bản 29, `NHAT_KY_CHAY.json`): " + "; ".join(f"`{k}` chrF chung {f(x)}, loss kiểm ep2 {f(l, 4)}" for k, x, l in sl) + ".", ""]
    t = doc("ft_run/kaggle_cheo_that/cheo/cheo_trang_thai.json") or {}
    if t.get("adapter"):
        ra += ["sha256 adapter tài khoản 2 (ghi trong trạng thái phiên thử chéo E2):", ""] + bang(["Adapter", "Đường dẫn trên Kaggle", "sha256"], [(k, f"`{v['duong_dan']}`", f"`{v['sha256']}`") for k, v in t["adapter"].items()]) + [""]
    return ra


def r_cheo_that():
    t = doc("ft_run/kaggle_cheo_that/cheo/cheo_trang_thai.json") or {}
    m = t.get("mau") or {}
    ra = ["## Mẫu và lượt sinh (`cheo_trang_thai.json`)", ""]
    ra += bang(["Mục", "Giá trị"], [("Ứng viên (sản phẩm zh→vi fold 0, mô tả nguyên, nguồn hai nhánh giống hệt)", f(m.get("so_ung_vien"))), ("Số sản phẩm chọn", f(m.get("so_chon"))),
                                     ("Trong đó thuộc mẫu cố định của các lần huấn luyện", f(m.get("so_co_dinh_trong_chon"))), ("Bỏ vì nguồn hai nhánh khác nhau", f(lay(m, "bo", "nguon_hai_nhanh_khac_nhau"))),
                                     ("Seed chọn mẫu", f(lay(t, "cau_hinh", "seed")))]) + [""]
    ra += bang(["Điều kiện (adapter / dạng đầu vào)", "Số mẫu", "Giây mỗi mẫu"], [(k, f(v.get("so_mau")), f(v.get("giay_moi_mau"), 3)) for k, v in (t.get("da_sinh") or {}).items()]) + [""]
    md = FT / "cheo_ket_qua" / "bang.md"
    if md.exists():
        ra += ["## Kết quả chấm trên máy (`cham_cheo.py` → `ft_run/cheo_ket_qua/bang.md`, chép nguyên)", ""]
        ra += [("#" + x) if x.startswith("## ") else x for x in md.read_text(encoding="utf-8").splitlines() if not x.startswith("# ")] + [""]
    return ra


def r_dong6():
    d = doc("ft_run/kaggle_dong6/bao_cao_dong6.json") or {}
    ra = ["## Sinh và chấm trên Kaggle (`bao_cao_dong6.json`)", "", f"Cấu hình: {jtxt(d.get('cau_hinh'))}. Môi trường: {jtxt(d.get('moi_truong'))}. Đạt: {f(d.get('dat'))}. Tổng {gio(d.get('tong_giay'))}.", ""]
    dong = []
    for v in d.get("viec", []):
        nh = lambda k: lay(v, "theo_nhom", k, "chrf_corpus")  # noqa: E731
        dong.append((v.get("tap"), v.get("adapter"), f(v.get("n_hang_eval")), f(v.get("n_luot_goi")), f(v.get("so_khoi_lech_dong")), pt(v.get("ti_le_rong")), f(lay(v, "chrf", "chrf_macro_huong")),
                     f(lay(v, "chrf", "chrf_cau_tb_macro_huong")), *(f(nh(k)) for k in NHOM4), f(v.get("giay_nap"), 0), f(v.get("giay_moi_luot"), 3)))
    ra += bang(["Bộ", "Adapter", "Hàng eval", "Lượt gọi", "Khối lệch dòng", "Rỗng", "chrF macro (corpus)", "chrF macro (câu)", "zh→vi title", "zh→vi mô tả", "vi→zh title", "vi→zh mô tả", "Giây nạp", "Giây/lượt"], dong) + [""]
    ra += ["Đường dẫn adapter trên Kaggle: " + "; ".join(f"`{v.get('duong_dan')}`" for v in d.get("viec", [])) + " (bản huấn luyện lại ở tài khoản 2, phiên E1).", ""]
    return ra + r_5mh(["Qwen3-4B", "Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16)"], "## Đặt cạnh Qwen3-4B vòng sàng lọc (`tong_hop_5mh.py`)")


def r_demo_gate():
    ra = ["## Cổng nén trên laptop (`demo/do_luong_tu.py`, RTX 3050 4 GB)", "",
          "So bản nén GGUF + LoRA trên máy với bản fp16 trên Kaggle T4, cùng 150 sản phẩm fold 0, hướng zh→vi, adapter `pair_f0_p11` (tài khoản 2). Ngưỡng đặt trước khi đo: chrF tụt ≤ 1,0; khối lệch dòng ≤ 2%; rỗng ≈ 0; title ≤ 3 giây; khối 20 dòng ≤ 20 giây.", ""]
    dong = []
    for ten in ("q4km-lora-pair-f0", "q5km-lora-pair-f0"):
        d = doc(f"ft_run/demo_do_luong_tu/ket_qua_{ten}.json")
        if not d:
            continue
        a, b, td = d["may_4bit"], d["kaggle_fp16"], d["toc_do"]
        ca, cb = lay(a, "chrf_diem_chinh", "chrf_cau_tb_macro_huong"), lay(b, "chrf_diem_chinh", "chrf_cau_tb_macro_huong")
        dong.append((ten, f(ca), f(cb), f(ca - cb), f(lay(a, "chrf_diem_chinh", "chrf_macro_huong")), f(lay(b, "chrf_diem_chinh", "chrf_macro_huong")), f"{a['so_khoi_lech_dong']}/{a['so_khoi']}",
                     pt(a["ti_le_rong"]), pt(d.get("ti_le_trung_khop_voi_fp16"), 1), f(td.get("token_moi_giay_trung_vi"), 1), f(td.get("giay_title_trung_vi")), f(td.get("giay_title_p90")),
                     f(td.get("giay_khoi_trung_vi")), f(td.get("giay_khoi_p90")), f(td.get("so_dong_khoi_trung_vi"), 0), f(td.get("het_tran"))))
    ra += bang(["Bản nén", "chrF câu TB máy", "chrF câu TB fp16", "Chênh", "chrF corpus máy", "chrF corpus fp16", "Khối lệch dòng", "Rỗng", "Trùng khớp fp16", "Token/giây",
                "Giây title (trung vị)", "Giây title (p90)", "Giây khối (trung vị)", "Giây khối (p90)", "Dòng/khối (trung vị)", "Hết trần sinh"], dong)
    return ra + [""]


def r_demo_kaggle():
    ra = ["## Output tải về", ""]
    p = FT / "demo_bo_dem_kaggle" / "may_chu" / "bo_dem.sqlite"
    if p.exists():
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            dem = {b: con.execute(f"select count(*) from {b}").fetchone()[0] for (b,) in con.execute("select name from sqlite_master where type='table'")}
        finally:
            con.close()
        ra += [f"`may_chu/bo_dem.sqlite`: số dòng theo bảng {jtxt(dem)}. Đây là bộ đệm bản dịch của 13 trang thử (13 title + 22 khối thuộc tính + 2 mục tự kiểm), chép sang `demo/trang_dich_san/` để demo không cần GPU.", ""]
    ra += ["Thư mục còn bản chép mã máy chủ (`may_chu/*.py`) đúng như lúc chạy.", ""]
    return ra


def r_8b_loi():
    ra = ["## Hai lần chạy thử, cả hai bị cấp GPU nhỏ", ""]
    dong = []
    for p in ("ft_run/kaggle_cuoi8b_khoi/bao_cao_thu_cuoi8b.json", "ft_run/kaggle_cuoi8b_khoi2/bao_cao_thu_cuoi8b.json"):
        d = doc(p) or {}
        loi = [x for x in str(d.get("loi") or "").strip().splitlines() if x.strip()][-1:]
        dong.append((f"`{p}`", lay(d, "moi_truong", "gpu"), f(lay(d, "moi_truong", "vram_gb"), 1), f(d.get("dat")), gio(d.get("tong_giay")), f"`{(loi or ['–'])[0][:240]}`"))
    return ra + bang(["Báo cáo", "GPU được cấp", "VRAM (GB)", "Đạt", "Thời lượng", "Dòng lỗi cuối"], dong) + [""]


def r_song_ngu(the):
    d = doc(the) or {}
    ra = [f"## Thẻ nguồn gốc `{the}`", "", f"Bước: `{d.get('buoc')}`, ngày chạy {d.get('ngay_chay')}. Mô hình theo thẻ: {jtxt(d.get('mo_hinh'))}; ngưỡng chọn: {jtxt(d.get('nguong'), 500)}.", ""]
    cb = d.get("chuan_bi") or {}
    if cb:
        ra += ["Chuẩn bị (bước 1–3):", ""] + bang(["Trường", "Giá trị"], [(k, jtxt(v, 300) if isinstance(v, (dict, list)) else f(v)) for k, v in cb.items() if k != "giay_to_kem"]) + [""]
    dong = []
    for k, v in (d.get("kaggle") or {}).items():
        if isinstance(v, dict) and v:
            dong.append((k, v.get("mo_hinh") or v.get("mo_hinh_cham") or "–", v.get("repo") or v.get("ban_dich_cua") or "–", v.get("backend") or (f"{v.get('gpu')} GPU" if v.get("gpu") else "–"),
                         v.get("bat_dau") or v.get("ngay") or "–", v.get("ket_thuc") or "–", f(v.get("tong") or v.get("can_cham")), gio(v.get("giay"))))
    if dong:
        ra += ["Các lượt dịch / chấm (thẻ `kaggle` trong file; `api_tuy_chon` = Gemma gọi qua API, `qwen3_14b` = Qwen3-14B-AWQ chạy vLLM trên Kaggle, `kiwi` = CometKiwi chấm trên Kaggle):", ""]
        ra += bang(["Thẻ", "Mô hình", "Repo / bản dịch của", "Backend / GPU", "Bắt đầu", "Kết thúc", "Số đơn vị", "Thời lượng"], dong) + [""]
    for k in ("don_vi", "cap"):
        v = d.get(k)
        if isinstance(v, dict):
            ra += [f"Kết quả chọn theo `{k}`:", ""] + bang(["Trường", "Giá trị"], [(kk, jtxt(vv) if isinstance(vv, dict) else f(vv)) for kk, vv in v.items()]) + [""]
    if d.get("file"):
        ra += [f"File ra: {jtxt(d['file'])}", ""]
    return ra


def r_ft_clean():
    d = doc("ft_clean/pheu_buoc0_4.json") or {}
    ra = ["## Phễu lọc (`ft_clean/pheu_buoc0_4.json`)", "", f"Vào: **{f(d.get('vao'))}** sản phẩm (`{lay(d, 'tieu_chi', 'nguon')}`, sha256 `{str(lay(d, 'tieu_chi', 'sha256_nguon'))[:16]}…`). Mỗi dòng chỉ bị loại theo lý do đầu tiên nó dính; không sửa nội dung.", ""]
    ra += bang(["Thứ tự", "Lý do loại", "Số loại", "Theo nguồn"], [(i + 1, b["ly_do"], f(b["loai"]), jtxt(b.get("theo_nguon"))) for i, b in enumerate(d.get("theo_buoc", []))])
    ra += [""]
    ra += bang(["Mục", "Giá trị"], [("Còn lại", f(d.get("con_lai"))), ("Theo nguồn", jtxt(d.get("con_lai_theo_nguon"))), ("Theo hướng", jtxt(d.get("con_lai_theo_huong"))),
                                     ("Theo nhóm hàng", jtxt(d.get("con_lai_theo_nhom_hang"))), ("Căn hàng chắc (số đoạn và số khớp)", f"{f(lay(d, 'can_hang_chac_trong_con_lai', 'so_dong'))} ({pt(lay(d, 'can_hang_chac_trong_con_lai', 'ti_le'), 1)})"),
                                     ("Cờ mềm (chỉ ghi, không loại)", jtxt(d.get("co_mem_trong_con_lai"))), ("Nhóm mô tả trùng", jtxt(d.get("nhom_mo_ta_trung_trong_con_lai"))),
                                     ("Trần mỗi khuôn thuộc tính", f(lay(d, "tieu_chi", "tran_khuon")))])
    return ra + [""]


def r_judge():
    ra = ["## 1. Thí điểm trên máy (29–31/10/2026, `data/processed/judge/`, `data/processed/so_cap/`)", "",
          "Mỗi dòng là một lần chạy thử trên mẫu nhỏ để chọn giao thức (một hay hai lượt hoán vị, có ngữ cảnh hay không, quy tắc v1 hay v2, gom nhiều mục mỗi lời gọi). Số đọc từ `tom_tat_*.json`.", ""]
    dong = []
    for p in sorted((DP / "judge").glob("tom_tat_*.json")) + sorted((DP / "so_cap").glob("tom_tat_*.json"), key=lambda x: x.stat().st_mtime):
        d = json.loads(p.read_text(encoding="utf-8"))
        dong.append((f"`{rel(p)}`", d.get("ten"), d.get("mo_hinh"), d.get("phien_ban", "–"), d.get("che_do", "–"), f(d.get("ok_lan_nay")), f(d.get("loi_lan_nay")),
                     f(int(d["tokens_vao"])) if d.get("tokens_vao") is not None else "–", f(int(d["tokens_ra"])) if d.get("tokens_ra") is not None else "–", f(d.get("giay"))))
    ra += bang(["File", "Lần chạy", "Mô hình chấm", "Phiên bản lời nhắc", "Chế độ", "Mục xong", "Lỗi", "Token vào", "Token ra", "Giây"], dong) + [""]
    ra += ["Kết quả so cặp mù hai thứ tự trên mẫu 100 mục mỗi loại (`bao_cao_*_100.json`, quy tắc v1): `gemma`/`qwen` = cả hai lượt cùng chọn; `khong_nhat_quan` = hai lượt chọn khác nhau.", ""]
    dong = []
    for loai in ("title", "cum", "pair"):
        b = doc(f"so_cap/bao_cao_{loai}_100.json") or {}
        k = b.get("ket_qua") or {}
        dong.append((loai, f(b.get("so_muc_ok")), f(k.get("gemma")), f(k.get("qwen")), f(k.get("hoa_that")), f(k.get("khong_nhat_quan")), pt(lay(b, "thien_vi_vi_tri", "ti_le_chon_vi_tri_1"), 1),
                     pt(b.get("ti_le_gemma_trong_so_co_nguoi_thang"), 1), pt(lay(b, "thien_vi_do_dai", "ti_le_ban_dai_hon_thang"), 1), pt(lay(b, "trung_voi_cometkiwi", "tong"), 1)))
    ra += bang(["Loại", "Mục", "Gemma", "Qwen", "Hòa thật", "Không nhất quán", "Chọn vị trí 1", "Gemma trong số có người thắng", "Bản dài hơn thắng", "Trùng CometKiwi"], dong) + [""]
    g = lay(doc("so_cap/so_sanh_gom.json"), "tat_ca") or {}
    if g:
        ra += ["Thử gom 15 mục mỗi lời gọi (`so_sanh_gom.json`, 450 mục, quy tắc v2) so với một mục mỗi lời gọi:", ""]
        ra += bang(["Chỉ số", "Một mục / lời gọi", "Gom"], [("Token mỗi mục", f(lay(g, "token_moi_muc", "mot_muc_mot_loi_goi")), f(lay(g, "token_moi_muc", "gom"))),
                                                          ("Gemma / Qwen", " / ".join(map(str, lay(g, "gemma_qwen", "mot_muc_mot_loi_goi") or [])), " / ".join(map(str, lay(g, "gemma_qwen", "gom") or []))),
                                                          ("Chọn vị trí 1", pt(lay(g, "chon_vi_tri_1", "mot_muc_mot_loi_goi"), 1), pt(lay(g, "chon_vi_tri_1", "gom"), 1)),
                                                          ("Khai tương đương", pt(lay(g, "ti_le_khai_tuong_duong", "mot_muc_mot_loi_goi"), 1), pt(lay(g, "ti_le_khai_tuong_duong", "gom"), 1)),
                                                          ("Trùng kết quả trên mục bền vững hai thứ tự", "–", f"{f(g.get('gom_trung_ket_qua_ben_vung'))}/{f(g.get('muc_ben_vung_hai_thu_tu'))} ({pt(g.get('ti_le_gom_trung_ben_vung'), 1)})")]) + [""]
    import pandas as pd
    ra += ["## 2. Kết quả trên bản phát hành (HF `ntquang0410/zh-vie_ecom`, `data/LLM/bilingual_judged`, commit `82aa9c9`)", "",
           "Đọc từ bốn file đã tải về `data/processed/bo_chon_hf/` (sha256 khớp `manifest.json`). Theo README của bản phát hành: DeepSeek chấm **4.439/4.446** mục cụm (7 mục bị nhà cung cấp chặn được rà riêng: 2 chọn Gemma, 4 tương đương, 1 chưa chắc); bản pair chấm **5.000/34.534** đơn vị tranh chấp dùng nhiều nhất (khoảng 84% mảnh), đơn vị chưa chấm giữ bản trước.", ""]
    dong, dong_dd = [], []
    for bo in ("cum", "pair"):
        d = pd.read_parquet(DP / f"bo_chon_hf/bo_{bo}_chon_deepseek.parquet", columns=["source", "judge_chon_title", "judge_chon_description"])
        tong = {}
        for s in d["judge_chon_description"].dropna():
            for k, v in re.findall(r"(\w+): (\d+)", s):
                tong[k] = tong.get(k, 0) + int(v)
        tc = d["judge_chon_title"].value_counts().to_dict()
        dong.append((bo, f(len(d)), jtxt(d["source"].value_counts().to_dict()), f(int(tc.get("gemma", 0))), f(int(tc.get("qwen", 0))),
                     f(tong.get("gemma")), f(tong.get("qwen")), f(tong.get("quy_tac")), f(tong.get("chua_ro"))))
        dd = pd.read_parquet(DP / f"bo_chon_hf/bo_{bo}_danh_dau.parquet")
        so = lambda c: int(dd[c].sum()) if c in dd else None  # noqa: E731
        dong_dd.append((bo, jtxt(dd["title_nhan"].value_counts().to_dict()), f(int(dd["title_da_doi"].sum())), f(so("mo_ta_so_manh_tranh_chap")), f(so("mo_ta_so_manh_judge_t1")),
                        f(so("mo_ta_so_manh_judge_t2")), f(so("mo_ta_so_manh_judge_tuong_duong")), f(so("mo_ta_so_manh_khong_nhat_quan")), f(so("mo_ta_so_manh_chua_cham")),
                        f(so("mo_ta_so_manh_da_doi")), f(so("mo_ta_so_manh_gemma_duoc_ap_dung")), f(so("mo_ta_so_manh_qwen_duoc_ap_dung"))))
    ra += ["Nguồn của nội dung cuối (`bo_*_chon_deepseek.parquet`). `judge_chon_title` là mô hình tạo bản title được giữ, không nhất thiết là lựa chọn trực tiếp của DeepSeek; `quy_tac` = mảnh khóa bằng từ điển/luật; `chua_ro` = chưa chấm hoặc không quyết, giữ hiện trạng. Mảnh cụm là một cặp tên–giá trị, mảnh pair là tên hoặc giá trị riêng, nên không so số đếm giữa hai bộ.", ""]
    ra += bang(["Bộ", "Sản phẩm", "Theo nguồn", "Title Gemma", "Title Qwen", "Mảnh mô tả Gemma", "Mảnh mô tả Qwen", "Mảnh quy tắc", "Mảnh chưa rõ"], dong) + [""]
    ra += ["Trạng thái Judge theo file đánh dấu (`bo_*_danh_dau.parquet`, cộng trên mọi sản phẩm):", ""]
    ra += bang(["Bộ", "Nhãn title", "Title đã đổi", "Mảnh tranh chấp", "Judge tầng 1", "Judge tầng 2", "Khai tương đương", "Không nhất quán", "Chưa chấm", "Mảnh đã đổi", "Áp Gemma", "Áp Qwen"], dong_dd)
    return ra + [""]


def r_chia_tap():
    dd = doc("ft_run/DU_LIEU_DONG_BANG.json") or {}
    sp = dd.get("split") or {}
    ra = ["## 1. Đóng băng dữ liệu và phép chia (`ft_run/DU_LIEU_DONG_BANG.json`)", ""]
    ra += bang(["Mục", "Giá trị"], [("Nguồn", f"HF `{lay(dd, 'nguon_du_lieu', 'hf_repo')}`, `{lay(dd, 'nguon_du_lieu', 'thu_muc')}`, commit `{lay(dd, 'nguon_du_lieu', 'commit')}`"),
                                     ("Phiên bản phép chia", sp.get("phien_ban")), ("Seed", str(sp.get("seed"))), ("Tỉ lệ test", pt(sp.get("ti_le_test"), 0)), ("Số fold của pool", f(sp.get("so_fold"))),
                                     ("Test khóa / pool", f"{f(sp.get('n_test'))} / {f(sp.get('n_pool'))} sản phẩm"), ("Ngưỡng Jaccard gom nhóm gần-trùng", jtxt(sp.get("nguong_jaccard"))),
                                     ("sha256 phép chia / danh sách test", f"`{str(sp.get('sha256_file'))[:16]}…` / `{str(sp.get('sha256_ds_test'))[:16]}…`"), ("Quy tắc", dd.get("quy_tac"))]) + [""]
    c = doc("ft_run/bao_cao_chia_tap.json") or {}
    ra += ["## 2. Kết quả chia (`ft_run/bao_cao_chia_tap.json`)", ""]
    ra += bang(["Mục", "Giá trị"], [("Số cặp gần-trùng theo góc nhìn", jtxt(c.get("so_cap_theo_goc_nhin"))), ("Số nhóm / nhóm từ 2 sản phẩm / nhóm lớn nhất", f"{f(c.get('so_nhom'))} / {f(c.get('so_nhom_tu_2_sp'))} / {f(c.get('nhom_lon_nhat'))}"),
                                     ("Test theo hướng", jtxt(c.get("test_theo_huong"))), ("Số sản phẩm mỗi fold", jtxt(c.get("fold_co"))), ("Tỉ lệ sản phẩm thuộc nhóm từ 2", jtxt(c.get("ti_le_sp_thuoc_nhom_tu_2")))]) + [""]
    ra += ["Phân tầng theo hướng × nhóm hàng:", ""] + bang(["Hướng", "Nhóm hàng", "Tổng", "Test", "Tỉ lệ test", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"],
                                                           [(b["huong"], b["category"], f(b["tong"]), f(b["test"]), pt(b["ti_le_test"], 1), *(f(b[f"fold{i}"]) for i in range(5))) for b in c.get("bang_tang", [])]) + [""]
    r = doc("ft_run/bao_cao_ro_ri.json") or {}
    ra += ["## 3. Kiểm rò rỉ (`ft_run/bao_cao_ro_ri.json`)", "", f"Kết luận trong file: **{r.get('ket_luan')}**. Title trùng chính xác giữa test và pool: {jtxt(r.get('trung_title_chinh_xac'))}.", ""]
    ra += bang(["So", "Góc nhìn", "Ngưỡng", "Số truy vấn", "Số vi phạm"], [("test với pool", x.get("goc_nhin"), f(x.get("nguong")), f(x.get("so_truy_van")), f(x.get("so_vi_pham"))) for x in r.get("test_vs_pool", [])])
    fv = r.get("fold_vs_cac_fold_khac", [])
    ra += ["", f"Fold với các fold khác: {len(fv)} phép so, tổng số vi phạm {sum(x.get('so_vi_pham', 0) for x in fv)}.",
           f"Tỉ lệ dòng mô tả của test đã có nguyên văn trong pool: {pt(r.get('thong_tin_dong_mo_ta_test_da_co_trong_pool'), 1)} (thuộc tính khuôn mẫu như \"Chất liệu: Cotton\"; đây là lý do các phiên B chấm riêng \"dòng mới\").", ""]
    t = doc("ft_run/loc_title/bao_cao.json") or {}
    ra += ["## 4. Lọc title bị gắn cờ (`ft_run/loc_title/bao_cao.json`)", "", "Cờ luật trên title của bộ sau Judge (toàn bộ 25.575 sản phẩm): `co_co` = số title dính ít nhất một cờ.", ""]
    ra += bang(["Hướng", "n", "Có cờ", "Hán trong bản Việt", "Việt trong bản Trung", "Âm tiết Việt", "Mất chữ Latin", "Lệch số", "Lệch hiệu"],
               [(h, f(x.get("n")), f(x.get("co_co")), f(x.get("han_trong_vi")), f(x.get("viet_trong_zh")), f(x.get("am_tiet_viet")), f(x.get("mat_latin")), f(x.get("so_lech")), f(x.get("hieu_lech")))
                for h, x in (t.get("toan_bo") or {}).items()]) + [""]
    dg = t.get("danh_gia") or {}
    ra += [f"Đánh giá cờ trên mẫu title đã chấm tay: {f(dg.get('so_title_da_cham'))} title đã chấm, {f(dg.get('so_title_loi'))} title lỗi, cờ bắt được {f(dg.get('bat_duoc_title_loi'))}; "
           f"trong {f(dg.get('co_co_trong_da_cham'))} title có cờ thì {f(dg.get('trong_do_loi'))} là lỗi thật. Theo từng cờ: {jtxt(dg.get('theo_co'), 600)}.", ""]
    return ra


def r_mau():
    d = doc("ft_run/mau/bao_cao_dung_mau.json") or {}
    ra = ["## 1. Lời dẫn (chép từ `scripts/ft_train/dung_mau.py`)", "", "Khuôn chat Qwen3, tắt suy luận (`enable_thinking=False`); loss chỉ tính trên phần bản dịch.", "", "```text",
          "Title:   Translate this product title from {SRC} to {TGT}. Output only the translation.\\n\\n{NGUON}",
          "Mô tả, pair: Translate this product's attribute list from {SRC} to {TGT}. Each line is one attribute written as \"name: value\". Keep the same number and order of lines, and output only the translation.",
          "             \\nProduct title (context only, do not translate): {TITLE}\\n\\n{NGUON}",
          "Mô tả, cụm:  ... The attributes are separated by \"; \" and each is written as \"name: value\". Keep the same number and order of attributes, separated by \"; \", and output only the translation. (+ cùng phần title ngữ cảnh)",
          "```", "", "## 2. Mẫu dựng được (`ft_run/mau/bao_cao_dung_mau.json`)", "",
          "Mỗi sản phẩm cho 2 mẫu: title và mô tả, theo đúng hướng gốc (1688 zh→vi, Tiki vi→zh). Mô tả quá `max_len` thì chia thành nhiều phần theo thuộc tính; thuộc tính đơn lẻ vẫn quá dài thì loại. Tập test chỉ được đếm, không dựng vào gói huấn luyện.", ""]
    dong = []
    for nh in ("cum", "pair"):
        p = lay(d, nh, "pool") or {}
        th, tl = p.get("theo_huong") or {}, p.get("theo_loai") or {}
        dong.append((nh, f(lay(d, nh, "max_len")), f(p.get("so_mau")), f(p.get("so_token")), f"{f(lay(th, 'zh->vi', 'so_mau'))} / {f(lay(th, 'vi->zh', 'so_mau'))}",
                     f"{f(lay(tl, 'title', 'so_mau'))} / {f(lay(tl, 'mo_ta', 'so_mau'))}", f"{f(lay(tl, 'title', 'so_token'))} / {f(lay(tl, 'mo_ta', 'so_token'))}",
                     f(p.get("so_sp_bi_chia_phan")), f(p.get("so_mau_bi_loai")), jtxt(p.get("ly_do_loai")), f(lay(d, nh, "test_chi_dem", "so_mau"))))
    ra += bang(["Bộ", "max_len", "Mẫu pool", "Token pool", "Mẫu zh→vi / vi→zh", "Mẫu title / mô tả", "Token title / mô tả", "Sản phẩm bị chia phần", "Mẫu bị loại", "Lý do loại", "Mẫu test (chỉ đếm)"], dong) + [""]
    ra += ["Số token ở đây đếm cả lời dẫn. Tỉ trọng title theo **token đích** (10,74% cụm, 10,86% pair) được tính từ chính các mẫu này lúc huấn luyện, xem cột `% title thực` ở B3–B7.", ""]
    ra += ["## 3. Dataset riêng tư đã đóng gói lên Kaggle", ""]
    dong = []
    for p, ten, tk in (("ft_run/goi_pool/MANIFEST.json", "ft-qwen3-4b-pool", "1"), ("ft_run/goi_pool_honghuynanh/MANIFEST.json", "ft-qwen3-4b-pool", "2"),
                       ("ft_run/goi_quota/MANIFEST.json", "ft-qwen3-4b-quota", "1, 2"), ("ft_run/goi_song_ngu/MANIFEST.json", "ft-song-ngu", "2"), ("ft_run/goi_do_khoa/MANIFEST.json", "ft-qwen3-4b-do-khoa", "1")):
        m = doc(p) or {}
        tom = {k: (v.get("so_mau") if isinstance(v, dict) else v) for k, v in (m.get("nhanh") or m.get("tap") or {}).items()}
        noi_dung = jtxt(tom, 300) if tom else f"{f(m.get('n_files'))} file (chỉ số mẫu, bảng tham chiếu, mẫu eval)"
        dong.append((f"`{ten}`", tk, noi_dung, f"`{p}`"))
    ra += bang(["Dataset", "Tài khoản", "Nội dung (số mẫu / số dòng)", "Manifest"], dong) + [""]
    ra += ["- `ft-qwen3-4b-pool`: mẫu pool hai bộ (không có test), dùng cho mọi phiên B, C, D, E.",
           "- `ft-qwen3-4b-quota`: bảng tham chiếu và mẫu eval cố định cho khảo sát quota (C, D). Bảng tỉ trọng natural theo 4 nhóm ở `goi_quota/meta.json`.",
           "- `ft-song-ngu`: dữ liệu từng dòng của bản pair cho OpenNMT và Moses (D3–D5), gồm biến thể học đủ và khớp 4M token.",
           "- `ft-qwen3-4b-do-khoa`: tập test khóa, chỉ gắn vào phiên đo test B8.", ""]
    return ra


# ---------------------------------------------------------------- danh sách phiên (phần chữ viết tay)
TK1, TK2 = "1 (`trinhnhatanh`)", "2 (`honghuynanh`)"
GHI_SANG_LOC = ["- Đây là **vòng sàng lọc** (rank 16, một lượt, khoảng 4M token tính toán), không phải công thức cuối (rank 64, 2 epoch). Điểm chỉ dùng để so các cấu hình hoặc mô hình với nhau.",
                "- Bộ `pair` trong khảo sát quota là **từng dòng** (title hoặc một dòng thuộc tính mỗi mẫu, 331.296 mẫu), khác adapter `pair_*` của công thức cuối (cả khối thuộc tính một lượt)."]
LUU_5MH = "Bảng tổng của cả năm mô hình ở phiên D6."

PHIEN = [
    # ---- A. dịch dữ liệu
    dict(ma="A1", ten="Bộ song ngữ 1: thuộc tính 1688 dịch theo đơn vị (Gemma, Qwen3-14B, CometKiwi)", giai_doan="A. Dữ liệu: dịch, lọc, Judge", tai_khoan=TK1, thoi_gian="15–22/10/2026",
         notebook="các notebook dịch và chấm của tài khoản 1 (thẻ `kaggle` bên dưới)", thu_muc=["song_ngu_1"],
         tep_chinh=["song_ngu_1/the_chuan_bi.json", "song_ngu_1/the_nguon_goc_20261022_2126.json"],
         muc_dich="Dịch tên và giá trị thuộc tính của sản phẩm 1688 sang tiếng Việt theo từng đơn vị bằng hai mô hình, chấm từng bản bằng CometKiwi (`wmt22-cometkiwi-da`, không cần tham chiếu) rồi chọn bản theo luật ngưỡng. Phần cố định (có/không, mã thật, tên hàng, từ điển đã duyệt) đi theo luật, không qua mô hình.",
         render=lambda: r_song_ngu("song_ngu_1/the_nguon_goc_20261022_2126.json"),
         ket_luan=["Bản cuối của bộ này là nguồn cho bộ cụm 1688 (A4)."],
         ghi_chu=["Ngưỡng chọn (`tau`, `kappa`, `delta`) ghi trong thẻ là tạm, đặt từ phân bố điểm, không có nhãn đúng/sai.",
                  "Thư mục còn các lần thử mô hình dịch khác: `hy_mt_thu` (HY-MT1.5-7B), `qwen14b_thu`, `qwen30b_thu`."]),
    dict(ma="A2", ten="Bộ song ngữ 2: thuộc tính Tiki (vi→zh) dịch theo đơn vị", giai_doan="A. Dữ liệu: dịch, lọc, Judge", tai_khoan=TK1, thoi_gian="23/10/2026", notebook="các notebook dịch và chấm của tài khoản 1",
         thu_muc=["song_ngu_2"], tep_chinh=["song_ngu_2/the_chuan_bi.json", "song_ngu_2/the_nguon_goc_20261023_1656.json"],
         muc_dich="Như A1 nhưng cho sản phẩm Tiki: gốc tiếng Việt, dịch sang tiếng Trung.", render=lambda: r_song_ngu("song_ngu_2/the_nguon_goc_20261023_1656.json"),
         ket_luan=["Bản cuối của bộ này là nguồn cho bộ cụm Tiki (A3)."]),
    dict(ma="A3", ten="Bộ song ngữ 3: rà soát theo cụm có ngữ cảnh (Tiki)", giai_doan="A. Dữ liệu: dịch, lọc, Judge", tai_khoan=TK1, thoi_gian="23–24/10/2026", notebook="`trinhnhatanh/song-ngu-3-cum`, `trinhnhatanh/song-ngu-3-cum-qwen`",
         thu_muc=["song_ngu_3"], tep_chinh=["song_ngu_3/the_chuan_bi_v2.json", "song_ngu_3/tom_tat_cum_20261024_1616.json"],
         muc_dich="Đưa cả cụm thuộc tính của một sản phẩm (đã dịch theo đơn vị ở A2) cho hai mô hình rà soát có ngữ cảnh: giữ, sửa hoặc dịch mới từng cặp; cặp đã khóa bằng luật không cho sửa.",
         render=lambda: ["## Thẻ chuẩn bị `song_ngu_3/the_chuan_bi_v2.json`", ""] + bang_json("song_ngu_3/the_chuan_bi_v2.json") + ["", "## Tóm tắt `song_ngu_3/tom_tat_cum_20261024_1616.json`", ""] + bang_json("song_ngu_3/tom_tat_cum_20261024_1616.json") + [""],
         ghi_chu=["`A` và `B` trong bảng là hai mô hình rà soát theo thứ tự trong mã (A = Gemma, B = Qwen).", "Thư mục còn các lần chấm mù mẫu (`cham_mau_*`, `cham_that_*`); file `khoa_KHONG_XEM.json` là khóa giải mã, không mở khi chấm."]),
    dict(ma="A4", ten="Bộ song ngữ 4: rà soát theo cụm có ngữ cảnh (1688)", giai_doan="A. Dữ liệu: dịch, lọc, Judge", tai_khoan=TK1, thoi_gian="24–26/10/2026",
         notebook="`trinhnhatanh/song-ngu-4-cum-gemma` (+ `-p1`, `-p2`, `-p3`), `trinhnhatanh/song-ngu-4-cum-qwen`, `trinhnhatanh/dl-group2` (Qwen dự phòng trên RTX)", thu_muc=["song_ngu_4"],
         tep_chinh=["song_ngu_4/the_chuan_bi.json", "song_ngu_4/tom_tat_cum_20261026_2123.json"],
         muc_dich="Như A3 cho 1688: mỗi cụm thuộc tính được Gemma rà soát (Qwen chỉ phủ một phần), luật trượt chặn bản sửa xấu.",
         render=lambda: ["## Thẻ chuẩn bị `song_ngu_4/the_chuan_bi.json`", ""] + bang_json("song_ngu_4/the_chuan_bi.json") + ["", "## Tóm tắt `song_ngu_4/tom_tat_cum_20261026_2123.json`", ""] + bang_json("song_ngu_4/tom_tat_cum_20261026_2123.json") + [""],
         ghi_chu=["`cum_qwen.chua_dich` lớn: Qwen chỉ dịch được một phần cụm trong thời gian có GPU; phần còn lại theo Gemma.",
                  "Sau bước này các bộ được Huy gộp thành bộ official v2 (`bilingual_official_context.parquet`), rồi qua lọc luật (A5) và Judge (A6)."]),
    dict(ma="A5", ten="Lọc luật cho fine-tune (trên máy)", giai_doan="A. Dữ liệu: dịch, lọc, Judge", tai_khoan="– (chạy trên máy)", notebook="– (`scripts/ft_clean/xay_ft_clean.py`)", thoi_gian="30/10/2026",
         thu_muc=["ft_clean"], tep_chinh=["ft_clean/pheu_buoc0_4.json"],
         muc_dich="Loại khỏi bộ official v2 (bản cụm, 25.956 sản phẩm, HF commit `4a23574`) các dòng chắc chắn hỏng theo luật cứng trước khi đưa vào Judge và fine-tune; cờ mềm chỉ ghi lại, không loại.",
         render=r_ft_clean, ket_luan=["Còn **25.575** sản phẩm (1688 14.837, Tiki 10.738); đây là tập sản phẩm của cả hai bộ cụm và pair sau Judge."],
         ghi_chu=["Luật `so_doan_lech` có thể loại oan khi giá trị chứa chuỗi giống \"nhãn:\" bên trong.", "Mã có 6 bài thử trên dữ liệu tự dựng (`scripts/ft_clean/test_ft_clean.py`)."]),
    dict(ma="A6", ten="Judge DeepSeek chọn bản dịch Gemma hay Qwen (cụm và pair)", giai_doan="A. Dữ liệu: dịch, lọc, Judge", tai_khoan="– (gọi API DeepSeek qua nhà cung cấp, không chạy trên Kaggle)",
         notebook="– (`scripts/judge/`: `so_cap_du_lieu.py`, `so_cap_chon.py`, `ghep_bo.py`, `chay_lon.ps1`)", thoi_gian="thí điểm 29–31/10/2026; bản phát hành tải về 03/11/2026",
         thu_muc=["judge", "so_cap", "bo_chon_hf"],
         tep_chinh=[*(rel(p) for p in sorted((DP / "judge").glob("tom_tat_*.json"))), *(rel(p) for p in sorted((DP / "so_cap").glob("tom_tat_*.json"))),
                    "so_cap/bao_cao_title_100.json", "so_cap/bao_cao_cum_100.json", "so_cap/bao_cao_pair_100.json", "so_cap/so_sanh_gom.json", "so_cap/so_sanh_quy_tac.json",
                    "bo_chon_hf/README.md", "bo_chon_hf/manifest.json"],
         muc_dich="Với mỗi mảnh mà bản Gemma và bản Qwen khác nhau (title nguyên câu; mô tả cụm theo từng cặp tên–giá trị; mô tả pair theo từng tên hoặc giá trị), cho DeepSeek (`deepseek-v4.1-flash`) chấm mù theo rubric rồi chọn một bản. Chỉ thay bản hiện có khi hai lượt hoán vị cùng chọn một mô hình; mục khai tương đương, không nhất quán hoặc chưa chấm thì giữ hiện trạng; chặn bản mới nếu gây sót chữ Hán (1688) hoặc thêm chữ Việt (Tiki).",
         render=r_judge,
         ket_luan=["Bản cuối của hai bộ (cụm, pair) là dữ liệu huấn luyện của mọi phiên B–F (HF commit `82aa9c9`, hash ở `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` mục 2.5).",
                   "Gemma thắng phần lớn mục có người thắng ở thí điểm; Judge gần như tất định (chạy lặp giống khoảng 97%), nên mục không nhất quán chỉ gỡ được khi thêm ngữ cảnh (tầng 2).",
                   "Ở bản phát hành, Judge chỉ đổi một phần nhỏ nội dung so với bản đầu vào (cột `Title đã đổi`, `Mảnh đã đổi`); phần lớn mảnh giữ hiện trạng vì tương đương, chưa chấm hoặc là mảnh luật. Cột `Judge tầng 2` bằng 0 nghĩa là lần chạy đầy đủ không dùng tầng 2."],
         ghi_chu=["**Lần chạy đầy đủ không có trên máy này**: chỉ có thí điểm và bản phát hành trên HF. Hướng dẫn chạy (`HUONG_DAN_JUDGE.md`) và gói dữ liệu riêng đã được chuẩn bị để gửi Huy (WALKTHROUGH 694–697); người chạy lần đầy đủ cần nhóm xác nhận khi viết báo cáo.",
                  "Các file truy vết chi tiết của lần chạy đầy đủ nằm ở revision HF `126a564f` (theo README bản phát hành), chưa tải về máy.",
                  "Judge thiên vị vị trí 1 (khoảng 63–74% ở thí điểm) nên bắt buộc hỏi hai thứ tự; tham chiếu cuối vẫn là bản dịch máy, chưa người kiểm toàn bộ.",
                  "Thư mục `so_cap/` còn các bảng mục `muc_*.parquet` (đầu vào của Judge) và kết quả thô `ket_qua_*.jsonl` của thí điểm."]),
    dict(ma="A7", ten="Chia tập theo sản phẩm, kiểm rò rỉ, gắn cờ title (trên máy)", giai_doan="A. Dữ liệu: dịch, lọc, Judge", tai_khoan="– (chạy trên máy)",
         notebook="– (`scripts/ft_train/chia_tap.py`, `kiem_ro_ri.py`, `loc_title.py`)", thoi_gian="04–05/11/2026",
         thu_muc=["ft_run"], tep_chinh=["ft_run/DU_LIEU_DONG_BANG.json", "ft_run/bao_cao_chia_tap.json", "ft_run/bao_cao_ro_ri.json", "ft_run/loc_title/bao_cao.json"],
         muc_dich="Đóng băng bản dữ liệu sau Judge, chia một phép chia dùng chung cho cả hai bộ: gom sản phẩm gần-trùng (title, mô tả) vào cùng nhóm để không rơi vào hai phía, khóa 10% làm test, pool còn lại chia 5 fold phân tầng theo hướng × nhóm hàng; kiểm rò rỉ; gắn cờ title nghi lỗi bằng luật.",
         render=r_chia_tap,
         ket_luan=["Không rò rỉ theo ngưỡng đã đặt (0 vi phạm test với pool và giữa các fold).", "Cờ title bắt được phần lớn title lỗi trên mẫu chấm tay nhưng cũng gắn cờ nhiều title không lỗi; Nhật Anh chọn loại mẫu title bị cờ ở bản cuối (B9), giữ mẫu mô tả."],
         ghi_chu=["Tên script ghi theo chức năng; mã nằm trong `scripts/ft_train/`.", "Dòng mô tả của test phần lớn đã có nguyên văn trong pool vì thuộc tính lặp theo khuôn; vì vậy điểm mô tả cao không đồng nghĩa mô hình dịch tốt câu mới, các phiên B có chỉ số \"dòng mới\" riêng.",
                  "Thư mục `ft_run/test_khoa/` chứa tập test; không mở khi viết tài liệu này."]),
    # ---- B. fine-tune 4B
    dict(ma="B0", ten="Dựng mẫu huấn luyện và đóng gói Dataset Kaggle (trên máy)", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan="– (dựng trên máy; Dataset ở tài khoản 1 và 2)",
         notebook="– (`scripts/ft_train/dung_mau.py`, `dong_goi_kaggle.py`, `dong_goi_quota.py`, `xuat_song_ngu.py`)", thoi_gian="04–12/11/2026",
         thu_muc=["ft_run/mau", "ft_run/goi_pool", "ft_run/goi_pool_honghuynanh", "ft_run/goi_quota", "ft_run/goi_song_ngu", "ft_run/goi_do_khoa"],
         tep_chinh=["ft_run/mau/bao_cao_dung_mau.json", "ft_run/mau/bao_cao_dung_mau.md", "ft_run/goi_pool/MANIFEST.json", "ft_run/goi_pool_honghuynanh/MANIFEST.json", "ft_run/goi_quota/MANIFEST.json",
                    "ft_run/goi_quota/meta.json", "ft_run/goi_song_ngu/MANIFEST.json", "ft_run/goi_do_khoa/MANIFEST.json"],
         muc_dich="Biến mỗi sản phẩm của pool thành mẫu hội thoại (lời dẫn + bản dịch tham chiếu) cho LLM, đếm token, rồi đóng gói thành Dataset riêng tư trên Kaggle cho các phiên huấn luyện; dựng thêm dữ liệu từng dòng cho hai mô hình cổ điển.",
         render=r_mau, ket_luan=["Mọi phiên B–E đọc mẫu từ các Dataset này; hash của gói pool khớp giữa tài khoản 1 và 2."]),
    dict(ma="B1", ten="B6: dò siêu tham số Qwen3-4B (lr × rank)", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="20, 21, 22 (nối nhau)", thoi_gian="05/11/2026",
         thu_muc=["ft_run/kaggle_b6"], muc_dich="Chọn lr và rank LoRA cho bản cụm trên fold 0: vòng 1 chạy 1 epoch cho 7 cấu hình (gồm LoRA chỉ attention), vòng 2 chạy tiếp tới 2 epoch cho 2 cấu hình tốt nhất, sinh 1.000 mẫu và chấm.",
         render=lambda: r_4b(["20", "21", "22"], ["B6"], ["B6 "]),
         ket_luan=["Chọn **lr 2e-4, rank 64, 2 epoch**: loss kiểm thấp nhất; chênh chrF với lr 1e-4 là hòa (bảng so sánh).", "Ba phiên nối nhau vì mỗi phiên cài trần 2,5 giờ; phiên 1, 2 dừng có trật tự, phiên 3 xong."],
         ghi_chu=["Mã `kaggle_train.py` dùng cho B6 là bản trước lần sửa 05/11 23:06, không còn bản lưu riêng.", "Báo cáo phân tích thêm: `kaggle_b6/phan_tich_chi_so/BAO_CAO.md`. Lần chạy thử nối phiên `kaggle_b6_noi` là khói, không thuộc B6 (xem INDEX)."]),
    dict(ma="B2", ten="B7 + B8 phần đầu: LoRA attention, pair fold 0, cụm fold 1 (dở)", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="25 (bản 26 nối tiếp bị hủy)", thoi_gian="05/11/2026",
         thu_muc=["ft_run/kaggle_b78"], muc_dich="Chấm lại LoRA attention so với LoRA mọi lớp; huấn luyện bản pair fold 0 với cấu hình đã chọn; bắt đầu cụm fold 1.",
         render=lambda: r_4b(["25"], ["B7+B8 phần đầu"], ["B7 "]),
         ket_luan=["LoRA chỉ attention kém LoRA mọi lớp rõ, nên giữ `target_modules = all-linear`.", "`cum_f1` dừng ở bước 1.102/1.800 do hết giờ; được nối ở B4 từ checkpoint."]),
    dict(ma="B3", ten="Thử tỉ trọng title fold 0 (~11% và 75%, cụm và pair)", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="29", thoi_gian="05–06/11/2026",
         thu_muc=["ft_run/kaggle_tl"], muc_dich="Đổi tỉ trọng title trong loss từ mốc 1 : 1 theo mẫu (~50% loss) sang ~11% (bằng tỉ lệ token đích) và 75%, bằng hệ số `w_title`; đo trên fold 0 cho cả hai bộ.",
         render=lambda: r_4b(["29"], ["Thử tỉ lệ fold 0"], ["Thử tỉ lệ", "Dòng mới fold 0"]),
         ket_luan=["Mức ~11% giúp rõ ở dòng mô tả mới hướng zh→vi (cả cụm và pair); chrF chung hòa.", "Mức 75% thua ở pair, hòa ở cụm, và bị cắt gradient gần như mọi bước title nên bỏ."],
         ghi_chu=["Fold 0 cũng là nơi phát hiện tín hiệu dòng mới, nên chỉ fold 1, 2 (B4–B7) là kiểm độc lập."]),
    dict(ma="B4", ten="B8 fold 1, 2 (mốc 1 : 1) + cụm ~11% fold 1", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="32", thoi_gian="06/11/2026",
         thu_muc=["ft_run/kaggle_b8f1"], muc_dich="Nối `cum_f1` từ checkpoint bản 25; chạy `cum_f1_p11` (~11%) và `cum_f2` (mốc) để kiểm tỉ trọng trên fold khác.",
         render=lambda: r_4b(["32"], ["B8 fold 1, 2 + cụm ~11% fold 1"], ["Fold 1"]), ket_luan=["Fold 1: mức ~11% vẫn lợi ở dòng mới mô tả zh→vi, title không tụt."]),
    dict(ma="B5", ten="Cụm ~11% fold 2 + luật 3 fold bản cụm", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="34", thoi_gian="06/11/2026",
         thu_muc=["ft_run/kaggle_b8f2"], muc_dich="Chạy `cum_f2_p11` rồi áp luật 3 fold đã khóa trước để quyết đổi tỉ trọng cho bản cụm.",
         render=lambda: r_4b(["34"], ["Cụm ~11% fold 2"], ["Fold 2", "Bản cụm, fold"]) + r_3fold(), ket_luan=["Bản cụm **đạt** luật 3 fold, chốt tỉ trọng title 10,74%."]),
    dict(ma="B6", ten="Kiểm lr 1e-4 (cụm) + pair fold 1", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="36", thoi_gian="06–07/11/2026",
         thu_muc=["ft_run/kaggle_tl2"], muc_dich="Kiểm lại lr ở mức tỉ trọng mới (lr 1e-4 so với 2e-4, cụm fold 0); chạy pair fold 1 ở mốc và ở ~11%.",
         render=lambda: r_4b(["36"], ["Kiểm lr 1e-4 (cụm) + pair fold 1"], ["Kiểm lr", "Pair fold 1"]), ket_luan=["lr 1e-4 **không đạt** luật lr, giữ lr 2e-4.", "Pair fold 1 đạt cổng dòng mới."]),
    dict(ma="B7", ten="Pair fold 2 + luật 3 fold bản pair", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="38 (bản 37 là khói)", thoi_gian="07/11/2026",
         thu_muc=["ft_run/kaggle_tl3"], muc_dich="Chạy `pair_f2`, `pair_f2_p11` rồi áp luật 3 fold riêng của pair.",
         render=lambda: r_4b(["38"], ["Pair fold 2"], ["Bản pair, fold"]) + r_3fold_pair(),
         ket_luan=["Pair **đạt** luật 3 fold riêng; Nhật Anh duyệt chốt 10,86%.", "Lợi chủ yếu ở zh→vi; title fold 0 của pair tụt thật."]),
    dict(ma="B8", ten="Đo test khóa MỘT lần", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3` + Dataset riêng `ft-qwen3-4b-do-khoa`", ban="41 (bản 39, 40 là khói)", thoi_gian="07/11/2026",
         thu_muc=["ft_run/kaggle_do_test"], bao_cao=["ft_run/kaggle_do_test/phien_1_20261107_080604/bao_cao_do_test.json"],
         muc_dich="Đo một lần duy nhất adapter fold 0 (`cum_f0_p11`, `pair_f0_p11` của tài khoản 1, kho `ft-qwen3-4b-tl-luu`) so với zero-shot trên toàn bộ tập test khóa; luật thắng đặt trước khi đo.",
         render=r_do_test, ket_luan=["Fine-tune thắng zero-shot ở **mọi nhánh, mọi hướng** (cận dưới xa 0)."],
         ghi_chu=["Không đổi tỉ lệ, lr hay adapter sau khi thấy số test. Bản cuối (B9) **không** đo lại trên test.", "Chênh lớn một phần vì tham chiếu là bản dịch máy cùng kiểu với dữ liệu huấn luyện."]),
    dict(ma="B9", ten="Bản cuối trên toàn pool (cum_cuoi, pair_cuoi)", giai_doan="B. Fine-tune Qwen3-4B (công thức chính)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3` → kho `trinhnhatanh/ft-qwen3-4b-cuoi-luu`", ban="43 (bản 42 là khói)", thoi_gian="07/11/2026",
         thu_muc=["ft_run/kaggle_cuoi", "ft_run/ban_cuoi"], muc_dich="Huấn luyện hai adapter cuối trên toàn pool 23.017 sản phẩm với cấu hình đã chốt; loại mẫu title của 6.386 sản phẩm bị `loc_title.py` gắn cờ, giữ mẫu mô tả.",
         render=r_cuoi, ket_luan=["Hai adapter cuối đã lưu và kiểm hash. Demo dùng `pair_cuoi`."],
         ghi_chu=["Không còn fold giữ lại nên không có điểm chất lượng; loss trong bảng là loss trên mẫu đã học (chỉ để bắt lỗi hỏng)."]),
    # ---- C. khảo sát quota
    dict(ma="C1", ten="Khảo sát quota vòng 1 (rank 16, max_len 960, 1.200 bước)", giai_doan="C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="47 (pair), 48 (cụm)", thoi_gian="09/11/2026",
         thu_muc=["ft_run/kaggle_quota_screen"], bao_cao=["ft_run/kaggle_quota_screen/version47/quota_report.json", "ft_run/kaggle_quota_screen/version48/quota_report.json"],
         muc_dich="Thử 5 bảng tỉ trọng (title/mô tả × 1688/Tiki) cho mỗi bộ bằng cách điều khiển tần suất lấy mẫu, cùng số bước.",
         render=lambda: r_quota(["ft_run/kaggle_quota_screen/version47/quota_report.json", "ft_run/kaggle_quota_screen/version48/quota_report.json"], GHI_SANG_LOC),
         ket_luan=["**Không dùng để chốt**, đã thay bằng C3, C4: `max_len` 960 làm bỏ nhiều mẫu dài (cột `Mẫu bị bỏ vì dài`), nên % title thực lệch xa % đặt, và mỗi cấu hình thấy số token khác nhau."],
         ghi_chu=["Bản 47 lỗi ở phần cụm (dòng LỖI ở bảng); phần cụm chạy lại ở bản 48.", "Bảng tổng hợp cũ của vòng này: `docs/BANG_KET_QUA_CUOI.md` (đã bị thay)."]),
    dict(ma="C2", ten="C0: adapter chính thức chấm trên eval của vòng sàng lọc", giai_doan="C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="50 (cụm), 51 (pair)", thoi_gian="09/11/2026",
         thu_muc=["ft_run/kaggle_c0"], bao_cao=["ft_run/kaggle_c0/c0_report_v50_cum.json", "ft_run/kaggle_c0/c0_report_v51_pair.json"],
         muc_dich="Chấm `cum_f0_p11`, `pair_f0_p11` (tài khoản 1, công thức chính: rank 64, 2 epoch) trên đúng eval fold 0 của khảo sát quota để đặt hai cách cạnh nhau.",
         render=r_c0, ket_luan=["Adapter công thức chính cao hơn mọi cấu hình của vòng sàng lọc, nhưng chênh gộp cả công thức (rank, max_len, số epoch), không tách được tác dụng của tỉ trọng."],
         ghi_chu=["Eval pair ở đây là từng dòng (1.198 mẫu), còn adapter `pair_f0_p11` học cả khối."]),
    dict(ma="C3", ten="Khảo sát quota vòng 2 (ngân sách token cố định, max_len 2.048)", giai_doan="C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="53", thoi_gian="10/11/2026",
         thu_muc=["ft_run/kaggle_quota_screen_v2", "ft_run/kernel_v53"], bao_cao=["ft_run/kaggle_quota_screen_v2/quota_report.json"],
         muc_dich="Làm lại vòng 1 với cùng ngân sách token đích cho mọi cấu hình (~8.000 token đích mỗi bước) và `max_len` 2.048 để không bỏ mẫu.",
         render=lambda: r_quota(["ft_run/kaggle_quota_screen_v2/quota_report.json"], GHI_SANG_LOC), ket_luan=["Một seed (42). C4 chạy lại 3 cấu hình mỗi bộ với 2 seed để chốt."],
         ghi_chu=["Báo cáo bản này chưa ghi điểm theo nhóm, nên không có cột title/mô tả và cột Chính.", "`kernel_v53/nlp-group3.ipynb` là notebook đúng bản 53, lưu để tái lập."]),
    dict(ma="C4", ten="Khảo sát quota vòng 3: 3 cấu hình × 2 bộ × 2 seed (bảng đã chốt)", giai_doan="C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc)", tai_khoan=TK1, notebook="`trinhnhatanh/nlp-group3`", ban="55", thoi_gian="10/11/2026",
         thu_muc=["ft_run/kaggle_quota_step1"], bao_cao=["ft_run/kaggle_quota_step1/quota_report.json"],
         muc_dich="Chốt bảng tỉ trọng: natural, balanced, title_heavy (cụm) và natural, balanced, attr_heavy (pair), seed 42 và 123, cùng ngân sách token.",
         render=lambda: r_quota(["ft_run/kaggle_quota_step1/quota_report.json"], GHI_SANG_LOC),
         ket_luan=["**natural** (title ~11%) cao nhất ở điểm Chính cho cả hai bộ, cả hai seed; balanced chỉ hơn ở macro 50/50 bản cụm.",
                   "Diễn giải và w hòa vốn: `docs/QUOTA_KET_LUAN.md`, `docs/TRONG_SO_TITLE_MO_TA_DA_CHOT.md`. Dòng natural của bảng này là dòng Qwen3-4B trong bảng năm mô hình (D6)."]),
    dict(ma="C5", ten="Tái lập ở tài khoản 2 (check1)", giai_doan="C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-group3`", thoi_gian="11/11/2026",
         thu_muc=["ft_run/kaggle_quota_check1_acc2"], bao_cao=["ft_run/kaggle_quota_check1_acc2/quota_report.json"],
         muc_dich="Kiểm môi trường tài khoản 2 bằng cách chạy lại đúng cụm natural seed 42 của C4 và so điểm.",
         render=r_check1, ket_luan=["Tái lập được C4: cùng số bước, cùng token đích, chênh chrF dưới 0,2 điểm (bảng đối chiếu), nên môi trường tài khoản 2 dùng được cho các phiên sau."]),
    dict(ma="C6", ten="Qwen3-4B natural với lr 1e-4", giai_doan="C. Khảo sát tỉ trọng bằng lấy mẫu quota (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-group3` (cụm), `honghuynanh/nlp-group3-pair` (pair)", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_quota_4b_lr1e4_acc2"], bao_cao=["ft_run/kaggle_quota_4b_lr1e4_acc2/cum/quota_report.json", "ft_run/kaggle_quota_4b_lr1e4_acc2/pair/quota_report.json"],
         muc_dich="Thử lr 1e-4 (lr chung trong cấu hình của Quang) cho 4B ở vòng sàng lọc, so với lr 2e-4 của C4.",
         render=lambda: r_quota(["ft_run/kaggle_quota_4b_lr1e4_acc2/cum/quota_report.json", "ft_run/kaggle_quota_4b_lr1e4_acc2/pair/quota_report.json"]),
         ket_luan=["lr 1e-4 kém lr 2e-4 rõ ở cả hai bộ (so với dòng natural của C4), vì vòng sàng lọc chỉ có 107–241 bước; giữ lr 2e-4 cho các LLM."]),
    # ---- D. năm mô hình
    dict(ma="D0", ten="Danh sách mẫu natural dùng chung cho các LLM (chạy khô, CPU)", giai_doan="D. So sánh năm mô hình (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-dry-chon`", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_dry_chon"], bao_cao=["ft_run/kaggle_dry_chon/quota_dry.json"],
         muc_dich="Xuất danh sách mẫu natural (seed 42 và 123, cụm và pair) bằng tokenizer Qwen để Qwen3-4B, Qwen3-8B và EcomGPT-7B học đúng cùng danh sách mẫu.",
         render=lambda: r_quota(["ft_run/kaggle_dry_chon/quota_dry.json"]), ket_luan=["Các file `chon_*_natural_s*.json` là danh sách mẫu được dùng ở D1, D2."],
         ghi_chu=["Chạy khô nên không có điểm; cột Bước là số bước ước tính."]),
    dict(ma="D1", ten="Qwen3-8B, vòng sàng lọc (cụm + pair, 2 seed)", giai_doan="D. So sánh năm mô hình (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-8b`", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_quota_nlp-8b"], bao_cao=["ft_run/kaggle_quota_nlp-8b/quota_report.json"],
         muc_dich="Fine-tune Qwen3-8B (revision `b968826d`) cùng cấu hình sàng lọc và cùng danh sách mẫu với 4B.",
         render=lambda: r_quota(["ft_run/kaggle_quota_nlp-8b/quota_report.json"]) + r_5mh(["Qwen3-4B", "Qwen3-8B"]),
         ket_luan=["Cụm: 8B hơn 4B rõ (khoảng tin cậy xa 0). Pair: 8B và 4B hòa theo chrF câu; chrF corpus của 8B thấp hơn do vài dòng dài (tên thành phần Latin mà 8B dịch sang chữ Hán)."],
         ghi_chu=["Gradient checkpointing bật; VRAM đỉnh khoảng 25 GB (cột VRAM), nên không vừa một card 24 GB.", LUU_5MH]),
    dict(ma="D2", ten="EcomGPT-7B, vòng sàng lọc (cụm + pair, 2 seed)", giai_doan="D. So sánh năm mô hình (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-ecom`", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_quota_nlp-ecom"], bao_cao=["ft_run/kaggle_quota_nlp-ecom/quota_report.json"],
         muc_dich="Fine-tune EcomGPT-7B (ModelScope `damo/nlp_ecomgpt_multilingual-7B-ecom`, nền BLOOMZ-7B1, đổi sang bf16 safetensors) với khuôn Alpaca, attention eager, cùng danh sách mẫu.",
         render=lambda: r_quota(["ft_run/kaggle_quota_nlp-ecom/quota_report.json"]) + r_5mh(["Qwen3-4B", "EcomGPT-7B"]),
         ket_luan=["Kém Qwen3-4B rõ ở cả hai bộ (khoảng tin cậy xa 0); tỉ lệ sinh lặp cao hơn 4B (cột Sinh lặp)."],
         ghi_chu=["Tokenizer BLOOM cho ít token hơn nên cùng danh sách mẫu lại ít bước hơn (cột Bước); chưa tách được lỗi do mô hình hay do ít bước và khuôn.", LUU_5MH]),
    dict(ma="D3", ten="OpenNMT Transformer nhỏ, học từ đầu (pair + dịch bản cụm)", giai_doan="D. So sánh năm mô hình (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-onmt`, `honghuynanh/nlp-onmt-cum` (T4, có Internet)", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_onmt_that", "ft_run/kaggle_onmt_cum"], bao_cao=["ft_run/kaggle_onmt_that/onmt_report.json", "ft_run/kaggle_onmt_cum/onmt_cum_report.json"],
         muc_dich="Mốc NMT cổ điển: Transformer d_model 256, 4 lớp, 8 đầu, ff 1024, SentencePiece unigram 16k mỗi ngôn ngữ, Noam lr đỉnh 2e-3, tối đa 20 epoch + dừng sớm, trung bình 5 checkpoint, beam 5; học trên dòng của bản pair.",
         render=lambda: r_onmt() + r_5mh(["Qwen3-4B", "OpenNMT (tu dau, train day du)"]), ket_luan=["Pair: hòa với 4B theo chrF câu; cụm: kém rõ. Yếu nhất ở vi→zh (ít dữ liệu hướng này)."],
         ghi_chu=["Chạy trên T4 vì cần Internet để cài thư viện và torch 2.2.2 không hỗ trợ GPU Blackwell.", "Bản cụm không huấn luyện riêng: dịch từng mảnh thuộc tính rồi nối.", LUU_5MH]),
    dict(ma="D4", ten="Moses vòng 1: chọn cách tách chữ Hán (từng chữ hay jieba)", giai_doan="D. So sánh năm mô hình (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-moses-zhvi`, `honghuynanh/nlp-moses-vizh` (CPU)", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_moses_zhvi", "ft_run/kaggle_moses_vizh"], bao_cao=["ft_run/kaggle_moses_zhvi/moses_report.json", "ft_run/kaggle_moses_vizh/moses_report.json"],
         muc_dich="Phrase-based SMT: eflomal + grow-diag-final-and, KenLM 5-gram, cụm tối đa 7, msd-bidirectional-fe, MERT trên dev 1.000 dòng; thử tách chữ Hán từng chữ và bằng jieba, chọn theo chrF dev.",
         render=lambda: r_moses(["ft_run/kaggle_moses_zhvi/moses_report.json", "ft_run/kaggle_moses_vizh/moses_report.json"]), ket_luan=["Chọn tách **từng chữ** theo chrF dev (bảng)."],
         ghi_chu=["Moses biên dịch ở notebook `honghuynanh/nlp-moses-build` (`./bjam -j4 --with-boost=/usr -q`), gói `moses_bin_scripts.tgz` (xem INDEX, nhóm khói)."]),
    dict(ma="D5", ten="Moses vòng 2: học đủ dữ liệu và khớp 4M token, kèm eval bản cụm", giai_doan="D. So sánh năm mô hình (vòng sàng lọc)", tai_khoan=TK2, notebook="`honghuynanh/nlp-moses-v2-zhvi`, `honghuynanh/nlp-moses-v2-vizh` (CPU)", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_moses_v2_zhvi", "ft_run/kaggle_moses_v2_vizh"], bao_cao=["ft_run/kaggle_moses_v2_zhvi/moses_report.json", "ft_run/kaggle_moses_v2_vizh/moses_report.json"],
         muc_dich="Hai biến thể: học toàn bộ dòng train, và học số dòng khớp ngân sách 4M token của LLM (seed 42); dịch cả eval bản cụm từng mảnh.",
         render=lambda: r_moses(["ft_run/kaggle_moses_v2_zhvi/moses_report.json", "ft_run/kaggle_moses_v2_vizh/moses_report.json"]) + r_5mh(["Qwen3-4B", "Moses (train day du)", "Moses (khop 4M token)"]),
         ket_luan=["Pair: Moses học đủ hơn 4B; khớp 4M token thì hòa. Cụm: Moses kém 4B ở cả hai biến thể."],
         ghi_chu=["MERT không tất định: hai lần chạy Moses học đủ lệch khoảng 0,25 điểm chính.", "Khoảng 87% dòng thuộc tính của eval pair đã có nguyên văn trong train, có lợi cho Moses.", LUU_5MH]),
    dict(ma="D6", ten="Tổng hợp năm mô hình (trên máy, không GPU)", giai_doan="D. So sánh năm mô hình (vòng sàng lọc)", tai_khoan="– (Qwen3-4B lấy từ C4, tài khoản 1; các mô hình khác tài khoản 2)", notebook="– (`scripts/ft_train/tong_hop_5mh.py`)", thoi_gian="12–14/11/2026",
         thu_muc=["ft_run/tong_hop_5mh", "ft_run/tong_hop_5mh_cum"], muc_dich="Chấm chung bản dịch của C4 (Qwen3-4B) và D1–D5 bằng một bộ chấm (`cham_diem.py`, `cham_bleu.py`), bootstrap theo sản phẩm, thêm BLEU/SacreBLEU/TER theo cách đo của bài VLSP.",
         render=lambda: r_5mh(["Qwen3-4B", "Qwen3-8B", "EcomGPT-7B", "OpenNMT (tu dau, train day du)", "Moses (train day du)", "Moses (khop 4M token)", "Qwen3-4B cong thuc cuoi (hoc fold 1-4, T4 fp16)"], "## Bảng năm mô hình"),
         ket_luan=["Cụm: Qwen3-8B hơn 4B rõ; 4B hơn Moses; EcomGPT và OpenNMT kém xa.", "Pair: 4B, 8B, OpenNMT, Moses khớp 4M token nằm trong khoảng nhiễu; Moses học đủ nhỉnh hơn.",
                   "Nhận xét đầy đủ: `docs/KE_HOACH_VA_KET_QUA_NAM_MO_HINH.md`; khoảng tin cậy SacreBLEU/TER: `tong_hop_5mh*/bang_metric_bai_bao.md` (số chép ở `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` mục 6)."],
         ghi_chu=["Dòng `Qwen3-4B cong thuc cuoi` đến từ phiên E3: khác lời dẫn và cách dịch nên không xếp hạng chung với năm dòng còn lại.", "Năm mô hình chưa đo trên tập test khóa."]),
    # ---- E. thử chéo, công thức cuối trên fold 0
    dict(ma="E1", ten="Huấn luyện lại adapter fold 0 ở tài khoản 2 (cum_f0_p11, pair_f0_p11)", giai_doan="E. Thử chéo và công thức chính trên eval năm mô hình", tai_khoan=TK2, notebook="`honghuynanh/nlp-group3` (cụm), `honghuynanh/nlp-group3-pair` (pair)", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_cheo_cum", "ft_run/kaggle_cheo_pair"], bao_cao=["ft_run/kaggle_cheo_cum/bao_cao_cheo_cum.json", "ft_run/kaggle_cheo_pair/bao_cao_cheo_pair.json"],
         muc_dich="Cần adapter chưa học fold 0 để đo trên eval fold 0 (adapter bản cuối đã học cả fold 0). Nhật Anh duyệt huấn luyện lại đúng cấu hình B3 (rank 64, 2 epoch, ~11% title) ở tài khoản 2, hai notebook chạy song song.",
         render=r_cheo_train, ket_luan=["Hai adapter này dùng cho E2 (thử chéo), E3 (công thức chính trên eval năm mô hình) và cổng nén demo (F1)."],
         ghi_chu=["Hai adapter là lần huấn luyện riêng nên hash khác bản tài khoản 1; số của E2, E3, F1 thuộc về adapter tài khoản 2, số của B8 (test) và C2 thuộc về adapter tài khoản 1."]),
    dict(ma="E2", ten="Thử chéo dạng pair và dạng cụm", giai_doan="E. Thử chéo và công thức chính trên eval năm mô hình", tai_khoan=TK2, notebook="`honghuynanh/nlp-cheo`", thoi_gian="12/11/2026",
         thu_muc=["ft_run/kaggle_cheo_that", "ft_run/cheo_ket_qua"], bao_cao=["ft_run/kaggle_cheo_that/bao_cao_cheo.json"],
         muc_dich="Cho mỗi adapter dịch cả hai dạng đầu vào (cụm và pair) trên 600 sản phẩm fold 0, mô tả zh→vi, cộng zero-shot; chấm trên máy với cả hai tham chiếu.",
         render=r_cheo_that, ket_luan=["Hai dạng ngang chất lượng khi dùng đúng adapter của dạng đó; dùng sai dạng thì tụt. Chọn pair hay cụm cho extension là vì tiện, không vì điểm."],
         ghi_chu=["Một fold, chỉ mô tả, không title."]),
    dict(ma="E3", ten="Công thức chính trên eval năm mô hình (\"dòng thứ sáu\", T4 fp16)", giai_doan="E. Thử chéo và công thức chính trên eval năm mô hình", tai_khoan=TK2, notebook="`honghuynanh/nlp-dong6`", thoi_gian="14/11/2026",
         thu_muc=["ft_run/kaggle_dong6"], bao_cao=["ft_run/kaggle_dong6/bao_cao_dong6.json"],
         muc_dich="Cho adapter công thức chính học fold 1–4 (E1) dịch đúng 1.200 mẫu eval của năm mô hình, để biết công thức chính hơn vòng sàng lọc bao nhiêu trên cùng mẫu.",
         render=r_dong6, ket_luan=["Công thức chính hơn 4B sàng lọc rõ ở cả hai bộ (khoảng tin cậy xa 0)."],
         ghi_chu=["Chạy trên Tesla T4 fp16 (đã mất RTX Pro 6000), khác bf16 lúc huấn luyện.", "Lời dẫn theo `dung_mau.py`, pair dịch cả khối rồi tách dòng, nên không xếp hạng chung với năm dòng sàng lọc.",
                  "Hai lần chạy thử trước đó (`kaggle_dong6_khoi`, `kaggle_dong6_khoi_fp16`) nằm ở nhóm khói trong INDEX."]),
    # ---- F. demo, 8B
    dict(ma="F1", ten="Cổng nén để chạy demo trên laptop (không phải Kaggle)", giai_doan="F. Demo và việc chưa chạy được", tai_khoan="–", notebook="– (máy Nhật Anh, `demo/do_luong_tu.py`)", thoi_gian="14/11/2026",
         thu_muc=["ft_run/demo_do_luong_tu"], muc_dich="Đo bản nén GGUF (Q4_K_M, Q5_K_M) của Qwen3-4B + LoRA trên RTX 3050 4 GB so với fp16 trên Kaggle, theo ngưỡng đặt trước.",
         render=r_demo_gate, ket_luan=["Q4_K_M + LoRA **không đạt** (tụt quá 1 điểm, có khối lệch dòng, có bản rỗng); Q5_K_M + LoRA **đạt**."],
         ghi_chu=["Đo bằng adapter học fold 1–4 (`pair_f0_p11`, E1); adapter demo `pair_cuoi` chỉ được suy ra là chịu nén tương tự.", "File GGUF ở `ft_run/demo_gguf/` (không liệt kê hash vì nặng)."]),
    dict(ma="F2", ten="Máy chủ dịch demo trên Kaggle + thử 13 trang 1688 thật", giai_doan="F. Demo và việc chưa chạy được", tai_khoan=TK1, notebook="`trinhnhatanh/demo-may-chu` (2 T4, có Internet, đường hầm Cloudflare)", thoi_gian="15/11/2026",
         thu_muc=["ft_run/demo_bo_dem_kaggle"], muc_dich="Chạy máy chủ dịch với `pair_cuoi` fp16 trên Kaggle, mở đường hầm cho extension, dịch 13 trang 1688 đã lưu và giữ bộ đệm để demo không cần GPU.",
         render=r_demo_kaggle, ket_luan=["13/13 trang đạt kiểm cấu trúc; đọc tay 8 trang mới khoảng 95% dòng đúng. Lỗi đọc tay và 5 trang chọn demo: `docs/TONG_HOP_KET_QUA_NHAT_ANH.md` mục 9."],
         ghi_chu=["Không có tham chiếu cho trang thật: đây là quan sát, không phải số đo.", "Đường hầm có 2 lần đứt ngẫu nhiên; extension đã thêm tự thử lại khi lỗi mạng (chưa kiểm trong Chrome thật)."]),
    dict(ma="F3", ten="Qwen3-8B theo công thức chính: không chạy được (bị cấp T4)", giai_doan="F. Demo và việc chưa chạy được", tai_khoan=TK2, notebook="`honghuynanh/nlp-8b-cuoi-cum`", thoi_gian="13/11/2026",
         thu_muc=["ft_run/kaggle_cuoi8b_khoi", "ft_run/kaggle_cuoi8b_khoi2"], bao_cao=["ft_run/kaggle_cuoi8b_khoi/bao_cao_thu_cuoi8b.json", "ft_run/kaggle_cuoi8b_khoi2/bao_cao_thu_cuoi8b.json"],
         muc_dich="Chạy khói Qwen3-8B bản cụm theo công thức chính (rank 64, 2 epoch, `w_title`) trước khi chạy thật, vì ở vòng sàng lọc 8B hơn 4B ở bản cụm.",
         render=r_8b_loi, ket_luan=["Hai lần khói liền được cấp Tesla T4 14,6 GB dù metadata giống notebook từng chạy RTX Pro 6000. Lần 1 hết bộ nhớ khi nạp 8B; sau đó thêm chốt chặn `vram_toi_thieu_gb` nên lần 2 dừng sau 30 giây. **8B theo công thức chính chưa có; phiên thật không được đẩy.**"]),
]

# phiên thử ngắn: (thư mục, kiểm cái gì, notebook)
N1, N2 = "trinhnhatanh/nlp-group3", "honghuynanh"
KHOI = [
    ("ft_run/kaggle_b5", "B5: chạy thử môi trường và mã huấn luyện, lần 1", N1), ("ft_run/kaggle_b5_lan2", "B5 lần 2", N1), ("ft_run/kaggle_b5_lan3", "B5 lần 3", N1),
    ("ft_run/kaggle_b5_lan4", "B5 lần 4", N1), ("ft_run/kaggle_do_toc_do", "đo tốc độ token/giây", N1), ("ft_run/kaggle_nguyen_nhan", "tìm nguyên nhân chậm", N1),
    ("ft_run/kaggle_b6_thu", "khói B6", N1), ("ft_run/kaggle_b6_thu2", "khói B6 lần 2", N1), ("ft_run/kaggle_b6_noi", "khói nối phiên B6 (phan1 → phan2)", N1),
    ("ft_run/kaggle_b78_thu", "khói B7 + B8 (hai phần + tổng hợp)", N1), ("ft_run/kaggle_tl_thu", "khói thử tỉ lệ (hai phần)", N1), ("ft_run/kaggle_tl3_thu", "khói pair fold 2 (bản 37)", N1),
    ("ft_run/kaggle_do_test_thu", "khói đo test (bản 39, 40)", N1), ("ft_run/kaggle_cuoi_thu", "khói bản cuối (bản 42)", N1), ("ft_run/kaggle_quota_smoke", "khói khảo sát quota (bản 46)", N1),
    ("ft_run/kaggle_quota_dry", "chạy khô quota trước C3 (bản 52)", N1), ("ft_run/kaggle_quota_dry2", "chạy khô quota trước C4 (bản 54)", N1),
    ("ft_run/kaggle_dry_mh", "chạy khô danh sách mẫu cho 8B và EcomGPT (CPU)", f"{N2}/nlp-dry-8b, nlp-dry-ecom"), ("ft_run/kaggle_smoke_mh", "khói 8B và EcomGPT (bộ nhớ, tốc độ)", f"{N2}/nlp-smoke-8b, nlp-smoke-ecom"),
    ("ft_run/kaggle_onmt_khoi", "khói OpenNMT", f"{N2}/nlp-onmt-khoi"), ("ft_run/kaggle_moses_build", "biên dịch Moses", f"{N2}/nlp-moses-build"),
    ("ft_run/kaggle_moses_diag", "chẩn đoán lỗi biên dịch Moses", f"{N2}/nlp-moses-diag"), ("ft_run/kaggle_moses_do", "đo thời gian Moses", f"{N2}/nlp-moses-do"),
    ("ft_run/kaggle_moses_khoi", "khói Moses (từng chữ, khớp 4M)", f"{N2}/nlp-moses-khoi"), ("ft_run/kaggle_cheo_khoi", "khói huấn luyện cho thử chéo", f"{N2}/nlp-group3"),
    ("ft_run/kaggle_cheo_khoi2", "khói sinh thử chéo", f"{N2}/nlp-cheo"), ("ft_run/kaggle_dong6_khoi", "khói dòng thứ sáu (bf16 trên T4)", f"{N2}/nlp-dong6"),
    ("ft_run/kaggle_dong6_khoi_fp16", "khói dòng thứ sáu (fp16 trên T4)", f"{N2}/nlp-dong6"),
]


def viet_phien(ph, truoc, sau):
    ra = [f"# {ph['ma']}. {ph['ten']}", "", "[← Mục lục](INDEX.md)" + (f" · [← {truoc['ma']}]({ten_tep(truoc)})" if truoc else "") + (f" · [{sau['ma']} →]({ten_tep(sau)})" if sau else ""), "",
          "Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py`. Mọi con số đọc từ file output đã tải về; phần mục đích, kết luận, ghi chú viết tay theo nhật ký. Tham chiếu của mọi điểm chrF/BLEU là bản dịch máy, chưa có người chấm.", ""]
    ra += phan_thong_tin(ph)
    ra += ["## Mục đích", "", ph["muc_dich"], ""]
    try:
        ra += ph["render"]()
    except Exception as e:  # không để một phiên lỗi làm hỏng cả bộ; main() báo phiên lỗi
        ra += [f"> LỖI khi đọc kết quả của phiên này: `{type(e).__name__}: {e}`", ""]
    if ph.get("ket_luan"):
        ra += ["## Kết luận của phiên", ""] + [f"- {x}" for x in ph["ket_luan"]] + [""]
    if ph.get("ghi_chu"):
        ra += ["## Ghi chú, giới hạn, sự cố", ""] + [f"- {x}" for x in ph["ghi_chu"]] + [""]
    ra += phan_tep(ph)
    return "\n".join(ra).rstrip() + "\n"


def viet_index():
    ra = ["# Mục lục kết quả Kaggle hướng Nhật Anh", "",
          f"Sinh tự động bởi `scripts/ft_train/xuat_ket_qua_kaggle.py` lúc {datetime.now(ICT).strftime('%d/%m/%Y %H:%M')} (ICT). Trang tổng hợp: [`docs/TONG_HOP_KET_QUA_NHAT_ANH.md`](../TONG_HOP_KET_QUA_NHAT_ANH.md). Nhật ký đầy đủ các phiên 4B (bản 20–43): [`docs/PHU_LUC_NHAT_KY_CHAY.md`](../PHU_LUC_NHAT_KY_CHAY.md).", "",
          "Mỗi phiên chạy thật một file. Output gốc nằm trên máy Nhật Anh ở `data/processed/` (không có trong git); mỗi file ghi sha256 của file gốc để đối chiếu. Mỗi file có: thông tin phiên (notebook, tài khoản, bản, thời gian, GPU), mục đích, bảng số, kết luận, ghi chú/giới hạn, danh sách file gốc.", "",
          "Tài khoản Kaggle: 1 = `trinhnhatanh` (tới 10/11/2026, và notebook demo 15/11), 2 = `honghuynanh` (từ 11/11/2026). GPU RTX Pro 6000 có được qua notebook gắn cuộc thi ARC Prize (nhóm báo đã được ban tổ chức đồng ý); từ 13/11/2026 notebook mới chỉ được cấp T4.", "",
          "Hai cách huấn luyện cần phân biệt khi đọc: **công thức chính** (B, E: rank 64, 2 epoch, `w_title`) và **vòng sàng lọc** (C, D: rank 16, một lượt, ~4M token, lấy mẫu quota). Điểm hai cách không so thẳng với nhau, trừ ở E3 (cùng eval)."]
    nhom = None
    for ph in PHIEN:
        if ph["giai_doan"] != nhom:
            nhom = ph["giai_doan"]
            ra += ["", f"## {nhom}", "", "| Mã | Phiên | Tài khoản | Bản Kaggle | Thời gian |", "|---|---|---|---|---|"]
        ra.append(f"| [{ph['ma']}]({ten_tep(ph)}) | [{ph['ten']}]({ten_tep(ph)}) | {ph.get('tai_khoan', '–')} | {ph.get('ban', '–')} | {ph.get('thoi_gian', '–')} |")
    ra += ["", "## Phiên chạy thử ngắn (khói, chạy khô, biên dịch): không có file riêng", "",
           "Các phiên này chỉ kiểm mã, môi trường, bộ nhớ hoặc tốc độ trước phiên thật; không dùng làm kết quả. Cột Đạt và Thời lượng đọc từ file báo cáo nếu có (`–` là phiên không ghi báo cáo, chỉ có log).", ""]
    dong = []
    for t, kiem, nb in KHOI:
        da, giay, gpu = [], 0.0, set()
        for p in sorted((DP / t).rglob("*.json")):
            if not re.match(r"(bao_cao_|quota_|.*_report\.json$)", p.name):
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            if "dat" in d:
                da.append(d.get("dat"))
            giay += float(d.get("tong_giay") or d.get("giay_tong") or 0)
            g = lay(d, "moi_truong", "gpu")
            if g:
                gpu.add(g)
        dong.append((f"`{t}`", kiem, f"`{nb}`", ", ".join(f(x) for x in da) if da else "–", gio(giay), ", ".join(sorted(gpu)) or "–", mtime(t)))
    ra += bang(["Thư mục", "Kiểm cái gì", "Notebook", "Đạt", "Thời lượng", "GPU", "Tải về lúc"], dong)
    ra += ["", "## Ngoài Kaggle", "",
           "- A5 (lọc luật), A6 (Judge DeepSeek, gọi API), D6 (tổng hợp năm mô hình), phần chấm của E2 và F1 (cổng nén) không chạy trên Kaggle nhưng có file riêng ở trên vì là bước chính.",
           "- Chia tập theo sản phẩm + test khóa và lọc title bị gắn cờ (`loc_title.py`): chạy trên máy; số liệu và hash ở trang tổng hợp mục 2.5, 2.6."]
    return "\n".join(ra).rstrip() + "\n"


def main():
    RA.mkdir(parents=True, exist_ok=True)
    for cu in RA.glob("*.md"):
        cu.unlink()
    for i, ph in enumerate(PHIEN):
        (RA / ten_tep(ph)).write_text(viet_phien(ph, PHIEN[i - 1] if i else None, PHIEN[i + 1] if i + 1 < len(PHIEN) else None), encoding="utf-8")
    (RA / "INDEX.md").write_text(viet_index(), encoding="utf-8")
    print("xong:", len(PHIEN), "phiên + INDEX.md ->", RA)
    loi = [p.name for p in RA.glob("*.md") if "LỖI khi đọc" in p.read_text(encoding="utf-8")]
    print("phiên lỗi:", loi or "không")
    thu_muc_co = {t for ph in PHIEN for t in ph["thu_muc"]} | {t for t, _, _ in KHOI}
    sot = [rel(p) for p in sorted(FT.glob("kaggle_*")) if p.is_dir() and rel(p) not in thu_muc_co]
    print("thư mục kaggle_* chưa có trong file nào:", sot or "không")


if __name__ == "__main__":
    main()
