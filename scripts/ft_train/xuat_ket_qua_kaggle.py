# -*- coding: utf-8 -*-
"""Sinh docs/ket_qua_kaggle/*.md + INDEX.md tu file ket qua that trong data/processed.

Chay: python -X utf8 scripts/ft_train/xuat_ket_qua_kaggle.py
Khong train, khong doc test.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FT = ROOT / "data" / "processed" / "ft_run"
SN = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "ket_qua_kaggle"
ICT = timezone(timedelta(hours=7))


# ------------------------------------------------------------------ tiện ích
def jdoc(p):
    return json.loads((FT / p).read_text(encoding="utf-8"))


def sha(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for k in iter(lambda: f.read(1 << 20), b""):
            h.update(k)
    return h.hexdigest()[:n]


def f2(x, n=2):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "—"


def ngay(ts):
    try:
        return datetime.fromtimestamp(float(ts), ICT).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return "—"


def bang(dau, dong):
    ra = ["| " + " | ".join(dau) + " |", "|" + "|".join(["---"] * len(dau)) + "|"]
    for d in dong:
        ra.append("| " + " | ".join(str(x) for x in d) + " |")
    return "\n".join(ra)


def nguon(paths, goc=None):
    goc = FT if goc is None else goc
    dong = []
    for p in paths:
        f = goc / p
        if not f.exists():
            continue
        dong.append((p, f"{f.stat().st_size:,}", sha(f)))
    if not dong:
        return "_Không có file nguồn._"
    return bang(["File (trong `data/processed/...`)", "Byte", "sha256 (16)"], dong)


def chrf_cua(o):
    """Lay chrf macro tu dict 'cham' hoac 'chrf'."""
    if not isinstance(o, dict):
        return None
    c = o.get("chrf") if "chrf" in o else o
    if isinstance(c, dict):
        return c.get("chrf_macro_huong")
    return None


def huong_cua(o):
    if not isinstance(o, dict):
        return {}
    c = o.get("chrf") if "chrf" in o else o
    if not isinstance(c, dict):
        return {}
    return {k: v.get("chrf_corpus") for k, v in (c.get("theo_huong") or {}).items()}


def gpu_cua(p):
    try:
        return jdoc(p).get("moi_truong", {}).get("gpu", "—")
    except Exception:
        return "—"


# ------------------------------------------------------------------ phiên
PHIEN = [
    {
        "slug": "00_dich_du_lieu_song_ngu",
        "title": "Dịch dữ liệu song ngữ (song_ngu 1 → 4)",
        "kind": "song_ngu",
        "mo_ta": "Sinh bản dịch Trung–Việt cho thuộc tính/title bằng Gemma/Qwen + từ điển, chọn bằng luật và Judge.",
        "notebook": "các phiên Kaggle tài khoản 1 và 2 (Gemma/Qwen)",
        "tai_khoan": "trinhnhatanh + honghuynanh",
        "ban": "không lưu trong file kết quả",
        "thoi_gian": "16–26/10/2026",
        "ghi_chu": "Đây là đầu vào của toàn bộ fine-tune; không phải phiên huấn luyện.",
    },
    {
        "slug": "01_moi_truong_b5",
        "title": "Kiểm môi trường và đo tốc độ (B5)",
        "kind": "b5",
        "mo_ta": "Chạy khói để kiểm GPU, tokenizer, loss giảm, lưu/nạp adapter và đo token/giây.",
        "dirs": ["kaggle_b5", "kaggle_b5_lan2", "kaggle_b5_lan3", "kaggle_b5_lan4", "kaggle_do_toc_do"],
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "không lưu (bản 1–4)",
        "thoi_gian": "04–05/11/2026",
        "ghi_chu": "Không dùng làm kết quả; chỉ để chốt cấu hình chạy.",
    },
    {
        "slug": "02_b6_do_sieu_tham_so",
        "title": "B6 — dò siêu tham số (fold 0, bản cụm)",
        "kind": "b6",
        "dirs": ["kaggle_b6", "kaggle_b6_noi"],
        "mo_ta": "Quét lr × rank, chọn cấu hình cho mọi bước sau.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "20, 21, 22 (3 phiên nối tiếp)",
        "thoi_gian": "05/11/2026 09:58 → 16:10",
        "ghi_chu": "Phiên 1–2 dừng ở trần 2,5 giờ; phiên 3 chạy nốt.",
    },
    {
        "slug": "03_b78_pair_f0_cum_f1",
        "title": "B7 + B8 — pair fold 0, cụm fold 1",
        "kind": "b78",
        "dirs": ["kaggle_b78"],
        "mo_ta": "Mốc 1:1 và mức ~11% title cho pair fold 0; mốc cụm fold 1 nối từ bước 1.102.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "25",
        "thoi_gian": "05/11/2026 18:13 → 20:43",
        "ghi_chu": "Dừng ở trần 2,5 giờ; bản 26 nối tiếp bị hủy, không có báo cáo.",
    },
    {
        "slug": "04_tl_ti_le_fold0",
        "title": "Thử tỉ trọng title fold 0 (cụm và pair)",
        "kind": "tl",
        "dirs": ["kaggle_tl"],
        "mo_ta": "So mốc 1:1 với ~11% và 75% title, cả hai bộ.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "29",
        "thoi_gian": "05/11/2026 23:20 → 06/11/2026 05:00",
        "ghi_chu": "Phiên 5,67 giờ; kết quả là cơ sở chọn ~11%.",
    },
    {
        "slug": "05_b8f1_fold1",
        "title": "B8 fold 1 — mốc 1:1 và cụm ~11%",
        "kind": "b78",
        "dirs": ["kaggle_b8f1"],
        "mo_ta": "Kiểm tỉ trọng ~11% trên fold 1 (kiểm độc lập đầu tiên).",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "32",
        "thoi_gian": "06/11/2026 10:28 → 14:05",
        "ghi_chu": "Nối cum_f1 từ bước 1.102 của bản 25.",
    },
    {
        "slug": "06_b8f2_fold2",
        "title": "B8 fold 2 — cụm ~11%",
        "kind": "b78",
        "dirs": ["kaggle_b8f2"],
        "mo_ta": "Kiểm độc lập thứ hai cho tỉ trọng ~11% ở cụm.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "34",
        "thoi_gian": "06/11/2026 16:51 → 18:20",
        "ghi_chu": "Hoàn tất luật 3 fold cho bản cụm.",
    },
    {
        "slug": "07_tl2_lr_va_pair_f1",
        "title": "Kiểm lr 1e-4 (cụm) và pair fold 1",
        "kind": "b78",
        "dirs": ["kaggle_tl2"],
        "mo_ta": "Kiểm lr 1e-4 ở mức ~11% và mốc pair fold 1.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "36",
        "thoi_gian": "06/11/2026 21:00 → 07/11/2026 01:10",
        "ghi_chu": "Lr 1e-4 không đạt; giữ 2e-4.",
    },
    {
        "slug": "08_tl3_pair_f2",
        "title": "Pair fold 2 — mốc 1:1 và ~11%",
        "kind": "b78",
        "dirs": ["kaggle_tl3"],
        "mo_ta": "Hoàn tất luật 3 fold riêng cho pair.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "38",
        "thoi_gian": "07/11/2026 02:31 → 05:17",
        "ghi_chu": "Bản 37 là khói.",
    },
    {
        "slug": "09_do_test_mot_lan",
        "title": "Đo test khóa MỘT lần",
        "kind": "do_test",
        "dirs": ["kaggle_do_test"],
        "mo_ta": "Zero-shot và adapter fold 0 trên toàn bộ test 2.558 sản phẩm (5.116 mục mỗi nhánh).",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "41",
        "thoi_gian": "07/11/2026 08:06 → 10:39",
        "ghi_chu": "Không huấn luyện. Bản 39–40 là khói. Không đo lại sau lần này.",
    },
    {
        "slug": "10_cuoi_toan_pool",
        "title": "Bản cuối trên toàn pool (cum_cuoi, pair_cuoi)",
        "kind": "b78",
        "dirs": ["kaggle_cuoi"],
        "mo_ta": "Huấn luyện 2 epoch trên 23.017 sản phẩm, loại 6.386 title bị cờ.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "43",
        "thoi_gian": "07/11/2026 10:55 → 14:07",
        "ghi_chu": "Không còn fold giữ lại nên không có điểm chất lượng; không đo test.",
    },
    {
        "slug": "11_cheo_dang_pair_cum",
        "title": "Thử chéo dạng pair ↔ cụm",
        "kind": "cheo",
        "dirs": ["kaggle_cheo_cum", "kaggle_cheo_pair", "kaggle_cheo_khoi", "kaggle_cheo_that"],
        "mo_ta": "Đo 600 sản phẩm fold 0, mô tả zh→vi, đổi dạng đầu vào và adapter.",
        "notebook": "nlp-group3 (các bản cheo)",
        "tai_khoan": "trinhnhatanh",
        "ban": "không lưu trong file kết quả",
        "thoi_gian": "08–09/11/2026",
        "ghi_chu": "Dùng adapter fold 0 cum_f0_p11 / pair_f0_p11.",
    },
    {
        "slug": "12_dong6_toc_do_t4",
        "title": "Đo tốc độ trên T4 fp16 (đóng gói máy chủ)",
        "kind": "dong6",
        "dirs": ["kaggle_dong6", "kaggle_dong6_khoi", "kaggle_dong6_khoi_fp16"],
        "mo_ta": "Đo thời gian nạp và sinh của adapter trên Tesla T4 để tính công suất máy chủ demo.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "không lưu trong file kết quả",
        "thoi_gian": "12/11/2026",
        "ghi_chu": "Chỉ đo tốc độ, không đo chất lượng.",
    },
    {
        "slug": "13_onmt_hoc_du",
        "title": "OpenNMT — học từ đầu, đủ dữ liệu",
        "kind": "onmt",
        "dirs": ["kaggle_onmt_that"],
        "mo_ta": "Baseline cổ điển: Transformer nhỏ của OpenNMT, huấn luyện trên dòng của bản pair.",
        "notebook": "các phiên onmt",
        "tai_khoan": "trinhnhatanh",
        "ban": "không lưu trong file kết quả",
        "thoi_gian": "10–11/11/2026",
        "ghi_chu": "Dùng để so trong bảng năm mô hình.",
    },
    {
        "slug": "14_moses_hai_huong",
        "title": "Moses — phrase-based, hai hướng",
        "kind": "moses",
        "dirs": ["kaggle_moses_v2_zhvi", "kaggle_moses_v2_vizh"],
        "mo_ta": "Baseline cổ điển: Moses với Hán tách từng chữ, hai biến thể (học đủ / khớp 4M token).",
        "notebook": "các phiên moses",
        "tai_khoan": "trinhnhatanh",
        "ban": "không lưu trong file kết quả",
        "thoi_gian": "10–12/11/2026",
        "ghi_chu": "Moses pair mạnh vì ~87% dòng thuộc tính đã có nguyên văn trong train.",
    },
    {
        "slug": "15_nam_mo_hinh",
        "title": "So sánh năm mô hình (eval fold 0)",
        "kind": "nam_mh",
        "dirs": ["kaggle_quota_nlp-8b", "kaggle_quota_nlp-ecom", "kaggle_quota_4b_lr1e4_acc2",
                 "kaggle_quota_check1_acc2", "tong_hop_5mh", "tong_hop_5mh_cum"],
        "mo_ta": "Qwen3-4B, Qwen3-8B, EcomGPT-7B, OpenNMT, Moses trên cùng eval fold 0.",
        "notebook": "nlp-group3 / nlp-8b / nlp-ecom",
        "tai_khoan": "trinhnhatanh + honghuynanh",
        "ban": "không lưu trong file kết quả",
        "thoi_gian": "10–13/11/2026",
        "ghi_chu": "Cấu hình sàng lọc rank 16, 4M token; không phải công thức cuối.",
    },
    {
        "slug": "16_quota_screen_960",
        "title": "Quota screen 960 (bản cũ, có lỗi bỏ mẫu)",
        "kind": "quota",
        "dirs": ["kaggle_quota_screen", "kaggle_quota_smoke"],
        "mo_ta": "5 bảng trọng số × 2 bộ, rank 16, max_len 960, 1.200 bước.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "47 (pair), 48 (cụm), 46 (smoke)",
        "thoi_gian": "09/11/2026",
        "ghi_chu": "Có lỗi: trainer bỏ mẫu dài >960 trong khi ngạch coi như cắt. Không dùng để chốt.",
    },
    {
        "slug": "17_quota_screen_2048",
        "title": "Quota screen 2.048 (đã sửa lỗi độ dài)",
        "kind": "quota",
        "dirs": ["kaggle_quota_screen_v2"],
        "mo_ta": "5 cấu hình × 2 bộ, cap 2.048, loss chuẩn hoá theo token đích, 1 seed.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "53",
        "thoi_gian": "10/11/2026",
        "ghi_chu": "Bản sửa đầu tiên; phát hiện metric 'Chính' sai trọng số ở pair, đã sửa ở bước sau.",
    },
    {
        "slug": "18_quota_step1_2seed",
        "title": "Quota bước 1 — 3 cấu hình × 2 bộ × 2 seed",
        "kind": "quota",
        "dirs": ["kaggle_quota_step1"],
        "mo_ta": "Kiểm lại sau khi sửa metric: title 10/90, guardrail, 2 seed.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "54 (dry), 55 (chạy thật)",
        "thoi_gian": "10/11/2026",
        "ghi_chu": "Kết quả dùng để chốt natural; xem docs/QUOTA_KET_LUAN.md.",
    },
    {
        "slug": "19_c0_baseline",
        "title": "C0 — chấm adapter official trên dev",
        "kind": "c0",
        "dirs": ["kaggle_c0"],
        "mo_ta": "Chấm cum_f0_p11 / pair_f0_p11 trên đúng bộ eval của screen.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "49 (lỗi adapter), 50 (cụm), 51 (pair)",
        "thoi_gian": "09/11/2026",
        "ghi_chu": "Dùng để so candidate với bản đang dùng.",
    },
    {
        "slug": "20_demo_may_chu",
        "title": "Demo — máy chủ dịch và đo lượng tử hoá",
        "kind": "demo",
        "mo_ta": "Đo Q4_K_M / Q5_K_M + LoRA trên RTX 3050, thử 13 trang 1688 thật.",
        "notebook": "demo cục bộ + Kaggle T4",
        "tai_khoan": "trinhnhatanh",
        "ban": "không lưu",
        "thoi_gian": "14–15/11/2026",
        "ghi_chu": "Q5_K_M đạt cổng; Q4_K_M bị loại.",
    },
    {
        "slug": "21_dry_run_ngach",
        "title": "Dry run ngạch và chọn mẫu",
        "kind": "dry",
        "dirs": ["kaggle_quota_dry", "kaggle_quota_dry2", "kaggle_dry_chon", "kaggle_dry_mh"],
        "mo_ta": "Dựng ngạch, kiểm tỉ lệ và độ dài trước khi train.",
        "notebook": "nlp-group3",
        "tai_khoan": "trinhnhatanh",
        "ban": "52, 54 (và các lần dry 8B/Ecom)",
        "thoi_gian": "09–10/11/2026",
        "ghi_chu": "0 GPU cho phần lớn; dùng để chặn lỗi trước khi chạy thật.",
    },
]


# ------------------------------------------------------------------ render
def md_dau(s, them=""):
    L = [f"# {s['title']}", "", s["mo_ta"], ""]
    L.append(bang(["", ""], [
        ("Notebook", s.get("notebook", "—")),
        ("Tài khoản", s.get("tai_khoan", "—")),
        ("Bản Kaggle", s.get("ban", "—")),
        ("Thời gian", s.get("thoi_gian", "—")),
        ("Dùng ở đâu", s.get("dung", "—")),
        ("Ghi chú", s.get("ghi_chu", "—")),
    ]))
    L.append("")
    return "\n".join(L)


def md_nguon(paths, goc=None):
    return "## Nguồn\n\n" + nguon(paths, goc) + "\n"


def render_song_ngu(s):
    p4 = "data/processed/song_ngu_4"
    t = json.loads((SN / "song_ngu_4" / "tom_tat_cum_20261026_2123.json").read_text(encoding="utf-8"))
    cb = json.loads((SN / "song_ngu_4" / "the_chuan_bi.json").read_text(encoding="utf-8"))
    L = [md_dau(s), "## 1. Quy mô\n"]
    L.append(bang(["Chỉ số", "Giá trị"], [
        ("Sản phẩm", f"{cb.get('san_pham', 0):,}"),
        ("Cụm (block) cần gọi", f"{cb.get('cum', 0):,}"),
        ("Cặp (cặp tên–giá trị) đưa vào", f"{cb.get('cap', 0):,}"),
        ("Trường cần xuất", f"{cb.get('truong_can_xuat', 0):,}"),
        ("Ký tự Trung cần xem", f"{cb.get('ky_tu_trung_can_xem', 0):,}"),
    ]))
    L.append("\n## 2. Kết quả chọn bản dịch (song_ngu_4)\n")
    L.append(bang(["Chỉ số", "Giá trị"], [
        ("Cặp giữ", f"{t.get('cap_giu', 0):,}"),
        ("Bỏ vì không có bản", f"{t.get('cap_bo_khong_co_ban', 0):,}"),
        ("Cách chọn", t.get("cach_chon", "—")),
        ("Gemma: ok / lỗi cấu trúc / chưa dịch", f"{t['cum_gemma']['ok']:,} / {t['cum_gemma']['loi_cau_truc']:,} / {t['cum_gemma']['chua_dich']:,}"),
        ("Qwen: ok / lỗi cấu trúc / chưa dịch", f"{t['cum_qwen']['ok']:,} / {t['cum_qwen']['loi_cau_truc']:,} / {t['cum_qwen']['chua_dich']:,}"),
    ]))
    L.append("\n## 3. Nguồn bản dịch cuối\n")
    nb = t.get("nguon_ban_dich", {})
    L.append(bang(["Nguồn", "Số mục"], [(k, f"{v:,}") for k, v in nb.items()]))
    L.append("\n- Độ tin cậy cấp: " + ", ".join(f"{k} {v:,}" for k, v in t.get("do_tin_cay_cap", {}).items()) + ".")
    L.append("- Bản cuối dùng cho fine-tune là bộ đã qua Judge; xem mục 2 của `TONG_HOP_KET_QUA_NHAT_ANH.md`.")
    L.append("\n" + md_nguon(["song_ngu_4/tom_tat_cum_20261026_2123.json",
                             "song_ngu_4/the_chuan_bi.json"], SN))
    return "\n".join(L)


def render_b5(s):
    L = [md_dau(s), "## 1. Các lần chạy\n"]
    dong = []
    for ten in ("kaggle_b5", "kaggle_b5_lan2", "kaggle_b5_lan3", "kaggle_b5_lan4"):
        p = f"{ten}/bao_cao_b5.json"
        try:
            d = jdoc(p)
        except Exception:
            continue
        m = d.get("moi_truong", {})
        dong.append((ten.replace("kaggle_", ""), f2(d.get("tong_giay", 0), 0), m.get("gpu", "—"),
                     m.get("vram_gb", "—"), d.get("token_moi_giay", "—"),
                     d.get("so_mau_pool", "—"), d.get("dat", "—")))
    L.append(bang(["Lần", "Giây", "GPU", "VRAM (GB)", "Token/s", "Mẫu pool", "Đạt"], dong))
    L.append("\n## 2. Đo tốc độ riêng\n")
    p = "kaggle_do_toc_do/bao_cao_b5.json"
    d = jdoc(p)
    L.append(bang(["Cấu hình", "Giây", "Token/s"], [
        (x.get("ten", "—"), f2(x.get("giay", 0), 1), x.get("token_moi_giay", "—"))
        for x in d.get("do_toc_do", [])]))
    L.append("\n" + md_nguon(["kaggle_b5/bao_cao_b5.json", "kaggle_b5_lan2/bao_cao_b5.json",
                             "kaggle_b5_lan3/bao_cao_b5.json", "kaggle_b5_lan4/bao_cao_b5.json",
                             "kaggle_do_toc_do/bao_cao_b5.json"]))
    return "\n".join(L)


def render_b6(s):
    L = [md_dau(s), "## 1. Cấu hình từng phần\n"]
    nhan = {
        "kaggle_b6/phien_1_20261105_095809": "phiên 1 (09:58)",
        "kaggle_b6/phien_2_20261105_123231": "phiên 2 (12:33)",
        "kaggle_b6/phien_3_20261105_150627": "phiên 3 (15:07)",
        "kaggle_b6_noi/phan1": "nối phần 1",
        "kaggle_b6_noi/phan2": "nối phần 2 (xong)",
    }
    dong = []
    for ten in ("kaggle_b6/phien_1_20261105_095809", "kaggle_b6/phien_2_20261105_123231",
                "kaggle_b6/phien_3_20261105_150627", "kaggle_b6_noi/phan1", "kaggle_b6_noi/phan2"):
        p = f"{ten}/bao_cao_b6.json"
        try:
            d = jdoc(p)
        except Exception:
            continue
        c = d.get("cau_hinh", {})
        dong.append((nhan.get(ten, ten), f2(d.get("tong_giay", 0), 0), c.get("so_buoc"), c.get("lr"), c.get("rank"),
                     d.get("dat"), len(d.get("b6", {}).get("cau_hinh", {}))))
    L.append(bang(["Phần", "Giây", "Bước", "lr", "rank", "Đạt", "Số cấu hình"], dong))

    L.append("\n## 2. Loss kiểm theo cấu hình (ghép 3 phiên)\n")
    gop = {}
    for ten in ("kaggle_b6/phien_1_20261105_095809", "kaggle_b6/phien_2_20261105_123231",
                "kaggle_b6/phien_3_20261105_150627"):
        d = jdoc(f"{ten}/bao_cao_b6.json")
        for k, v in (d.get("b6", {}).get("cau_hinh") or {}).items():
            gop.setdefault(k, {})
            for e in ("1", "2"):
                if e in (v.get("diem") or {}):
                    gop[k][e] = v["diem"][e]
    dong = []
    for k in sorted(gop):
        dong.append((k, f2(gop[k].get("1"), 4), f2(gop[k].get("2"), 4)))
    L.append(bang(["Cấu hình", "Loss kiểm ep1", "Loss kiểm ep2"], dong))

    L.append("\n## 3. Điểm hai cấu hình được chấm\n")
    d = jdoc("kaggle_b6/phien_3_20261105_150627/b6/b6_trang_thai.json")
    dong = []
    for k, v in (d.get("cham") or {}).items():
        c = v.get("chrf", {})
        nh = v.get("theo_nhom") or {}
        def tb(loai):
            a = (nh.get(f"vi->zh/{loai}") or {}).get("chrf_corpus")
            b = (nh.get(f"zh->vi/{loai}") or {}).get("chrf_corpus")
            return (a + b) / 2 if (a is not None and b is not None) else (a or b)
        dong.append((k, f2(c.get("chrf_macro_huong")), f2(tb("title")), f2(tb("mo_ta")),
                     f2(v.get("giay_moi_mau"), 3)))
    L.append(bang(["Cấu hình", "chrF macro", "Title (TB 2 hướng)", "Mô tả (TB 2 hướng)", "Giây/mẫu"], dong))
    L.append("\n- Chọn **lr 2e-4, rank 64**: loss thấp nhất; chênh chrF so 1e-4 là +0,33 [−0,12; +0,81], hòa.")
    L.append("- LoRA chỉ attention kém hơn all-linear: −2,20 [−2,75; −1,65].")
    L.append("\n" + md_nguon(["kaggle_b6/phien_3_20261105_150627/bao_cao_b6.json",
                             "kaggle_b6/phien_1_20261105_095809/bao_cao_b6.json",
                             "kaggle_b6/phien_2_20261105_123231/bao_cao_b6.json"]))
    return "\n".join(L)


def nn(x):
    return "—" if x is None else x


def _job_bang(d, ten_file="b78"):
    """Bang cac job trong bao_cao_b78/b8f2/cuoi."""
    viec = (d.get(ten_file) or {}).get("viec") or {}
    dong = []
    for k, v in viec.items():
        c = v.get("cham") or {}
        ch = c.get("chrf") or {}
        h = c.get("theo_huong") or {}
        ghi = ""
        if v.get("nguon_adapter"):
            ghi = f"dùng lại adapter {v['nguon_adapter']}" + (" (khớp B6)" if v.get("khop_b6") else "")
        elif not c:
            ghi = "chưa chấm"
        cauh = str(nn(v.get("cau_hinh")))
        lr = v.get("lr")
        if lr is not None and abs(float(lr) - 2e-4) > 1e-12:
            cauh += f" (lr {float(lr):g})"
        dong.append((k, nn(v.get("nhanh")), nn(v.get("fold")), cauh,
                     nn(v.get("epoch_xong")), nn(v.get("buoc")),
                     f2(ch.get("chrf_macro_huong")),
                     f2((h.get("vi->zh") or {}).get("chrf_corpus")),
                     f2((h.get("zh->vi") or {}).get("chrf_corpus")),
                     f2(c.get("giay_moi_mau"), 3), ghi))
    if not dong:
        return "_Không có job nào trong báo cáo._"
    return bang(["Việc", "Bộ", "Fold", "Cấu hình", "Epoch", "Bước", "chrF macro", "vi→zh", "zh→vi", "Giây/mẫu", "Ghi chú"], dong)


def render_b78(s):
    L = [md_dau(s), "## 1. Việc đã chạy\n"]
    for dten in s.get("dirs", []):
        for f in sorted((FT / dten).rglob("bao_cao_*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "b78" not in d:
                continue
            L.append(f"### `{f.relative_to(FT)}`\n")
            c = d.get("cau_hinh", {})
            L.append(f"- Thời lượng: **{f2(d.get('tong_giay', 0), 0)} giây**; GPU: `{(d.get('moi_truong') or {}).get('gpu', '—')}`; mô hình: `{', '.join(d.get('mo_hinh') or [])}`.")
            L.append(f"- Cấu hình: lr {c.get('lr')}, rank {c.get('rank')}, max_len {c.get('max_len')}, micro_bs {c.get('micro_bs')}, seed {c.get('seed')}.")
            L.append("")
            L.append(_job_bang(d))
            L.append("")
    L.append(md_nguon([str(f.relative_to(FT)) for dt in s.get("dirs", []) for f in (FT / dt).rglob("bao_cao_*.json")]))
    return "\n".join(L)


def render_tl(s):
    L = [md_dau(s), "## 1. Mốc và các mức\n"]
    d = jdoc("kaggle_tl/phien_1_20261105_231941/tong_hop/bao_cao_ti_le.json")
    for tap in ("cum", "pair"):
        x = d["nhanh"][tap]
        L.append(f"### {tap}\n")
        dong = []
        for ma, v in x.get("muc", {}).items():
            so = v.get("so") or {}
            dong.append((f"{ma} ({v.get('ten', '')})", f2(v.get("p_muc_tieu", 0) * 100), f2(v.get("p_thuc", 0) * 100),
                         f2(v.get("w_title"), 5), f2(v.get("diem_e2"), 3),
                         f2(so.get("chrf")), f2(so.get("title")), f2(so.get("mo_ta")), v.get("hong")))
        L.append(bang(["Mức", "% title đặt", "% title thực", "w_title", "Loss kiểm ep2", "chrF", "Title", "Mô tả", "Hỏng"], dong))
        moc = x.get("moc", {}).get("so", {})
        L.append(f"\nMốc 1:1: chrF {f2(moc.get('chrf'))}, title {f2(moc.get('title'))}, mô tả {f2(moc.get('mo_ta'))}.\n")
    L.append("\n## 2. Kết luận đã khóa\n")
    L.append("- Cụm: mức ~11% đạt 3 fold (xem mục 3 của `TONG_HOP_KET_QUA_NHAT_ANH.md`).")
    L.append("- Pair: mức 75% thua; ~11% đạt luật 3 fold riêng.")
    L.append("\n" + md_nguon(["kaggle_tl/phien_1_20261105_231941/tong_hop/bao_cao_ti_le.json",
                             "kaggle_tl/phien_1_20261105_231941/tong_hop/kiem_dong_moi.json"]))
    return "\n".join(L)


def render_do_test(s):
    L = [md_dau(s), "## 1. Kết quả trên test khóa (5.116 mục mỗi nhánh)\n"]
    d = jdoc("kaggle_do_test/phien_1_20261107_080604/tong_hop/tong_hop_do_test.json")
    dong = []
    for tap in ("cum", "pair"):
        for h, v in d[tap]["huong"].items():
            c = v["chung"]
            dong.append((tap, h, c.get("n_muc"), f2(c.get("chrf_zs")), f2(c.get("chrf_ft")),
                         f"+{f2(c.get('chenh'))} [{f2(c.get('thap'))}; {f2(c.get('cao'))}]"))
    L.append(bang(["Bộ", "Hướng", "Mục", "Zero-shot", "Fine-tune", "Chênh [95%]"], dong))
    L.append("\n## 2. Ghi chú\n")
    L.append("- Đo **một lần** theo luật khóa trước: thắng khi cận dưới 95% > 0.")
    L.append("- Sau lần này không đo lại test; mọi so sánh sau đó chỉ trên dev.")
    L.append("\n" + md_nguon(["kaggle_do_test/phien_1_20261107_080604/tong_hop/tong_hop_do_test.json"]))
    return "\n".join(L)


def render_cheo(s):
    L = [md_dau(s), "## 1. Bảng đo (600 sản phẩm fold 0, mô tả zh→vi)\n"]
    d = json.loads((FT / "cheo_ket_qua/ket_qua.json").read_text(encoding="utf-8"))
    dong = []
    for x in d.get("bang", []):
        dong.append((x.get("adapter"), x.get("dang_dau_vao"), x.get("n"),
                     f2(x.get("chrf_cau_tb")),
                     f2(x.get("dung_so_manh"), 4), f2(x.get("giu_so"), 4),
                     x.get("ro_loi_dan"), x.get("rong")))
    L.append(bang(["Adapter", "Dạng vào", "n", "chrF câu TB", "Đúng số mảnh", "Giữ số", "Rò lời dẫn", "Rỗng"], dong))
    L.append("\n## 2. Kết luận\n")
    L.append("- Dùng đúng adapter cho đúng dạng thì hai dạng tương đương (+0,04 [−0,16; +0,23]).")
    L.append("- Đưa sai dạng làm giảm 0,47–1,01 điểm.")
    L.append("\n" + md_nguon(["cheo_ket_qua/ket_qua.json", "cheo_ket_qua/bang.md"]))
    return "\n".join(L)


def render_dong6(s):
    L = [md_dau(s), "## 1. Tốc độ trên T4 fp16\n"]
    for dten in s.get("dirs", []):
        for f in sorted((FT / dten).glob("bao_cao_dong6.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            m = d.get("moi_truong", {})
            L.append(f"### `{f.relative_to(FT)}`\n")
            L.append(f"- GPU: `{m.get('gpu')}`; VRAM {m.get('vram_gb')} GB; dtype `{m.get('dtype')}`; tổng {f2(d.get('tong_giay', 0), 0)} giây.\n")
            dong = []
            for v in d.get("viec", []):
                dong.append((nn(v.get("tap")), nn(v.get("adapter")), nn(v.get("n_hang_eval")), nn(v.get("n_luot_goi")),
                             f2(v.get("giay_nap"), 0), f2(v.get("giay_sinh"), 0), f2(v.get("giay_moi_luot"), 2)))
            if dong:
                L.append(bang(["Bộ", "Adapter", "Mục eval", "Lượt gọi", "Giây nạp", "Giây sinh", "Giây/lượt"], dong))
            else:
                L.append("_Không có job nào._")
            L.append("")
    L.append("## 2. Dùng để làm gì\n")
    L.append("- Tính công suất máy chủ demo trên T4 (1,66–1,70 giây/lượt).")
    L.append("- Không đo chất lượng.")
    L.append("\n" + md_nguon([str(f.relative_to(FT)) for dt in s.get("dirs", []) for f in (FT / dt).glob("bao_cao_dong6.json")]))
    return "\n".join(L)


def render_onmt(s):
    L = [md_dau(s), "## 1. Điểm trên eval fold 0\n"]
    d = json.loads((FT / "kaggle_onmt_that/cham_onmt.json").read_text(encoding="utf-8"))
    dong = [(k, f2(v)) for k, v in (d.get("tom_tat") or {}).items()
            if isinstance(v, (int, float)) and not k.startswith("ti_le") and k != "thieu_hyp"]
    L.append(bang(["Nhóm", "chrF"], dong))
    rep = json.loads((FT / "kaggle_onmt_that/onmt_report.json").read_text(encoding="utf-8"))
    L.append("\n## 2. Huấn luyện\n")
    dong = []
    for v in rep.get("viec", []):
        dong.append((v.get("huong"), v.get("dong_train"), v.get("token_sp_train"),
                     v.get("steps"), f2(v.get("giay_train"), 0), f2(v.get("giay_tong"), 0)))
    L.append(bang(["Hướng", "Dòng train", "Token", "Bước", "Giây train", "Giây tổng"], dong))
    L.append("\n- OpenNMT yếu ở vi→zh vì chỉ có khoảng 61 nghìn dòng hướng này (token ít hơn zh→vi khoảng 4 lần).")
    L.append("\n" + md_nguon(["kaggle_onmt_that/cham_onmt.json", "kaggle_onmt_that/onmt_report.json"]))
    return "\n".join(L)


def render_moses(s):
    L = [md_dau(s), "## 1. Hai hướng\n"]
    dong = []
    for dten in s.get("dirs", []):
        f = FT / dten / "moses_report.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for v in d.get("viec", []):
            dong.append((v.get("huong"), v.get("seg"), v.get("bien_the"), v.get("dong_train"),
                         v.get("dev_chrf"), v.get("n_eval"), f2(v.get("giay", 0), 0)))
    L.append(bang(["Hướng", "Tách", "Biến thể", "Dòng train", "chrF dev", "n eval", "Giây"], dong))
    L.append("\n## 2. Ghi chú\n")
    L.append("- Moses pair mạnh vì ~87% dòng thuộc tính eval đã có nguyên văn trong train; lợi thế mất khi dịch cả khối.")
    L.append("\n" + md_nguon([f"{d}/moses_report.json" for d in s.get("dirs", [])]))
    return "\n".join(L)


def render_nam_mh(s):
    L = [md_dau(s), "## 1. Bảng tổng hợp (từ `tong_hop_5mh*`)\n"]
    for ten, nhan in (("tong_hop_5mh", "Pair"), ("tong_hop_5mh_cum", "Cụm")):
        p = FT / ten / "ket_qua.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        L.append(f"### {nhan}\n")
        dong = []
        for k, v in d.items():
            if k.startswith("_"):
                continue
            dong.append((k, v.get("loai"), f2(v.get("tb_chinh")), f2(v.get("tb_macro")),
                         f2(v.get("lech_chinh_giua_lan"))))
        L.append(bang(["Mô hình", "Loại", "Chính", "Macro", "Lệch giữa lần"], dong))
        L.append("")
    L.append("## 2. Ghi chú\n")
    L.append("- Ba LLM dùng cấu hình sàng lọc rank 16, 4M token, 2 seed; không phải công thức cuối.")
    L.append("- Dòng `Qwen3-4B công thức cuối` đo trên T4 fp16, lời dẫn khác nên không xếp hạng chung.")
    L.append("- Số chi tiết (BLEU/SacreBLEU/TER) nằm trong `bang_metric_bai_bao.md` của từng thư mục.")
    L.append("\n" + md_nguon(["tong_hop_5mh/ket_qua.json", "tong_hop_5mh_cum/ket_qua.json",
                             "tong_hop_5mh/bang_metric_bai_bao.md", "tong_hop_5mh_cum/bang_metric_bai_bao.md"]))
    return "\n".join(L)


def render_quota(s):
    L = [md_dau(s), "## 1. Kết quả từng cấu hình\n"]
    da_xem = set()
    for dten in s.get("dirs", []):
        for f in sorted((FT / dten).rglob("quota_report.json")):
            h = sha(f)
            if h in da_xem:
                continue
            da_xem.add(h)
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            L.append(f"### `{f.relative_to(FT)}`\n")
            kiem = d.get("kiem_tung_tap") or {}
            for tap, k in kiem.items():
                nh = k.get("natural") or {}
                L.append(f"- {tap}: bỏ {k.get('n_bo')} mẫu, dài nhất {k.get('dai_nhat_giu')}, "
                         f"natural title {f2(sum(v for a, v in nh.items() if a.endswith('_title')) * 100)}%.")
            dong, loi, co_nhom = [], [], False
            for j in d.get("viec", []):
                if "loi" in j:
                    loi.append(f"{j.get('tap')}/{j.get('config')}: {j['loi'][:60]}")
                    continue
                nh = j.get("theo_nhom") or {}
                if nh:
                    co_nhom = True
                t = (nh.get("zh->vi/title") or {}).get("chrf_corpus")
                m = (nh.get("zh->vi/mo_ta") or {}).get("chrf_corpus")
                chinh = (0.10 * t + 0.90 * m) if (t is not None and m is not None) else None
                dong.append((j.get("tap"), j.get("config"), j.get("seed", ""),
                             f2(j.get("chrf", {}).get("chrf_macro_huong")),
                             f2(chinh), f2(t), f2(m), j.get("so_buoc"), f2(j.get("lech_max"), 4)))
            if dong:
                L.append(bang(["Bộ", "Cấu hình", "Seed", "Macro", "Chính 10/90", "Title zh→vi", "Mô tả zh→vi", "Bước", "Lệch ngạch"], dong))
            if loi:
                L.append(f"\n- {len(loi)} job lỗi (bỏ khỏi bảng): " + "; ".join(loi[:5]) + ".")
            if dong and not co_nhom:
                L.append("\n- Bản này **chưa lưu điểm tách title/mô tả**, nên không tính được cột Chính 10/90; chỉ có macro.")
            L.append("")
    L.append("## 2. Đọc kết quả\n")
    L.append("- `natural` là bảng dữ liệu tự nhiên; các mức khác là can thiệp có chủ đích.")
    L.append("- Với bản 960: có lỗi bỏ mẫu dài, **không dùng để chốt**.")
    L.append("- Kết luận cuối ở `docs/QUOTA_KET_LUAN.md`.")
    L.append("\n" + md_nguon([str(f.relative_to(FT)) for dt in s.get("dirs", []) for f in (FT / dt).rglob("quota_report.json")]))
    return "\n".join(L)


def render_c0(s):
    L = [md_dau(s), "## 1. Điểm C0 trên dev\n"]
    dong, da_xem = [], set()
    for f in sorted((FT / "kaggle_c0").glob("c0_report*.json")):
        h = sha(f)
        if h in da_xem:
            continue
        da_xem.add(h)
        d = json.loads(f.read_text(encoding="utf-8"))
        for j in d.get("viec", []):
            if "loi" in j:
                continue
            hh = j.get("theo_huong") or {}
            dong.append((f.name, j.get("tap"), j.get("c0"), j.get("n_eval"),
                         f2(j.get("chrf", {}).get("chrf_macro_huong")),
                         f2((hh.get("vi->zh") or {}).get("chrf_corpus")),
                         f2((hh.get("zh->vi") or {}).get("chrf_corpus"))))
    L.append(bang(["File", "Bộ", "Adapter", "n", "chrF", "vi→zh", "zh→vi"], dong))
    L.append("\n- C0 cụm 76,49 (1.200 mẫu) và pair 74,55 (1.198 mẫu), chấm trên đúng eval của screen.")
    L.append("- Bản 49 lỗi đường dẫn adapter (`khong_thay_adapter`), đã sửa ở bản 50–51.")
    L.append("- Dùng để so với các bảng quota; không phải điểm test.")
    L.append("\n" + md_nguon([str(f.relative_to(FT)) for f in (FT / "kaggle_c0").glob("c0_report*.json")]))
    return "\n".join(L)


def render_demo(s):
    L = [md_dau(s), "## 1. Lượng tử hoá trên RTX 3050\n"]
    dong = []
    for f in sorted((FT / "demo_do_luong_tu").glob("ket_qua_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        m4 = d.get("may_4bit") or {}
        kg = d.get("kaggle_fp16") or {}
        def cau_tb(x):
            nh = x.get("chrf_theo_nhom") or {}
            t = (nh.get("zh->vi/title") or {}).get("chrf_cau_tb")
            m = (nh.get("zh->vi/mo_ta") or {}).get("chrf_cau_tb")
            return (t + m) / 2 if (t is not None and m is not None) else None
        def corpus(x):
            return (x.get("chrf_diem_chinh") or {}).get("chrf_macro_huong")
        dong.append((d.get("mo_hinh"), d.get("so_san_pham"), d.get("so_hang_eval"),
                     f2(cau_tb(kg)), f2(cau_tb(m4)),
                     f2(corpus(kg)), f2(corpus(m4)),
                     m4.get("so_khoi_lech_dong"), f2(m4.get("ti_le_rong"), 4)))
    L.append(bang(["Mô hình", "Sản phẩm", "Dòng eval", "chrF câu TB zh→vi fp16", "4-bit",
                   "chrF corpus fp16", "4-bit", "Khối lệch dòng", "Tỉ lệ rỗng"], dong))
    L.append("\n## 2. Thử trang 1688 thật\n")
    p = FT / "demo_thu_trang_bo_dem.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        xong = sum(1 for x in d if str(x.get("trang_thai", "")).startswith("Xong"))
        L.append(f"- Số trang: **{len(d)}**; đạt kiểm cấu trúc: **{xong}/{len(d)}**.")
    L.append("- Q5_K_M + LoRA đạt cổng (tụt 0,14 điểm); Q4_K_M bị loại (tụt 2,81).")
    L.append("\n" + md_nguon(["demo_do_luong_tu/ket_qua_q5km-lora-pair-f0.json",
                             "demo_do_luong_tu/ket_qua_q4km-lora-pair-f0.json",
                             "demo_thu_trang_bo_dem.json"]))
    return "\n".join(L)


def render_dry(s):
    L = [md_dau(s), "## 1. Các lần dry\n"]
    dong, da = [], set()
    for dten in s.get("dirs", []):
        for f in sorted((FT / dten).rglob("quota_dry.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            kiem = d.get("kiem_tung_tap") or {}
            for tap, k in kiem.items():
                khoa = (tap, k.get("n_bo"), k.get("dai_nhat_giu"), k.get("ngan_sach_token_dich"))
                if khoa in da:
                    continue
                da.add(khoa)
                n_loi = sum(1 for j in d.get("viec", []) if "loi" in j)
                dong.append((tap, k.get("n_tat_ca"), k.get("n_bo"), k.get("dai_nhat_giu"),
                             k.get("ngan_sach_token_dich"), len(d.get("viec", [])), n_loi))
    L.append(bang(["Bộ", "Mẫu", "Bỏ", "Dài nhất", "Ngân sách token đích", "Số job", "Job lỗi"], dong))
    L.append("\n- Dry run dùng để chặn lỗi độ dài/ngạch trước khi tốn GPU; các bộ 4B/8B/Ecom dùng chung một bảng ngạch nên số giống nhau.")
    L.append("\n" + md_nguon([str(f.relative_to(FT)) for dt in s.get("dirs", []) for f in (FT / dt).rglob("quota_dry.json")]))
    return "\n".join(L)


RENDER = {
    "song_ngu": render_song_ngu,
    "b5": render_b5,
    "b6": render_b6,
    "b78": render_b78,
    "tl": render_tl,
    "do_test": render_do_test,
    "cheo": render_cheo,
    "dong6": render_dong6,
    "onmt": render_onmt,
    "moses": render_moses,
    "nam_mh": render_nam_mh,
    "quota": render_quota,
    "c0": render_c0,
    "demo": render_demo,
    "dry": render_dry,
}

DUNG = {
    "00_dich_du_lieu_song_ngu": "đầu vào của mọi bước fine-tune và Judge",
    "01_moi_truong_b5": "chốt cấu hình chạy (micro_bs, ngân sách, tốc độ)",
    "02_b6_do_sieu_tham_so": "chọn lr 2e-4, rank 64 cho mọi bước sau",
    "03_b78_pair_f0_cum_f1": "mốc pair fold 0 và cụm fold 1",
    "04_tl_ti_le_fold0": "chọn tỉ trọng title ~11%, bỏ 75%",
    "05_b8f1_fold1": "kiểm độc lập fold 1 (luật 3 fold)",
    "06_b8f2_fold2": "kiểm độc lập fold 2, đủ luật 3 fold cho cụm",
    "07_tl2_lr_va_pair_f1": "loại lr 1e-4; mốc pair fold 1",
    "08_tl3_pair_f2": "đủ luật 3 fold cho pair",
    "09_do_test_mot_lan": "điểm test duy nhất của bản fold 0",
    "10_cuoi_toan_pool": "adapter giao cho demo, không có điểm",
    "11_cheo_dang_pair_cum": "chọn dạng cho extension (pair hay cụm)",
    "12_dong6_toc_do_t4": "tính công suất máy chủ demo trên T4",
    "13_onmt_hoc_du": "dòng OpenNMT trong bảng năm mô hình",
    "14_moses_hai_huong": "dòng Moses trong bảng năm mô hình",
    "15_nam_mo_hinh": "so sánh 5 mô hình trên eval fold 0",
    "16_quota_screen_960": "bản thử đầu, bị lỗi bỏ mẫu — chỉ để đối chiếu",
    "17_quota_screen_2048": "bản sửa lỗi độ dài, 1 seed",
    "18_quota_step1_2seed": "chốt bảng natural (2 seed, metric 10/90)",
    "19_c0_baseline": "so candidate với adapter đang dùng",
    "20_demo_may_chu": "chọn Q5_K_M cho demo, kiểm 13 trang thật",
    "21_dry_run_ngach": "chặn lỗi độ dài/ngạch trước khi chạy GPU",
}
for _s in PHIEN:
    _s.setdefault("dung", DUNG.get(_s["slug"], "—"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in PHIEN:
        noi_dung = RENDER[s["kind"]](s)
        noi_dung += "\n\n---\n\n*Tài liệu sinh tự động từ file kết quả bằng `scripts/ft_train/xuat_ket_qua_kaggle.py`. "
        noi_dung += "Số lấy từ JSON gốc, không gõ tay; file gốc không nằm trong git.*\n"
        (OUT / f"{s['slug']}.md").write_text(noi_dung, encoding="utf-8")
        print("ghi", s["slug"], len(noi_dung))

    # INDEX
    L = ["# Mục lục kết quả Kaggle", "",
         "Mỗi phiên chạy thật một file. Các lần chạy khói/thử gom ở bảng cuối, không dùng làm kết quả.", "",
         "## 1. Các phiên có kết quả", "",
         "| # | Phiên | File | Ghi chú |", "|---|---|---|---|"]
    for s in PHIEN:
        so = s["slug"].split("_")[0]
        L.append(f"| {so} | {s['title']} | [`{s['slug']}.md`]({s['slug']}.md) | {s.get('ghi_chu', '')} |")
    L.append("")
    L.append("## 2. Các lần chạy khói / thử (không dùng làm kết quả)")
    L.append("")
    L.append("| Thư mục | Ghi chú |")
    L.append("|---|---|")
    da_co = {d for s in PHIEN for d in s.get("dirs", [])}
    for d in sorted([p for p in FT.glob("kaggle_*") if p.is_dir()]):
        if d.name in da_co:
            continue
        if "cuoi8b" in d.name:
            ghi = "8B bản cuối: T4 không đủ VRAM, dừng ở chốt chặn, không có kết quả"
        elif "thu" in d.name or "khoi" in d.name:
            ghi = "chạy khói/thử, không dùng làm kết quả"
        else:
            ghi = "thử/dò, không dùng làm kết quả"
        L.append(f"| `{d.name}` | {ghi} |")
    L.append("")
    (OUT / "INDEX.md").write_text("\n".join(L), encoding="utf-8")
    print("ghi INDEX")


if __name__ == "__main__":
    main()
