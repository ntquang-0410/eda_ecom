# -*- coding: utf-8 -*-
"""Bộ song ngữ 4 (Kaggle): hiệu đính CẢ CỤM thuộc tính của một sản phẩm 1688 (Trung -> Việt) trong MỘT lần gọi, theo cách v2 của bộ 3:
tên sản phẩm làm ngữ cảnh, mục KHOÁ (n_vi/v_vi) chỉ để đọc, mục có BẢN NHÁP (draft_n/draft_v, bản dịch đơn vị của bộ 1) thì giữ nguyên nếu
đúng, chỉ sửa khi ngữ cảnh cho thấy sai; mục không có bản nào thì dịch mới. Dùng lại phần nạp mô hình và gọi API của dich.py (cùng thư mục
trong Dataset).

Vào: cum_can_dich_zv.parquet (một dòng một cụm: muc_json, can_json = trường mô hình phải xuất). Mô hình trả về MỘT mảng JSON, mỗi mục cần
xuất một đối tượng {"i", "n"?, "v"?, "c"}. Kiểm cấu trúc: đọc được JSON, đủ mục, đủ trường, không rỗng, không trùng. Sai cấu trúc hoặc còn sót
chữ Hán thì hỏi lại MỘT lần (lượt hội thoại thứ hai), giữ bản tốt hơn. Ghi từng lô ra <ra>/<mo_hinh>/lo_*.parquet; chạy tiếp được (bỏ qua cụm
đã có, kể cả trong các thư mục --cu). Gọi API lỗi thì không ghi, lần sau dịch lại. Luật chặt từng trường làm ở laptop (song_ngu_4.ket_qua).
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dich as D  # noqa: E402

# v3 (25/10/2026, sau khi rà toàn bộ lần thử 200 sản phẩm): thêm bảng thuật ngữ cho các lỗi lặp lại của bản dịch bộ 1 mà Gemma không tự
# sửa (拉链 thành "ngăn kéo" ở 354 cặp, 防水台 thành "đế chống nước" 23, 揭阳 thành "Khiết/Triết Dương" 31; Gemma dịch mới cũng lặp lại
# 拉链 và 防水台, và dịch 跨境不出国内 ngược nghĩa), nói rõ hơn "không sửa chỉ vì văn phong" (lần thử có ~10 chỗ bỏ chữ "Màu", đổi
# "Ba năm" thành "3 năm") và dùng một cách dịch thống nhất trong một sản phẩm.
# v4 (25/10/2026, sau lần thử v3): v3 sửa đúng gần hết thuật ngữ cũ, bớt hẳn sửa vì văn phong, nhưng (1) cách dịch 跨境不出国内 có dấu
# phẩy nên trượt luật lech_so_muc và cặp bị bỏ: đổi sang cách dịch không dấu phẩy, cấm dấu ", " trong một mục; (2) Gemma giữ nguyên các
# lỗi dịch từng chữ lặp lại hàng loạt của bộ 1, đếm trên cả bộ: 独立站 "Trạm độc lập" 4.386 cặp, 低帮 "Thân thấp" 473, 单/双镜头
# "Đơn/Song thấu kính" 179, 小白鞋 "Giày trắng nhỏ" 213, 莓茶 "Trà dâu" 20, 无印风 "không nhãn hiệu" 8: đưa vào bảng thuật ngữ.
# v4b (25/10/2026, sau lần thử v4): chỉ đổi cách dịch 无印风 "Phong cách Muji" (trượt latin_moi + mat_viet nên lùi về nháp sai) thành
# "Phong cách tối giản Nhật"; test kiểm mọi cách dịch của bảng thuật ngữ qua đủ luật. Phần còn lại giữ đúng bản v4 đã chạy thử.
PHIEN_BAN_CAU_LENH = "cum-zh_vi-v4b-2026-10-25"
HAN = re.compile("[㐀-䶿一-鿿豈-﫿]")
DANH_MUC_VI = {"bags": "túi xách", "mother_baby": "mẹ và bé", "home": "đồ gia dụng", "shoes": "giày dép", "food": "thực phẩm",
               "fashion": "thời trang", "beauty": "mỹ phẩm, làm đẹp", "electronics": "đồ điện tử", "auto": "phụ kiện ô tô"}

HE_THONG = (
    "You are a professional e-commerce translator. You receive the attribute list of ONE product from the Chinese wholesale site "
    "1688.com and must produce its Vietnamese version, in the style of Vietnamese e-commerce listings (Shopee, Tiki).\n"
    "Input: the product title (Chinese, context only, never translate it) and a JSON array of items. Each item has \"i\" (item number), "
    "\"n\" (attribute name, Chinese) and \"v\" (attribute value, Chinese), and may also have:\n"
    "- \"n_vi\" / \"v_vi\": the FIXED Vietnamese translation of that field, already reviewed. Never output these fields and never change "
    "them; use them only as context. A very long fixed value is shown shortened, ending with \" …\".\n"
    "- \"draft_n\" / \"draft_v\": a DRAFT Vietnamese translation of that field, made WITHOUT seeing the rest of the product.\n"
    "Your task, for every field that is not fixed:\n"
    "- If it has a draft: check the draft against the Chinese and against the WHOLE product (the title and all other items). Keep the "
    "draft EXACTLY as it is when it is correct. Change it ONLY when: the context shows its meaning is wrong (for example a colour, size, "
    "part, flavour, material, age group or product type that the title or the other items make clear); it leaves Chinese characters or "
    "untranslated pinyin; it drops or adds meaning; or it uses \"cân\" as a unit for 斤 (1 斤 = 0.5 kg: convert to kg, for example "
    "2斤 -> 1 kg); or it contradicts the glossary below. Never rewrite a correct draft only to change the style or word choice: for "
    "example do not change \"Màu nâu, Màu đen\" to \"Nâu, Đen\", \"Ba năm\" to \"3 năm\" or \"Loại đóng gói cơ bản\" to \"Đóng gói cơ "
    "bản\".\n"
    "- If it has no draft: translate it.\n"
    "Output ONLY a JSON array, with no markdown, no code fence and no explanation. Output one object for each item that has at least one "
    "non-fixed field, in the same order: {\"i\": item number, \"n\": Vietnamese name (only when the name is not fixed), \"v\": Vietnamese "
    "value (only when the value is not fixed), \"c\": \"\" when you kept the draft(s) unchanged, otherwise a short English reason for the "
    "change}.\n"
    "Rules for every field you output:\n"
    "- Vietnamese only: no Chinese character and no pinyin for ordinary words. Chinese place names use their Vietnamese names "
    "(浙江 -> Chiết Giang, 义乌 -> Nghĩa Ô, 广州 -> Quảng Châu); never replace them with other places.\n"
    "- Keep numbers, units, sizes, model codes, Latin-letter brand names and bracket tokens such as [SĐT], [LIÊN HỆ], [URL] exactly.\n"
    "- A list separated by \", \" stays a list with the same number of items, in the same order, separated by \", \". Never put "
    "\", \" inside one item.\n"
    "- Do not put the attribute name or a label before a value unless the Chinese value itself starts with it. Do not add information "
    "that is not in the Chinese.\n"
    "- Within one product, translate the same Chinese term the same way every time.\n"
    "Glossary (always follow it, also when correcting drafts):\n"
    "- 拉链 = khóa kéo (a zipper), never \"ngăn kéo\" (a drawer): 拉链暗袋 -> Túi khóa kéo ẩn, 夹层拉链袋 -> Ngăn phụ có khóa kéo, "
    "拉链 (closure) -> Khóa kéo.\n"
    "- 防水台 (shoes) = đế độn mũi (a platform under the toe), never \"chống nước\".\n"
    "- 跨境 = xuyên biên giới (cross-border export): 跨境不出国内 -> Chỉ bán xuất khẩu.\n"
    "- 独立站 (a sales platform) = Website độc lập (an independent online shop), never \"Trạm độc lập\".\n"
    "- 低帮 / 中帮 / 高帮 (shoe collar height) = Cổ thấp / Cổ vừa / Cổ cao, never \"Thân thấp\" or \"Thân trung\".\n"
    "- 镜头 (camera) = ống kính, never \"thấu kính\": 单镜头 -> Một ống kính, 双镜头 -> Ống kính kép.\n"
    "- 小白鞋 = Giày trắng (white sneakers), never \"Giày trắng nhỏ\".\n"
    "- 莓茶 = Trà dây (vine tea), never \"Trà dâu\". 无印风 = Phong cách tối giản Nhật, never \"không nhãn hiệu\".\n"
    "- 揭阳 = Yết Dương; 纯色 = Màu trơn; 单排扣 = Một hàng khuy; 中腰 = Cạp vừa.\n"
    "Example input items (title: 婴儿硅胶奶嘴宽口径):\n"
    '[{"i": 1, "n": "品牌", "v": "贝来", "n_vi": "Thương hiệu", "v_vi": "Beilai"}, '
    '{"i": 2, "n": "颜色", "v": "透明", "n_vi": "Màu sắc", "draft_v": "Minh bạch"}, '
    '{"i": 3, "n": "适用年龄", "v": "0-6个月", "draft_n": "Độ tuổi phù hợp", "draft_v": "0-6 tháng"}]\n'
    "Example output:\n"
    '[{"i": 2, "v": "Trong suốt", "c": "透明 is a colour here: transparent, not honest"}, '
    '{"i": 3, "n": "Độ tuổi phù hợp", "v": "0-6 tháng", "c": ""}]')


def mo_ta_can(can):
    return "; ".join(f"item {i}: {', '.join(f)}" for i, f in can.items())


def cau_lenh(cat, tieu_de, muc_json, can_json, ten_khac="", phan=0, so_phan=1):
    can = json.loads(can_json)
    u = f"Product category: {cat} ({DANH_MUC_VI.get(cat, cat)}).\nProduct title (context only, do not translate): {tieu_de or '(none)'}\n"
    if so_phan > 1:
        u += (f"This is part {phan + 1} of {so_phan} of this product's attributes (the product is long). Attribute names in the other "
              f"parts (context only): {ten_khac}\n")
    u += "Items (JSON):\n" + json.dumps(json.loads(muc_json), ensure_ascii=False) + "\n"
    u += f"Output exactly {len(can)} objects, with exactly these fields besides \"i\" and \"c\": {mo_ta_can(can)}."
    return [{"role": "system", "content": HE_THONG}, {"role": "user", "content": u}]


def phan_tich(text, can):
    """can: {"3": ["v"], "13": ["n", "v"], ...}. Bóc câu trả lời thành {i: {"n"?, "v"?, "c"}} đủ mọi mục và trường phải xuất. Mục thừa (mục
    toàn khoá, số lạ) bị bỏ qua và đếm ở khoá "_thua". Trả về (dict, "") hoặc (None, lý do)."""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b <= a:
        return None, "khong_co_mang_json"
    try:
        x = json.loads(s[a:b + 1])
    except Exception:
        return None, "json_hong"
    if not isinstance(x, list):
        return None, "khong_phai_mang"
    out, thua = {}, 0
    for m in x:
        if not isinstance(m, dict):
            return None, "muc_khong_phai_doi_tuong"
        try:
            i = str(int(m.get("i")))
        except Exception:
            return None, "so_thu_tu_hong"
        if i not in can:
            thua += 1
            continue
        if i in out:
            return None, f"trung_muc_{i}"
        o = {"c": m.get("c") if isinstance(m.get("c"), str) else ""}
        for f in can[i]:
            g = m.get(f)
            if not isinstance(g, str) or not g.strip():
                return None, f"muc_{i}_thieu_{f}"
            o[f] = g.strip()
        out[i] = o
    thieu = [i for i in can if i not in out]
    if thieu:
        return None, f"thieu_muc_{thieu[0]}"
    out["_thua"] = thua
    return out, ""


def muc_sot_han(out, can):
    """Số thứ tự các mục mà trường phải xuất còn chữ Hán."""
    return [i for i in can if any(HAN.search(out[i][f]) for f in can[i])]


def hoi_lai(msgs, tra_loi, can, ly_do=None, sot=None):
    if ly_do:
        yeu_cau = (f"Your reply was not usable ({ly_do}). Reply again with ONLY the JSON array: exactly {len(can)} objects, with exactly "
                   f"these fields besides \"i\" and \"c\": {mo_ta_can(can)}. No text before or after the array.")
    else:
        yeu_cau = (f"Items {', '.join(sot)} still contain Chinese characters. Translate them fully into Vietnamese (no Chinese character). "
                   f"Reply again with the FULL JSON array: exactly {len(can)} objects, fields: {mo_ta_can(can)}.")
    return msgs + [{"role": "assistant", "content": tra_loi or ""}, {"role": "user", "content": yeu_cau}]


def so_token_ra(so_ky_tu_nhap, so_ky_tu_can, so_truong, tran):
    return int(min(tran, 512 + 2 * so_ky_tu_nhap + 4 * so_ky_tu_can + 40 * so_truong))


def dich_lo(be, lo, lap, tran):
    msgs = [cau_lenh(c, t, m, cj, k, p, s) for c, t, m, cj, k, p, s in
            zip(lo.category, lo.tieu_de, lo.muc_json, lo.can_json, lo.ten_khac, lo.phan, lo.so_phan)]
    can = [json.loads(x) for x in lo.can_json]
    toi_da = [so_token_ra(n, k, t, tran) for n, k, t in zip(lo.so_ky_tu_nhap, lo.so_ky_tu_can, lo.so_truong_can)]
    ra = [None] * len(lo)
    chay = [j for j in range(len(lo)) if can[j]]
    for j in range(len(lo)):
        if not can[j]:      # cụm toàn mục khoá: không gọi mô hình
            ra[j] = {"tho": "", "ly_do_dung": "khong_can", "muc": {"_thua": 0}, "ly_do": "", "sot": [], "hoi_lai": False, "sua_duoc": False}
    kq1 = be.sinh([msgs[j] for j in chay], [toi_da[j] for j in chay], lap) if chay else []
    for j, (x, r) in zip(chay, kq1):
        x = D.don_dep(x)
        out, ly = phan_tich(x, can[j]) if r not in ("loi_api", "qua_dai") else (None, r)
        ra[j] = {"tho": x, "ly_do_dung": r, "muc": out, "ly_do": ly, "sot": muc_sot_han(out, can[j]) if out else [],
                 "hoi_lai": False, "sua_duoc": False}
    lai = [j for j in chay if ra[j]["ly_do_dung"] not in ("loi_api", "qua_dai") and (ra[j]["muc"] is None or ra[j]["sot"])]
    if lai:
        kq2 = be.sinh([hoi_lai(msgs[j], ra[j]["tho"], can[j], ra[j]["ly_do"] or None, ra[j]["sot"]) for j in lai],
                      [toi_da[j] for j in lai], lap)
        for j, (x2, r2) in zip(lai, kq2):
            ra[j]["hoi_lai"] = True
            x2 = D.don_dep(x2)
            if r2 in ("loi_api", "qua_dai"):
                continue
            out2, _ = phan_tich(x2, can[j])
            if out2 is None:
                continue
            sot2 = muc_sot_han(out2, can[j])
            if ra[j]["muc"] is None or len(sot2) < len(ra[j]["sot"]):
                ra[j].update(tho=x2, ly_do_dung=r2, muc=out2, ly_do="", sot=sot2, sua_duoc=True)
    return ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", required=True, help="cum_can_dich_zv.parquet")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--mo_hinh", required=True, choices=list(D.MO_HINH))
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf", "gia"])
    ap.add_argument("--chi_mau_thu", action="store_true", help="chỉ dịch các cụm có mau_thu = True (200 sản phẩm thử)")
    ap.add_argument("--song_song", type=int, default=None)
    ap.add_argument("--api_base_url", default="")
    ap.add_argument("--api_model", default="")
    ap.add_argument("--api_key_env", default="")
    ap.add_argument("--api_extra", default="")
    ap.add_argument("--rpm", type=int, default=0)
    ap.add_argument("--toi_da_yeu_cau", type=int, default=0)
    ap.add_argument("--ngan_sach_usd", type=float, default=0.0)
    ap.add_argument("--gio_toi_da", type=float, default=8.0)
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--cu", action="append", default=[], help="thư mục đầu ra của phiên bản trước")
    a = ap.parse_args()
    t0 = time.time()
    cfg = D.MO_HINH[a.mo_hinh]
    if "api" in cfg:
        if a.api_base_url.strip():
            cfg = dict(cfg, base_url=a.api_base_url.strip())
        if a.api_model.strip():
            cfg = dict(cfg, api=a.api_model.strip())
        if a.api_key_env.strip():
            cfg = dict(cfg, key_env=a.api_key_env.strip())
    ra = Path(a.ra) / a.mo_hinh
    ra.mkdir(parents=True, exist_ok=True)
    CUM = pd.read_parquet(a.vao)
    assert "can_json" in CUM.columns and "so_ky_tu_can" in CUM.columns, "file vào không phải cum_can_dich_zv.parquet của bộ 4"
    if a.chi_mau_thu:
        CUM = CUM[CUM.mau_thu]
    for k, cu in enumerate(a.cu):
        for f in sorted(Path(cu).glob(f"{a.mo_hinh}/lo_*.parquet")):
            dich = ra / f"lo_cu{k}_{f.name}"
            if not dich.exists():
                shutil.copy2(f, dich)
    da = set()
    for f in ra.glob("lo_*.parquet"):
        da |= set(pd.read_parquet(f, columns=["ma_cum"]).ma_cum)
    con = CUM[~CUM.ma_cum.isin(da)].sort_values("trong_so", kind="stable").reset_index(drop=True)
    D.log(f"{a.mo_hinh}: tổng {len(CUM):,} cụm, đã có {len(CUM) - len(con):,}, còn {len(con):,} | câu lệnh {PHIEN_BAN_CAU_LENH}")
    the = {"mo_hinh": a.mo_hinh, "phien_ban_cau_lenh": PHIEN_BAN_CAU_LENH, "huong": "zh_vi", "loai": "cum_zv",
           "bat_dau": time.strftime("%Y-%m-%d %H:%M:%S"), "tong": int(len(CUM)), "da_co_truoc": int(len(CUM) - len(con)),
           "chi_mau_thu": a.chi_mau_thu, "tham_so": {"temperature": 0.0, "repetition_penalty": cfg["lap"],
                                                    "max_tokens": "min(tran, 512 + 2*ky_tu_nhap + 4*ky_tu_trung_can + 40*so_truong)",
                                                    "hoi_lai": 1}}
    f_the = Path(a.ra) / f"the_{a.mo_hinh}.json"
    if len(con) == 0:
        the.update(ket_thuc=time.strftime("%Y-%m-%d %H:%M:%S"), ghi_chu="đã đủ")
        f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    if cfg.get("gia_usd_1m") and a.backend != "gia" and a.ngan_sach_usd <= 0:
        raise SystemExit(f"{a.mo_hinh} là mô hình có giá: bắt buộc đặt --ngan_sach_usd.")
    song_song = a.song_song if a.song_song is not None else int(cfg.get("song_song", 8))
    be, ten_be, repo, loi_vllm = D.tai_backend(cfg, a.backend, song_song, json.loads(a.api_extra) if a.api_extra.strip() else None)
    if hasattr(be, "tk"):
        if cfg.get("gia_usd_1m"):
            be.ngan_sach = a.ngan_sach_usd
        be.toi_da_goi = a.toi_da_yeu_cau
        be.rpm = a.rpm
        be.han_chot = t0 + a.gio_toi_da * 3600
    tran = D.TOKEN_RA_API if ten_be == "api" else D.DO_DAI_NGU_CANH
    sha = {"gia": "gia", "api": "api (không có mã phiên bản; ghi ngày gọi ở bat_dau)"}.get(ten_be) or D.sha_mo_hinh(repo)
    the.update(backend=ten_be, repo=repo, sha_repo=sha, loi_vllm=loi_vllm, phien_ban=be.phien_ban,
               rpm=a.rpm, toi_da_yeu_cau=a.toi_da_yeu_cau, ngan_sach_usd=a.ngan_sach_usd)
    D.log("Nạp xong sau", round(time.time() - t0), "giây; backend", ten_be)
    so = [int(m.group(1)) for f in ra.glob("lo_*.parquet") for m in [re.fullmatch(r"lo_(\d+)", f.stem)] if m]
    so_lo = max(so) + 1 if so else 0
    dem = {"xong": 0, "loi_cau_truc": 0, "con_sot_han": 0, "hoi_lai": 0, "sua_duoc": 0, "loi_api": 0}
    t_dich, het_gio = time.time(), False
    for i in range(0, len(con), a.lo):
        if time.time() - t0 > a.gio_toi_da * 3600:
            het_gio = True
            D.log("Hết giờ cho phép, dừng êm để lưu đầu ra")
            break
        lo = con.iloc[i:i + a.lo]
        kq = dich_lo(be, lo, cfg["lap"], tran)
        ok = [k["ly_do_dung"] != "loi_api" for k in kq]
        dem["loi_api"] += len(ok) - sum(ok)
        if not any(ok):
            the["dung_vi_loi_api"] = getattr(be, "tk", {}).get("loi_cuoi", "") or "cả lô lỗi API"
            D.log("Cả lô đều lỗi API, dừng:", the["dung_vi_loi_api"])
            break
        df = pd.DataFrame({"ma_cum": lo.ma_cum.values,
                           "ket_qua": [json.dumps(k["muc"], ensure_ascii=False) if k["muc"] is not None else "" for k in kq],
                           "trang_thai": ["ok" if k["muc"] is not None else "loi_cau_truc" for k in kq],
                           "ly_do": [k["ly_do"] for k in kq], "so_muc_sot_han": [len(k["sot"]) for k in kq],
                           "hoi_lai": [k["hoi_lai"] for k in kq], "sua_duoc": [k["sua_duoc"] for k in kq],
                           "ly_do_dung": [k["ly_do_dung"] for k in kq],
                           "tho": [k["tho"][:20000] for k in kq], "backend": ten_be})[ok]
        tam = ra / f".tam_{so_lo:05d}.parquet"
        df.to_parquet(tam, index=False)
        tam.rename(ra / f"lo_{so_lo:05d}.parquet")
        so_lo += 1
        dem["xong"] += len(df)
        dem["loi_cau_truc"] += int((df.trang_thai != "ok").sum())
        dem["con_sot_han"] += int((df.so_muc_sot_han > 0).sum())
        dem["hoi_lai"] += int(df.hoi_lai.sum())
        dem["sua_duoc"] += int(df.sua_duoc.sum())
        td = dem["xong"] / max(time.time() - t_dich, 1)
        D.log(f"{a.mo_hinh}: xong {dem['xong']:,}/{len(con):,} cụm ({td:.2f} cụm/giây) | sai cấu trúc {dem['loi_cau_truc']} | "
              f"còn sót Hán {dem['con_sot_han']} | hỏi lại {dem['hoi_lai']}, sửa được {dem['sua_duoc']} | lỗi API {dem['loi_api']}")
        the.update(dem=dem, toc_do_cum_giay=round(td, 3))
        if hasattr(be, "tk"):
            the["api"] = be.tk
        f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    the.update(ket_thuc=time.strftime("%Y-%m-%d %H:%M:%S"), dung_vi_het_gio=het_gio, dem=dem,
               con_lai=int(len(con) - dem["xong"]), giay=round(time.time() - t0))
    if hasattr(be, "tk"):
        the["api"] = be.tk
    f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    D.log("Kết thúc", a.mo_hinh, "| còn lại", len(con) - dem["xong"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
