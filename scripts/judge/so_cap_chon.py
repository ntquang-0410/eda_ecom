"""SO CẶP MÙ: nhờ judge (mặc định deepseek-v4.1-flash, cấu hình ở cau_hinh.py) chọn bản tốt hơn giữa Gemma và Qwen.

Mỗi mục luôn được hỏi ĐỦ HAI THỨ TỰ; chỉ khi hai lượt cùng chọn MỘT mô hình thì mới có người thắng. Không có luật 'hòa thì lấy
bản nào đó' ở đây; việc xử lý các mục không có người thắng là quyết định riêng ở bước ráp dữ liệu.

  python so_cap_du_lieu.py                                  # dựng bảng mục (miễn phí, không gọi API)
  python so_cap_chon.py --loai title --thu 60 --tat-suy-luan --ten t60      # thí điểm, chia tầng
  python so_cap_chon.py --loai pair --thu 60 --tat-suy-luan --ten p60
  python so_cap_chon.py --loai title --margin-toi-da 0.05 --tat-ca --ten title_005
  python so_cap_chon.py --bao-cao t60                       # thống kê + kiểm tra thiên vị
Khóa API nhập ẩn bằng chay_judge.ps1 (xem README)."""
import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chay_judge as CJ  # noqa: E402
import so_cap_du_lieu as DL  # noqa: E402
import so_cap_gom as SG  # noqa: E402
import so_cap_rubric as SR  # noqa: E402

CH = CJ.CH
RA = DL.RA


# ---------------------------------------------------------------- chấm một mục (đủ hai thứ tự)
CHE_DO = "hai_luot"          # 'hai_luot' (mặc định: đủ hai thứ tự) hoặc 'mot_luot' (một lần, thứ tự ngẫu nhiên theo băm item_id)


def cham_muc(muc, khoa, goi, max_token=None):
    luot, tv, tr, giay, usd = [], 0, 0, 0.0, {"chi_phi": 0.0, "list": 0.0, "allow": 0.0}
    nguon_tinh = None
    thu_tu_can_hoi = (SR.thu_tu_bang_bam(muc["item_id"]),) if CHE_DO == "mot_luot" else ("gq", "qg")
    for thu_tu in thu_tu_can_hoi:
        prompt, anh_xa = SR.dung_nguoi_dung(muc, thu_tu)
        tn = [{"role": "system", "content": SR.he_thong(muc)}, {"role": "user", "content": prompt}]
        nd, v, r, kt, phu = goi(khoa, tn, max_token)
        tv, tr, giay = tv + v, tr + r, giay + (phu.get("giay") or 0)
        usd["chi_phi"] += phu.get("chi_phi") or 0.0
        usd["list"] += phu.get("list_cost") or 0.0
        usd["allow"] += phu.get("allowance") or 0.0
        nguon_tinh = phu.get("nguon") or nguon_tinh
        if kt == "length":
            return {"trang_thai": "loi", "ly_do": "bi_cat_het_token", "tokens_vao": tv, "tokens_ra": tr, **_usd(usd)}
        try:
            kq = SR.doc_ket_qua_hien_hanh(nd)
        except (ValueError, json.JSONDecodeError) as e:
            return {"trang_thai": "loi", "ly_do": f"doc_loi:{e}", "tra_loi_tho": nd[:300],
                    "tokens_vao": tv, "tokens_ra": tr, **_usd(usd)}
        chon = "hoa_that" if kq["winner"] == "tie" else anh_xa[kq["winner"]]
        ban = {"thu_tu": thu_tu, "vi_tri_chon": kq["winner"], "chon": chon, "tieu_chi": kq["criterion"], "ly_do": kq["reason"]}
        if "tuong_duong" in kq:                                         # rubric v2
            ban.update({"tuong_duong": kq["tuong_duong"], "so_loi_1": kq["so_loi_1"], "so_loi_2": kq["so_loi_2"]})
        luot.append(ban)
    ket_qua = luot[0]["chon"] if len(luot) == 1 else SR.gop(luot[0]["chon"], luot[1]["chon"])
    return {"trang_thai": "ok", "ket_qua": ket_qua, "luot": luot,
            "tokens_vao": tv, "tokens_ra": tr, "giay_goi": round(giay, 1), "nguon_tinh_phi": nguon_tinh, **_usd(usd)}


def _usd(u):
    return {"chi_phi_usd": u["chi_phi"], "list_cost_usd": u["list"], "allowance_usd": u["allow"]}


def ban_ghi(muc, kq):
    cho_tran = {k: muc.get(k) for k in ("item_id", "kind", "source", "huong", "category", "san_pham_n")}
    if pd.notna(muc.get("kiwi_g")):                  # title và đơn vị bộ pair có điểm Kiwi (chỉ để phân tích sau)
        cho_tran.update({"kiwi_g": muc.get("kiwi_g"), "kiwi_q": muc.get("kiwi_q"), "margin": muc.get("margin"),
                         "bin_margin": muc.get("bin_margin"), "ma_don_vi": muc.get("ma_don_vi")})
    cho_tran.update({"dai_gemma": len(muc["cand_gemma"]), "dai_qwen": len(muc["cand_qwen"]),
                     "mo_hinh": CH.MO_HINH, "nha_cung_cap": CH.NHA_CUNG_CAP, "phien_ban": SR.phien_ban_muc(muc),
                     "che_do": CHE_DO, "tham_so_them": dict(CJ._them) or None})
    if SR.co_ngu_canh(muc):                          # để so với lần chạy KHÔNG ngữ cảnh của cùng cặp (so_sanh_ngu_canh.py)
        cho_tran.update({"dai_ngu_canh": len(muc["ngu_canh"]), "id_khong_ngu_canh": muc.get("id_khong_ngu_canh")})
    if muc.get("tang") == 2:                         # tầng 2 của title / pair: ghi đè kết quả tầng 1 của CHÍNH mục này (ghep_bo.py)
        cho_tran["tang"] = 2
    return {**cho_tran, **kq}


def ids_khong_nhat_quan(ra, ten_dot):
    """item_id mà bản ghi ok CUỐI CÙNG (qua các đợt, đợt sau ghi đè đợt trước) có kết quả 'khong_nhat_quan'."""
    cuoi = {}
    for ten in ten_dot:
        for dong in open(Path(ra) / f"ket_qua_{ten}.jsonl", encoding="utf-8"):
            d = json.loads(dong)
            if d.get("trang_thai") == "ok":
                cuoi[d["item_id"]] = d["ket_qua"]
    return {i for i, k in cuoi.items() if k == "khong_nhat_quan"}


def da_xong(duong):
    xong = set()
    if Path(duong).exists():
        for dong in open(duong, encoding="utf-8"):
            try:
                d = json.loads(dong)
            except json.JSONDecodeError:
                continue
            if d.get("trang_thai") == "ok" and str(d.get("phien_ban", "")).startswith(SR.phien_ban_goc()) \
                    and d.get("che_do", "hai_luot") == CHE_DO:
                xong.add(d["item_id"])
    return xong


def chay(muc_df, ten, khoa, goi, ra=None, song_song=CH.SO_LUONG_SONG_SONG, tran_usd=None, tran_token=None,
         max_token=None, in_tien_do=True, gom=1):
    ra = Path(ra or RA)
    ra.mkdir(parents=True, exist_ok=True)
    duong = ra / f"ket_qua_{ten}.jsonl"
    xong = da_xong(duong)
    can = [{**r, "_thu_tu": i} for i, r in enumerate(muc_df.to_dict("records")) if r["item_id"] not in xong]
    if gom > 1 and not (CHE_DO == "mot_luot" and SR.RUBRIC == "v2"):
        raise ValueError("gom > 1 chỉ dùng với --rubric v2 --che-do mot_luot")
    print(f"[{ten}] tổng {len(muc_df)} | đã xong {len(xong)} | còn {len(can)} | mô hình {CH.MO_HINH} | {SR.phien_ban_goc()} | {CHE_DO}"
          + (f" | gom {gom}" if gom > 1 else ""))
    dung, khoa_ghi = threading.Event(), threading.Lock()
    dem = {"ok": 0, "loi": 0, "tv": 0, "tr": 0, "usd": 0.0, "list": 0.0, "allow": 0.0, "ngoai_le": None}
    t0 = time.time()

    def lam(nhom):
        """nhom: danh sách mục (1 mục ở chế độ thường, nhiều mục ở chế độ gom). Trả danh sách bản ghi."""
        if dung.is_set():
            return None
        try:
            kqs = SG.cham_gom(nhom, khoa, goi, max_token) if gom > 1 else [(nhom[0], cham_muc(nhom[0], khoa, goi, max_token))]
        except CJ.LoiNang as e:
            dem["ngoai_le"] = str(e)
            dung.set()
            return None
        except Exception as e:
            kqs = [(m, {"trang_thai": "loi", "ly_do": f"goi_loi:{str(e)[:200]}"}) for m in nhom]
        with khoa_ghi:
            for _, kq in kqs:
                dem["tv"] += kq.get("tokens_vao", 0)
                dem["tr"] += kq.get("tokens_ra", 0)
                dem["list"] += kq.get("list_cost_usd") or 0.0
                dem["allow"] += kq.get("allowance_usd") or 0.0
                d_ = kq.get("allowance_usd") or kq.get("list_cost_usd") or \
                    ((kq.get("tokens_vao", 0) * CH.GIA_VAO + kq.get("tokens_ra", 0) * CH.GIA_RA) / 1e6)
                dem["usd"] += max(kq.get("chi_phi_usd") or 0.0, d_)
            if tran_token and dem["tv"] + dem["tr"] >= tran_token and not dung.is_set():
                print(f"[{ten}] chạm trần token {tran_token}: dừng gửi thêm (chạy lại để tiếp tục).")
                dung.set()
            if tran_usd and dem["usd"] >= tran_usd and not dung.is_set():
                print(f"[{ten}] chạm trần chi phí {tran_usd} USD: dừng gửi thêm (chạy lại để tiếp tục).")
                dung.set()
        return [ban_ghi(m, kq) for m, kq in kqs]

    with open(duong, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=song_song) as ex:
        try:
            cong_viec = SG.chia_gom(can, gom) if gom > 1 else [[r] for r in can]
            futs = [ex.submit(lam, nhom) for nhom in cong_viec]
            xong_n = 0
            for fu in as_completed(futs):
                bs = fu.result()
                if bs is None:
                    continue
                with khoa_ghi:
                    for b in bs:
                        f.write(json.dumps(b, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                for b in bs:
                    dem["ok" if b["trang_thai"] == "ok" else "loi"] += 1
                    xong_n += 1
                    if in_tien_do and xong_n % 100 == 0:
                        print(f"  {xong_n}/{len(can)} | ok {dem['ok']} | lỗi {dem['loi']} | token {dem['tv'] + dem['tr']} | "
                              f"~{dem['usd']:.3f} USD | {time.time() - t0:.0f}s")
        except KeyboardInterrupt:
            print("Ctrl+C: dừng, kết quả đã ghi vẫn giữ nguyên.")
            dung.set()
            ex.shutdown(wait=False, cancel_futures=True)
    tom = {"ten": ten, "mo_hinh": CH.MO_HINH, "phien_ban": SR.phien_ban_goc(), "che_do": CHE_DO, "xong_truoc": len(xong), "ok_lan_nay": dem["ok"],
           "loi_lan_nay": dem["loi"], "tokens_vao": dem["tv"], "tokens_ra": dem["tr"], "list_cost_usd": round(dem["list"], 4),
           "allowance_usd": round(dem["allow"], 4), "chi_phi_dung_cho_tran_usd": round(dem["usd"], 4),
           "giay": round(time.time() - t0), "dung_do": dem["ngoai_le"]}
    (ra / f"tom_tat_{ten}.json").write_text(json.dumps(tom, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tom, ensure_ascii=False))
    if dem["ngoai_le"]:
        print("DỪNG VÌ LỖI NẶNG:", dem["ngoai_le"])
        return 2
    return 0


# ---------------------------------------------------------------- lấy mẫu chia tầng
def chon_mau(df, n, hat=CH.HAT_GIONG):
    """Chia ĐỀU cho các tầng (title: nguồn x mức margin Kiwi; pair: nguồn) để mỗi tầng đủ dòng, kể cả tầng margin nhỏ."""
    k = df.kind.iloc[0]
    cot = ["source", "bin_margin"] if k == "title" else (["source", "kind"] if k.startswith("unit_") else ["source"])
    nhom = list(df.groupby(cot))
    quota, du = divmod(n, len(nhom))
    out = []
    for i, (_, g) in enumerate(nhom):
        out.append(g.sample(n=min(len(g), quota + (1 if i < du else 0)), random_state=hat))
    return pd.concat(out).sort_values("item_id")


# ---------------------------------------------------------------- thống kê + kiểm tra thiên vị
def bao_cao(duong):
    cuoi = {}
    for dong in open(duong, encoding="utf-8"):
        d = json.loads(dong)
        cuoi[d["item_id"]] = d
    ok = [d for d in cuoi.values() if d["trang_thai"] == "ok"]
    n = len(ok)
    r = {"so_muc_ok": n, "so_muc_loi": len(cuoi) - n}
    if not n:
        return r
    kq = Counter(d["ket_qua"] for d in ok)
    r["ket_qua"] = {k: kq.get(k, 0) for k in ("gemma", "qwen", "hoa_that", "khong_nhat_quan")}
    r["ti_le"] = {k: round(v / n, 3) for k, v in r["ket_qua"].items()}
    cac_luot = [l for d in ok for l in d["luot"]]
    khong_hoa = [l for l in cac_luot if l["vi_tri_chon"] in (1, 2)]
    r["thien_vi_vi_tri"] = {"ti_le_chon_vi_tri_1": round(sum(l["vi_tri_chon"] == 1 for l in khong_hoa) / max(len(khong_hoa), 1), 3),
                            "ghi_chu": "gần 0,5 là tốt; lệch xa nghĩa là judge chọn theo vị trí"}
    qd = [d for d in ok if d["ket_qua"] in ("gemma", "qwen")]
    r["so_muc_co_nguoi_thang"] = len(qd)
    if qd:
        r["ti_le_gemma_trong_so_co_nguoi_thang"] = round(sum(d["ket_qua"] == "gemma" for d in qd) / len(qd), 3)
        r["theo_nguon"] = {s: {"co_nguoi_thang": sum(1 for d in qd if d["source"] == s),
                               "ti_le_gemma": round(sum(d["ket_qua"] == "gemma" for d in qd if d["source"] == s) /
                                                    max(sum(1 for d in qd if d["source"] == s), 1), 3)}
                           for s in sorted({d["source"] for d in qd})}
        dai = [d for d in qd if d["dai_gemma"] != d["dai_qwen"]]
        if dai:
            r["thien_vi_do_dai"] = {"ti_le_ban_dai_hon_thang": round(sum((d["dai_gemma"] > d["dai_qwen"]) == (d["ket_qua"] == "gemma")
                                                                       for d in dai) / len(dai), 3),
                                    "ghi_chu": "gần 0,5 là tốt (nền thực tế xem thêm ti_le_ban_dai_hon_la_gemma)",
                                    "ti_le_ban_dai_hon_la_gemma": round(sum(d["dai_gemma"] > d["dai_qwen"] for d in dai) / len(dai), 3)}
        r["tieu_chi_quyet_dinh"] = dict(Counter(d["luot"][0]["tieu_chi"] for d in qd))
        kt = [d for d in qd if d.get("kiwi_g") is not None and d["kiwi_g"] == d["kiwi_g"] and d.get("bin_margin")]
        if kt:
            r["trung_voi_cometkiwi"] = {"tong": round(sum((d["kiwi_g"] > d["kiwi_q"]) == (d["ket_qua"] == "gemma") for d in kt) / len(kt), 3),
                                        "theo_margin": {b: round(sum((d["kiwi_g"] > d["kiwi_q"]) == (d["ket_qua"] == "gemma")
                                                                    for d in kt if d["bin_margin"] == b) /
                                                                max(sum(1 for d in kt if d["bin_margin"] == b), 1), 3)
                                                        for b in sorted({d["bin_margin"] for d in kt})}}
    return r


# ---------------------------------------------------------------- bản giả để thử máy móc
_MAU = re.compile(r"Translation 1 \([A-Za-z]+\):\n(.*?)\n\nTranslation 2 \([A-Za-z]+\):\n(.*)$", re.S)


def goi_gia(khoa, tn, max_token=None):
    """Chọn bản dài hơn (chỉ để thử máy móc, không phải chất lượng thật)."""
    if "numbered ITEMS" in tn[0]["content"]:          # chế độ GOM: nhiều mục trong một lời gọi
        cap = re.findall(r"Translation 1 \([A-Za-z]+\): (.*)\nTranslation 2 \([A-Za-z]+\): (.*)", tn[-1]["content"])
        kq = [{"i": i, "d1": [], "d2": [], "eq": len(a) == len(b), "w": 1 if len(a) >= len(b) else 2, "c": "N"} for i, (a, b) in enumerate(cap, 1)]
        return json.dumps({"results": kq}), len(tn[-1]["content"]) // 3, 12 * len(kq), "stop", \
            {"chi_phi": 0.0, "list_cost": 0.0, "allowance": 0.0, "giay": 0.0}
    m = _MAU.search(tn[-1]["content"])
    a, b = m.group(1), m.group(2)
    if SR.RUBRIC == "v2":                            # rubric v2: không có 'tie'; hai bản dài bằng nhau = khai báo equivalent, chọn bản 1
        nd = json.dumps({"defects_1": [], "defects_2": [], "equivalent": len(a) == len(b), "winner": 1 if len(a) >= len(b) else 2,
                         "criterion": "fluency", "reason": "gia"})
        return nd, len(tn[-1]["content"]) // 3, 20, "stop", {"chi_phi": 0.0, "list_cost": 0.0, "allowance": 0.0, "giay": 0.0}
    w = "tie" if len(a) == len(b) else (1 if len(a) > len(b) else 2)
    nd = json.dumps({"winner": w, "criterion": "fluency", "reason": "gia"})
    return nd, len(tn[-1]["content"]) // 3, 20, "stop", {"chi_phi": 0.0, "list_cost": 0.0, "allowance": 0.0, "giay": 0.0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--thu", type=int, metavar="N")
    g.add_argument("--tat-ca", action="store_true")
    g.add_argument("--bao-cao", metavar="TEN")
    p.add_argument("--loai", choices=["title", "cum", "cum_ctx", "cum_lai", "pair", "title_ctx", "pair_ctx"],
                   help="title = tiêu đề (dùng chung hai bộ); cum = cặp (tên: giá trị) bộ CỤM, KHÔNG ngữ cảnh, gộp trùng theo cặp; "
                        "cum_ctx = (sản phẩm, cặp) bộ CỤM, CÓ ngữ cảnh sản phẩm; cum_lai = LAI (cặp dùng bởi 1 sản phẩm có ngữ cảnh, cặp dùng bởi >=2 sản phẩm không; dùng cho lần chạy chính); pair = đơn vị (tên/giá trị) bộ PAIR; "
                        "title_ctx / pair_ctx = TẦNG 2 của title / pair (cùng item_id với tầng 1, có ngữ cảnh; dùng với --chi-khong-nhat-quan)")
    p.add_argument("--chi-khong-nhat-quan", metavar="TEN[,TEN]",
                   help="chỉ chạy các mục mà kết quả TẦNG 1 (các đợt ket_qua_<TEN>.jsonl, bản ghi cuối của mỗi mục) là 'khong_nhat_quan'; "
                        "dùng với title_ctx / pair_ctx")
    p.add_argument("--theo-ctx", metavar="TEN",
                   help="chỉ với --loai cum: chỉ chạy các cặp (không ngữ cảnh) ứng với những mục trong kết quả cum_ctx TEN, để so hai cách")
    p.add_argument("--margin-toi-da", type=float, help="title: chỉ chấm các mục |kiwi_g - kiwi_q| nhỏ hơn số này")
    p.add_argument("--ten")
    p.add_argument("--song-song", type=int, default=CH.SO_LUONG_SONG_SONG)
    p.add_argument("--tran-usd", type=float)
    p.add_argument("--tran-token", type=int)
    p.add_argument("--max-token-ra", type=int)
    p.add_argument("--tat-suy-luan", action="store_true")
    p.add_argument("--rubric", choices=["v1", "v2"], default="v1",
                   help="v1 = lời nhắc cũ (có đáp án hòa); v2 = bộ quy tắc siết chặt (liệt kê lỗi, bắt buộc chọn 1 trong 2, cờ equivalent)")
    p.add_argument("--gom", type=int, default=1, metavar="K",
                   help="gom K mục vào MỘT lời gọi (chỉ với --rubric v2 --che-do mot_luot) để tiết kiệm token; mục có ngữ cảnh gom ít hơn (max(3, K//2))")
    p.add_argument("--top", type=int, metavar="N",
                   help="chỉ giữ N mục có san_pham_n (số sản phẩm dùng) cao nhất, xếp giảm dần: ưu tiên mục phủ nhiều dữ liệu nhất")
    p.add_argument("--che-do", choices=["hai_luot", "mot_luot"], default="hai_luot",
                   help="hai_luot = hỏi đủ hai thứ tự (mặc định); mot_luot = một lần, thứ tự ngẫu nhiên theo băm cố định của item_id")
    p.add_argument("--nha-cung-cap", choices=list(CH.CAC_NHA_CUNG_CAP), default="opencode")
    p.add_argument("--rpm", type=int)
    p.add_argument("--ra")
    p.add_argument("--vao", help="file parquet mục (mặc định muc_<loai>.parquet trong thư mục ra)")
    p.add_argument("--gia", action="store_true")
    a = p.parse_args(argv)
    ra = Path(a.ra) if a.ra else RA
    if a.bao_cao:
        txt = json.dumps(bao_cao(ra / f"ket_qua_{a.bao_cao}.jsonl"), ensure_ascii=False, indent=1)
        (ra / f"bao_cao_{a.bao_cao}.json").write_text(txt, encoding="utf-8")       # để Claude đọc thẳng từ file, khỏi dán từ terminal
        print(txt)
        return 0
    if not a.loai:
        p.error("cần --loai title|pair")
    CH.chon_nha_cung_cap(a.nha_cung_cap)
    SR.set_rubric(a.rubric)
    global CHE_DO
    CHE_DO = a.che_do
    if a.rpm:
        CH.RPM = a.rpm
    if a.tat_suy_luan:
        CJ._them.update(CH.THAM_SO_TAT_SUY_LUAN)
    df = pd.read_parquet(a.vao or RA / f"muc_{a.loai}.parquet")      # bảng mục luôn đọc từ thư mục chuẩn; --ra chỉ là nơi GHI kết quả
    if a.margin_toi_da is not None:
        df = df[df.margin.fillna(9.9) < a.margin_toi_da]
    if a.loai in ("title_ctx", "pair_ctx"):
        df = df.assign(tang=2)                       # ghi vào bản ghi để ghep_bo.py biết kết quả này thuộc tầng 2
    if a.chi_khong_nhat_quan:
        if a.loai not in ("title_ctx", "pair_ctx"):
            p.error("--chi-khong-nhat-quan chỉ đi với --loai title_ctx hoặc pair_ctx")
        df = df[df.item_id.isin(ids_khong_nhat_quan(ra, [t for t in a.chi_khong_nhat_quan.split(",") if t]))]
        if df.empty:
            p.error("không có mục 'khong_nhat_quan' nào ở tầng 1 trong các đợt đã cho")
    elif a.loai in ("title_ctx", "pair_ctx") and not a.gia:
        p.error("tầng 2 phải chạy kèm --chi-khong-nhat-quan (không chạy lại các mục tầng 1 đã quyết)")
    if a.theo_ctx:
        if a.loai != "cum":
            p.error("--theo-ctx chỉ đi với --loai cum")
        ids = {json.loads(x)["id_khong_ngu_canh"] for x in open(ra / f"ket_qua_{a.theo_ctx}.jsonl", encoding="utf-8")
               if json.loads(x).get("id_khong_ngu_canh")}
        df = df[df.item_id.isin(ids)]
        if df.empty:
            p.error(f"không có mục nào ứng với kết quả {a.theo_ctx}")
    if a.top:
        df = df.sort_values(["san_pham_n", "item_id"], ascending=[False, True], kind="stable").head(a.top)
    if a.thu:
        df, ten = chon_mau(df, a.thu), a.ten or f"{a.loai}_{a.thu}"
    else:
        ten = a.ten or f"{a.loai}_tat_ca"
    if a.gia:
        return chay(df, "gia_" + ten, "gia", goi_gia, ra, a.song_song, a.tran_usd, a.tran_token, a.max_token_ra, gom=a.gom)
    try:
        khoa = CJ.lay_khoa()
    except CJ.LoiNang as e:
        print(e)
        return 2
    return chay(df, ten, khoa, CJ.goi_api, ra, a.song_song, a.tran_usd, a.tran_token, a.max_token_ra, gom=a.gom)


if __name__ == "__main__":
    sys.exit(main())
