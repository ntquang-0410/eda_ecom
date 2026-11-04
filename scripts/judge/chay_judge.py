"""Chạy Claude-style judge (mô hình cố định: xem cau_hinh.MO_HINH) trên bản cụm bilingual_official_context.

Chỉ dùng thư viện chuẩn + pandas/pyarrow đã có trong .venv. KHÔNG bao giờ in hoặc ghi khóa API.

Ví dụ (chạy bằng chay_judge.ps1 để nhập khóa ẩn, hoặc tự đặt biến môi trường OPENCODE_API_KEY):
  python chay_judge.py --kiem-tra              # thử địa chỉ + tên mô hình + 1 lời gọi nhỏ
  python chay_judge.py --chi-luat              # kiểm tra quy tắc cứng trên cả 26.038 dòng, miễn phí
  python chay_judge.py --thu 300               # thí điểm 300 sản phẩm chia tầng
  python chay_judge.py --tat-ca                # toàn bộ 26.038 dòng (chạy tiếp được nếu đứt)
"""
import argparse
import getpass
import json
import os
import random
import sys
import threading
import time
import urllib.error
import uuid
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cau_hinh as CH  # noqa: E402
import luat_kiem  # noqa: E402
import rubric  # noqa: E402


class LoiNang(Exception):
    """Lỗi không thử lại được (khóa sai, sai địa chỉ...): dừng cả đợt chạy."""


# ---------------------------------------------------------------- khóa + gọi API
def lay_khoa():
    khoa = os.environ.get(CH.TEN_BIEN_KHOA, "").strip()
    if not khoa and sys.stdin is not None and sys.stdin.isatty():
        khoa = getpass.getpass(f"Dán khóa {CH.TEN_BIEN_KHOA} (gõ ẩn, không hiện): ").strip()
    if not khoa:
        raise LoiNang(f"Chưa có khóa. Đặt biến môi trường {CH.TEN_BIEN_KHOA} hoặc chạy chay_judge.ps1.")
    return khoa


_bo_response_format = threading.Event()
_them = {}  # tham số thêm vào thân yêu cầu (thử nghiệm, đặt bằng --extra-body); rỗng = không thêm
_tho = {}   # phản hồi thô của lời gọi gần nhất, chỉ để --kiem-tra in ra cho dễ chẩn đoán (không chứa khóa)
# Docs OpenCode Go yêu cầu client tự nhận diện bằng user agent riêng và gửi x-opencode-session ổn định (hỗ trợ định tuyến + cache).
_PHIEN = uuid.uuid4().hex
_khoa_rpm, _tiep_theo = threading.Lock(), [0.0]


def _tieu_de(khoa):
    h = {"Authorization": f"Bearer {khoa}", "Content-Type": "application/json", "User-Agent": "ecom-judge/1.0"}
    if CH.GUI_PHIEN_OPENCODE:
        h["x-opencode-session"] = _PHIEN
    return h


def _cho_luot():
    """Giữ số lời gọi không vượt CH.RPM mỗi phút (chung cho mọi luồng)."""
    if not CH.RPM:
        return
    with _khoa_rpm:
        bay_gio = time.monotonic()
        cho = max(0.0, _tiep_theo[0] - bay_gio)
        _tiep_theo[0] = max(bay_gio, _tiep_theo[0]) + 60.0 / CH.RPM
    if cho:
        time.sleep(cho)


def goi_api(khoa, thong_diep, max_token=None, url=None):
    """Một lời gọi chat completions. Trả (noi_dung, tokens_vao, tokens_ra, finish_reason, phu).

    phu = {"chi_phi": usage.cost (USD, theo trang docs v2 của OpenCode), "nguon": opencode.billing_source}."""
    url = (url or CH.BASE_URL) + "/chat/completions"
    thu_lai = 0
    bat_dau = time.monotonic()
    while True:
        _cho_luot()
        body = {"model": CH.MO_HINH, "messages": thong_diep, "temperature": CH.NHIET_DO,
                "max_tokens": max_token or CH.MAX_TOKEN_RA}
        if not _bo_response_format.is_set():
            body["response_format"] = {"type": "json_object"}
        body.update(_them)
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), method="POST", headers=_tieu_de(khoa))
        try:
            with urllib.request.urlopen(req, timeout=CH.THOI_GIAN_CHO) as r:
                tho = r.read().decode("utf-8")
            _tho["cuoi"] = tho
            d = json.loads(tho)
            ch = d["choices"][0]
            u = d.get("usage") or {}
            msg = ch.get("message") or {}
            # NVIDIA (30/10): với response_format json_object, message.content = null còn JSON nằm trong reasoning_content.
            # Ưu tiên content; nếu rỗng thì lấy reasoning_content (rubric.doc_ket_qua vẫn kiểm lược đồ nên không lấy nhầm lời suy nghĩ).
            noi, tu = (msg.get("content") or ""), "content"
            if not noi.strip() and msg.get("reasoning_content"):
                noi, tu = str(msg["reasoning_content"]), "reasoning_content"
            op = u.get("opencode") or d.get("opencode") or {}     # đo 30/10: cổng Go đặt trong usage.opencode
            phu = {"chi_phi": u.get("cost"), "nguon": op.get("billing_source"),
                   "list_cost": op.get("list_cost"), "allowance": op.get("allowance"),
                   "suy_luan": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "tu": tu,
                   "giay": round(time.monotonic() - bat_dau, 1), "so_lan_thu": thu_lai + 1}   # tổng thời gian kể cả chờ thử lại
            return (noi, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), ch.get("finish_reason"), phu)
        except urllib.error.HTTPError as e:
            noi = e.read().decode("utf-8", "replace")[:300]
            if e.code in (401, 403):
                goi_y = (" Nếu dùng gói Go ngoài Âu/Mỹ: vào Console > Settings > Privacy > Regions, chọn Global (cần chủ/quản trị)."
                         if CH.NHA_CUNG_CAP == "opencode" else "")
                raise LoiNang(f"HTTP {e.code} ({CH.NHA_CUNG_CAP}): khóa không hợp lệ hoặc không có quyền với mô hình này. "
                              f"Kiểm tra khóa dán đúng nhà cung cấp ({CH.TEN_BIEN_KHOA}).{goi_y} {noi}")
            if e.code == 402:
                goi_y = (" Nếu có gói Go: kiểm tra BASE_URL có đoạn /go/ (đường Go), hoặc đã hết hạn mức Go "
                         "(bật Extra Usage ở trang Go để dùng số dư khi hết hạn mức)." if CH.NHA_CUNG_CAP == "opencode" else "")
                raise LoiNang(f"HTTP 402 ({CH.NHA_CUNG_CAP}): tài khoản hết số dư/tín dụng.{goi_y} {noi}")
            if e.code == 404:
                raise LoiNang(f"HTTP 404 tại {url}: sai địa chỉ hoặc tên mô hình. Chạy --kiem-tra. {noi}")
            if e.code == 400 and "response_format" in noi.lower() and not _bo_response_format.is_set():
                _bo_response_format.set()
                continue
            if e.code in (429, 500, 502, 503, 504) and thu_lai < CH.SO_LAN_THU_LAI:
                cho = float(e.headers.get("Retry-After") or 0) or min(2 ** thu_lai * 2, 60)
                thu_lai += 1
                time.sleep(cho + random.random())
                continue
            raise RuntimeError(f"HTTP {e.code}: {noi}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if thu_lai < CH.SO_LAN_THU_LAI:
                thu_lai += 1
                time.sleep(min(2 ** thu_lai * 2, 60) + random.random())
                continue
            raise RuntimeError(f"mạng: {e}")


def goi_gia(khoa, thong_diep, max_token=None, url=None):
    """Bản giả chỉ để thử máy móc (--gia): điểm theo độ dài, không phải chất lượng thật."""
    nd = thong_diep[-1]["content"]
    mau = 5 if len(nd) % 7 else 3
    kq = {k: 5 for k in CH.CHIEU}
    kq["so_don_vi"] = mau
    kq["ket_luan"] = "dat" if mau == 5 else "can_xem"
    kq["loi"] = [] if mau == 5 else [{"loai": "so_don_vi", "doan_nguon": "x", "doan_dich": "y", "goi_y": "z"}]
    return json.dumps(kq), len(nd) // 3, 60, "stop", {"chi_phi": 0.0, "nguon": "gia"}


# ---------------------------------------------------------------- dữ liệu
def doc_du_lieu(duong_dan=None):
    import pyarrow.parquet as pq
    cot = ["product_id", "source", "category", "title_zh", "title_vi",
           "description_zh", "description_vi", "desc_nguon_dich"]
    df = pq.read_table(duong_dan or CH.FILE_VAO, columns=cot).to_pandas()
    for c in cot:
        df[c] = df[c].astype(object).where(df[c].notna(), "")
    return df


def chon_mau(df, n, hat=CH.HAT_GIONG):
    """Mẫu chia tầng theo (source, category, desc_nguon_dich), tỉ lệ thuận kích thước tầng."""
    rng = random.Random(hat)
    tang = {k: list(g.index) for k, g in df.groupby(["source", "category", "desc_nguon_dich"])}
    tong = len(df)
    phan = {k: n * len(v) / tong for k, v in tang.items()}
    nguyen = {k: int(p) for k, p in phan.items()}
    du = n - sum(nguyen.values())
    for k in sorted(phan, key=lambda k: phan[k] - nguyen[k], reverse=True)[:du]:
        nguyen[k] += 1
    chon = []
    for k, v in sorted(tang.items()):
        rng.shuffle(v)
        chon += v[:nguyen[k]]
    return df.loc[sorted(chon)]


# ---------------------------------------------------------------- chấm một sản phẩm
def cham_mot(hang, khoa, goi, max_token, doi_tuong="mo_ta"):
    he_thong, dung, co_luat = rubric.doi_tuong(doi_tuong)
    tn = [{"role": "system", "content": he_thong}, {"role": "user", "content": dung(hang)}]
    nd, tv, tr, kt, phu = goi(khoa, tn, max_token)
    ban = {"product_id": hang["product_id"], "doi_tuong": doi_tuong, "source": hang["source"], "category": hang["category"],
           "desc_nguon_dich": hang["desc_nguon_dich"], "mo_hinh": CH.MO_HINH, "nha_cung_cap": CH.NHA_CUNG_CAP,
           "rubric": rubric.phien_ban(), "tokens_vao": tv, "tokens_ra": tr,
           "chi_phi_usd": phu.get("chi_phi"), "nguon_tinh_phi": phu.get("nguon"),
           "suy_luan_tokens": phu.get("suy_luan"), "tham_so_them": dict(_them) or None,
           "noi_dung_tu": phu.get("tu"), "json_mode": not _bo_response_format.is_set(),
           "giay_goi": phu.get("giay"), "so_lan_thu": phu.get("so_lan_thu"),
           "list_cost_usd": phu.get("list_cost"), "allowance_usd": phu.get("allowance"),
           "chi_phi_niem_yet_uoc_usd": round((tv * CH.GIA_VAO + tr * CH.GIA_RA) / 1e6, 6)}
    if kt == "length":
        return {**ban, "trang_thai": "loi", "ly_do": "bi_cat_het_token"}
    try:
        kq = rubric.doc_ket_qua(nd)
    except (ValueError, json.JSONDecodeError) as e:
        return {**ban, "trang_thai": "loi", "ly_do": f"doc_loi:{e}", "tra_loi_tho": nd[:500]}
    cam = kq["ket_luan"] != "dat" or min(kq[k] for k in CH.CHIEU) <= CH.NGUONG_CO
    return {**ban, "trang_thai": "ok", **kq, "co_cam_co": cam, "luat": luat_kiem.kiem(hang) if co_luat else None}


def da_xong(duong_dan):
    xong = set()
    if Path(duong_dan).exists():
        with open(duong_dan, encoding="utf-8") as f:
            for dong in f:
                try:
                    d = json.loads(dong)
                except json.JSONDecodeError:
                    continue
                if d.get("trang_thai") == "ok" and d.get("rubric") == rubric.phien_ban():   # đổi rubric thì chấm lại
                    xong.add(d["product_id"])
    return xong


def chay(df, ten, khoa, goi, ra=None, song_song=CH.SO_LUONG_SONG_SONG, tran_token=None,
         max_token=None, in_tien_do=True, tran_usd=None, doi_tuong="mo_ta"):
    ra = Path(ra or CH.THU_MUC_RA)
    ra.mkdir(parents=True, exist_ok=True)
    duong = ra / f"ket_qua_{ten}.jsonl"
    xong = da_xong(duong)
    can = [r for r in df.to_dict("records") if r["product_id"] not in xong]
    print(f"[{ten}] tổng {len(df)} | đã xong {len(xong)} | còn {len(can)} | mô hình {CH.MO_HINH} | rubric {rubric.phien_ban()}")
    dung, khoa_ghi = threading.Event(), threading.Lock()
    dem = {"ok": 0, "loi": 0, "co": 0, "tv": 0, "tr": 0, "usd": 0.0, "usd_bao": 0.0, "list": 0.0, "allow": 0.0, "ngoai_le": None}
    t0 = time.time()

    def lam(r):
        if dung.is_set():
            return None
        try:
            kq = cham_mot(r, khoa, goi, max_token, doi_tuong)
            with khoa_ghi:                      # tính token ngay trong luồng chấm để trần có hiệu lực tức thì
                dem["tv"] += kq.get("tokens_vao", 0)
                dem["tr"] += kq.get("tokens_ra", 0)
                dem["usd_bao"] += kq.get("chi_phi_usd") or 0.0
                dem["list"] += kq.get("list_cost_usd") or 0.0
                dem["allow"] += kq.get("allowance_usd") or 0.0
                # đại lượng dùng cho trần: allowance (phần tính vào hạn mức Go) nếu có, không thì list_cost, không thì ước từ token
                dung_cho_tran = kq.get("allowance_usd")
                if dung_cho_tran is None:
                    dung_cho_tran = kq.get("list_cost_usd")
                if dung_cho_tran is None:
                    dung_cho_tran = kq.get("chi_phi_niem_yet_uoc_usd") or 0.0
                dem["usd"] += max(kq.get("chi_phi_usd") or 0.0, dung_cho_tran)
                if tran_token and dem["tv"] + dem["tr"] >= tran_token and not dung.is_set():
                    print(f"[{ten}] chạm trần token {tran_token}: dừng gửi thêm (chạy lại để tiếp tục).")
                    dung.set()
                if tran_usd and dem["usd"] >= tran_usd and not dung.is_set():
                    print(f"[{ten}] chạm trần chi phí {tran_usd} USD (allowance, hoặc list_cost, hoặc ước từ token): dừng gửi thêm.")
                    dung.set()
            return kq
        except LoiNang as e:
            dem["ngoai_le"] = str(e)
            dung.set()
            return None
        except Exception as e:  # lỗi tạm thời đã hết lần thử: ghi lại, lần sau chạy tiếp sẽ thử lại
            return {"product_id": r["product_id"], "trang_thai": "loi", "ly_do": f"goi_loi:{str(e)[:200]}"}

    with open(duong, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=song_song) as ex:
        try:
            futs = [ex.submit(lam, r) for r in can]
            for i, fu in enumerate(as_completed(futs), 1):
                kq = fu.result()
                if kq is None:
                    continue
                with khoa_ghi:
                    f.write(json.dumps(kq, ensure_ascii=False) + "\n")
                    f.flush()
                dem["ok" if kq["trang_thai"] == "ok" else "loi"] += 1
                dem["co"] += bool(kq.get("co_cam_co"))
                if in_tien_do and i % 50 == 0:
                    print(f"  {i}/{len(can)} | ok {dem['ok']} | lỗi {dem['loi']} | cờ {dem['co']} | "
                          f"token {dem['tv'] + dem['tr']} | {time.time() - t0:.0f}s")
        except KeyboardInterrupt:
            print("Ctrl+C: dừng, kết quả đã ghi vẫn giữ nguyên.")
            dung.set()
            ex.shutdown(wait=False, cancel_futures=True)
    tom = {"ten": ten, "doi_tuong": doi_tuong, "mo_hinh": CH.MO_HINH, "rubric": rubric.phien_ban(), "xong_truoc": len(xong),
           "ok_lan_nay": dem["ok"], "loi_lan_nay": dem["loi"], "cam_co_lan_nay": dem["co"],
           "tokens_vao": dem["tv"], "tokens_ra": dem["tr"], "chi_phi_usd": round(dem["usd_bao"], 4),
           "list_cost_usd": round(dem["list"], 4), "allowance_usd": round(dem["allow"], 4),
           "chi_phi_dung_cho_tran_usd": round(dem["usd"], 4),
           "giay": round(time.time() - t0),
           "dung_do": dem["ngoai_le"]}
    (ra / f"tom_tat_{ten}.json").write_text(json.dumps(tom, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tom, ensure_ascii=False))
    if dem["ngoai_le"]:
        print("DỪNG VÌ LỖI NẶNG:", dem["ngoai_le"])
        return 2
    return 0


# ---------------------------------------------------------------- các lệnh phụ
def kiem_tra(khoa):
    """Thử địa chỉ + tên mô hình + một lời gọi rất nhỏ. Không in khóa."""
    ung_vien = []
    cs = CH.BASE_URL
    if cs.endswith("/openai/v1"):
        ung_vien.append(cs[:-len("/openai/v1")] + "/v1/models")
    ung_vien.append(cs + "/models")
    thay = None
    for u in ung_vien:
        try:
            req = urllib.request.Request(u, headers=_tieu_de(khoa))
            with urllib.request.urlopen(req, timeout=30) as r:
                ds = [m.get("id") for m in json.loads(r.read().decode("utf-8")).get("data", [])]
            print(f"GET {u} -> {len(ds)} mô hình")
            deep = [m for m in ds if m and "deepseek" in m.lower()]
            print("  mô hình DeepSeek có trong danh sách:", deep)
            thay = CH.MO_HINH in ds
            print(f"  '{CH.MO_HINH}' {'CÓ' if thay else 'KHÔNG CÓ'} trong danh sách")
            break
        except urllib.error.HTTPError as e:
            print(f"GET {u} -> HTTP {e.code}")
        except Exception as e:
            print(f"GET {u} -> {type(e).__name__}")
    print("Đang gọi thử 1 lời gọi chat (chờ tối đa ~60 giây, thử lại 1 lần)...", flush=True)
    cu = (CH.THOI_GIAN_CHO, CH.SO_LAN_THU_LAI)
    CH.THOI_GIAN_CHO, CH.SO_LAN_THU_LAI = 60, 1
    try:
        nd, tv, tr, kt, phu = goi_api(khoa, [{"role": "user", "content": "Reply with the single word OK."}], max_token=50)
        print(f"POST {CH.BASE_URL}/chat/completions -> OK | trả lời (từ {phu.get('tu')}): {nd.strip()[:60]!r} | token {tv}+{tr} | {kt} | "
              f"chi phí {phu['chi_phi']} USD | nguồn tính phí {phu['nguon']} | list_cost {phu.get('list_cost')} | allowance {phu.get('allowance')}")
        print("Phản hồi thô (800 ký tự đầu):", _tho.get("cuoi", "")[:800])
        return 0
    except LoiNang as e:
        print("LỖI NẶNG:", e)
    except Exception as e:
        print("LỖI:", e)
    finally:
        CH.THOI_GIAN_CHO, CH.SO_LAN_THU_LAI = cu
    return 2


def chi_luat(df, ra):
    import pandas as pd
    ra = Path(ra or CH.THU_MUC_RA)
    ra.mkdir(parents=True, exist_ok=True)
    kq = pd.DataFrame([{"product_id": r["product_id"], "source": r["source"], "category": r["category"],
                        **luat_kiem.kiem(r)} for r in df.to_dict("records")])
    duong = ra / "luat_kiem.csv"
    kq.to_csv(duong, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {duong} ({len(kq)} dòng)")
    for c in ("co_luat", "han_sot", "so_doan_lech", "ti_le_la", "lech_so", "rong"):
        print(f"  {c}: {int((kq[c] > 0).sum())} dòng")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--kiem-tra", action="store_true")
    g.add_argument("--chi-luat", action="store_true")
    g.add_argument("--thu", type=int, metavar="N")
    g.add_argument("--tat-ca", action="store_true")
    g.add_argument("--mau-tu", metavar="TEN",
                   help="chấm đúng các product_id đã có trong ket_qua_<TEN>.jsonl (so sánh cấu hình trên CÙNG mẫu; id không còn trong đầu vào bị bỏ)")
    p.add_argument("--doi-tuong", choices=["mo_ta", "mo_ta_pair", "title"], default="mo_ta",
                   help="mo_ta = mô tả bản CỤM (mặc định); mo_ta_pair = mô tả bản PAIR (tự lấy ft_buoc4_giu_pair.parquet); "
                        "title = tiêu đề (giống hệt ở cả hai bản)")
    p.add_argument("--ten", help="tên đợt chạy (mặc định: thu_N hoặc toan_bo; thêm tiền tố nếu --doi-tuong khác mo_ta)")
    p.add_argument("--song-song", type=int, default=CH.SO_LUONG_SONG_SONG)
    p.add_argument("--tran-token", type=int, help="dừng khi tổng token (vào+ra) chạm số này")
    p.add_argument("--nha-cung-cap", choices=list(CH.CAC_NHA_CUNG_CAP), default="opencode",
                   help="opencode (gói Go, mặc định) hoặc nvidia (build.nvidia.com, dịch vụ dùng thử, khóa NVIDIA_API_KEY)")
    p.add_argument("--rpm", type=int, help="giới hạn số lời gọi mỗi phút (mặc định theo nhà cung cấp)")
    p.add_argument("--tran-usd", type=float,
                   help="dừng khi tổng chi phí (giá niêm yết ước từ token, hoặc usage.cost nếu lớn hơn) chạm số này")
    p.add_argument("--max-token-ra", type=int, default=None)
    p.add_argument("--khong-json-mode", action="store_true",
                   help="không gửi response_format=json_object (thử khi cổng trả content rỗng hoặc JSON nằm sai chỗ)")
    p.add_argument("--tat-suy-luan", action="store_true",
                   help='THỬ NGHIỆM: gửi {"thinking":{"type":"disabled"}} (tiện hơn --extra-body vì khỏi lo dấu nháy trong PowerShell)')
    p.add_argument("--muc-suy-luan", choices=["low", "medium", "high"],
                   help='THỬ NGHIỆM: gửi reasoning_effort (cổng có nhận hay không chưa kiểm chứng)')
    p.add_argument("--extra-body", help='THỬ NGHIỆM: JSON gộp vào thân yêu cầu, vd. \'{"thinking":{"type":"disabled"}}\' '
                   "(tham số có được cổng chấp nhận hay không chưa kiểm chứng; ghi vào từng dòng kết quả)")
    p.add_argument("--ra", help="thư mục ghi kết quả (mặc định data/processed/judge)")
    p.add_argument("--vao", help="file parquet đầu vào (mặc định bản cụm)")
    p.add_argument("--gia", action="store_true", help="chỉ để thử máy móc, không gọi API")
    a = p.parse_args(argv)

    CH.chon_nha_cung_cap(a.nha_cung_cap)
    if a.khong_json_mode:
        _bo_response_format.set()
    if a.rpm:
        CH.RPM = a.rpm
    if a.extra_body:
        _them.update(json.loads(a.extra_body))
    if a.tat_suy_luan:
        _them.update(CH.THAM_SO_TAT_SUY_LUAN)
    if a.muc_suy_luan:
        _them["reasoning_effort"] = a.muc_suy_luan
    if a.kiem_tra:
        return kiem_tra(lay_khoa())
    df = doc_du_lieu(a.vao or (CH.FILE_VAO_PAIR if a.doi_tuong == "mo_ta_pair" else None))
    if a.chi_luat:
        return chi_luat(df, a.ra)
    tien_to = "" if a.doi_tuong == "mo_ta" else a.doi_tuong + "_"
    if a.mau_tu:
        ids = {json.loads(x)["product_id"] for x in open(CH.THU_MUC_RA / f"ket_qua_{a.mau_tu}.jsonl", encoding="utf-8")}
        truoc = len(ids)
        df, ten = df[df.product_id.isin(ids)], a.ten or f"{tien_to}{a.mau_tu}_lai"
        print(f"--mau-tu {a.mau_tu}: {len(df)}/{truoc} product_id còn trong đầu vào")
    elif a.thu:
        df, ten = chon_mau(df, a.thu), a.ten or f"{tien_to}thu_{a.thu}"
    else:
        ten = a.ten or f"{tien_to}toan_bo"
    if a.gia:
        return chay(df, "gia_" + ten, "gia", goi_gia, a.ra, a.song_song, a.tran_token, a.max_token_ra,
                    tran_usd=a.tran_usd, doi_tuong=a.doi_tuong)
    try:
        khoa = lay_khoa()
    except LoiNang as e:
        print(e)
        return 2
    return chay(df, ten, khoa, goi_api, a.ra, a.song_song, a.tran_token, a.max_token_ra, tran_usd=a.tran_usd,
                doi_tuong=a.doi_tuong)


if __name__ == "__main__":
    sys.exit(main())
