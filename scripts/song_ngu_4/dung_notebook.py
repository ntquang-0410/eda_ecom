# -*- coding: utf-8 -*-
"""Dựng hai notebook Kaggle của bộ song ngữ 4 (1688 Trung -> Việt, hiệu đính cả cụm theo sản phẩm):

    notebooks/song-ngu-4-cum-gemma-kaggle.ipynb   (không GPU, Internet: Gemma 4 gọi API Google AI Studio; anh bấm Commit trên web)
    notebooks/song-ngu-4-cum-qwen-kaggle.ipynb    (GPU T4 x2: Qwen3-14B-AWQ chạy vLLM; Claude đẩy và chạy bằng Kaggle API)

Chạy từ gốc repo:  .venv\\Scripts\\python.exe -X utf8 scripts\\song_ngu_4\\dung_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")

CAU_HINH = '''# ===== Cấu hình (chỉ sửa ô này) =====
CHE_DO = "thu"                 # "thu": 200 sản phẩm mẫu | "that": cả bộ (chỉ khi lần thử đạt và anh duyệt)
SO_PHAN = 1                    # chia các cụm còn lại thành SO_PHAN phần theo product_id, notebook này chỉ dịch phần CHI_SO_PHAN (kaggle_api.py day ... phan=i/K đặt)
CHI_SO_PHAN = 0
CHI_CAC_CUM = ""               # chỉ dịch các cụm trong danh sách này (JSON danh sách ma_cum, nén zlib + base64; kaggle_api.py day ... cum=<file> đặt), rỗng = không lọc
PHIEN_TIEP = False             # True: phiên sau (kaggle_api.py day ... tiep đặt), BẮT BUỘC thấy các lô đã dịch của phiên trước, không thì dừng ngay
CHAY_A = True                  # A = Gemma 4 qua API (không GPU)
CHAY_B = True                  # B = Qwen3-14B-AWQ trên vLLM (hai GPU)
MO_HINH_API = "gemma-4-26b-a4b-it"
TEN_SECRETS = ["GEMINI_API_KEY_aoCw", "GEMINI_API_KEY_eVYg", "GEMINI_API_KEY_rsqw", "GEMINI_API_KEY_XIBA", "GEMINI_API_KEY_XY2w",
               "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"]   # tên secret nào chưa có thì bỏ qua
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
API_EXTRA = '{"reasoning_effort": "minimal"}'   # như bộ 1-3: cách Google nhận để tắt suy nghĩ của Gemma 4
SONG_SONG_MOI_KHOA = 4
RPM = 20                       # trần yêu cầu mỗi phút cho mỗi khoá
TOI_DA_YEU_CAU = 2000 if CHE_DO == "thu" else 40000      # cả bộ ~16.900 cụm, kể cả hỏi lại và thử lại
NGAN_SACH_USD = 2.0 if CHE_DO == "thu" else 20.0          # chặn an toàn theo giá giả định; gói miễn phí không tốn tiền
GIO_TOI_DA = 1.5 if CHE_DO == "thu" else 11.0             # Kaggle cho tối đa 12 giờ một phiên; hết giờ dừng êm, phiên sau dịch tiếp
GOI_VLLM = "vllm==0.32.0"      # cùng bản với bộ 1-3
PHIEN_BAN_CAN = "cum-zh_vi-v4b-2026-10-25"   # câu lệnh phải đúng bản này (chặn chạy nhầm Dataset cũ)
'''

CHUAN_BI = '''import glob, json, os, subprocess, sys, time
from pathlib import Path

def _loai(p):
    try:
        return json.loads((Path(p).parent / "cau_hinh_kaggle.json").read_text(encoding="utf-8")).get("loai")
    except Exception:
        return None

FILE_VAO = "cum_can_dich_zv.parquet"
ds = [p for p in glob.glob(f"/kaggle/input/**/{FILE_VAO}", recursive=True) if _loai(p) == "cum_zv"]
assert ds, f"Chưa gắn Dataset của bộ 4 có {FILE_VAO} (cùng cau_hinh_kaggle.json có loai = cum_zv) (Add Data)."
VAO = Path(ds[0]).parent
assert (VAO / "dich.py").exists() and (VAO / "dich_cum_zv.py").exists(), "Dataset thiếu dich.py hoặc dich_cum_zv.py."
assert PHIEN_BAN_CAN in (VAO / "dich_cum_zv.py").read_text(encoding="utf-8"), (
    f"dich_cum_zv.py trong Dataset không phải bản {PHIEN_BAN_CAN}: bấm vào Dataset song-ngu-4-cum-vao ở khung Input bên phải, chọn phiên bản "
    "mới nhất (Check for updates / Version), rồi Save & Run All lại.")
TEN_RA = "ra_cum_zv"
RA = Path(f"/kaggle/working/{TEN_RA}" if CHE_DO == "that" else f"/kaggle/working/{TEN_RA}_thu")
RA.mkdir(parents=True, exist_ok=True)
# Dataset ra-cu: Kaggle bỏ mất tầng ra_cum_zv khi giải nén zip nên chỉ dựa vào tên thư mục mô hình (api_tuy_chon | qwen3_14b), không dựa tên thư mục ông
CU = [] if CHE_DO == "thu" else sorted({str(Path(p).parent.parent) for p in glob.glob("/kaggle/input/**/lo_*.parquet", recursive=True)
                                        if Path(p).parent.name in ("api_tuy_chon", "qwen3_14b")})
assert CU or not PHIEN_TIEP, ("Đây là phiên tiếp theo nhưng không thấy các lô đã dịch của phiên trước (/kaggle/input/**/api_tuy_chon/lo_*.parquet): "
                              "gắn Dataset song-ngu-4-cum-ra-cu (Add Data) rồi chạy lại. Dừng để khỏi dịch lại từ đầu.")
try:
    so_gpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
except FileNotFoundError:      # máy không có GPU (notebook Gemma)
    so_gpu = 0
VAO_CHAY = VAO / FILE_VAO
if SO_PHAN > 1:          # chia theo product_id (mọi cụm của một sản phẩm nằm cùng phần); crc32 ổn định giữa các máy, không phụ thuộc cụm nào đã dịch
    import zlib
    import pandas as pd
    _c = pd.read_parquet(VAO / FILE_VAO)
    _c = _c[[zlib.crc32(str(x).encode()) % SO_PHAN == CHI_SO_PHAN for x in _c.product_id]]
    VAO_CHAY = Path("/tmp/cum_phan.parquet")
    _c.to_parquet(VAO_CHAY, index=False)
    print(f"Phần {CHI_SO_PHAN}/{SO_PHAN}: {len(_c):,} cụm của {_c.product_id.nunique():,} sản phẩm (chưa trừ cụm đã dịch)")
if CHI_CAC_CUM:          # dự phòng: chỉ dịch đúng các cụm Gemma hỏng hoặc trượt luật (danh sách nhúng sẵn trong ô cấu hình)
    assert SO_PHAN == 1, "CHI_CAC_CUM và SO_PHAN không dùng cùng nhau"
    import base64, zlib, json as _json
    import pandas as pd
    _ds = set(_json.loads(zlib.decompress(base64.b64decode(CHI_CAC_CUM)).decode("utf-8")))
    _c = pd.read_parquet(VAO / FILE_VAO)
    _c = _c[_c.ma_cum.isin(_ds)]
    assert len(_c) == len(_ds), f"danh sách có {len(_ds)} cụm nhưng chỉ thấy {len(_c)} trong dữ liệu vào: Dataset cum-vao không khớp"
    VAO_CHAY = Path("/tmp/cum_chon.parquet")
    _c.to_parquet(VAO_CHAY, index=False)
    print(f"Chỉ dịch {len(_c):,} cụm trong danh sách ({_c.product_id.nunique():,} sản phẩm)")
# Chế độ OFFLINE cho Qwen (notebook gắn cuộc thi ARC Prize + GPU RTX Pro 6000 không bật được Internet): có Dataset thư viện (vllm-*.whl) và Dataset mô hình
# (Qwen3-14B-AWQ/config.json) gắn vào thì cài vLLM và nạp mô hình từ đó. Không phụ thuộc cấu trúc thư mục bên trong Dataset (Kaggle có thể bỏ tầng ngoài).
_whl = sorted(glob.glob("/kaggle/input/**/vllm-*.whl", recursive=True))
_mh = sorted(glob.glob("/kaggle/input/**/Qwen3-14B-AWQ/config.json", recursive=True))
OFFLINE = bool(CHAY_B and _whl and _mh)
if OFFLINE:
    WHEELS, MO_HINH_DIR = str(Path(_whl[0]).parent), str(Path(_mh[0]).parent)
    _rv = Path(MO_HINH_DIR) / "REVISION.txt"
    SHA_MO_HINH = _rv.read_text(encoding="utf-8").split()[-1] if _rv.exists() else "khong_ro"
    print("OFFLINE: thư viện", WHEELS, "| mô hình", MO_HINH_DIR, "| revision", SHA_MO_HINH)
print("Dữ liệu vào:", VAO, "| thư mục ra:", RA, "| đầu ra cũ:", CU or "không có", "| số GPU:", so_gpu)
if CHAY_B:
    assert so_gpu >= 1, "Chưa bật GPU (Settings > Accelerator: GPU T4 x2)."

KHOA_ENV = []
if CHAY_A:
    _loi = {}
    try:
        from kaggle_secrets import UserSecretsClient
        _us = UserSecretsClient()
        for _ten in TEN_SECRETS:
            try:
                os.environ[_ten] = _us.get_secret(_ten)     # không in, không ghi file
                KHOA_ENV.append(_ten)
            except Exception as e:
                _loi[_ten] = f"{type(e).__name__}: {str(e)[:200]}"   # thông báo lỗi của Kaggle không chứa giá trị khoá
    except Exception as e:
        print("Không đọc được Kaggle Secrets:", type(e).__name__, str(e)[:200])
    print(f"Đã nạp {len(KHOA_ENV)} khoá Gemini:", KHOA_ENV)
    for _ten, _l in list(_loi.items())[:3]:
        print("  không nạp được", _ten, "->", _l)
    # Secret KHÔNG đọc được khi notebook được chạy bằng Kaggle API (ConnectionError, thử 23 và 24/10/2026): Gemma phải bấm Commit trên web
    assert KHOA_ENV, ("Không có khoá Gemini nào. Add-ons > Secrets: bật Attach các secret GEMINI_API_KEY_... cho notebook này, rồi "
                      "Save Version > Save & Run All (Commit) trên web.")
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["UV_CACHE_DIR"] = "/tmp/uv_cache"
'''

GIO_CAO_DIEM = '''import datetime
_h = datetime.datetime.utcnow().hour
print(f"Giờ UTC hiện tại: {_h}h (giờ Việt Nam {(_h + 7) % 24}h, miền Đông Mỹ {(_h - 4) % 24}h).")
if CHAY_A and 12 <= _h <= 22:
    print("!! Đang là giờ làm việc bên Mỹ: Gemma qua API có thể chậm gấp ~8 lần (đo 24/10/2026). Nên chạy từ 7h đến 13h giờ Việt Nam.")
'''

INTERNET = '''if OFFLINE:
    print("Chạy OFFLINE: thư viện và mô hình lấy từ Dataset gắn vào notebook, không cần Internet.")
else:
    import urllib.request
    try:
        urllib.request.urlopen("https://generativelanguage.googleapis.com/", timeout=20)
        print("Có kết nối Internet.")
    except Exception as e:
        if type(e).__name__ == "HTTPError":
            print("Có kết nối Internet (HTTP", e.code, ").")
        else:
            raise SystemExit(f"Không ra được Internet ({type(e).__name__}). Bật Settings > Internet On rồi chạy lại.")
'''

CAI = '''# Môi trường riêng cho vLLM (Python 3.11, cài bằng uv), không đụng gói có sẵn của Kaggle
if CHAY_B and OFFLINE:
    # Python 3.12 của chính máy (thư viện trong Dataset dựng cho Python 3.12), cài từ Dataset, không Internet
    !uv venv /tmp/env_dich --python {sys.executable} -q
    !uv pip install -q --no-index --find-links {WHEELS} --python /tmp/env_dich/bin/python pandas pyarrow huggingface_hub "{GOI_VLLM}"
    !/tmp/env_dich/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.device_count() else '')"
elif CHAY_B:
    !pip -q install uv
    !uv venv /tmp/env_dich --python 3.11 -q
    !uv pip install -q --python /tmp/env_dich/bin/python pandas pyarrow huggingface_hub "{GOI_VLLM}"
    !/tmp/env_dich/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.device_count())"
'''

CHAY = '''def bat(lenh, gpu, log, them_env=None):
    return subprocess.Popen(lenh, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", **(them_env or {})))

chung = ["--vao", str(VAO_CHAY), "--ra", str(RA), "--gio_toi_da", str(GIO_TOI_DA)] + \\
        (["--chi_mau_thu"] if CHE_DO == "thu" else []) + sum([["--cu", c] for c in CU], [])
viec = []
if CHAY_A:
    lenh_a = [sys.executable, str(VAO / "dich_cum_zv.py"), "--mo_hinh", "api_tuy_chon", "--api_base_url", API_BASE_URL,
              "--api_model", MO_HINH_API, "--api_key_env", ",".join(KHOA_ENV), "--api_extra", API_EXTRA,
              "--song_song", str(SONG_SONG_MOI_KHOA * len(KHOA_ENV)), "--rpm", str(RPM), "--toi_da_yeu_cau", str(TOI_DA_YEU_CAU),
              "--ngan_sach_usd", str(NGAN_SACH_USD), "--lo", "50" if CHE_DO == "thu" else "200"] + chung
    viec.append(("A", bat(lenh_a, "", str(RA / "log_A.log")), RA / "log_A.log"))
if CHAY_B:
    lenh_b = ["/tmp/env_dich/bin/python", str(VAO / "dich_cum_zv.py"), "--mo_hinh", "qwen3_14b", "--backend", "vllm",
              "--lo", "500"] + chung
    env_b = {"DICH_THU_MUC_MO_HINH": MO_HINH_DIR, "DICH_SHA_MO_HINH": SHA_MO_HINH, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"} if OFFLINE else None
    viec.append(("B", bat(lenh_b, ",".join(str(i) for i in range(so_gpu)), str(RA / "log_B.log"), env_b), RA / "log_B.log"))
t0 = time.time()
while any(p.poll() is None for _, p, _ in viec):
    time.sleep(60)
    for k, _, log in viec:
        dong = Path(log).read_text(errors="ignore").strip().splitlines()
        print(f"[{(time.time() - t0) / 60:.0f} phút] {k}:", dong[-1][:220] if dong else "")
for k, p, log in viec:
    print(f"--- {k}: mã thoát {p.returncode}, 15 dòng cuối ---")
    print("\\n".join(Path(log).read_text(errors="ignore").strip().splitlines()[-15:]))
'''

TOM_TAT = '''# Tóm tắt: số cụm đã dịch, sai cấu trúc, còn sót chữ Hán, hỏi lại; vài cụm mẫu để xem nhanh
import pandas as pd
CUM = pd.read_parquet(VAO_CHAY)
if CHE_DO == "thu":
    CUM = CUM[CUM.mau_thu]
for k, m in [("A", "api_tuy_chon"), ("B", "qwen3_14b")]:
    fs = sorted((RA / m).glob("lo_*.parquet"))
    if not fs:
        print(k, m, ": chưa có kết quả")
        continue
    K = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("ma_cum", keep="last")
    print(f"{k} {m}: {len(K)}/{len(CUM)} cụm | trạng thái {K.trang_thai.value_counts().to_dict()} | còn sót Hán {(K.so_muc_sot_han > 0).sum()} | "
          f"hỏi lại {int(K.hoi_lai.sum())}, sửa được {int(K.sua_duoc.sum())} | lý do lỗi {K[K.trang_thai != 'ok'].ly_do.value_counts().head(5).to_dict()}")
    the = RA / f"the_{m}.json"
    if the.exists():
        t = json.loads(the.read_text(encoding="utf-8"))
        print("   thời gian", t.get("giay"), "giây | tốc độ", t.get("toc_do_cum_giay"), "cụm/giây | API",
              {x: t.get("api", {}).get(x) for x in ["yeu_cau", "thu_lai", "loi_api", "token_vao", "token_ra"]} if t.get("api") else "")
    for _, r in K[K.trang_thai == "ok"].head(2).iterrows():
        print("   ví dụ", r.ma_cum, r.ket_qua[:300])
print("\\nCác file đầu ra (tab Output):")
for f in sorted(RA.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(RA.parent)}  {f.stat().st_size:,} byte")
'''

GEMMA_MD = """# Bộ song ngữ 4 (1688 Trung -> Việt), hiệu đính cả cụm: mô hình A = Gemma 4 (gọi API Google, KHÔNG dùng GPU)

Mỗi sản phẩm gửi **một lần**: tên sản phẩm (ngữ cảnh), các thuộc tính tiếng Trung, bản khoá (từ điển, quy tắc) và bản nháp (bản dịch đơn vị của
bộ 1). Mô hình giữ nguyên bản nháp nếu đúng, chỉ sửa khi ngữ cảnh cho thấy sai. Luật chặt từng trường làm ở laptop sau khi tải về.

Notebook này chỉ chạy **Gemma 4**. Mô hình B (Qwen3-14B) chạy ở notebook riêng `song-ngu-4-cum-qwen`.

**Lần đầu (một lần):** Add-ons > **Secrets**: bật **Attach** các secret khoá Gemini (GEMINI_API_KEY_...) cho notebook này.
**Bắt buộc bấm trên web:** khi chạy bằng Kaggle API, Kaggle Secrets không đọc được. Mở notebook, **Edit**, rồi **Save Version > Save & Run All (Commit)**.

**Giờ chạy:** Gemma qua API nghẽn vào giờ làm việc bên Mỹ (đo 24/10/2026: chậm gấp ~8 lần). Nên bấm Commit **từ 7h đến 13h giờ Việt Nam**.

Thử 200 sản phẩm: `CHE_DO = "thu"` (ước dưới 30 phút). Cả bộ: `CHE_DO = "that"` (16.889 cụm, ước ~13 giờ, cần 2 phiên).
Hết giờ hoặc bị ngắt: phiên bản mới, **Add Data > Notebook Output** của phiên bản trước, Save & Run All lại; chỉ dịch phần còn thiếu.
Đầu ra: `ra_cum_zv/` (cả bộ) hoặc `ra_cum_zv_thu/` (thử). Claude tải về bằng Kaggle API.
"""

QWEN_MD = """# Bộ song ngữ 4 (1688 Trung -> Việt), hiệu đính cả cụm: mô hình B = Qwen3-14B (GPU T4 x2 của Kaggle)

Notebook này chỉ chạy **Qwen3-14B-AWQ** (vLLM 0.32.0, trọng số tải từ Hugging Face, chạy tại chỗ; không gọi API ngoài, không cần secret).
Mô hình A (Gemma 4) chạy ở notebook riêng `song-ngu-4-cum-gemma`.

Chạy được bằng Kaggle API (Claude đẩy và chạy nền). Settings: GPU T4 x2, Internet On, Dataset `song-ngu-4-cum-vao`.
Thử 200 sản phẩm: `CHE_DO = "thu"`. Cả bộ: `CHE_DO = "that"` (16.889 cụm, ước ~4-6 giờ).
Hết giờ hoặc bị ngắt: phiên bản mới, **Add Data > Notebook Output** của phiên bản trước; chỉ dịch phần còn thiếu.
Đầu ra: `ra_cum_zv/` (cả bộ) hoặc `ra_cum_zv_thu/` (thử).
"""


def o(loai, s):
    c = {"cell_type": loai, "metadata": {}, "source": s.splitlines(keepends=True)}
    if loai == "code":
        c.update(outputs=[], execution_count=None)
    return c


def kiem_cu_phap(nb, ten):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            compile("\n".join((l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("!", "%")) else l for l in s.splitlines()),
                    f"{ten}[{i}]", "exec")


def dung(ten, chay_a, chay_b, md):
    ch = CAU_HINH.replace("CHAY_A = True ", f"CHAY_A = {chay_a} ").replace("CHAY_B = True ", f"CHAY_B = {chay_b} ")
    assert f"CHAY_A = {chay_a} " in ch and f"CHAY_B = {chay_b} " in ch
    o_giua = [o("code", GIO_CAO_DIEM)] if chay_a else []
    nb = {"cells": [o("markdown", md), o("code", ch)] + o_giua + [o("code", CHUAN_BI), o("code", INTERNET), o("code", CAI),
                                                                   o("code", CHAY), o("code", TOM_TAT)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    kiem_cu_phap(nb, ten)
    p = ROOT / "notebooks" / f"song-ngu-4-cum-{ten}-kaggle.ipynb"
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(p.name, p.stat().st_size, "byte,", len(nb["cells"]), "ô")


dung("gemma", True, False, GEMMA_MD)
dung("qwen", False, True, QWEN_MD)
