# -*- coding: utf-8 -*-
"""Dựng notebook Kaggle của bộ song ngữ 3 (dịch cả cụm thuộc tính theo sản phẩm, Tiki Việt -> Trung):

    notebooks/song-ngu-3-cum-kaggle.ipynb   (GPU T4 x2 + Internet: Gemma 4 gọi API và Qwen3-14B chạy vLLM, song song)

Chạy từ gốc repo:  .venv\\Scripts\\python.exe scripts\\song_ngu_3\\dung_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")

MD = """# Bộ song ngữ thứ ba (Tiki): dịch CẢ CỤM thuộc tính của mỗi sản phẩm, Việt sang Trung

Mỗi sản phẩm gửi **một lần** (danh sách JSON các thuộc tính), để mô hình thấy các thuộc tính khác của cùng sản phẩm. Hai mô hình chạy
**song song** trong notebook này: **A = Gemma 4** gọi API Google AI Studio (không dùng GPU) và **B = Qwen3-14B** chạy vLLM trên hai GPU.
Không chấm CometKiwi ở lần thử (quyết định "cả cụm có hơn không" dựa vào chấm tay so với bản dịch đơn vị của bộ 2).

**Cách chạy**
1. **File > Import notebook** chọn file này.
2. Settings: **Accelerator: GPU T4 x2**, **Internet: On**.
3. **Add-ons > Secrets**: dùng lại các secret khoá Gemini đã có ở notebook bộ 2 (tên trong danh sách `TEN_SECRETS` ở ô Cấu hình), bật
   **Attach** cho notebook này. Không cần HF_TOKEN (Qwen3-14B-AWQ tải công khai). Không dán khoá vào notebook hay chat.
4. **Add Data**: tạo Dataset riêng tư MỚI từ thư mục `data/processed/song_ngu_3/vao_kaggle/` (4 file: `cum_can_dich.parquet`,
   `cau_hinh_kaggle.json`, `dich.py`, `dich_cum.py`) rồi gắn vào. Không cần gỡ Dataset của bộ 2 (notebook tự chọn đúng Dataset có
   `loai = cum`).
5. Để `CHE_DO = "thu"` (200 sản phẩm), **Save Version > Save & Run All (Commit)**. Kaggle chạy nền; đóng tab hay tắt máy vẫn chạy.
6. Xong (ước dưới 30 phút, phần lớn là cài vLLM và nạp mô hình), vào tab **Output**, tải thư mục `ra_cum_thu/` về
   `data/processed/song_ngu_3/tu_kaggle_thu/` trên laptop.

**Chạy cả bộ (chỉ khi lần thử cho thấy cả cụm hơn rõ):** sửa `CHE_DO = "that"`, Commit. Hết giờ thì tự dừng êm; muốn làm tiếp:
phiên bản mới, **Add Data > Notebook Output** của phiên bản trước, Commit lại (chỉ dịch phần còn thiếu).
"""

CAU_HINH = '''# ===== Cấu hình (chỉ sửa ô này) =====
CHE_DO = "thu"                 # "thu": 200 sản phẩm mẫu | "that": cả bộ (chỉ khi lần thử đạt)
PHIEN_BAN = "v2"               # "v2": tên sản phẩm + mục khoá + hiệu đính bản nháp (anh chọn 23/10/2026) | "v1": dịch cả cụm thô
CHAY_A = True                  # A = Gemma 4 qua API (không GPU)
CHAY_B = True                  # B = Qwen3-14B-AWQ trên vLLM (hai GPU)
MO_HINH_API = "gemma-4-26b-a4b-it"
TEN_SECRETS = ["GEMINI_API_KEY_aoCw", "GEMINI_API_KEY_eVYg", "GEMINI_API_KEY_rsqw", "GEMINI_API_KEY_XIBA", "GEMINI_API_KEY_XY2w",
               "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"]   # tên secret nào chưa có thì bỏ qua
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
API_EXTRA = '{"reasoning_effort": "minimal"}'   # như bộ 2: cách Google nhận để tắt suy nghĩ của Gemma 4
SONG_SONG_MOI_KHOA = 4
RPM = 20                       # trần yêu cầu mỗi phút cho mỗi khoá
TOI_DA_YEU_CAU = 15000
NGAN_SACH_USD = 8.0            # chặn an toàn theo giá giả định; gói miễn phí không tốn tiền
GIO_TOI_DA = 1.0 if CHE_DO == "thu" else 7.5
GOI_VLLM = "vllm==0.32.0"      # cùng bản với lần chạy bộ 2 (23/10/2026) để so được
'''

CHUAN_BI = '''import glob, json, os, subprocess, sys, time
from pathlib import Path

def _loai(p):
    try:
        return json.loads((Path(p).parent / "cau_hinh_kaggle.json").read_text(encoding="utf-8")).get("loai")
    except Exception:
        return None

FILE_VAO = "cum_can_dich_v2.parquet" if PHIEN_BAN == "v2" else "cum_can_dich.parquet"
ds = [p for p in glob.glob(f"/kaggle/input/**/{FILE_VAO}", recursive=True) if _loai(p) == "cum"]
assert ds, f"Chưa gắn Dataset của bộ 3 có {FILE_VAO} (cùng cau_hinh_kaggle.json có loai = cum) (Add Data), hoặc Dataset là bản cũ."
VAO = Path(ds[0]).parent
assert (VAO / "dich.py").exists() and (VAO / "dich_cum.py").exists(), "Dataset thiếu dich.py hoặc dich_cum.py."
if PHIEN_BAN == "v2":
    assert "dich_lo_v2" in (VAO / "dich_cum.py").read_text(encoding="utf-8"), "dich_cum.py trong Dataset là bản cũ (chưa có v2)."
TEN_RA = f"ra_cum_{PHIEN_BAN}" if PHIEN_BAN != "v1" else "ra_cum"
RA = Path(f"/kaggle/working/{TEN_RA}" if CHE_DO == "that" else f"/kaggle/working/{TEN_RA}_thu")
RA.mkdir(parents=True, exist_ok=True)
CU = [] if CHE_DO == "thu" else sorted({str(Path(p).parent.parent) for p in glob.glob("/kaggle/input/**/lo_*.parquet", recursive=True)
                                        if Path(p).parent.parent.name == TEN_RA})
try:
    so_gpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
except FileNotFoundError:      # máy không có GPU (chạy riêng Gemma)
    so_gpu = 0
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
    if not KHOA_ENV:
        # Secret KHÔNG đọc được khi notebook được chạy bằng Kaggle API (kaggle kernels push): ConnectionError, thử 23/10/2026.
        # Khi đó vẫn chạy B (Qwen, không cần khoá); A phải chạy bằng Save & Run All (Commit) trên web.
        print("!! Không có khoá Gemini: BỎ QUA mô hình A ở lần chạy này, vẫn chạy B. Muốn chạy A: mở notebook trên web, "
              "Save Version > Save & Run All (Commit).")
        CHAY_A = False
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["UV_CACHE_DIR"] = "/tmp/uv_cache"
'''

INTERNET = '''import urllib.request
try:
    urllib.request.urlopen("https://generativelanguage.googleapis.com/", timeout=20)
    print("Có kết nối tới Google.")
except Exception as e:
    if type(e).__name__ == "HTTPError":
        print("Có kết nối tới Google (HTTP", e.code, ").")
    else:
        raise SystemExit(f"Không ra được Internet ({type(e).__name__}). Bật Settings > Internet On rồi chạy lại.")
'''

CAI = '''# Môi trường riêng cho vLLM (Python 3.11, cài bằng uv), không đụng gói có sẵn của Kaggle
if CHAY_B:
    !pip -q install uv
    !uv venv /tmp/env_dich --python 3.11 -q
    !uv pip install -q --python /tmp/env_dich/bin/python pandas pyarrow huggingface_hub "{GOI_VLLM}"
    !/tmp/env_dich/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.device_count())"
'''

CHAY = '''def bat(lenh, gpu, log):
    return subprocess.Popen(lenh, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1"))

chung = ["--vao", str(VAO / FILE_VAO), "--ra", str(RA), "--gio_toi_da", str(GIO_TOI_DA)] + \\
        (["--chi_mau_thu"] if CHE_DO == "thu" else []) + sum([["--cu", c] for c in CU], [])
viec = []
if CHAY_A:
    lenh_a = [sys.executable, str(VAO / "dich_cum.py"), "--mo_hinh", "api_tuy_chon", "--api_base_url", API_BASE_URL,
              "--api_model", MO_HINH_API, "--api_key_env", ",".join(KHOA_ENV), "--api_extra", API_EXTRA,
              "--song_song", str(SONG_SONG_MOI_KHOA * len(KHOA_ENV)), "--rpm", str(RPM), "--toi_da_yeu_cau", str(TOI_DA_YEU_CAU),
              "--ngan_sach_usd", str(NGAN_SACH_USD), "--lo", "50" if CHE_DO == "thu" else "200"] + chung
    viec.append(("A", bat(lenh_a, "", str(RA / "log_A.log")), RA / "log_A.log"))
if CHAY_B:
    lenh_b = ["/tmp/env_dich/bin/python", str(VAO / "dich_cum.py"), "--mo_hinh", "qwen3_14b", "--backend", "vllm",
              "--lo", "500"] + chung
    viec.append(("B", bat(lenh_b, ",".join(str(i) for i in range(so_gpu)), str(RA / "log_B.log")), RA / "log_B.log"))
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

TOM_TAT = '''# Tóm tắt: số cụm đã dịch, sai cấu trúc, còn sót tiếng Việt, hỏi lại; vài cụm mẫu để xem nhanh
import pandas as pd
CUM = pd.read_parquet(VAO / FILE_VAO)
if CHE_DO == "thu":
    CUM = CUM[CUM.mau_thu]
for k, m in [("A", "api_tuy_chon"), ("B", "qwen3_14b")]:
    fs = sorted((RA / m).glob("lo_*.parquet"))
    if not fs:
        print(k, m, ": chưa có kết quả")
        continue
    K = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("ma_cum", keep="last")
    print(f"{k} {m}: {len(K)}/{len(CUM)} cụm | trạng thái {K.trang_thai.value_counts().to_dict()} | còn sót Việt {(K.so_muc_sot_viet > 0).sum()} | "
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


def o(loai, s):
    c = {"cell_type": loai, "metadata": {}, "source": s.splitlines(keepends=True)}
    if loai == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {"cells": [o("markdown", MD), o("code", CAU_HINH), o("code", CHUAN_BI), o("code", INTERNET), o("code", CAI), o("code", CHAY),
                o("code", TOM_TAT)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
def kiem_cu_phap(nb, ten):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            compile("\n".join((l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("!", "%")) else l for l in s.splitlines()),
                    f"{ten}[{i}]", "exec")


def thay(s, cu, moi):
    assert s.count(cu) == 1, f"khuôn đã đổi, không thấy đúng 1 chỗ: {cu[:60]!r} (thấy {s.count(cu)})"
    return s.replace(cu, moi)


def bien_the(nb, chay_a, chay_b):
    """Bản riêng cho một mô hình: đặt CHAY_A/CHAY_B, nâng trần chạy thật, viết lại ô giải thích."""
    import copy
    n = copy.deepcopy(nb)
    s = "".join(n["cells"][1]["source"])
    s = thay(s, "CHAY_A = True ", f"CHAY_A = {chay_a} ")
    s = thay(s, "CHAY_B = True ", f"CHAY_B = {chay_b} ")
    s = thay(s, "TOI_DA_YEU_CAU = 15000", "TOI_DA_YEU_CAU = 30000     # ước cả bộ ~11.300 lần gửi (kể cả thử lại); hạn mức ngày của Google 14.400 mỗi dự án")
    s = thay(s, "NGAN_SACH_USD = 8.0 ", "NGAN_SACH_USD = 12.0")
    s = s.replace("NGAN_SACH_USD = 12.0           # chặn an toàn theo giá giả định; gói miễn phí không tốn tiền",
                  "NGAN_SACH_USD = 12.0           # chặn an toàn theo giá giả định (ước cả bộ ~4,3 USD giả định); gói miễn phí không tốn tiền")
    n["cells"][1]["source"] = s.splitlines(keepends=True)
    return n


GEMMA_MD = """# Bộ song ngữ 3 (Tiki), dịch cả cụm: mô hình A = Gemma 4 (gọi API Google, KHÔNG dùng GPU)

Notebook này chỉ chạy **Gemma 4**. Mô hình B (Qwen3-14B) chạy ở notebook riêng `song-ngu-3-cum-qwen`.

**Bắt buộc bấm trên web:** khi notebook chạy bằng Kaggle API, Kaggle Secrets không đọc được (ConnectionError). Mở notebook trên web, bấm
**Edit**, rồi **Save Version > Save & Run All (Commit)**. Secret khoá Gemini đã gắn (Attach) sẵn cho notebook này.

**Giờ chạy:** Gemma qua API bị nghẽn vào giờ cao điểm bên Mỹ (đo 24/10/2026: 226 giây lúc 8h sáng giờ VN so với 1.843 giây lúc chiều VN, cùng đầu vào).
Nên bấm Commit **từ 7h sáng đến trưa giờ Việt Nam**. Ô cấu hình in giờ UTC để đối chiếu.

Chạy thử 200 sản phẩm: `CHE_DO = "thu"`. Chạy cả bộ: `CHE_DO = "that"` (ước ~4 giờ ở giờ thoáng, 13.208 cụm, 7.931 cụm cần gọi mô hình).
Hết giờ hoặc bị ngắt: tạo phiên bản mới, **Add Data > Notebook Output** của phiên bản trước, Save & Run All lại; chỉ dịch phần còn thiếu.
Đầu ra: thư mục `ra_cum_v2/` (chạy thật) hoặc `ra_cum_v2_thu/` (thử). Em (Claude) tải về bằng Kaggle API.
"""

QWEN_MD = """# Bộ song ngữ 3 (Tiki), dịch cả cụm: mô hình B = Qwen3-14B (chạy trên GPU T4 x2 của Kaggle)

Notebook này chỉ chạy **Qwen3-14B-AWQ** (vLLM 0.32.0, tải trọng số từ Hugging Face về máy Kaggle, chạy tại chỗ; không gọi API ngoài, không cần
secret nào). Mô hình A (Gemma 4) chạy ở notebook riêng `song-ngu-3-cum` (không GPU).

Chạy được bằng Kaggle API (Claude đẩy và chạy nền, tắt máy vẫn chạy). Settings: GPU T4 x2, Internet On, Dataset `song-ngu-3-cum-vao`.
Chạy thử 200 sản phẩm: `CHE_DO = "thu"`. Chạy cả bộ: `CHE_DO = "that"` (ước ~2,7 giờ dịch + ~4 phút nạp mô hình).
Hết giờ hoặc bị ngắt: phiên bản mới, **Add Data > Notebook Output** của phiên bản trước; chỉ dịch phần còn thiếu.
Đầu ra: thư mục `ra_cum_v2/` (chạy thật) hoặc `ra_cum_v2_thu/` (thử).
"""

GIO_CAO_DIEM = '''import datetime
_h = datetime.datetime.utcnow().hour
print(f"Giờ UTC hiện tại: {_h}h (giờ Việt Nam {(_h + 7) % 24}h, miền Đông Mỹ {(_h - 4) % 24}h).")
if CHAY_A and 12 <= _h <= 22:
    print("!! Đang là giờ làm việc bên Mỹ: Gemma qua API có thể chậm gấp ~8 lần (đo 24/10/2026). Nên chạy từ 7h đến 13h sáng giờ Việt Nam.")
'''

kiem_cu_phap(nb, "khuon")
ra = ROOT / "notebooks" / "song-ngu-3-cum-kaggle.ipynb"
ra.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(ra.name, ra.stat().st_size, "byte,", len(nb["cells"]), "ô (khuôn chung, không đẩy lên Kaggle)")
for ten, a, b, md in (("gemma", True, False, GEMMA_MD), ("qwen", False, True, QWEN_MD)):
    n = bien_the(nb, a, b)
    n["cells"][0]["source"] = md.splitlines(keepends=True)
    if ten == "gemma":      # in cảnh báo giờ cao điểm ngay sau ô cấu hình
        n["cells"].insert(2, {"cell_type": "code", "metadata": {}, "source": GIO_CAO_DIEM.splitlines(keepends=True),
                              "outputs": [], "execution_count": None})
    kiem_cu_phap(n, ten)
    p = ROOT / "notebooks" / f"song-ngu-3-cum-{ten}-kaggle.ipynb"
    p.write_text(json.dumps(n, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(p.name, p.stat().st_size, "byte,", len(n["cells"]), "ô")
