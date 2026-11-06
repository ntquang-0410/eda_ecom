"""B0, B1b: đóng gói lên Kaggle cho fine-tune Qwen3-4B + LoRA. Token Kaggle đọc từ ~/.kaggle (script không in, không ghi token).

  python -X utf8 scripts/ft_train/dong_goi_kaggle.py thu_vien    # notebook có Internet: tải wheel peft, sacrebleu, ... cho Python 3.12
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py mo_hinh     # notebook có Internet: tải Qwen/Qwen3-4B (mạng Kaggle nhanh hơn máy)
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py du_lieu     # Dataset RIÊNG TƯ chỉ gồm mẫu POOL
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py chay_b5     # ghi đè notebook nlp-group3 (Nhật Anh cho phép 05/11/2026) và chạy thử B5
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py b6_thu      # chạy khói B6 phần 1; rồi b6_thu_tiep: phiên mới nối phiên trước
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py b6_tu_dong moi   # B6 thật qua nhiều phiên tối đa 2,5 giờ; dừng giữa chừng thì: b6_tu_dong tiep
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py luu_b6      # lưu adapter B6 sang notebook riêng, phải chạy trước mọi lần đẩy lại nlp-group3
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py b78_thu     # chạy khói B7, B8 phần 1; rồi b78_thu_tiep; chạy thật: b78_tu_dong moi | tiep
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py luu_tl      # lưu đầu ra khảo sát tỉ lệ sang notebook riêng; rồi b8f1_thu_noi, b8f1_thu_luong; chạy thật: b8f1 moi | tiep
  python -X utf8 scripts/ft_train/dong_goi_kaggle.py luu_b8f1    # lưu đầu ra bản 32 sang notebook riêng; rồi b8f2_thu; chạy thật: b8f2 moi | tiep

KHÓA TEST: Dataset huấn luyện không chứa split.parquet, parquet gốc hay test_khoa/. Trước khi đẩy, kiem_goi() đọc từng dòng và dừng nếu
gặp mã sản phẩm test. Notebook huấn luyện vì vậy không có dữ liệu test để đọc nhầm.
"""
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chung as C  # noqa: E402

NGUOI_DUNG = "trinhnhatanh"
KAGGLE = str(Path(sys.executable).parent / "kaggle")
THU_MUC_API = C.GOC / "notebooks" / "kaggle_api"
GOI_POOL = C.FT_RUN / "goi_pool"
SLUG_THU_VIEN, SLUG_MO_HINH, SLUG_POOL = "ft-qwen3-4b-thu-vien", "ft-qwen3-4b-mo-hinh", "ft-qwen3-4b-pool"
# Notebook gắn cuộc thi mới có RTX Pro 6000; ảnh môi trường giữ nguyên của nlp-group3 (Python 3.12, torch 2.10, transformers 5.0, đo 26/10/2026)
ANH_RTX = "gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9"
CUOC_THI = "arc-prize-2026-arc-agi-3"
FILE_GOI = {"pool_cum.jsonl.gz", "pool_pair.jsonl.gz", "MANIFEST.json", "dataset-metadata.json"}

THU_VIEN = r'''import json, subprocess
from pathlib import Path

OUT = Path("/kaggle/working/thu_vien")
OUT.mkdir(parents=True, exist_ok=True)
# --no-deps: máy đích đã có torch, transformers, huggingface_hub; không tải lại torch cho nhẹ. Cài ở đích bằng pip --no-index --find-links.
GOI = ["peft", "accelerate", "sacrebleu", "portalocker", "regex", "tabulate", "colorama", "lxml", "safetensors", "psutil"]

def sh(c):
    print(">>", c[:300], flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    print((r.stdout or "")[-3000:], (r.stderr or "")[-3000:], flush=True)
    return r.returncode

sh("which uv || pip install -q uv")
assert sh("uv python install 3.12") == 0
assert sh("uv venv /tmp/py312 --python 3.12 --seed") == 0
assert sh(f"/tmp/py312/bin/python -m pip download --no-cache-dir --no-deps -d {OUT} --only-binary=:all: " + " ".join(GOI)) == 0
ws = sorted(OUT.glob("*.whl"))
print(len(ws), "wheel:", [p.name for p in ws], flush=True)
for g in GOI:
    assert any(p.name.lower().startswith(g.replace("-", "_")) for p in ws), f"thieu wheel {g}"
(OUT / "MANIFEST.json").write_text(json.dumps({"goi": GOI, "files": [p.name for p in ws]}, indent=1), encoding="utf-8")
print("XONG", flush=True)
'''

MO_HINH = r'''import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, model_info

REPO = "Qwen/Qwen3-4B"
OUT = Path("/kaggle/working/Qwen3-4B")
sha = model_info(REPO).sha
print("revision:", sha, flush=True)
snapshot_download(REPO, local_dir=str(OUT), revision=sha)
shutil.rmtree(OUT / ".cache", ignore_errors=True)
fs = [p for p in OUT.rglob("*") if p.is_file()]
tong = sum(p.stat().st_size for p in fs)
print(f"{len(fs)} file, {tong / 2**30:.2f} GiB", flush=True)
for p in sorted(fs, key=lambda p: -p.stat().st_size)[:8]:
    print(f"  {p.stat().st_size / 2**20:9.1f} MiB  {p.relative_to(OUT)}")
for can in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
    assert (OUT / can).exists(), f"thieu {can}"
assert any(p.suffix == ".safetensors" for p in fs), "khong co trong so .safetensors"
assert tong < 19 * 2**30, "vuot gioi han 20 GB"
(OUT / "REVISION.txt").write_text(f"{REPO}\n{sha}\n", encoding="utf-8")
print("XONG", flush=True)
'''


# Lưu adapter B6 sang notebook riêng trước khi đẩy lại nlp-group3: mỗi lần đẩy thay đầu ra mới nhất của nlp-group3, nên chỉ để ở đó là mất.
SLUG_B6_LUU = "ft-qwen3-4b-b6-luu"
B6_LUU = r'''import hashlib, json, shutil
from pathlib import Path

ds = sorted(Path("/kaggle/input").rglob("b6_trang_thai.json"))
assert len(ds) == 1, ds
b6 = ds[0].parent
st = json.loads(ds[0].read_text(encoding="utf-8"))
assert st.get("xong") is True, "B6 chua xong"
OUT = Path("/kaggle/working/b6_luu")
OUT.mkdir(parents=True)
shutil.copy2(ds[0], OUT / ds[0].name)
for ten, v in st["cau_hinh"].items():
    for e in range(1, v["epoch_xong"] + 1):
        vao = b6 / ten / f"ep{e}"
        assert (vao / "adapter_model.safetensors").exists(), vao
        ra = OUT / ten / f"ep{e}"
        ra.mkdir(parents=True)
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(vao / f, ra / f)
    for f in sorted((b6 / ten).glob("sinh_ep*.jsonl")):
        shutil.copy2(f, OUT / ten / f.name)
mf = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as a:
            for k in iter(lambda: a.read(2**20), b""):
                h.update(k)
        mf[str(p.relative_to(OUT)).replace("\\", "/")] = {"byte": p.stat().st_size, "sha256": h.hexdigest()}
(OUT / "MANIFEST.json").write_text(json.dumps(mf, indent=1), encoding="utf-8")
print(len(mf), "file", round(sum(v["byte"] for v in mf.values()) / 2**30, 2), "GiB", flush=True)
print("XONG", flush=True)
'''


# Lưu đầu ra B7, B8 (kể cả checkpoint đang dở) sang notebook riêng, cùng lý do với B6_LUU
SLUG_B78_LUU = "ft-qwen3-4b-b78-luu"
B78_LUU = r'''import hashlib, json, shutil
from pathlib import Path

ds = sorted(Path("/kaggle/input").rglob("b78_trang_thai.json"))
assert len(ds) == 1, ds
st = json.loads(ds[0].read_text(encoding="utf-8"))
print("cum_f1 buoc", st["viec"]["cum_f1"]["buoc"], "pair_f0 epoch", st["viec"]["pair_f0"]["epoch_xong"], flush=True)
OUT = Path("/kaggle/working/b78")
shutil.copytree(ds[0].parent, OUT)
assert (OUT / "cum_f1" / "moi_nhat" / "opt.pt").exists() and (OUT / "pair_f0" / "ep2" / "adapter_model.safetensors").exists()
mf = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as a:
            for k in iter(lambda: a.read(2**20), b""):
                h.update(k)
        mf[str(p.relative_to(OUT)).replace("\\", "/")] = {"byte": p.stat().st_size, "sha256": h.hexdigest()}
Path("/kaggle/working/MANIFEST_B78.json").write_text(json.dumps(mf, indent=1), encoding="utf-8")
print(len(mf), "file", round(sum(v["byte"] for v in mf.values()) / 2**30, 2), "GiB", flush=True)
print("XONG", flush=True)
'''


# Lưu đầu ra bản 29 (khảo sát tỉ lệ, bước 2) sang notebook riêng, cùng lý do: adapter mức ~11% chỉ còn ở đó, lần đẩy kế sẽ thay
SLUG_TL_LUU = "ft-qwen3-4b-tl-luu"
TL_LUU = r'''import hashlib, json, shutil
from pathlib import Path

ds = sorted(Path("/kaggle/input").rglob("tl_trang_thai.json"))
assert len(ds) == 1, ds
st = json.loads(ds[0].read_text(encoding="utf-8"))
assert st.get("xong") is True and not st.get("bat_thuong"), "ban khao sat ti le chua xong"
OUT = Path("/kaggle/working/tl")
shutil.copytree(ds[0].parent, OUT)
for ten, v in st["viec"].items():
    print(ten, "epoch", v["epoch_xong"], "chrF", v["cham"]["chrf"]["chrf_macro_huong"], flush=True)
    assert (OUT / ten / "ep2" / "adapter_model.safetensors").exists() and (OUT / ten / "sinh_ep2.jsonl").exists(), ten
mf = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as a:
            for k in iter(lambda: a.read(2**20), b""):
                h.update(k)
        mf[str(p.relative_to(OUT)).replace("\\", "/")] = {"byte": p.stat().st_size, "sha256": h.hexdigest()}
Path("/kaggle/working/MANIFEST_TL.json").write_text(json.dumps(mf, indent=1), encoding="utf-8")
print(len(mf), "file", round(sum(v["byte"] for v in mf.values()) / 2**30, 2), "GiB", flush=True)
print("XONG", flush=True)
'''


# Lưu đầu ra bản 32 (B8 mức 1 : 1 và ~11% fold 1) sang notebook riêng, cùng lý do: adapter cum_f1, cum_f1_p11, cum_f2 chỉ còn ở đó, lần đẩy kế sẽ thay
SLUG_B8F1_LUU = "ft-qwen3-4b-b8f1-luu"
B8F1_LUU = r'''import hashlib, json, shutil
from pathlib import Path

ds = sorted(Path("/kaggle/input").rglob("b78_trang_thai.json"))
assert len(ds) == 1, ds
st = json.loads(ds[0].read_text(encoding="utf-8"))
assert st.get("xong") is True and not st.get("bat_thuong"), "ban B8 chua xong"
OUT = Path("/kaggle/working/b78")
shutil.copytree(ds[0].parent, OUT)
for ten in ("pair_f0", "cum_f1", "cum_f1_p11", "cum_f2"):
    v = st["viec"][ten]
    print(ten, "epoch", v["epoch_xong"], "chrF", v["cham"]["chrf"]["chrf_macro_huong"], flush=True)
    assert (OUT / ten / "ep2" / "adapter_model.safetensors").exists() and (OUT / ten / "sinh_ep2.jsonl").exists(), ten
mf = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as a:
            for k in iter(lambda: a.read(2**20), b""):
                h.update(k)
        mf[str(p.relative_to(OUT)).replace("\\", "/")] = {"byte": p.stat().st_size, "sha256": h.hexdigest()}
Path("/kaggle/working/MANIFEST_B8F1.json").write_text(json.dumps(mf, indent=1), encoding="utf-8")
print(len(mf), "file", round(sum(v["byte"] for v in mf.values()) / 2**30, 2), "GiB", flush=True)
print("XONG", flush=True)
'''


# Lưu đầu ra bản 34 (cum_f2_p11, ~11% fold 2) sang notebook riêng, cùng lý do: adapter chỉ còn ở đó, lần đẩy kế sẽ thay
SLUG_B8F2_LUU = "ft-qwen3-4b-b8f2-luu"
B8F2_LUU = r'''import hashlib, json, shutil
from pathlib import Path

ds = sorted(Path("/kaggle/input").rglob("b8f2_trang_thai.json"))
assert len(ds) == 1, ds
st = json.loads(ds[0].read_text(encoding="utf-8"))
assert st.get("xong") is True and not st.get("bat_thuong"), "ban b8f2 chua xong"
OUT = Path("/kaggle/working/b8f2")
shutil.copytree(ds[0].parent, OUT)
for ten in ("cum_f2_p11",):
    v = st["viec"][ten]
    print(ten, "epoch", v["epoch_xong"], "chrF", v["cham"]["chrf"]["chrf_macro_huong"], flush=True)
    assert (OUT / ten / "ep2" / "adapter_model.safetensors").exists() and (OUT / ten / "sinh_ep2.jsonl").exists(), ten
mf = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as a:
            for k in iter(lambda: a.read(2**20), b""):
                h.update(k)
        mf[str(p.relative_to(OUT)).replace("\\", "/")] = {"byte": p.stat().st_size, "sha256": h.hexdigest()}
Path("/kaggle/working/MANIFEST_B8F2.json").write_text(json.dumps(mf, indent=1), encoding="utf-8")
print(len(mf), "file", round(sum(v["byte"] for v in mf.values()) / 2**30, 2), "GiB", flush=True)
print("XONG", flush=True)
'''


def goi(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    print(out[-1500:])
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {args[0]} {args[1] if len(args) > 1 else ''} lỗi (mã {r.returncode})")
    return out


def tao_notebook(o_ma, tieu_de):
    o = [{"cell_type": "markdown", "metadata": {}, "source": [tieu_de]}]
    o += [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": m.splitlines(keepends=True)} for m in o_ma]
    return {"cells": o, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def day_dong_goi(slug, ma, tieu_de, internet=True, kernel_sources=()):
    thu = THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(tao_notebook([ma], tieu_de), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": internet, "dataset_sources": [],
        "competition_sources": [], "kernel_sources": list(kernel_sources), "model_sources": []}, indent=1), encoding="utf-8")
    goi("kernels", "push", "-p", str(thu))


def kiem_goi(ids_test, duong_dan):
    """Đọc từng dòng file mẫu đã nén: dừng nếu có mã test, fold ngoài 0..4. Trả (số mẫu, số sp)."""
    sp, n = set(), 0
    with gzip.open(duong_dan, "rt", encoding="utf-8") as f:
        for d in f:
            m = json.loads(d)
            if m["product_id"] in ids_test:
                raise SystemExit(f"RÒ RỈ TEST trong {duong_dan.name}: {m['product_id']}")
            if not 0 <= m["fold5"] <= 4:
                raise SystemExit(f"fold sai trong {duong_dan.name}: {m['id']}")
            sp.add(m["product_id"])
            n += 1
    return n, len(sp)


def du_lieu():
    tach = C.doc_split()
    ids_test = set(tach.product_id[tach.la_test])
    GOI_POOL.mkdir(parents=True, exist_ok=True)
    mf = {"sha256_ds_test": C.bam_ds_test(ids_test), "sha256_split": C.sha256_file(C.FILE_SPLIT), "nhanh": {}}
    for nh in ("cum", "pair"):
        vao, ra = C.FT_RUN / "mau" / f"pool_{nh}.jsonl", GOI_POOL / f"pool_{nh}.jsonl.gz"
        with open(vao, "rb") as a, open(ra, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as b:
            shutil.copyfileobj(a, b)
        n, so_sp = kiem_goi(ids_test, ra)
        if so_sp != int((~tach.la_test).sum()):
            raise SystemExit(f"{nh}: thiếu sản phẩm pool, có {so_sp}")
        mf["nhanh"][nh] = {"so_mau": n, "so_sp": so_sp, "sha256_jsonl": C.sha256_file(vao), "sha256_gz": hashlib.sha256(ra.read_bytes()).hexdigest()}
    (GOI_POOL / "MANIFEST.json").write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")
    (GOI_POOL / "dataset-metadata.json").write_text(json.dumps({"title": SLUG_POOL, "id": f"{NGUOI_DUNG}/{SLUG_POOL}",
                                                                "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    co = {p.name for p in GOI_POOL.iterdir()}
    assert co == FILE_GOI, f"thư mục tải lên có file lạ hoặc thiếu file: {sorted(co ^ FILE_GOI)}"
    print(json.dumps(mf["nhanh"], indent=1))
    print("dung lượng MB:", {p.name: round(p.stat().st_size / 2**20, 1) for p in GOI_POOL.iterdir()})
    if "ready" in goi("datasets", "status", f"{NGUOI_DUNG}/{SLUG_POOL}", check=False):
        goi("datasets", "version", "-p", str(GOI_POOL), "-m", "cap nhat mau pool")
    else:
        goi("datasets", "create", "-p", str(GOI_POOL))      # không có --public: riêng tư


def chay_b5(cau_hinh=None, noi_phien_truoc=False, them_nguon=()):
    """Ghi đè notebook nlp-group3: nhúng mã huấn luyện vào notebook, Dataset chỉ chứa dữ liệu pool.
    noi_phien_truoc: gắn đầu ra phiên trước của chính notebook để chạy tiếp (không tải checkpoint về máy rồi tải lên lại).
    them_nguon: notebook khác cần gắn thêm (kho lưu B6)."""
    goc = Path(__file__).resolve().parent
    cfg = {"che_do": "thu_b5", "nhanh": "cum", "fold_kiem": 0, "so_buoc": 100, "lr": 1e-4, "rank": 16, "micro_bs": 32, "tich_luy": 1,
           "ngan_sach_token": 4096, "gc": False, "max_len": 2048, "seed": 42, "so_mau_kiem": 300, "so_mau_sinh": 12}
    cfg.update(cau_hinh or {})
    ghi = "".join(f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
                  for ten in ("cham_diem.py", "kaggle_train.py"))
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cfg) + "\n"      # repr: True/False hợp lệ trong Python (json.dumps ra true/false)
    o2 = "# Ghi mã huấn luyện và bộ chấm (nhúng sẵn, phiên tắt Internet)\nfrom pathlib import Path\n" + ghi
    o3 = "import sys\nsys.path.insert(0, '.')\nimport kaggle_train\nkaggle_train.main(CAU_HINH)\n"
    tieu = ("# Fine-tune Qwen3-4B + LoRA (scripts/ft_train)\n\nGPU RTX Pro 6000 (notebook gắn cuộc thi), phiên tắt Internet. "
            "Dữ liệu: chỉ mẫu pool; không có dữ liệu test trong notebook này.")
    slug = "nlp-group3"
    thu = THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(tao_notebook([o1, o2, o3], tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "docker_image": ANH_RTX, "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [f"{NGUOI_DUNG}/{SLUG_POOL}"], "competition_sources": [CUOC_THI],
        "kernel_sources": [f"{NGUOI_DUNG}/{SLUG_THU_VIEN}", f"{NGUOI_DUNG}/{SLUG_MO_HINH}"] + [f"{NGUOI_DUNG}/{s}" for s in them_nguon] + ([f"{NGUOI_DUNG}/{slug}"] if noi_phien_truoc else []),
        "model_sources": []}, indent=1), encoding="utf-8")
    goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaRtxPro6000")


CAU_HINH_B6 = {"che_do": "b6", "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "gio_toi_da": 2.5}
THU_B6 = {"che_do": "b6", "so_train": 320, "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 3, "gio_toi_da": 1.0,
          "chi_cau_hinh": ["lr5e-5_r16", "lr1e-4_r16", "lr2e-4_r64", "lr1e-4_r16_attn"]}
CAU_HINH_B78 = {"che_do": "b78", "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "gio_toi_da": 2.5}
THU_B78 = {"che_do": "b78", "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 250, "rut_gon_sinh": 3, "so_train": 320, "so_mau_ca_fold": 200, "gio_toi_da": 1.0}
# Khảo sát tỉ lệ title : mô tả, bước 2 (plan mục 0). Một phiên 7,5 giờ, dưới giới hạn Kaggle 9 hay 12 giờ, nên chạy hết kể cả khi máy tắt.
# ~11% lấy đúng tỉ lệ token đích đo riêng từng bản; ngưỡng dừng chỉ để bắt lần chạy hỏng (mốc 75,3–75,8), không dùng chọn tỉ lệ.
VIEC_TL_B2 = [{"ten": "cum_f0_p11", "nhanh": "cum", "fold": 0, "p_title": 0.1074, "ca_fold": False},
              {"ten": "cum_f0_p75", "nhanh": "cum", "fold": 0, "p_title": 0.75, "ca_fold": False},
              {"ten": "pair_f0_p11", "nhanh": "pair", "fold": 0, "p_title": 0.1086, "ca_fold": False},
              {"ten": "pair_f0_p75", "nhanh": "pair", "fold": 0, "p_title": 0.75, "ca_fold": False}]
CAU_HINH_TL_B2 = {"che_do": "b78", "ten_trang_thai": "tl", "viec": VIEC_TL_B2, "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "gio_toi_da": 7.5,
                  "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05}
THU_TL = {"che_do": "b78", "ten_trang_thai": "tl", "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 250, "rut_gon_sinh": 3, "so_train": 320, "gio_toi_da": 1.0,
          "viec": [{"ten": "cum_f0_p11", "nhanh": "cum", "fold": 0, "p_title": 0.1074, "ca_fold": False},
                   {"ten": "pair_f0_p75", "nhanh": "pair", "fold": 0, "p_title": 0.75, "ca_fold": False}]}
# Nhật Anh duyệt 06/11/2026 (Prompt 794): B8 mức 1 : 1 của bản cụm, kèm kiểm tín hiệu ~11% trên fold 1. Nối trạng thái B8 trong kho lưu, cum_f1 chạy tiếp từ bước 1.102.
# Cùng tên trạng thái "b78" và cùng dấu vân tay với phiên gốc, nên mã từ chối chạy nếu cách chạy khác đi. p_title 0.1074 đúng mức đã thử trên fold 0.
VIEC_B8_F1 = [{"ten": "cum_f1", "nhanh": "cum", "fold": 1, "ca_fold": True},
              {"ten": "cum_f1_p11", "nhanh": "cum", "fold": 1, "p_title": 0.1074, "ca_fold": False},
              {"ten": "cum_f2", "nhanh": "cum", "fold": 2, "ca_fold": True}]
CAU_HINH_B8_F1 = {"che_do": "b78", "ten_trang_thai": "b78", "viec": VIEC_B8_F1, "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "gio_toi_da": 7.5,
                  "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05}
# Khói luồng đủ việc trên fold 1, 2: trạng thái riêng, không đụng trạng thái B8 thật; điểm trong khói không có ý nghĩa nên không bật ngưỡng dừng
THU_B8_F1 = {"che_do": "b78", "ten_trang_thai": "thu_b8", "viec": VIEC_B8_F1, "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 250, "rut_gon_sinh": 3, "so_train": 320,
             "so_mau_ca_fold": 200, "gio_toi_da": 1.0}
# Nhật Anh duyệt 06/11/2026 (Prompt 796): kiểm ~11% trên fold 2, mốc 1 : 1 đã có ở cum_f2 của bản 32. Trạng thái riêng, không nối phiên nào, không cần kho lưu.
# Trần 3 giờ cho một việc ~1,35 giờ, để lần chạy lỗi không đốt quota.
VIEC_B8_F2 = [{"ten": "cum_f2_p11", "nhanh": "cum", "fold": 2, "p_title": 0.1074, "ca_fold": False}]
CAU_HINH_B8_F2 = {"che_do": "b78", "ten_trang_thai": "b8f2", "viec": VIEC_B8_F2, "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "gio_toi_da": 3.0,
                  "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05}
THU_B8_F2 = {"che_do": "b78", "ten_trang_thai": "thu_b8f2", "viec": VIEC_B8_F2, "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 250, "rut_gon_sinh": 3, "so_train": 320, "gio_toi_da": 0.5}
# Nhật Anh duyệt 06/11/2026 (Prompt 800), bước 2 và 3 của kế hoạch tiếp: (2) kiểm lr 1e-4 ở mức ~11% bản cụm, fold 0, so với cum_f0_p11 (lr 2e-4, đã có);
# (3) bản pair trên fold 1: mốc 1 : 1 (pair_f1) và ~11% (pair_f1_p11, p_title 0.1086 đúng mức đã thử trên fold 0). Trạng thái riêng "tl2", không nối phiên nào, không cần kho lưu.
# Ba việc ~4,1 giờ [ƯỚC]; trần 6 giờ chỉ để lần chạy lỗi không đốt quota. Tất cả ca_fold False (không cần loss cả fold).
VIEC_TL2 = [{"ten": "cum_f0_p11_lr1e4", "nhanh": "cum", "fold": 0, "p_title": 0.1074, "lr": 1e-4, "ca_fold": False},
            {"ten": "pair_f1", "nhanh": "pair", "fold": 1, "ca_fold": False},
            {"ten": "pair_f1_p11", "nhanh": "pair", "fold": 1, "p_title": 0.1086, "ca_fold": False}]
CAU_HINH_TL2 = {"che_do": "b78", "ten_trang_thai": "tl2", "viec": VIEC_TL2, "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "gio_toi_da": 6.0,
                "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05}
THU_TL2 = {"che_do": "b78", "ten_trang_thai": "thu_tl2", "viec": VIEC_TL2, "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 250, "rut_gon_sinh": 3, "so_train": 320, "gio_toi_da": 0.75}


def trang_thai_kernel():
    r = subprocess.run([KAGGLE, "kernels", "status", f"{NGUOI_DUNG}/nlp-group3"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr).strip()


def cho_xong(giay=120):
    while True:
        s = trang_thai_kernel()
        if not any(x in s for x in ("RUNNING", "QUEUED")) and ("COMPLETE" in s or "ERROR" in s or "CANCEL" in s):
            return s
        time.sleep(giay)


def tai_ket_qua(ra, ten="b6"):
    """Tải kết quả nhẹ của phiên vừa xong về máy (số liệu, bản dịch đã sinh, log), không tải checkpoint nặng."""
    ra.mkdir(parents=True, exist_ok=True)
    subprocess.run([KAGGLE, "kernels", "output", f"{NGUOI_DUNG}/nlp-group3", "-p", str(ra), "--file-pattern", rf"(bao_cao_{ten}\.json|{ten}_trang_thai\.json|.*\.log|.*sinh.*\.jsonl)"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    tep = list(ra.rglob(f"{ten}_trang_thai.json"))
    return json.loads(tep[0].read_text(encoding="utf-8")) if tep else None


def chay_nhieu_phien(ten, cau_hinh, thu_muc, bat_dau_moi, them_nguon=(), so_phien_toi_da=4):
    """Chạy qua nhiều phiên, cùng cách với b6_tu_dong. Không tự bật thêm việc: xong hoặc bất thường thì dừng, người quyết việc tiếp theo.
    Phiên chạy trên Kaggle, máy tắt thì phiên vẫn chạy và tự lưu; bật lại máy chạy "tiep" để lấy kết quả."""
    goc = C.FT_RUN / thu_muc
    s = trang_thai_kernel()
    if "RUNNING" in s or "QUEUED" in s:
        raise SystemExit(f"nlp-group3 đang chạy, không đẩy đè: {s}")

    def ket_thuc(st):
        if st.get("bat_thuong"):
            raise SystemExit(f"DỪNG DO BẤT THƯỜNG: {st['bat_thuong']}")
        if st.get("xong"):
            print(ten.upper(), "XONG", flush=True)
            return True
        return False

    if not bat_dau_moi:
        st = tai_ket_qua(goc / time.strftime("phien_truoc_%Y%m%d_%H%M%S"), ten)
        if st and ket_thuc(st):
            return
    for lan in range(1, so_phien_toi_da + 1):
        ra = goc / time.strftime(f"phien_{lan}_%Y%m%d_%H%M%S")
        chay_b5(cau_hinh, noi_phien_truoc=not (bat_dau_moi and lan == 1), them_nguon=them_nguon)
        time.sleep(90)
        s = cho_xong()
        st = tai_ket_qua(ra, ten)
        print(time.strftime("%H:%M:%S"), "phien", lan, s, flush=True)
        if st:
            print(json.dumps({"xong": st.get("xong"), "bat_thuong": st.get("bat_thuong"), "so_cap_attn": st.get("attn", {}).get("so_cap"),
                              "viec": {t: {"epoch": v["epoch_xong"], "buoc": v["buoc"], "diem": v["diem"], "p_thuc": v.get("p_thuc"),
                                           "chrf": (v.get("cham") or {}).get("chrf", {}).get("chrf_macro_huong")} for t, v in st["viec"].items()}}, ensure_ascii=False), flush=True)
        if "COMPLETE" not in s:
            raise SystemExit(f"Phiên {lan} không hoàn tất: {s}. Xem {ra}. Sửa rồi chạy lệnh với tham số tiep")
        if not st:
            raise SystemExit(f"Phiên {lan} xong mà không có trạng thái, xem {ra}")
        if ket_thuc(st):
            return
    raise SystemExit(f"Hết {so_phien_toi_da} phiên mà {ten} chưa xong")


def b78_tu_dong(bat_dau_moi):
    """B7, B8. Chốt chặn không tự bật thêm việc: xong thì chạy tong_hop_b78.py rồi báo Nhật Anh."""
    chay_nhieu_phien("b78", CAU_HINH_B78, "kaggle_b78", bat_dau_moi, [SLUG_B6_LUU])


def b6_tu_dong(bat_dau_moi, so_phien_toi_da=6):
    """Chạy B6 qua nhiều phiên: mỗi phiên tự dừng ở gio_toi_da, tải kết quả về máy, phiên sau nối đầu ra phiên trước. Lỗi thì dừng, không tự chạy lại."""
    goc = C.FT_RUN / "kaggle_b6"
    s = trang_thai_kernel()
    if "RUNNING" in s or "QUEUED" in s:
        raise SystemExit(f"nlp-group3 đang chạy, không đẩy đè: {s}")
    if not bat_dau_moi:     # vòng trước bị ngắt: giữ kết quả phiên đã xong trước khi đẩy phiên mới; B6 đã xong thì không đẩy nữa
        st = tai_ket_qua(goc / time.strftime("phien_truoc_%Y%m%d_%H%M%S"))
        if st and st.get("xong"):
            print("B6 XONG", st.get("xep_hang_chrf"), st.get("so_cap_chrf"), flush=True)
            return
    for lan in range(1, so_phien_toi_da + 1):
        ra = goc / time.strftime(f"phien_{lan}_%Y%m%d_%H%M%S")
        chay_b5(CAU_HINH_B6, noi_phien_truoc=not (bat_dau_moi and lan == 1))
        time.sleep(90)
        s = cho_xong()
        st = tai_ket_qua(ra)
        print(time.strftime("%H:%M:%S"), "phien", lan, s, flush=True)
        if st:
            print(json.dumps({"chon": st.get("chon"), "xong": st.get("xong"), "cau_hinh": {t: {"epoch": v["epoch_xong"], "diem": v["diem"]} for t, v in st["cau_hinh"].items()}},
                             ensure_ascii=False), flush=True)
        if "COMPLETE" not in s:
            raise SystemExit(f"Phiên {lan} không hoàn tất: {s}. Xem {ra}. Sửa rồi chạy: dong_goi_kaggle.py b6_tu_dong tiep")
        if st and st.get("xong"):
            print("B6 XONG", st.get("xep_hang_chrf"), st.get("so_cap_chrf"), flush=True)
            return
        if not st:
            raise SystemExit(f"Phiên {lan} xong mà không có trạng thái, xem {ra}")
    raise SystemExit(f"Hết {so_phien_toi_da} phiên mà B6 chưa xong")


def main():
    viec = sys.argv[1] if len(sys.argv) > 1 else ""
    if viec == "thu_vien":
        day_dong_goi(SLUG_THU_VIEN, THU_VIEN, "# Đóng gói thư viện fine-tune (Internet bật)")
    elif viec == "mo_hinh":
        day_dong_goi(SLUG_MO_HINH, MO_HINH, "# Đóng gói Qwen/Qwen3-4B (Internet bật)")
    elif viec == "du_lieu":
        du_lieu()
    elif viec == "luu_b6":      # lưu adapter B6 sang notebook riêng, không GPU; chạy trước mọi lần đẩy lại nlp-group3
        day_dong_goi(SLUG_B6_LUU, B6_LUU, "# Lưu adapter B6 (không GPU, không Internet)", internet=False, kernel_sources=[f"{NGUOI_DUNG}/nlp-group3"])
    elif viec == "luu_b78":     # lưu đầu ra B7, B8 đang dở sang notebook riêng, không GPU; chạy trước khi đẩy lại nlp-group3
        day_dong_goi(SLUG_B78_LUU, B78_LUU, "# Lưu đầu ra B7, B8 (không GPU, không Internet)", internet=False, kernel_sources=[f"{NGUOI_DUNG}/nlp-group3"])
    elif viec == "luu_tl":      # lưu đầu ra khảo sát tỉ lệ (bản 29) sang notebook riêng, không GPU; chạy trước khi đẩy lại nlp-group3
        day_dong_goi(SLUG_TL_LUU, TL_LUU, "# Lưu đầu ra khảo sát tỉ lệ, bước 2 (không GPU, không Internet)", internet=False, kernel_sources=[f"{NGUOI_DUNG}/nlp-group3"])
    elif viec == "b8f1_thu_noi":  # khói 1: nối đúng checkpoint thật cum_f1 từ bước 1.102, chạy 3 bước rồi dừng; không đụng kho lưu
        chay_b5({**CAU_HINH_B8_F1, "dung_sau_buoc": 3}, them_nguon=[SLUG_B78_LUU])
    elif viec == "b8f1_thu_luong":  # khói 2: đủ các việc trên fold 1, 2 với dữ liệu rút gọn, trạng thái riêng
        chay_b5(THU_B8_F1)
    elif viec == "b8f1":        # b8f1 moi | b8f1 tiep: B8 mức 1 : 1 và kiểm ~11% trên fold 1, một phiên, máy tắt vẫn chạy
        chay_nhieu_phien("b78", CAU_HINH_B8_F1, "kaggle_b8f1", sys.argv[2] == "moi", [SLUG_B78_LUU])
    elif viec == "luu_b8f1":    # lưu đầu ra bản 32 sang notebook riêng, không GPU; chạy trước khi đẩy lại nlp-group3
        day_dong_goi(SLUG_B8F1_LUU, B8F1_LUU, "# Lưu đầu ra B8 mức 1 : 1 và ~11% fold 1 (không GPU, không Internet)", internet=False, kernel_sources=[f"{NGUOI_DUNG}/nlp-group3"])
    elif viec == "b8f2_thu":    # khói: đủ luồng cum_f2_p11 với dữ liệu rút gọn, trạng thái riêng
        chay_b5(THU_B8_F2)
    elif viec == "b8f2":        # b8f2 moi | b8f2 tiep: kiểm ~11% trên fold 2, một phiên, máy tắt vẫn chạy
        chay_nhieu_phien("b8f2", CAU_HINH_B8_F2, "kaggle_b8f2", sys.argv[2] == "moi")
    elif viec == "luu_b8f2":    # lưu đầu ra bản 34 sang notebook riêng, không GPU; chạy trước khi đẩy lại nlp-group3
        day_dong_goi(SLUG_B8F2_LUU, B8F2_LUU, "# Lưu đầu ra cụm ~11% fold 2 (không GPU, không Internet)", internet=False, kernel_sources=[f"{NGUOI_DUNG}/nlp-group3"])
    elif viec == "tl2_thu":     # khói: đủ luồng 3 việc (lr 1e-4, pair fold 1 hai mức) với dữ liệu rút gọn, trạng thái riêng
        chay_b5(THU_TL2)
    elif viec == "tl2":         # tl2 moi | tl2 tiep: kiểm lr 1e-4 + pair fold 1, một phiên, máy tắt vẫn chạy
        chay_nhieu_phien("tl2", CAU_HINH_TL2, "kaggle_tl2", sys.argv[2] == "moi")
    elif viec == "chay_b5":
        chay_b5()
    elif viec == "b6_thu":      # chạy khói phần 1: dừng giữa chừng, kết thúc phiên như khi hết giờ
        chay_b5({**THU_B6, "dung_sau_buoc": 3})
    elif viec == "b6_thu_tiep":  # chạy khói phần 2: phiên MỚI nối đầu ra phiên trước, chạy tới hết
        chay_b5(THU_B6, noi_phien_truoc=True)
    elif viec == "b6_tu_dong":  # b6_tu_dong moi | b6_tu_dong tiep
        b6_tu_dong(sys.argv[2] == "moi")
    elif viec == "b78_thu":     # chạy khói B7, B8 phần 1: dừng giữa pair_f0, kết thúc phiên như khi hết giờ
        chay_b5({**THU_B78, "dung_sau_buoc": 3}, them_nguon=[SLUG_B6_LUU])
    elif viec == "b78_thu_tiep":  # chạy khói phần 2: phiên MỚI nối đầu ra phiên trước, chạy tới hết
        chay_b5(THU_B78, noi_phien_truoc=True, them_nguon=[SLUG_B6_LUU])
    elif viec == "b78_tu_dong":  # b78_tu_dong moi | b78_tu_dong tiep
        b78_tu_dong(sys.argv[2] == "moi")
    elif viec == "tl_thu":      # chạy khói khảo sát tỉ lệ phần 1: dừng giữa việc đầu, kết thúc phiên như khi hết giờ
        chay_b5({**THU_TL, "dung_sau_buoc": 3})
    elif viec == "tl_thu_tiep":  # chạy khói phần 2: phiên MỚI nối đầu ra phiên trước, chạy tới hết
        chay_b5(THU_TL, noi_phien_truoc=True)
    elif viec == "tl_b2":       # tl_b2 moi | tl_b2 tiep: bước 2 của plan mục 0, một phiên, máy tắt vẫn chạy
        chay_nhieu_phien("tl", CAU_HINH_TL_B2, "kaggle_tl", sys.argv[2] == "moi")
    elif viec == "do_nguyen_nhan":
        chay_b5({"che_do": "do_nguyen_nhan", "so_lo_do": 40})
    elif viec == "do_toc_do":
        chay_b5({"che_do": "do_toc_do", "so_buoc_do": 23, "cac_cau_hinh": [
            {"micro_bs": 8, "ngan_sach": 8192, "gc": True}, {"micro_bs": 32, "ngan_sach": 16384, "gc": True},
            {"micro_bs": 32, "ngan_sach": 32768, "gc": True}, {"micro_bs": 32, "ngan_sach": 4096, "gc": False},
            {"micro_bs": 32, "ngan_sach": 6144, "gc": False}]})
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
