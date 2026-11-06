# -*- coding: utf-8 -*-
"""Đóng gói Qwen để chạy OFFLINE trên GPU RTX Pro 6000 (notebook gắn cuộc thi ARC Prize không bật được Internet). Hai notebook chạy trên Kaggle
(có Internet, CPU, không cần khoá) vì mạng của Kaggle nhanh hơn máy anh; mỗi cái tạo một thư mục đầu ra, anh bấm "New Dataset" ở tab Output
để lưu thành Dataset (/kaggle/working giới hạn 20 GB nên tách làm hai):

    song-ngu-4-dong-goi-thu-vien   -> thu_vien/*.whl   (vllm 0.32.0 + mọi phụ thuộc cho Python 3.12, manylinux x86_64; cài bằng uv --no-index)
    song-ngu-4-dong-goi-mo-hinh    -> Qwen3-14B-AWQ/   (bản chụp đầy đủ của Qwen/Qwen3-14B-AWQ trên Hugging Face, ~9,3 GB)

Môi trường đích đo ngày 26/10/2026 (notebook gắn cuộc thi, RTX PRO 6000 Blackwell, compute 12.0, driver 580, CUDA toolkit 12.8):
Python 3.12.12, pip 24.1.2, có uv, KHÔNG có ensurepip, torch 2.10.0+cu128, transformers 5.0.0, chưa có vllm.

Chạy từ gốc repo:  .venv\\Scripts\\python.exe -X utf8 scripts\\song_ngu_4\\dung_dong_goi.py [thu_vien|mo_hinh|ca_hai]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")
KAGGLE = str(Path(sys.executable).parent / "kaggle")

THU_VIEN = r'''import json, os, subprocess, sys
from pathlib import Path

OUT = Path("/kaggle/working/thu_vien")
OUT.mkdir(parents=True, exist_ok=True)
YEU_CAU = ["vllm==0.32.0", "pandas", "pyarrow", "huggingface_hub"]     # cùng bản vllm với bộ 1-3
CHUOI = " ".join(f"'{y}'" for y in YEU_CAU)

def sh(c, tail=3000):
    print(">>", c[:300], flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    print((r.stdout or "")[-tail:], (r.stderr or "")[-tail:], flush=True)
    return r.returncode

# Lần 1 và 2 (26/10/2026) lỗi: `pip download --platform ...` chạy trên Python 3.13 của máy chủ Kaggle nhưng điều kiện môi trường (numpy 2.4, ...)
# vẫn tính theo 3.13 -> bộ wheel không khớp Python 3.12 của máy đích. Cách đúng: dựng Python 3.12 thật rồi tải bằng pip của nó.
print("glibc:", subprocess.run("ldd --version | head -1", shell=True, capture_output=True, text=True).stdout.strip(), flush=True)
sh("which uv || pip install -q uv")
assert sh("uv python install 3.12") == 0
assert sh("uv venv /tmp/py312 --python 3.12 --seed") == 0
PY = "/tmp/py312/bin/python"
sh(f"{PY} --version; {PY} -m pip --version")
assert sh(f"{PY} -m pip download --no-cache-dir -d {OUT} --only-binary=:all: {CHUOI}") == 0, "pip download lỗi: xem log phía trên"
ws = sorted(OUT.glob("*"))
tong = sum(p.stat().st_size for p in ws)
print(f"{len(ws)} file, {tong / 2**30:.2f} GiB")
for p in sorted(ws, key=lambda p: -p.stat().st_size)[:8]:
    print(f"  {p.stat().st_size / 2**20:9.1f} MiB  {p.name}")
assert tong < 19 * 2**30, "vượt giới hạn 20 GB của /kaggle/working"
# máy đích dùng glibc 2.35: không được có wheel đòi manylinux_2_36 trở lên
import re
cao = [p.name for p in ws for m in re.finditer(r"manylinux_2_(\d+)", p.name) if int(m.group(1)) > 35]
assert not cao, f"có wheel đòi glibc > 2.35: {cao[:5]}"
# kiểm THẬT: cài offline (không Internet) vào một môi trường Python 3.12 sạch, rồi kiểm phụ thuộc
assert sh("uv venv /tmp/env_kiem --python 3.12") == 0
rc2 = sh(f"uv pip install --no-index --find-links {OUT} --python /tmp/env_kiem/bin/python {CHUOI}", tail=4000)
rc3 = sh("uv pip check --python /tmp/env_kiem/bin/python") if rc2 == 0 else 1
sh("uv pip list --python /tmp/env_kiem/bin/python | grep -i -E '^(vllm|torch|numpy|pandas|pyarrow|huggingface|transformers|triton|flashinfer) '")
(OUT / "MANIFEST.json").write_text(json.dumps({"yeu_cau": YEU_CAU, "python": "3.12", "so_file": len(ws), "tong_byte": tong, "cai_offline_dat": rc2 == 0,
                                               "pip_check_dat": rc3 == 0, "files": [{"ten": p.name, "byte": p.stat().st_size} for p in ws]},
                                              indent=1), encoding="utf-8")
assert rc2 == 0, "cài offline KHÔNG đạt: thiếu gói trong bộ wheel, xem log"
assert rc3 == 0, "uv pip check báo xung đột phụ thuộc, xem log"
print("XONG: bộ thư viện cài offline được trên Python 3.12.")
'''

MO_HINH = r'''import json, os, subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, model_info

REPO = "Qwen/Qwen3-14B-AWQ"
OUT = Path("/kaggle/working/Qwen3-14B-AWQ")
sha = model_info(REPO).sha
print("revision:", sha, flush=True)
snapshot_download(REPO, local_dir=str(OUT), revision=sha)
fs = [p for p in OUT.rglob("*") if p.is_file() and ".cache" not in p.parts]
tong = sum(p.stat().st_size for p in fs)
print(f"{len(fs)} file, {tong / 2**30:.2f} GiB")
for p in sorted(fs, key=lambda p: -p.stat().st_size)[:10]:
    print(f"  {p.stat().st_size / 2**20:9.1f} MiB  {p.relative_to(OUT)}")
for need in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
    assert (OUT / need).exists(), f"thiếu {need}"
assert any(p.suffix == ".safetensors" for p in fs), "không có file trọng số .safetensors"
assert tong < 19 * 2**30, "vượt giới hạn 20 GB"
import shutil
shutil.rmtree(OUT / ".cache", ignore_errors=True)      # bộ nhớ đệm của huggingface_hub, không cần trong Dataset
(OUT / "REVISION.txt").write_text(f"{REPO}\n{sha}\n", encoding="utf-8")
print("XONG: mô hình đã tải đủ.")
'''


def dung_va_chay(slug, ma, tieu_de):
    thu = ROOT / "notebooks" / "kaggle_api" / slug
    thu.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"# {tieu_de}\n", "Chạy trên Kaggle (Internet bật, CPU), lưu đầu ra thành Dataset."]},
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ma.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (thu / f"{slug}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"trinhnhatanh/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1), encoding="utf-8")
    r = subprocess.run([KAGGLE, "kernels", "push", "-p", str(thu)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(slug, "->", (r.stdout + r.stderr).strip()[-200:])
    assert r.returncode == 0, f"đẩy {slug} lỗi"


if __name__ == "__main__":
    que = sys.argv[1] if len(sys.argv) > 1 else "ca_hai"
    if que in ("thu_vien", "ca_hai"):
        dung_va_chay("song-ngu-4-dong-goi-thu-vien", THU_VIEN, "Đóng gói thư viện vLLM cho Qwen offline (Python 3.12)")
    if que in ("mo_hinh", "ca_hai"):
        dung_va_chay("song-ngu-4-dong-goi-mo-hinh", MO_HINH, "Đóng gói mô hình Qwen3-14B-AWQ cho Qwen offline")
