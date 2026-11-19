# -*- coding: utf-8 -*-
"""Mở máy Colab T4 và khởi động OpenNMT-py hai seed (43, 44) cùng lúc trên một card, 18/11/2026 (phản biện vòng 2, mục 2.2(b)/2.3: khoảng tin cậy theo seed cho câu "hòa 4B";
đã có seed 42 ở D3). Không đọc test: chỉ gửi các tệp song ngữ train/dev/eval fold 0 (cùng Dataset ft-song-ngu của lần D3), đã chọn theo tên, không mở nội dung.

  python -I -X utf8 scripts/colab/onmt_colab.py          # mở máy, cài, khởi động; in unit trước
Việc theo dõi và tải kết quả: onmt_theo_doi.py (hoặc làm tay). Máy phải được trả bằng `colab stop -s onmt3` sau khi tải xong."""
import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
FT = GOC / "data" / "processed" / "ft_run"
GOI = FT / "goi_song_ngu"
COLAB = Path(r"C:\Users\ADMIN\tools\colab-cli\venv\Scripts\colab.exe")
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
PHIEN = "onmt3"
SEEDS = (43, 44)
TEN_CAN = [f"{h}.{k}" for h in ("zh-vi", "vi-zh") for k in ("train.src", "train.tgt", "dev.src", "dev.tgt", "eval.src", "eval.id", "eval.nhom", "evalcum.src", "evalcum.id", "evalcum_items.jsonl")]


def colab(args, timeout=300, kiem=True):
    r = subprocess.run([str(COLAB), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=ENV)
    if kiem and r.returncode:
        raise SystemExit(f"colab {' '.join(args[:3])} loi rc={r.returncode}: {(r.stderr or r.stdout)[-500:]}")
    return r.stdout + r.stderr


def exec_py(ma, timeout=300):
    p = FT / "_tmp_onmt_exec.py"
    p.write_text(ma, encoding="utf-8", newline="\n")
    return colab(["exec", "-s", PHIEN, "-f", str(p), "--timeout", str(timeout)], timeout=timeout + 60)


def dung_tar():
    ra = FT / "onmt_colab" / "data.tar.gz"
    ra.parent.mkdir(parents=True, exist_ok=True)
    thieu = [t for t in TEN_CAN if not (GOI / t).exists()]
    if thieu:
        raise SystemExit(f"thieu tep: {thieu}")
    with tarfile.open(ra, "w:gz") as t:
        for ten in TEN_CAN:
            t.add(GOI / ten, arcname=ten)
    return ra, hashlib.sha256(ra.read_bytes()).hexdigest()


def main():
    print("unit truoc:", colab(["usage"]).strip().replace("\n", " | "))
    if f"[{PHIEN}]" in colab(["sessions"], kiem=False):
        raise SystemExit(f"Phien {PHIEN} da ton tai, khong ghi de.")
    tar, sha = dung_tar()
    print("tar du lieu:", tar.stat().st_size, "byte, sha256", sha[:16])
    print(colab(["new", "-s", PHIEN, "--gpu", "T4"]).strip())
    try:
        gpu = exec_py("import subprocess\nprint(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)\n", 90)
        print("GPU:", gpu.strip())
        if "T4" not in gpu:
            raise SystemExit(f"GPU khong phai T4: {gpu.strip()}")
        exec_py("import os\nfor d in ('/content/code','/content/kg','/content/kq'):\n    os.makedirs(d, exist_ok=True)\nprint('ok')\n", 60)
        colab(["upload", "-s", PHIEN, str(GOC / "scripts" / "ft_train" / "kaggle_onmt.py"), "/content/code/kaggle_onmt.py"])
        colab(["upload", "-s", PHIEN, str(Path(__file__).resolve().parent / "onmt_seed_vm.py"), "/content/code/onmt_seed_vm.py"])
        colab(["upload", "-s", PHIEN, str(tar), "/content/kg/data.tar.gz"], timeout=600)
        kq = exec_py(
            "import subprocess, hashlib\n"
            f"assert hashlib.sha256(open('/content/kg/data.tar.gz','rb').read()).hexdigest() == '{sha}', 'hash tar sai'\n"
            "def sh(c):\n"
            "    r = subprocess.run(['bash','-c','set -o pipefail; '+c], capture_output=True, text=True)\n"
            "    print('>>', c[:100]); print((r.stdout or '')[-600:], (r.stderr or '')[-600:], 'rc', r.returncode, flush=True)\n"
            "    assert r.returncode == 0, c\n"
            "sh('uv venv --python 3.11 /content/onmt_env')\n"
            "sh(\"uv pip install --python /content/onmt_env/bin/python -q 'torch==2.2.2' 'numpy<2' 'OpenNMT-py==3.5.1' sentencepiece\")\n"
            "sh(\"/content/onmt_env/bin/python -c \\\"import torch, onmt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'onmt', onmt.__version__); assert torch.cuda.is_available()\\\"\")\n", 900)
        print(kq[-900:].strip())
        print(exec_py(
            "import subprocess, sys, threading, time\n"
            f"for s in {list(SEEDS)}:\n"
            "    subprocess.Popen([sys.executable, '/content/code/onmt_seed_vm.py', str(s)], stdout=open(f'/content/kq/s{s}.log', 'w'), stderr=subprocess.STDOUT, start_new_session=True)\n"
            "# bao hiem: het 5 gio thi tu tra may du the nao (ASCII: colab exec doc file bang charmap)\n"
            "from google.colab import runtime\n"
            "threading.Timer(5 * 3600, runtime.unassign).start()\n"
            "print('da khoi dong', time.strftime('%H:%M:%S'))\n", 60).strip())
    except BaseException:
        print("LOI khi dung may, tra may de khoi ton unit")
        colab(["stop", "-s", PHIEN], kiem=False)
        raise
    print("xong. Theo doi bang: colab exec -s", PHIEN, "(doc /content/kq/s43.log, s44.log). Nho `colab stop -s", PHIEN, "` sau khi tai xong.")


if __name__ == "__main__":
    main()
