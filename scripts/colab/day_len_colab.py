# -*- coding: utf-8 -*-
"""Mở máy Colab, đẩy mã + dữ liệu pool, khởi động bộ bao job trên máy Colab và tiến trình theo dõi (WMI) trên máy nhà.

Chạy:  python -I -X utf8 scripts/colab/day_len_colab.py {khoi|that} [--gpu G4]
Không đọc test_khoa. Dữ liệu đẩy lên chỉ gồm pool_*.jsonl.gz (đã kiểm không có test lúc đóng gói), mã cham_diem.py, kaggle_train.py và bộ bao job.
In số compute unit trước khi mở máy. Máy chỉ được trả khi: tiến trình theo dõi tải xong, đồng hồ an toàn hết hạn, hoặc người dùng `colab stop`.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cau_hinh_tlc as CH  # noqa: E402

GOC = CH.GOC
FT = GOC / "data" / "processed" / "ft_run"
COLAB = Path(r"C:\Users\ADMIN\tools\colab-cli\venv\Scripts\colab.exe")
KHOA_SSH = Path(r"C:\Users\ADMIN\.ssh\colab_ed25519.pub")
PYTHON_NHA = GOC / ".venv" / "Scripts" / "python.exe"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def colab(args, timeout=300, kiem=True):
    r = subprocess.run([str(COLAB), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=ENV)
    if kiem and r.returncode:
        raise SystemExit(f"colab {' '.join(args[:3])} loi rc={r.returncode}: {(r.stderr or r.stdout)[-500:]}")
    return r.stdout + r.stderr


def exec_py(phien, ma, timeout=300):
    p = FT / "_tmp_exec.py"
    p.write_text(ma, encoding="utf-8", newline="\n")      # không BOM: BOM làm SyntaxError trên máy Colab
    return colab(["exec", "-s", phien, "-f", str(p), "--timeout", str(timeout)], timeout=timeout + 60)


def khoi_dong_wmi(dong_lenh):
    """Tạo tiến trình ẩn độc lập với terminal/VS Code (Start-Process chết theo phiên)."""
    # ShowWindow=0 là BẮT BUỘC: thiếu nó tiến trình có cửa sổ console, đóng cửa sổ là chết (đo 17/11/2026: chết sau ~17 giây; có ShowWindow=0 sống > 2 phút)
    ps = ("$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
          "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=%s; CurrentDirectory=%s; ProcessStartupInformation=$si}; "
          "Write-Output ('ReturnValue=' + $r.ReturnValue + ' PID=' + $r.ProcessId)" % ("'" + dong_lenh.replace("'", "''") + "'", "'" + str(GOC).replace("'", "''") + "'"))
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return r.stdout.strip() or r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("che_do", choices=["khoi", "that"])
    ap.add_argument("--gpu", default="G4")
    a = ap.parse_args()
    job = CH.tao_job(a.che_do)
    phien = job["phien"]
    ra = FT / f"colab_{phien}"
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    print("job:", phien, "viec:", job["viec"], "tran_gio", job["tran_gio"], "chunk_gio", job["chunk_gio"])
    print("unit truoc:", colab(["usage"]).strip().replace("\n", " | "))
    if f"[{phien}]" in colab(["sessions"], kiem=False):
        raise SystemExit(f"Phien {phien} da ton tai tren server, khong ghi de. Dung `colab stop -s {phien}` neu chac chan.")
    print(colab(["new", "-s", phien, "--gpu", a.gpu]).strip())
    try:
        gpu = exec_py(phien, "import subprocess\nprint(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)\n", 90)
        print("GPU:", gpu.strip())
        if "6000" not in gpu or "97" not in gpu:
            raise SystemExit(f"GPU khong dung RTX PRO 6000 95 GiB: {gpu.strip()}")
        exec_py(phien, "import os\nfor d in ('/content/job/code','/content/job/pool','/content/kq'):\n    os.makedirs(d, exist_ok=True)\nprint('ok')\n", 60)
        for f in ("cham_diem.py", "kaggle_train.py"):
            colab(["upload", "-s", phien, str(GOC / "scripts" / "ft_train" / f), f"/content/job/code/{f}"], timeout=300)
        colab(["upload", "-s", phien, str(Path(__file__).resolve().parent / "giam_sat_vm.py"), "/content/giam_sat_vm.py"])
        for f in job["pool_sha256"]:
            colab(["upload", "-s", phien, str(CH.GOI_POOL / f), f"/content/job/pool/{f}"], timeout=600)
        colab(["upload", "-s", phien, str(ra / "job.json"), "/content/job/job.json"])
        print(exec_py(phien, "import sys, threading\nsys.path.insert(0, '/content')\nimport giam_sat_vm as G\n"
                             "threading.Thread(target=G.chay, args=('/content/job/job.json',), name='giam_sat').start()\nprint('da khoi dong bo bao job')\n", 60).strip())
    except BaseException:
        print("LOI khi dung may, tra may de khoi ton unit")
        colab(["stop", "-s", phien], kiem=False)
        raise
    cfg_theo_doi = {"phien": phien, "colab_exe": str(COLAB), "khoa_ssh": str(KHOA_SSH), "thu_muc_ra": str(ra), "chu_ky_giay": 300, "chu_ky_ckpt_giay": 1800,
                    "giu_ckpt": 2, "tran_theo_doi_gio": job["tran_gio"] + 4}
    (ra / "theo_doi_cau_hinh.json").write_text(json.dumps(cfg_theo_doi, indent=1), encoding="utf-8")
    dl = f'"{PYTHON_NHA}" -I -X utf8 "{Path(__file__).resolve().parent / "theo_doi_may_nha.py"}" --cau-hinh "{ra / "theo_doi_cau_hinh.json"}"'
    print("WMI:", khoi_dong_wmi(dl))
    print("xong. Theo doi:", ra / "theo_doi.log")


if __name__ == "__main__":
    main()
