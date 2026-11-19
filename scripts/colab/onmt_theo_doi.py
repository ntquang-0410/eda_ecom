# -*- coding: utf-8 -*-
"""Theo dõi phiên Colab onmt3: mỗi seed xong (có /content/kq/s<seed>.DONE) thì tải ket_qua.tar.gz về data/processed/ft_run/colab_onmt/s<seed>/ (ssh cat, kiểm gzip mở được
và đủ tệp), xong cả hai (hoặc quá hạn) thì `colab stop` và kiểm lại bằng sessions/usage. Chạy độc lập bằng WMI ShowWindow=0, ghi log colab_onmt/theo_doi.log."""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
RA = GOC / "data" / "processed" / "ft_run" / "colab_onmt"
COLAB = r"C:\Users\ADMIN\tools\colab-cli\venv\Scripts\colab.exe"
KHOA = r"C:\Users\ADMIN\.ssh\colab_ed25519.pub"
PHIEN = "onmt3"
SEEDS = (43, 44)
TRAN_GIO = 5.0
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
LOG = RA / "theo_doi.log"


def ghi(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")


def chay(args, timeout, gop_loi=True):
    """Chạy lệnh, ghi đầu ra ra tệp tạm, hết hạn thì giết CẢ CÂY tiến trình. Không dùng capture_output: colab ssh sinh tiến trình cháu giữ ống mở nên subprocess.run treo
    sau hết hạn (đo 18/11/2026: bộ theo dõi đứng im 17 phút)."""
    import tempfile
    with tempfile.TemporaryFile() as f:
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT if gop_loi else subprocess.DEVNULL, env=ENV)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            return 1, b"HET_HAN"
        f.seek(0)
        return p.returncode, f.read()


def colab(args, timeout=240):
    rc, out = chay([COLAB, *args], timeout)
    return rc, out.decode("utf-8", "replace")


def doc_tren_may(lenh, timeout=120):
    return chay([COLAB, "ssh", "-s", PHIEN, "-i", KHOA, "--", lenh], timeout, gop_loi=False)      # stderr riêng: tải tệp nhị phân qua stdout


def tai(seed):
    d = RA / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    for ten in ("ket_qua.tar.gz",):
        for lan in range(3):
            rc, out = doc_tren_may(f"cat /content/kq/s{seed}/{ten}", 300)
            if rc == 0 and len(out) > 1000:
                (d / ten).write_bytes(out)
                try:
                    with tarfile.open(d / ten) as t:
                        names = t.getnames()
                    ghi(f"seed {seed}: tai {ten} {len(out)} byte, {len(names)} tep")
                    return True
                except Exception as e:
                    ghi(f"seed {seed}: gzip hong lan {lan}: {e}")
            else:
                ghi(f"seed {seed}: tai loi lan {lan} rc={rc} {len(out)} byte")
            time.sleep(20)
    return False


def main():
    RA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ghi("bat dau theo doi " + PHIEN)
    da_tai, xong = set(), {}
    while time.time() - t0 < TRAN_GIO * 3600 and len(da_tai) < len(SEEDS):
        rc, out = doc_tren_may("cat /content/kq/s43.DONE /content/kq/s44.DONE 2>/dev/null; echo; tail -n 2 /content/kg/s43/out/onmt/zh-vi_train/train.log 2>/dev/null | cut -c1-200")
        txt = out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)
        if rc != 0 and not txt.strip():
            ghi("khong ket noi duoc may (se thu lai)")
        for s in SEEDS:
            if s in da_tai:
                continue
            rc2, o2 = doc_tren_may(f"cat /content/kq/s{s}.DONE 2>/dev/null")
            if rc2 == 0 and o2.strip().startswith(b"{"):
                tt = json.loads(o2.decode("utf-8", "replace"))
                xong[s] = tt
                ghi(f"seed {s} DONE: {tt}")
                if tt.get("loi"):
                    da_tai.add(s)               # lỗi: không có gì để tải; để người xem log
                elif tai(s):
                    da_tai.add(s)
        ghi("cho: " + " | ".join(l.strip() for l in txt.splitlines() if l.strip())[-300:])
        if len(da_tai) < len(SEEDS):
            time.sleep(300)
    # tai them log train day du cho moi seed de doi chieu (nho)
    ghi(f"ket thuc theo doi: da tai {sorted(da_tai)}, xong {sorted(xong)}")
    for lan in range(3):
        colab(["stop", "-s", PHIEN])
        time.sleep(5)
        rc, sess = colab(["sessions"])
        if f"[{PHIEN}]" not in sess:
            break
    rc, u = colab(["usage"])
    ghi("sau stop: " + sess.strip().replace("\n", " | ")[:200] + " || " + u.strip().replace("\n", " | ")[:200])
    (RA / "THEO_DOI_XONG.json").write_text(json.dumps({"da_tai": sorted(da_tai), "xong": xong}, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
