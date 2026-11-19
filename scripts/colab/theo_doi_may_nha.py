# -*- coding: utf-8 -*-
"""Tiến trình theo dõi chạy NỀN trên máy nhà (khởi động bằng WMI để tắt VS Code không làm nó chết).

Việc làm, lặp mỗi chu_ky_giay:
  1. đọc trạng thái máy Colab qua `colab ssh -- python3 /content/giam_sat_vm.py doc`;
  2. tải mọi gói kết quả mới (kq/*.json có sha256), kiểm hash, giải nén vào <thu_muc_ra>;
  3. sao lưu checkpoint: tải kq/ckpt.tar nếu mới và đã quá chu_ky_ckpt_giay kể từ lần trước, giữ 2 bản gần nhất;
  4. khi máy Colab ghi DONE.json: tải nốt, kiểm hash, tạo /content/TAI_XONG, `colab stop`, kiểm lại bằng `colab sessions` + `colab usage`, rồi thoát.
Lỗi mạng/CLI: thử lại có chờ; nhiều lần liên tiếp thì ghi log và vẫn tiếp tục vòng (không bỏ máy). Mất phiên trên server thì thoát.
Nếu bản thân tiến trình này chết, đồng hồ an toàn trên máy Colab (cho_tai_phut sau DONE) sẽ tự trả máy.
Chạy:  python -I -X utf8 theo_doi_may_nha.py --cau-hinh <file.json>
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[[0-9;]*m")


# ---------------------------------------------------------------- hàm thuần (có test)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for k in iter(lambda: f.read(2 ** 20), b""):
            h.update(k)
    return h.hexdigest()


def bo_ansi(s):
    return ANSI.sub("", s or "")


def tach_json(van_ban):
    """Lấy phần giữa <<JSON>> và <<FIN>> (lệnh ssh có thể kèm cảnh báo trước/sau)."""
    m = re.search(r"<<JSON>>(.*?)<<FIN>>", van_ban, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def phien_con_tren_server(dau_ra_sessions, ten):
    return any(l.strip().startswith(f"[{ten}]") for l in bo_ansi(dau_ra_sessions).splitlines())


def can_tai_ckpt(mf, da_tai_sha, lan_cuoi, bay_gio, chu_ky):
    """Tải checkpoint khi có bản mới (khác sha đã tải) và đã đủ chu kỳ kể từ lần tải trước."""
    if not mf or mf.get("sha256") == da_tai_sha:
        return False
    return bay_gio - lan_cuoi >= chu_ky


def goi_can_tai(danh_sach_kq, da_tai):
    """Gói kết quả/chung chưa tải hoặc đã đổi hash (chung bị ghi đè mỗi lượt). Bỏ ckpt (xử lý riêng)."""
    return [m for m in danh_sach_kq if m and m.get("loai") != "ckpt" and da_tai.get(m["ten"]) != m["sha256"]]


def da_tai_du(done, da_tai, danh_sach_kq):
    """DONE đã ghi và mọi gói trong DONE.goi (trừ ckpt) đã tải đúng hash."""
    if not done:
        return False
    hien = {m["ten"]: m for m in danh_sach_kq if m}
    for ten in done.get("goi", []):
        m = hien.get(ten)
        if m is None or m.get("loai") == "ckpt":
            continue
        if da_tai.get(ten) != m["sha256"]:
            return False
    return True


def giai_nen_an_toan(tar_path, dich):
    dich = Path(dich)
    dich.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            goc = (dich / m.name).resolve()
            if not str(goc).startswith(str(dich.resolve())):
                raise RuntimeError(f"TAR_DUONG_DAN_LA:{m.name}")
        try:
            t.extractall(dich, filter="data")
        except TypeError:
            t.extractall(dich)


# ---------------------------------------------------------------- chạy thật
class May:
    def __init__(self, cfg):
        self.cfg = cfg
        self.exe = cfg["colab_exe"]
        self.phien = cfg["phien"]
        self.ra = Path(cfg["thu_muc_ra"])
        self.ra.mkdir(parents=True, exist_ok=True)
        self.tt_p = self.ra / "theo_doi_trang_thai.json"
        self.nk_p = self.ra / "theo_doi.log"
        self.tt = json.loads(self.tt_p.read_text(encoding="utf-8")) if self.tt_p.exists() else {"da_tai": {}, "ckpt_sha": None, "ckpt_luc": 0, "ckpt_giu": []}
        self.env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def log(self, *a):
        s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
        with open(self.nk_p, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def luu(self, **kw):
        self.tt.update(kw)
        self.tt["cap_nhat"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tam = self.tt_p.with_suffix(".tmp")
        tam.write_text(json.dumps(self.tt, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tam.replace(self.tt_p)

    def colab(self, args, timeout=120):
        r = subprocess.run([self.exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=self.env)
        return r.returncode, bo_ansi(r.stdout), bo_ansi(r.stderr)

    def thu_lai(self, args, timeout=120, lan=3):
        loi = ""
        for i in range(lan):
            try:
                rc, o, e = self.colab(args, timeout)
                if rc == 0:
                    return rc, o, e
                loi = (e or o)[-300:]
            except subprocess.TimeoutExpired:
                loi = "timeout"
            time.sleep(10 * (i + 1))
        self.log("colab", " ".join(args[:3]), "loi sau", lan, "lan:", loi)
        return 1, "", loi

    def ssh(self, lenh, timeout=120):
        return self.thu_lai(["ssh", "-s", self.phien, "-i", self.cfg["khoa_ssh"], "--", lenh], timeout)

    def doc_may(self):
        rc, o, e = self.ssh("python3 /content/giam_sat_vm.py doc")
        return tach_json(o) if rc == 0 else None

    def tai_phan(self, mf, d, ph):
        """Tải một phần 50 MB, kiểm byte + sha256; phần đã có và đúng thì bỏ qua (tải tiếp được sau khi bị ngắt)."""
        p = d / ph["ten"]
        if p.exists() and p.stat().st_size == ph["byte"] and sha256_file(p) == ph["sha256"]:
            return True
        tam = d / (ph["ten"] + ".tai")
        for lan in range(3):
            tam.unlink(missing_ok=True)
            # Đo 17/11/2026 trên máy CPU: `ssh -- cat` truyền nhị phân đúng hash, 200 MB trong 52 giây (3,8 MB/s) và không nạp cả file vào JSON base64 như `colab download`
            # (lệnh này hỏng với file 1,6 GB). Luồng thẳng ra file, không qua PowerShell nên không bị đổi mã.
            try:
                with open(tam, "wb") as f:
                    r = subprocess.run([self.exe, "ssh", "-s", self.phien, "-i", self.cfg["khoa_ssh"], "--", f"cat /content/kq/{mf['ten']}.parts/{ph['ten']}"],
                                       stdout=f, stderr=subprocess.PIPE, timeout=420, env=self.env)
                rc = r.returncode
            except subprocess.TimeoutExpired:
                rc = 1
            if rc == 0 and tam.exists() and tam.stat().st_size == ph["byte"] and sha256_file(tam) == ph["sha256"]:
                tam.replace(p)
                return True
            time.sleep(5 * (lan + 1))
        tam.unlink(missing_ok=True)
        return False

    def tai(self, mf, dich_tar):
        """Tải các phần của gói về thư mục riêng theo sha (song song vài luồng), ghép lại, kiểm sha256 cả tar; đúng thì True.
        Một lệnh `colab download` cho file 1,6 GB hỏng liên tục nên không tải cả file."""
        from concurrent.futures import ThreadPoolExecutor
        dich_tar.parent.mkdir(parents=True, exist_ok=True)
        d = dich_tar.parent / f"{mf['ten']}.phan_{mf['sha256'][:12]}"
        d.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=int(self.cfg.get("tai_song_song", 2))) as ex:      # đo: song song không tăng băng thông (~3,5 MB/s tổng); ssh giới hạn số kết nối đồng thời
            ket = list(ex.map(lambda ph: self.tai_phan(mf, d, ph), mf["phan"]))
        if not all(ket):
            self.log("tai thieu phan", mf["ten"], sum(1 for k in ket if not k), "/", len(ket), "(se thu lai o vong sau)")
            return False
        ghep = dich_tar.with_suffix(".ghep")
        with open(ghep, "wb") as out:
            for ph in mf["phan"]:
                with open(d / ph["ten"], "rb") as f:
                    shutil.copyfileobj(f, out, 2 ** 20)
        if ghep.stat().st_size != mf["byte"] or sha256_file(ghep) != mf["sha256"]:
            self.log("HASH SAI sau khi ghep", mf["ten"])
            ghep.unlink(missing_ok=True)
            shutil.rmtree(d, ignore_errors=True)
            return False
        ghep.replace(dich_tar)
        shutil.rmtree(d, ignore_errors=True)
        return True

    def tai_goi_ket_qua(self, mf):
        tar = self.ra / "goi" / f"{mf['ten']}.tar"
        if not self.tai(mf, tar):
            return False
        giai_nen_an_toan(tar, self.ra / "ket_qua")
        self.tt["da_tai"][mf["ten"]] = mf["sha256"]
        self.luu()
        self.log("da tai + kiem hash", mf["ten"], f"{mf['byte'] / 2 ** 20:.1f} MB")
        return True

    def sao_luu_ckpt(self, mf):
        ten = f"ckpt_{time.strftime('%Y%m%d_%H%M%S')}_{mf.get('viec')}_buoc{mf.get('buoc')}.tar"
        if not self.tai(mf, self.ra / "ckpt" / ten):
            return False
        giu = self.tt.get("ckpt_giu", []) + [ten]
        for cu in giu[:-int(self.cfg.get("giu_ckpt", 2))]:
            (self.ra / "ckpt" / cu).unlink(missing_ok=True)
        self.luu(ckpt_sha=mf["sha256"], ckpt_luc=time.time(), ckpt_giu=giu[-int(self.cfg.get("giu_ckpt", 2)):])
        self.log("sao luu checkpoint", ten, f"{mf['byte'] / 2 ** 20:.0f} MB")
        return True

    def ket_thuc(self, trang_thai):
        self.ssh("touch /content/TAI_XONG")
        rc, o, e = self.thu_lai(["stop", "-s", self.phien], timeout=180)
        time.sleep(10)
        _, ss, _ = self.colab(["sessions"], 60)
        _, us, _ = self.colab(["usage"], 60)
        con = phien_con_tren_server(ss, self.phien)
        self.log("stop rc", rc, "con tren server:", con, "|", us.strip().replace("\n", " | "))
        self.luu(xong=True, trang_thai_may=trang_thai, phien_con_tren_server=con, usage=us.strip())
        return not con

    def chay(self):
        chu_ky = int(self.cfg.get("chu_ky_giay", 300))
        han = time.time() + float(self.cfg.get("tran_theo_doi_gio", 16)) * 3600
        mat_phien, loi_lien_tiep = 0, 0
        self.log("bat dau theo doi", self.phien)
        while time.time() < han:
            try:
                rc, ss, _ = self.colab(["sessions"], 60)
                if rc == 0 and not phien_con_tren_server(ss, self.phien):
                    mat_phien += 1
                    self.log("phien khong con tren server", mat_phien)
                    if mat_phien >= 3:
                        self.luu(xong=True, trang_thai_may="MAT_PHIEN")
                        return 2
                    time.sleep(60)
                    continue
                mat_phien = 0
                tt = self.doc_may()
                if tt is None:
                    loi_lien_tiep += 1
                    self.log("khong doc duoc trang thai may", loi_lien_tiep)
                    if loi_lien_tiep >= 12:
                        self.log("CANH BAO: 12 lan doc loi lien tiep, van tiep tuc")
                    time.sleep(chu_ky)
                    continue
                loi_lien_tiep = 0
                st = tt.get("status") or {}
                self.luu(may=st, done=tt.get("done"))
                kq = tt.get("kq") or []
                for mf in goi_can_tai(kq, self.tt["da_tai"]):
                    self.tai_goi_ket_qua(mf)
                ck = next((m for m in kq if m and m.get("loai") == "ckpt"), None)
                # Đo 18/11/2026: máy Colab biến mất 10–20 phút sau DONE dù hạn chờ là 30 phút, nên khi đã DONE tuyệt đối không để tải checkpoint (8 phút) chặn việc tải kết quả
                if not tt.get("done") and can_tai_ckpt(ck, self.tt.get("ckpt_sha"), self.tt.get("ckpt_luc", 0), time.time(), int(self.cfg.get("chu_ky_ckpt_giay", 1800))):
                    self.sao_luu_ckpt(ck)
                done = tt.get("done")
                if done:
                    kq2 = (self.doc_may() or {}).get("kq") or kq
                    for mf in goi_can_tai(kq2, self.tt["da_tai"]):
                        self.tai_goi_ket_qua(mf)
                    ck2 = next((m for m in kq2 if m and m.get("loai") == "ckpt"), None)
                    if done.get("status") != "OK" and ck2 and ck2.get("sha256") != self.tt.get("ckpt_sha"):
                        self.sao_luu_ckpt(ck2)
                    if da_tai_du(done, self.tt["da_tai"], kq2):
                        ok = self.ket_thuc(done.get("status"))
                        self.log("KET THUC", done.get("status"), "da dung may" if ok else "CHUA DUNG DUOC MAY, kiem tay")
                        return 0 if ok else 3
                    self.log("DONE nhung chua tai du, thu lai")
            except Exception:  # noqa: BLE001
                self.log("loi vong lap", traceback.format_exc()[-600:])
            time.sleep(chu_ky)
        self.log("het han theo doi, chua xong")
        self.luu(het_han=True)
        return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cau-hinh", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.cau_hinh).read_text(encoding="utf-8"))
    sys.exit(May(cfg).chay())


if __name__ == "__main__":
    main()
