# -*- coding: utf-8 -*-
"""Bộ bao job chạy TRÊN máy Colab (chỉ dùng thư viện chuẩn; không chạy ở máy nhà).

Việc làm: dựng môi trường như phiên Kaggle (/kaggle/input, /kaggle/working), chạy kaggle_train.main theo từng lượt 30 phút
(mỗi lượt kết thúc bằng checkpoint của chính mã huấn luyện), đóng gói kết quả và checkpoint thành file .tar kèm sha256 để máy nhà tải,
ghi nhịp tim status.json, và khi xong thì ghi DONE.json rồi chờ máy nhà báo "TAI_XONG". Quá hạn chờ thì tự trả máy (runtime.unassign).

Giao thức thư mục /content:
  status.json   nhịp tim, ghi mỗi ~20 giây
  kq/<ten>.tar + kq/<ten>.json   gói kết quả; .json (có sha256) chỉ xuất hiện SAU khi .tar đã đóng xong
  DONE.json     job kết thúc (status: OK | BAT_THUONG | LOI | TIMEOUT | DUNG_THEO_YEU_CAU | LOI_GIAM_SAT)
  TAI_XONG      máy nhà tạo sau khi tải và kiểm hash xong -> bộ bao thôi chờ
  DUNG          máy nhà tạo để yêu cầu dừng sớm
Chạy bằng exec (trong kernel, để runtime.unassign hoạt động):
  import sys, threading; sys.path.insert(0, '/content'); import giam_sat_vm as G
  threading.Thread(target=G.chay, args=('/content/job/job.json',), name='giam_sat').start()
Đọc trạng thái từ máy nhà:  python3 /content/giam_sat_vm.py doc
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

GOC = Path("/content")
KQ = GOC / "kq"
TRANG_THAI = GOC / "status.json"
DONE = GOC / "DONE.json"
TAI_XONG = GOC / "TAI_XONG"
DUNG = GOC / "DUNG"
NHAT_KY = GOC / "giam_sat.log"
# Không dùng /kaggle/...: trên Colab /kaggle là hệ thống chỉ đọc (lỗi Errno 30, thử 17/11/2026). kaggle_train.VAO/RA được gán lại trong mã mỗi lượt (chay_mot_luot).
KAGGLE_VAO = GOC / "kg" / "input"
KAGGLE_RA = GOC / "kg" / "working"
LIBS = GOC / "libs"
MA = GOC / "code"
_TT = {}
_KHOA = threading.Lock()


# ---------------------------------------------------------------- hàm thuần (có test ở máy nhà)
def bay_gio():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ghi_json(duong_dan, obj):
    tam = Path(str(duong_dan) + ".tmp")
    tam.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tam.replace(duong_dan)


def sha256_file(duong_dan):
    h = hashlib.sha256()
    with open(duong_dan, "rb") as f:
        for k in iter(lambda: f.read(2 ** 20), b""):
            h.update(k)
    return h.hexdigest()


TAM_PHAN = 100 * 2 ** 20   # `colab download` một file 1,6 GB hỏng liên tục (17/11/2026: "Download failed"): cắt phần 100 MB, tải bằng `ssh -- cat` và kiểm hash từng phần


def chia_phan(tar, thu_muc_phan, kich_phan=TAM_PHAN):
    """Cắt tar thành các phần liên tiếp 0000, 0001, ... kèm sha256 từng phần. Trả danh sách {ten, byte, sha256}."""
    shutil.rmtree(thu_muc_phan, ignore_errors=True)
    Path(thu_muc_phan).mkdir(parents=True)
    ds = []
    with open(tar, "rb") as f:
        i = 0
        while True:
            buf = f.read(kich_phan)
            if not buf:
                break
            p = Path(thu_muc_phan) / f"{i:04d}"
            p.write_bytes(buf)
            ds.append({"ten": p.name, "byte": len(buf), "sha256": hashlib.sha256(buf).hexdigest()})
            i += 1
    return ds


def tao_goi(ten, thu_muc_goc, danh_sach_tep, thu_muc_ra=None, them=None, kich_phan=TAM_PHAN):
    """Đóng danh_sach_tep (đường dẫn tuyệt đối dưới thu_muc_goc) thành <ten>.tar, cắt phần vào <ten>.parts/; ghi <ten>.json (sha256 cả tar + từng phần) SAU CÙNG.
    Xóa .json cũ trước để người tải không đọc nhầm hash của bản đang bị thay."""
    ra = Path(thu_muc_ra or KQ)
    ra.mkdir(parents=True, exist_ok=True)
    tar, part, js = ra / f"{ten}.tar", ra / f"{ten}.tar.part", ra / f"{ten}.json"
    js.unlink(missing_ok=True)
    with tarfile.open(part, "w") as t:
        for p in danh_sach_tep:
            t.add(p, arcname=str(Path(p).relative_to(thu_muc_goc)).replace(os.sep, "/"), recursive=False)
    part.replace(tar)
    phan = chia_phan(tar, ra / f"{ten}.parts", kich_phan)
    mf = {"ten": ten, "tar": tar.name, "byte": tar.stat().st_size, "sha256": sha256_file(tar), "luc": bay_gio(), "so_tep": len(danh_sach_tep), "phan": phan}
    mf.update(them or {})
    ghi_json(js, mf)
    return mf


def tep_ket_qua_viec(thu_ra, ten, giu_adapter):
    """Các tệp của một việc đã xong (ep2 + bản dịch). Không lấy moi_nhat (checkpoint, ~1,6 GB, đã bị xóa khi xong việc)."""
    d = Path(thu_ra) / ten
    ds = []
    if (d / "sinh_ep2.jsonl").exists():
        ds.append(d / "sinh_ep2.jsonl")
    if (d / "ep2").exists():
        for p in sorted((d / "ep2").rglob("*")):
            if p.is_file() and (giu_adapter or p.suffix not in (".safetensors", ".bin")):
                ds.append(p)
    return ds


def viec_xong(st, ten, thu_ra):
    v = (st.get("viec") or {}).get(ten) or {}
    return bool(v.get("cham")) and (Path(thu_ra) / ten / "sinh_ep2.jsonl").exists() and (Path(thu_ra) / ten / "ep2").exists()


def quyet_dinh_sau_luot(rc, st, so_loi_lien_tiep, buoc_truoc, buoc_nay):
    """Sau mỗi lượt: trả (hanh_dong, so_loi_moi). hanh_dong: 'XONG' | 'BAT_THUONG' | 'LOI' | 'TIEP'.
    Lượt lỗi lần 1 thì thử lại; lần 2 liên tiếp thì LOI. Lượt thoát bình thường mà không tiến bước nào ba lần thì cũng là LOI (kẹt)."""
    if st.get("bat_thuong"):
        return "BAT_THUONG", so_loi_lien_tiep
    if st.get("xong") is True and rc == 0:
        return "XONG", 0
    if rc != 0:
        so = so_loi_lien_tiep + 1
        return ("LOI" if so >= 2 else "TIEP"), so
    if buoc_nay <= buoc_truoc:
        so = so_loi_lien_tiep + 1
        return ("LOI" if so >= 3 else "TIEP"), so
    return "TIEP", 0


def tong_buoc_da_chay(st):
    return sum(int(v.get("buoc") or 0) for v in (st.get("viec") or {}).values())


# ---------------------------------------------------------------- nhật ký, trạng thái
def log(*a):
    s = f"{bay_gio()} " + " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(NHAT_KY, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    except OSError:
        pass


def cap_nhat(**kw):
    with _KHOA:
        _TT.update(kw)
        _TT["ts"] = bay_gio()
        _TT["ts_epoch"] = time.time()
        ghi_json(TRANG_THAI, _TT)


def doi_tuong_trang_thai(tt, ten):
    p = KAGGLE_RA / tt / f"{tt}_trang_thai.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


# ---------------------------------------------------------------- cài đặt
def sh(cmd, timeout=3600, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return r.returncode, (r.stdout or "")[-1500:], (r.stderr or "")[-1500:]


def cai_du_lieu(job):
    (KAGGLE_VAO / "pool").mkdir(parents=True, exist_ok=True)
    KAGGLE_RA.mkdir(parents=True, exist_ok=True)
    for ten, sha in job["pool_sha256"].items():
        nguon = GOC / "job" / "pool" / ten
        if sha256_file(nguon) != sha:
            raise RuntimeError(f"POOL_SAI_HASH:{ten}")
        shutil.copy2(nguon, KAGGLE_VAO / "pool" / ten)
    MA.mkdir(parents=True, exist_ok=True)
    for p in (GOC / "job" / "code").glob("*.py"):
        shutil.copy2(p, MA / p.name)


def cai_mo_hinh(job):
    mh = job["mo_hinh"]
    dich = KAGGLE_VAO / "mo_hinh" / mh["repo"].split("/")[-1]
    if (dich / "REVISION.txt").exists():
        return
    code = ("from huggingface_hub import snapshot_download\n"
            f"snapshot_download({mh['repo']!r}, revision={mh['revision']!r}, local_dir={str(dich)!r})\n")
    rc, o, e = sh([sys.executable, "-c", code], timeout=3600)
    if rc:
        raise RuntimeError(f"TAI_MO_HINH_LOI:{e}")
    shutil.rmtree(dich / ".cache", ignore_errors=True)
    for can in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (dich / can).exists():
            raise RuntimeError(f"MO_HINH_THIEU:{can}")
    if not any(p.suffix == ".safetensors" for p in dich.iterdir()):
        raise RuntimeError("MO_HINH_KHONG_CO_SAFETENSORS")
    (dich / "REVISION.txt").write_text(f"{mh['repo']}\n{mh['revision']}\n", encoding="utf-8")


VENV = GOC / "venv"
KIEM_THU_VIEN = ("import torch,transformers,peft,accelerate,pandas,sacrebleu;assert torch.cuda.is_available();"
                 "print(torch.__version__,transformers.__version__,peft.__version__)")


def python_chay():
    """Python dùng cho mỗi lượt huấn luyện: venv đã ghim nếu có, không thì Python của Colab."""
    p = VENV / "bin" / "python"
    return str(p) if (VENV / ".da_ghim").exists() and p.exists() else sys.executable


def moi_truong_chay():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def cai_thu_vien(job):
    """Dựng venv SẠCH và ghim torch/transformers/peft như phiên Kaggle đã chạy khảo sát tỉ lệ (2.10.0, 5.0.0, 0.18.1).
    Lần thử đầu (--target trộn với gói có sẵn của Colab) hỏng import; venv sạch tránh xung đột. Hỏng thì chạy bằng bản có sẵn của Colab
    (gỡ torchao cũ mà peft mới không chịu) và GHI RÕ vào status."""
    ly_do = None
    if job.get("ghim_phien_ban", True):
        pv = job.get("phien_ban", {"torch": "2.10.0", "transformers": "5.0.0", "peft": "0.18.1"})
        shutil.rmtree(VENV, ignore_errors=True)
        # Colab không có ensurepip cho venv thường (thử 17/11/2026: lỗi ở `-m venv`), nhưng có sẵn `uv`.
        uv = shutil.which("uv") or "/usr/local/bin/uv"
        rc, o, e = sh([uv, "venv", str(VENV), "--python", sys.executable], timeout=300)
        if rc == 0:
            cmd = [uv, "pip", "install", "--python", str(VENV / "bin" / "python"), "-q", "--link-mode", "copy", "--index-strategy", "unsafe-best-match",
                   f"torch=={pv['torch']}", f"transformers=={pv['transformers']}", f"peft=={pv['peft']}", "accelerate", "sacrebleu", "pandas", "numpy",
                   "--index-url", "https://download.pytorch.org/whl/cu128", "--extra-index-url", "https://pypi.org/simple"]
            rc, o, e = sh(cmd, timeout=3000)
            if rc == 0:
                (VENV / ".da_ghim").write_text("1")
                rc2, o2, e2 = sh([python_chay(), "-c", KIEM_THU_VIEN])
                if rc2 == 0:
                    return {"ghim": True, "ban": o2.strip()}
                ly_do = "import hong: " + e2[-300:]
            else:
                ly_do = "pip loi: " + e[-300:]
        else:
            ly_do = "tao venv loi: " + e[-300:]
        log("ghim phien ban that bai:", ly_do)
        shutil.rmtree(VENV, ignore_errors=True)
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], timeout=300)
    rc, o, e = sh([sys.executable, "-m", "pip", "install", "-q", "sacrebleu"], timeout=900)
    rc2, o2, e2 = sh([sys.executable, "-c", KIEM_THU_VIEN])
    return {"ghim": False, "ly_do": ly_do or "khong_yeu_cau_ghim", "ban": (o2 or e2[-200:]).strip(), "pip_sacrebleu_rc": rc}


# ---------------------------------------------------------------- vòng chạy
def doc_dong_cuoi(duong_dan, n=300):
    try:
        with open(duong_dan, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            return f.read().decode("utf-8", "replace").strip().splitlines()[-1][:n]
    except (OSError, IndexError):
        return ""


def gpu_nho():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def dong_goi_ket_qua(job, da_dong_goi):
    tt = job["ten_trang_thai"]
    thu_ra = KAGGLE_RA / tt
    st = doi_tuong_trang_thai(tt, None)
    for ten in job["viec"]:
        if ten not in da_dong_goi and viec_xong(st, ten, thu_ra):
            ds = tep_ket_qua_viec(thu_ra, ten, job.get("giu_adapter", True))
            tao_goi(f"ket_qua_{ten}", thu_ra, ds, them={"loai": "ket_qua", "viec": ten})
            da_dong_goi.add(ten)
            log("da dong goi ket qua", ten)
    # gói chung (nhỏ): trạng thái, báo cáo, nhật ký; ghi đè mỗi lượt
    chung = [p for p in [thu_ra / f"{tt}_trang_thai.json", *KAGGLE_RA.glob("bao_cao_*.json"), NHAT_KY, TRANG_THAI, *GOC.glob("log_luot_*.txt")] if Path(p).exists()]
    if chung:
        tao_goi("chung", Path("/"), chung, them={"loai": "chung"})
    return st


def dong_goi_checkpoint(job, st):
    """Checkpoint của việc đang chạy dở (moi_nhat + trạng thái), để tải về làm bản sao lưu."""
    tt = job["ten_trang_thai"]
    for ten in job["viec"]:
        mn = KAGGLE_RA / tt / ten / "moi_nhat"
        if mn.exists() and (mn / "opt.pt").exists() and not viec_xong(st, ten, KAGGLE_RA / tt):
            ds = [p for p in mn.rglob("*") if p.is_file()] + [KAGGLE_RA / tt / f"{tt}_trang_thai.json"]
            v = (st.get("viec") or {}).get(ten) or {}
            tao_goi("ckpt", KAGGLE_RA / tt, ds, them={"loai": "ckpt", "viec": ten, "buoc": v.get("buoc"), "tong_buoc": v.get("tong_buoc"), "epoch_xong": v.get("epoch_xong")})
            log("da dong goi checkpoint", ten, "buoc", v.get("buoc"))
            return
    (KQ / "ckpt.json").unlink(missing_ok=True)


def chay_mot_luot(job, so_luot, t_bat_dau, tran_giay):
    cfg = dict(job["cfg"])
    cfg["gio_toi_da"] = job["chunk_gio"]
    cfg_p = GOC / "job" / f"cfg_luot.json"
    ghi_json(cfg_p, cfg)
    log_p = GOC / f"log_luot_{so_luot:03d}.txt"
    code = (f"import json,sys\nfrom pathlib import Path\nsys.path.insert(0,{str(MA)!r})\nimport kaggle_train as K\n"
            f"K.VAO=Path({str(KAGGLE_VAO)!r})\nK.RA=Path({str(KAGGLE_RA)!r})\nK.cai_thu_vien=lambda: None\nK.main(json.load(open({str(cfg_p)!r})))\n")
    with open(log_p, "w", encoding="utf-8") as lg:
        proc = subprocess.Popen([python_chay(), "-c", code], stdout=lg, stderr=subprocess.STDOUT, env=moi_truong_chay(), cwd=str(MA))
        ly_do_dung = None
        while proc.poll() is None:
            time.sleep(20)
            st = doi_tuong_trang_thai(job["ten_trang_thai"], None)
            cap_nhat(luot=so_luot, dong_cuoi=doc_dong_cuoi(log_p), gpu=gpu_nho(), tien_do={t: {"epoch_xong": v.get("epoch_xong"), "buoc": v.get("buoc"), "tong_buoc": v.get("tong_buoc")}
                                                                                             for t, v in (st.get("viec") or {}).items()})
            if DUNG.exists():
                ly_do_dung = "DUNG_THEO_YEU_CAU"
            elif time.time() - t_bat_dau > tran_giay:
                ly_do_dung = "TIMEOUT"
            if ly_do_dung:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
    return proc.returncode, ly_do_dung


def cho_tai_va_tra_may(job):
    han = time.time() + job.get("cho_tai_phut", 180) * 60
    while time.time() < han and not TAI_XONG.exists():
        cap_nhat(giai_doan="cho_tai", con_lai_phut=round((han - time.time()) / 60, 1))
        time.sleep(20)
    log("tai xong" if TAI_XONG.exists() else "het han cho, tu tra may")
    cap_nhat(giai_doan="tra_may", tai_xong=TAI_XONG.exists())
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception:  # noqa: BLE001
        log("unassign loi", traceback.format_exc()[-400:])


def ket_thuc(job, trang_thai, ghi_chu, da_dong_goi):
    try:
        st = dong_goi_ket_qua(job, da_dong_goi)
        dong_goi_checkpoint(job, st)
    except Exception:  # noqa: BLE001
        log("dong goi cuoi loi", traceback.format_exc()[-600:])
    goi = sorted(p.stem for p in KQ.glob("*.json"))
    ghi_json(DONE, {"status": trang_thai, "luc": bay_gio(), "goi": goi, "ghi_chu": ghi_chu})
    cap_nhat(giai_doan="done", status=trang_thai)
    log("DONE", trang_thai, ghi_chu)
    cho_tai_va_tra_may(job)


def chay(duong_job):
    job = json.loads(Path(duong_job).read_text(encoding="utf-8"))
    for p in (DONE, TAI_XONG, DUNG):
        p.unlink(missing_ok=True)
    KQ.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tran = job["tran_gio"] * 3600
    da_dong_goi = set()
    cap_nhat(phien=job["phien"], giai_doan="cai_dat", bat_dau=bay_gio(), tran_gio=job["tran_gio"], viec=job["viec"])
    try:
        cai_du_lieu(job)
        cai_mo_hinh(job)
        tv = cai_thu_vien(job)
        cap_nhat(thu_vien=tv)
        log("cai dat xong", tv)
        loi_lt, so_luot = 0, 0
        while True:
            so_luot += 1
            tt = job["ten_trang_thai"]
            buoc_truoc = tong_buoc_da_chay(doi_tuong_trang_thai(tt, None))
            cap_nhat(giai_doan="huan_luyen", luot=so_luot)
            rc, ly_do = chay_mot_luot(job, so_luot, t0, tran)
            st = doi_tuong_trang_thai(tt, None)
            log("het luot", so_luot, "rc", rc, "buoc", tong_buoc_da_chay(st))
            if ly_do:
                return ket_thuc(job, ly_do, f"dung o luot {so_luot}", da_dong_goi)
            dong_goi_ket_qua(job, da_dong_goi)
            hd, loi_lt = quyet_dinh_sau_luot(rc, st, loi_lt, buoc_truoc, tong_buoc_da_chay(st))
            if hd == "TIEP":
                dong_goi_checkpoint(job, st)
                continue
            return ket_thuc(job, {"XONG": "OK", "BAT_THUONG": "BAT_THUONG", "LOI": "LOI"}[hd], st.get("bat_thuong") or f"rc={rc}", da_dong_goi)
    except Exception:  # noqa: BLE001
        log("LOI GIAM SAT", traceback.format_exc())
        ket_thuc(job, "LOI_GIAM_SAT", traceback.format_exc()[-800:], da_dong_goi)


def doc():
    """In tóm tắt cho máy nhà: <<JSON>>{...}<<FIN>>."""
    def js(p):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    out = {"status": js(TRANG_THAI), "done": js(DONE), "tai_xong": TAI_XONG.exists(), "kq": [js(p) for p in sorted(KQ.glob("*.json"))] if KQ.exists() else [], "may": time.time()}
    print("<<JSON>>" + json.dumps(out, ensure_ascii=False, default=str) + "<<FIN>>")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "doc":
        doc()
