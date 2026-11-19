# -*- coding: utf-8 -*-
"""Chạy THẬT song song trên Kaggle L4 x4 (tài khoản trinhnhatanh), 18/11/2026: ba việc khảo sát tỉ trọng title mức 25%, 35% và đối chứng 11% (bản CỤM, fold 0, 1 seed, trả lời
phản biện vòng 2 mục 3.3.5) và bản dịch zero-shot Qwen3-4B trên eval 1.200 mẫu (mục 1.2(d)). Mỗi card một tiến trình riêng:

  card 0  cum_f0_p25      card 1  cum_f0_p35      card 2  cum_f0_p11 (đối chứng, p_title 0,1074 như phiên Kaggle cũ)      card 3  zero-shot Qwen3-4B (cụm rồi pair)

Cùng cấu hình các phiên đã đo p11 và p75 (kaggle_train.chay_b78): seed 42, lr 2e-4, rank 64, 2 epoch, micro_bs 32, ngân sách 4.096 token, max_len 2.048, 750 mẫu kiểm và 250 mẫu sinh
mỗi nhóm. KHÁC phiên cũ (ghi vào báo cáo, không giấu): (1) card L4 23 GB thay vì RTX Pro 6000 nên bật gradient checkpointing (không đổi phép tính, chỉ đổi bộ nhớ), (2) ngân sách gom lô
của loss kiểm 4.096 thay 8.192 và của sinh 12.000 thay 32.768 để không tràn bộ nhớ (gom lô khác có thể làm số học bf16 lệch rất nhẹ), (3) thư viện của ảnh Kaggle (torch/transformers),
(4) lần chạy trên Kaggle thay vì Colab. Hệ quả: phải coi việc đối chứng p11 là thước đo độ lệch do môi trường, không so thẳng với điểm cũ mà không qua nó.

Không đọc test: không gắn Dataset test; kiem_khoa_test chạy ở tiến trình cha trước khi tốn GPU (chỉ bỏ qua thư mục dữ liệu của cuộc thi gắn kèm, xem CUOC_THI_BO_QUA).
Chạy:  python -I -X utf8 scripts/colab/kaggle_that.py            (không --cho: chỉ đẩy lên, không chờ)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import dong_goi_kaggle as DG  # noqa: E402

KAGGLE = GOC / ".venv" / "Scripts" / "kaggle.exe"
SLUG = "nlp-tlc-song-song"
THU = GOC / "notebooks" / "kaggle_api" / SLUG
SLUG_QUOTA = "ft-qwen3-4b-quota"
FILE_MA = ("cham_diem.py", "kaggle_train.py", "kaggle_quota.py", "kaggle_cheo.py", "kaggle_dong6.py", "kaggle_zero_shot.py")

VIEC = [("cum_f0_p25", "cum", 0.25), ("cum_f0_p35", "cum", 0.35), ("cum_f0_p11", "cum", 0.1074)]


def cfg_chung():
    return {"che_do": "b78", "micro_bs": 32, "tich_luy": 1, "ngan_sach_token": 4096, "gc": True, "max_len": 2048, "seed": 42, "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250,
            "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05, "vram_toi_thieu_gb": 20, "gio_toi_da": 10.5}


def cau_hinh():
    return {"viec": [{"ten": t, "nhanh": n, "fold": 0, "p_title": p, "ca_fold": False} for t, n, p in VIEC], "chung": cfg_chung(),
            "zero_shot": {"tap": ["cum", "pair"], "dtype": "bf16", "bs": 32, "ngan_sach": 12000}}


CHAY = r'''import hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import kaggle_train as K
RA = Path("/kaggle/working")
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
(RA / "nvidia_smi.txt").write_text(smi, encoding="utf-8")
print(smi, flush=True)
bo_nho = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split("\n")
the = [(int(l.split(",")[0]), int(l.split(",")[1])) for l in bo_nho if l.strip()]
print("cards:", the, flush=True)
if len(the) < 4 or min(m for _, m in the) < 22000:
    raise SystemExit(f"KHONG_PHAI_L4x4: {the}")
K.kiem_khoa_test()          # trước khi tốn GPU: có tệp test của mình trong input thì dừng ngay
K.cai_thu_vien()
print(subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True).stdout[-300:], flush=True)   # ảnh Kaggle: torchao 0.10 không hợp peft
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
import torch, transformers, peft
mt = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "python": sys.version.split()[0],
      "ma_nguon": {n: sha(n) for n in FILE_MA}, "du_lieu": {str(p.relative_to("/kaggle/input")): sha(p) for pat in ("pool_*.jsonl*", "eval_*_sample.csv") for p in sorted(Path("/kaggle/input").rglob(pat)) if p.is_file()}}
(RA / "moi_truong_that.json").write_text(json.dumps(mt, indent=1), encoding="utf-8")
print("moi truong:", json.dumps({k: v for k, v in mt.items() if k != "du_lieu"}), flush=True)

dung, dinh = threading.Event(), {}
def lay_mau():
    while not dung.is_set():
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for l in o.strip().splitlines():
            i, m, u = [int(x) for x in l.split(",")]
            d = dinh.setdefault(i, {"mib_dinh": 0, "util_tb_tong": 0, "so_mau": 0})
            d["mib_dinh"] = max(d["mib_dinh"], m); d["util_tb_tong"] += u; d["so_mau"] += 1
        time.sleep(10)
threading.Thread(target=lay_mau, daemon=True).start()

DAU = ("import sys, json\nsys.path.insert(0, '.')\nimport kaggle_train as K\nK.cai_thu_vien = lambda: None\n"
       "_ln, _lt, _s = K.loss_nhom, K.loss_tong, K.sinh\n"
       "K.loss_nhom = lambda *a, **k: _ln(*a, **{**k, 'ngan_sach': 4096})\n"
       "K.loss_tong = lambda *a, **k: _lt(*a, **{**k, 'ngan_sach': 4096})\n"
       "K.sinh = lambda *a, **k: _s(*a, **{'ngan_sach': 12000, **k})\n")
t0, cacs = time.time(), []
for i, v in enumerate(CAU_HINH["viec"]):
    cfg = dict(CAU_HINH["chung"], ten_trang_thai="tlc_" + v["ten"], viec=[v])
    ma = DAU + "K.main(json.loads(%r))\n" % json.dumps(cfg)
    cacs.append((v["ten"], i, subprocess.Popen([sys.executable, "-c", ma], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)),
                                               stdout=open(RA / f"log_{v['ten']}.log", "w"), stderr=subprocess.STDOUT)))
ma = DAU + "import kaggle_zero_shot as Z\nZ.main(json.loads(%r))\n" % json.dumps(CAU_HINH["zero_shot"])
cacs.append(("zero_shot", 3, subprocess.Popen([sys.executable, "-c", ma], env=dict(os.environ, CUDA_VISIBLE_DEVICES="3"), stdout=open(RA / "log_zero_shot.log", "w"), stderr=subprocess.STDOUT)))
xong = {}
while len(xong) < len(cacs):
    time.sleep(600)
    for ten, i, p in cacs:
        if ten not in xong and p.poll() is not None:
            xong[ten] = {"rc": p.returncode, "giay": round(time.time() - t0)}
    dong = []
    for ten, i, p in cacs:
        try:
            ls = [l for l in (RA / f"log_{ten}.log").read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        except Exception:
            ls = []
        dong.append(f"{ten}: {'XONG rc=' + str(xong[ten]['rc']) if ten in xong else 'dang chay'} | {(ls[-1] if ls else '')[:160]}")
    print(f"[{round((time.time() - t0) / 60)} phut]\n" + "\n".join(dong), flush=True)
dung.set()
for d in dinh.values():
    d["util_tb"] = round(d.pop("util_tb_tong") / max(d["so_mau"], 1), 1)
(RA / "tong_ket_that.json").write_text(json.dumps({"tong_giay": round(time.time() - t0, 1), "ket_thuc": xong, "dinh_theo_card": dinh}, indent=1), encoding="utf-8")
print("XONG", json.dumps(xong), round(time.time() - t0), "giay", json.dumps(dinh), flush=True)
'''


def dung_notebook(acc, cuoc_thi):
    goc = GOC / "scripts" / "ft_train"
    ghi = "".join(f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n" for ten in FILE_MA)
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cau_hinh()) + "\nFILE_MA = " + repr(FILE_MA) + "\n"
    o2 = "from pathlib import Path\n" + ghi
    cells = [o1, o2, CHAY]
    THU.mkdir(parents=True, exist_ok=True)
    tieu = "# Khảo sát tỉ trọng title (cụm, fold 0, 25% 35% đối chứng 11%) + zero-shot Qwen3-4B trên Kaggle L4 x4, mỗi card một tiến trình\n\nKhông có dữ liệu test."
    (THU / f"{SLUG}.ipynb").write_text(json.dumps(DG.tao_notebook(cells, tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    meta = {"id": f"{DG.NGUOI_DUNG}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": acc,
            "dataset_sources": [f"{DG.NGUOI_DUNG}/{DG.SLUG_POOL}", f"{DG.NGUOI_DUNG}/{SLUG_QUOTA}"], "competition_sources": [cuoc_thi] if cuoc_thi else [],
            "kernel_sources": [f"{DG.NGUOI_DUNG}/{DG.SLUG_THU_VIEN}", f"{DG.NGUOI_DUNG}/{DG.SLUG_MO_HINH}"], "model_sources": []}
    (THU / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="NvidiaL4")
    ap.add_argument("--cuoc-thi", default="gemma-4-developer-agent")
    ap.add_argument("--chi-dung", action="store_true", help="chỉ dựng notebook, không đẩy")
    a = ap.parse_args()
    dung_notebook(a.acc, a.cuoc_thi)
    if a.chi_dung:
        print("da dung", THU)
        return
    r = subprocess.run([str(KAGGLE), "kernels", "push", "-p", str(THU), "--accelerator", a.acc], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-800:])


if __name__ == "__main__":
    main()
