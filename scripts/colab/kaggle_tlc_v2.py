# -*- coding: utf-8 -*-
"""Kaggle L4 x4 (tài khoản trinhnhatanh), lần 2 sau sự cố 18/11/2026: khảo sát tỉ trọng title bản CỤM (25%, 35% và đối chứng 11%, fold 0, seed 42), MỖI CARD MỘT VIỆC.

Sự cố lần 1: bộ bao `K.loss_tong = lambda *a, **k: _lt(*a, **{**k, 'ngan_sach': 4096})` bị gọi với ngan_sach theo vị trí -> TypeError ở bước đo loss cuối epoch 1, lúc đó chưa lưu
checkpoint nên mất 5,67 giờ. Sửa lần 2:
  (1) KHÔNG bọc hàm nữa: đổi giá trị mặc định (__defaults__) của loss_tong/loss_kiem/loss_nhom/sinh, có assert chữ ký cũ nên đổi sai là dừng ngay trước khi tốn GPU.
  (2) kaggle_train.chay_b78 có cờ cfg["bao_ve"]: lưu checkpoint NGAY khi hết epoch (trước bước đo), lưu định kỳ mỗi cfg["luu_moi_buoc"] bước (ghi vào thư mục tạm rồi đổi tên),
      lỗi ở bước đo phụ (loss_nhom, loss cả fold) chỉ ghi lại chứ không bỏ việc; tiến độ ghi cùng checkpoint nên chạy nối tiếp đúng bước.
  (3) Ba chế độ cùng MỘT mã chạy: `thu` (thử ngắn trên đúng L4x4 thật, có dừng giữa chừng rồi chạy nối tiếp, có thử bộ nhớ ở mẫu dài nhất), `that` (chạy thật, dừng có trật tự ở
      --gio giờ), `tiep` (phiên nối tiếp: gắn đầu ra phiên trước làm nguồn, kiểm trạng thái + checkpoint có sẵn TRƯỚC khi tốn GPU).
Không đọc test: không gắn Dataset test, kiem_khoa_test chạy ở tiến trình cha trước khi tốn GPU.

  python -I -X utf8 scripts/colab/kaggle_tlc_v2.py thu                       # chỉ dựng notebook (--day để đẩy lên)
  python -I -X utf8 scripts/colab/kaggle_tlc_v2.py that --gio 4.4 --day
  python -I -X utf8 scripts/colab/kaggle_tlc_v2.py tiep --gio 10.5 --nguon nlp-tlc-v2-that --day
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))

KAGGLE = GOC / ".venv" / "Scripts" / "kaggle.exe"
SLUG_QUOTA = "ft-qwen3-4b-quota"
FILE_MA = ("cham_diem.py", "kaggle_train.py", "kaggle_quota.py", "kaggle_cheo.py", "kaggle_dong6.py")
TEN_NB = {"thu": "nlp-tlc-v2-thu", "that": "nlp-tlc-v2-that", "tiep": "nlp-tlc-v2-tiep"}

VIEC = [("cum_f0_p25", 0.25), ("cum_f0_p35", 0.35), ("cum_f0_p11", 0.1074)]     # card 0, 1, 2

# Đổi mặc định ngân sách gom lô cho card L4 23 GB (loss 8.192 -> 4.096, sinh 32.768 -> 12.000); chữ ký cũ được assert để đổi sai là dừng ngay.
DAU = ("import sys, json\nsys.path.insert(0, '.')\nimport kaggle_train as K\nK.cai_thu_vien = lambda: None\n"
       "assert K.loss_tong.__defaults__ == (8192,) and K.loss_kiem.__defaults__ == (8192,) and K.loss_nhom.__defaults__ == (8192,), K.loss_tong.__defaults__\n"
       "assert K.sinh.__defaults__ == (32, 32768, 'dich', None), K.sinh.__defaults__\n"
       "K.loss_tong.__defaults__ = (4096,)\nK.loss_kiem.__defaults__ = (4096,)\nK.loss_nhom.__defaults__ = (4096,)\nK.sinh.__defaults__ = (32, 12000, 'dich', None)\n")


def cfg_chung(gio):
    return {"che_do": "b78", "micro_bs": 32, "tich_luy": 1, "ngan_sach_token": 4096, "gc": True, "max_len": 2048, "seed": 42, "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250,
            "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05, "vram_toi_thieu_gb": 20, "gio_toi_da": gio, "bao_ve": True, "luu_moi_buoc": 50}


def cau_hinh(che_do, gio):
    ch = cfg_chung(gio)
    if che_do == "thu":
        # Việc A: chạy 200 mẫu, DỪNG ở bước 5 (giữa epoch 1) rồi chạy tiếp ở tiến trình mới: thử lưu định kỳ, dừng, nối tiếp, cuối epoch, đo loss, lưu ep2, chấm.
        # Việc B: chạy thẳng một mạch, kiểm tra nhiều mẫu hơn ở các bước đo. Card 2: thử bộ nhớ ở các mẫu DÀI NHẤT (loss kiểm, loss cả lô, sinh).
        nho = {"so_train": 200, "so_kiem_moi_nhom": 250, "so_sinh_moi_nhom": 8, "so_mau_ca_fold": 600, "luu_moi_buoc": 3, "chrf_toi_thieu": None, "gio_toi_da": 1.0}
        viec = [{"ten": "thu_a", "nhanh": "cum", "fold": 0, "p_title": 0.25, "card": 0, "cfg_rieng": dict(nho, dung_sau_buoc=5), "hai_lan": True},
                {"ten": "thu_b", "nhanh": "cum", "fold": 0, "p_title": 0.1074, "card": 1, "cfg_rieng": dict(nho), "hai_lan": False}]
    else:
        viec = [{"ten": t, "nhanh": "cum", "fold": 0, "p_title": p, "card": i, "cfg_rieng": {}, "hai_lan": False} for i, (t, p) in enumerate(VIEC)]
    ra = {"che_do": che_do, "viec": viec, "chung": ch, "stress": che_do == "thu"}
    if che_do == "tiep":
        tep = GOC / "data" / "processed" / "ft_run" / "kaggle_tlc_v2_that" / "SHA256_checkpoint_phien1.txt"
        bam = {}
        for dong in tep.read_text(encoding="utf-8").splitlines():
            h, _, rel = dong.strip().partition(" *")
            if h and rel and not rel.endswith("tien_do.json"):
                bam[rel.replace("\\", "/")] = h
        assert len(bam) == 6, bam
        ra["bam_ck"] = bam
    return ra


STRESS = r'''
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import json, sys, time, torch
sys.path.insert(0, ".")
import kaggle_train as K
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
thu_mh = K.tim_mot("REVISION.txt").parent
tok = AutoTokenizer.from_pretrained(str(thu_mh))
tok.padding_side = "right"
ma, _ = K.xay_mau(K.doc_mau("cum"), tok, 2048)
goc = K.nap_goc(thu_mh); goc.config.use_cache = False
m = get_peft_model(goc, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.enable_input_require_grads()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
dung = sorted(ma, key=lambda x: -len(x["ids"]))
# 1 bước huấn luyện thật trên lô dài nhất (để trạng thái bộ nhớ giống lúc đo: có bộ tối ưu, có cache của huấn luyện)
m.train()
lo = [x for x in dung[:2]]
ids, lab, mask = K.lo_tensor(lo, tok, "cuda:0")
m(input_ids=ids, attention_mask=mask, labels=lab).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
kq = {"mib_sau_buoc": torch.cuda.max_memory_allocated() // 2**20}
t = time.time(); ln, dm = K.loss_nhom(m, tok, dung[:96], "cuda:0"); m.train()
kq["loss_nhom_96_dai_nhat"] = {"giay": round(time.time() - t, 1), "dm": dm, "mib_dinh": torch.cuda.max_memory_allocated() // 2**20}
t = time.time(); tg, dem = K.loss_tong(m, tok, dung[:96], 32, "cuda:0"); m.train()
kq["loss_tong_96_dai_nhat"] = {"giay": round(time.time() - t, 1), "mib_dinh": torch.cuda.max_memory_allocated() // 2**20}
mau_sinh = [x for x in K.doc_mau("cum") if x["phan"] == 0]
mau_sinh = sorted(mau_sinh, key=lambda x: -len(x["messages"][1]["content"]))[:32]
t = time.time(); du = K.sinh(m, tok, mau_sinh, "cuda:0"); m.train()
kq["sinh_32_dai_nhat"] = {"giay": round(time.time() - t, 1), "mib_dinh": torch.cuda.max_memory_allocated() // 2**20, "so_rong": sum(1 for r in du if not r["hyp"].strip())}
kq["tong_mib_card"] = torch.cuda.get_device_properties(0).total_memory // 2**20
kq["dat"] = kq["sinh_32_dai_nhat"]["mib_dinh"] < 21000
open("/kaggle/working/stress_bo_nho.json", "w").write(json.dumps(kq, indent=1))
print("STRESS XONG", json.dumps(kq), flush=True)
'''

CHAY = r'''import hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import kaggle_train as K
RA = Path("/kaggle/working")
VAO = Path("/kaggle/input")
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
(RA / "nvidia_smi.txt").write_text(smi, encoding="utf-8")
print(smi, flush=True)
bo_nho = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split("\n")
the = [(int(l.split(",")[0]), int(l.split(",")[1])) for l in bo_nho if l.strip()]
print("cards:", the, flush=True)
if len(the) < 4 or min(m for _, m in the) < 22000:
    raise SystemExit(f"KHONG_PHAI_L4x4: {the}")
K.kiem_khoa_test()          # trước khi tốn GPU: có tệp test của mình trong input thì dừng ngay
CH = CAU_HINH["che_do"]
if CH == "tiep":
    # phiên nối tiếp: mỗi việc phải có sẵn trạng thái VÀ checkpoint (hoặc bản ep2) từ phiên trước, thiếu thì DỪNG NGAY, không chạy lại từ đầu
    for v in CAU_HINH["viec"]:
        tt = sorted(VAO.rglob(f"tlc_{v['ten']}_trang_thai.json"))
        assert tt, f"KHONG_CO_TRANG_THAI_PHIEN_TRUOC:{v['ten']}"
        st = max((json.loads(p.read_text(encoding="utf-8")) for p in tt), key=lambda s: s["viec"][v["ten"]]["buoc"])["viec"][v["ten"]]
        ck = [p for p in VAO.rglob("opt.pt") if v["ten"] in str(p)] + [p for p in VAO.rglob("ep2") if v["ten"] in str(p)]
        assert st["buoc"] > 0 and ck, f"KHONG_CO_CHECKPOINT_PHIEN_TRUOC:{v['ten']}:buoc={st['buoc']}"
        print("noi tiep", v["ten"], "buoc", st["buoc"], "epoch_xong", st["epoch_xong"], flush=True)
    # checkpoint tải lên phải ĐÚNG từng bit so với bản đã kiểm ở phiên 1 (băm sha256)
    for rel, bam in CAU_HINH["bam_ck"].items():
        fp = [p for p in sorted(VAO.rglob(rel.split("/")[-1])) if str(p).replace("\\", "/").endswith(rel)]
        assert len(fp) == 1, f"CHECKPOINT_THIEU_HOAC_TRUNG:{rel}:{len(fp)}"
        h = hashlib.sha256()
        with open(fp[0], "rb") as f:
            for kh in iter(lambda: f.read(1 << 24), b""):
                h.update(kh)
        assert h.hexdigest() == bam, f"CHECKPOINT_SAI_BAM:{rel}"
        print("checkpoint khop bam:", rel, flush=True)
    # dữ liệu phải GIỐNG HỆT phiên trước (tài khoản khác có thể giữ bản pool khác): băm nội dung đã giải nén, so với MANIFEST của goi_pool
    import gzip
    BAM = {"pool_cum": "51acf0e04890013b8b0d94d6ce059356e4de128d7c82ef79bd9889116adfd766", "pool_pair": "c47fc93a10c5cca7fa642e436eff67739744ae67c3f48b73bbc99455e6434cc4"}
    for ten_p, bam in BAM.items():
        fp = [p for p in sorted(VAO.rglob(ten_p + ".jsonl*")) if p.is_file()]
        assert fp, "KHONG_THAY_POOL:" + ten_p
        with open(fp[0], "rb") as f:
            nen = f.read(2) == b"\x1f\x8b"
        h = hashlib.sha256()
        with (gzip.open(fp[0], "rb") if nen else open(fp[0], "rb")) as f:
            for kh in iter(lambda: f.read(1 << 20), b""):
                h.update(kh)
        assert h.hexdigest() == bam, f"POOL_KHAC_PHIEN_TRUOC:{ten_p}:{h.hexdigest()}"
        print("pool khop MANIFEST:", ten_p, flush=True)
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
(RA / "moi_truong.json").write_text(json.dumps(mt, indent=1), encoding="utf-8")
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

xong, t0 = {}, time.time()
def chay_viec(v):
    """Chạy một việc trên card của nó. hai_lan: lần 1 có dung_sau_buoc, lần 2 là TIẾN TRÌNH MỚI chạy nối tiếp (thử đường nối tiếp thật)."""
    kq = []
    for lan in (1, 2) if v["hai_lan"] else (1,):
        cfg = dict(CAU_HINH["chung"], ten_trang_thai="tlc_" + v["ten"], viec=[{k: v[k] for k in ("ten", "nhanh", "fold", "p_title")} | {"ca_fold": False}], **v["cfg_rieng"])
        if lan == 2:
            cfg.pop("dung_sau_buoc")
        ma = DAU + "K.main(json.loads(%r))\n" % json.dumps(cfg)
        with open(RA / f"log_{v['ten']}.log", "a") as lg:
            p = subprocess.Popen([sys.executable, "-c", ma], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(v["card"])), stdout=lg, stderr=subprocess.STDOUT)
            rc = p.wait()
        kq.append(rc)
        if rc != 0:
            break
    xong[v["ten"]] = {"rc": kq, "giay": round(time.time() - t0)}
luong = []
for v in CAU_HINH["viec"]:
    luong.append(threading.Thread(target=chay_viec, args=(v,), daemon=True)); luong[-1].start()
if CAU_HINH["stress"]:
    def chay_stress():
        with open(RA / "log_stress.log", "w") as lg:
            rc = subprocess.Popen([sys.executable, "-c", DAU + STRESS], env=dict(os.environ, CUDA_VISIBLE_DEVICES="2"), stdout=lg, stderr=subprocess.STDOUT).wait()
        xong["stress"] = {"rc": rc, "giay": round(time.time() - t0)}
    luong.append(threading.Thread(target=chay_stress, daemon=True)); luong[-1].start()
ten_log = [v["ten"] for v in CAU_HINH["viec"]] + (["stress"] if CAU_HINH["stress"] else [])
n_can = len(ten_log)
cho = 120 if CH == "thu" else 600
while len(xong) < n_can:
    time.sleep(cho)
    dong = []
    for ten in ten_log:
        try:
            ls = [l for l in (RA / f"log_{ten}.log").read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        except Exception:
            ls = []
        dong.append(f"{ten}: {'XONG ' + json.dumps(xong[ten]) if ten in xong else 'dang chay'} | {(ls[-1] if ls else '')[:170]}")
    du = subprocess.run(["df", "-m", "/kaggle/working"], capture_output=True, text=True).stdout.split("\n")[-2].split()
    print(f"[{round((time.time() - t0) / 60)} phut] dia dung {du[2]} MB, con {du[3]} MB\n" + "\n".join(dong), flush=True)
dung.set()
for d in dinh.values():
    d["util_tb"] = round(d.pop("util_tb_tong") / max(d["so_mau"], 1), 1)
(RA / "tong_ket.json").write_text(json.dumps({"che_do": CH, "tong_giay": round(time.time() - t0, 1), "ket_thuc": xong, "dinh_theo_card": dinh}, indent=1), encoding="utf-8")
print("XONG", json.dumps(xong), round(time.time() - t0), "giay", json.dumps(dinh), flush=True)
'''


def dung_notebook(che_do, gio, acc, cuoc_thi, nguon, nguon_ds=None, thu_vien=None):
    import dong_goi_kaggle as DG
    slug = TEN_NB[che_do]
    thu = GOC / "notebooks" / "kaggle_api" / slug
    goc = GOC / "scripts" / "ft_train"
    ghi = "".join(f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n" for ten in FILE_MA)
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cau_hinh(che_do, gio)) + "\nFILE_MA = " + repr(FILE_MA) + "\n"
    o2 = "from pathlib import Path\n" + ghi
    o3 = "DAU = " + repr(DAU) + "\nSTRESS = " + repr(STRESS) + "\n"
    cells = [o1, o2, o3, CHAY]
    thu.mkdir(parents=True, exist_ok=True)
    tieu = f"# Khảo sát tỉ trọng title (cụm, fold 0), Kaggle L4 x4, chế độ {che_do}, mỗi card một việc, có checkpoint chạy nối tiếp\n\nKhông có dữ liệu test."
    (thu / f"{slug}.ipynb").write_text(json.dumps(DG.tao_notebook(cells, tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ks = [f"{DG.NGUOI_DUNG}/{thu_vien or DG.SLUG_THU_VIEN}", f"{DG.NGUOI_DUNG}/{DG.SLUG_MO_HINH}"] + ([f"{DG.NGUOI_DUNG}/{nguon}"] if nguon else [])
    meta = {"id": f"{DG.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": acc,
            "dataset_sources": [f"{DG.NGUOI_DUNG}/{DG.SLUG_POOL}", f"{DG.NGUOI_DUNG}/{SLUG_QUOTA}"] + ([f"{DG.NGUOI_DUNG}/{nguon_ds}"] if nguon_ds else []), "competition_sources": [cuoc_thi] if cuoc_thi else [],
            "kernel_sources": ks, "model_sources": []}
    (thu / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return thu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("che_do", choices=["thu", "that", "tiep"])
    ap.add_argument("--gio", type=float, default=None, help="trần giờ thực của mỗi tiến trình (tính từ lúc tiến trình bắt đầu); quá trần thì lưu checkpoint rồi dừng có trật tự")
    ap.add_argument("--acc", default="NvidiaL4")
    ap.add_argument("--cuoc-thi", default="gemma-4-developer-agent")
    ap.add_argument("--nguon", default=None, help="chế độ tiep: slug notebook phiên trước (đầu ra của nó được gắn làm nguồn)")
    ap.add_argument("--nguon-ds", default=None, help="chế độ tiep sang TÀI KHOẢN KHÁC: slug Dataset (cùng tài khoản đang đăng nhập) chứa đầu ra phiên trước đã tải về rồi tải lên")
    ap.add_argument("--thu-vien", default=None, help="slug notebook thư viện (mặc định ft-qwen3-4b-thu-vien); tài khoản 2 dùng bản peft 0.21.2 để GIỐNG phiên 1")
    ap.add_argument("--day", action="store_true", help="đẩy lên Kaggle (mặc định chỉ dựng notebook)")
    a = ap.parse_args()
    if a.che_do == "tiep" and not (a.nguon or a.nguon_ds):
        raise SystemExit("che do tiep can --nguon (cung tai khoan) hoac --nguon-ds (khac tai khoan)")
    if a.nguon and a.nguon_ds:
        raise SystemExit("chi chon mot trong --nguon / --nguon-ds")
    if a.che_do != "thu" and a.gio is None:
        raise SystemExit("can --gio cho che do that/tiep")
    thu = dung_notebook(a.che_do, a.gio if a.gio is not None else 1.0, a.acc, a.cuoc_thi, a.nguon, a.nguon_ds, a.thu_vien)
    print("da dung", thu)
    if not a.day:
        return
    r = subprocess.run([str(KAGGLE), "kernels", "push", "-p", str(thu), "--accelerator", a.acc], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-800:])


if __name__ == "__main__":
    main()
