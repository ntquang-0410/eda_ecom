# -*- coding: utf-8 -*-
"""Chạy thử trên Kaggle L4 x4 (tài khoản trinhnhatanh): bộ nhớ đỉnh và tốc độ huấn luyện Qwen3-4B + LoRA r64 khi bật gradient checkpointing, MỖI CARD MỘT TIẾN TRÌNH.
Không đọc test; chỉ gắn Dataset pool, notebook thư viện và notebook mô hình như các phiên 4B cũ. Không in token.

Chạy:  python -I -X utf8 scripts/colab/kaggle_l4x4_thu.py [--acc NvidiaL4X4] [--cuoc-thi gemma-4-developer-agent]
Kết quả: Kaggle ghi /kaggle/working/{nvidia_smi.txt, vram_l4.json, bao_cao_l4_<i>.json, l4_<i>.log}; tải về bằng `kaggle kernels output`.
Máy chủ có thể IM LẶNG dùng GPU khác nếu tên bộ tăng tốc sai (SDK ghi rõ), nên notebook in nvidia-smi và dừng nếu không đủ 4 card ≥ 22 GiB."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import dong_goi_kaggle as DG  # noqa: E402

KAGGLE = GOC / ".venv" / "Scripts" / "kaggle.exe"
SLUG = "nlp-l4x4-thu"
THU = GOC / "notebooks" / "kaggle_api" / SLUG

CAU_HINH_DO = {"che_do": "do_toc_do", "nhanh": "cum", "fold_kiem": 0, "so_buoc": 100, "lr": 2e-4, "rank": 64, "micro_bs": 32, "tich_luy": 1, "ngan_sach_token": 4096,
               "gc": True, "max_len": 2048, "seed": 42, "so_mau_kiem": 300, "so_mau_sinh": 12, "so_buoc_do": 23,
               "cac_cau_hinh": [{"micro_bs": 32, "ngan_sach": 4096, "gc": True}, {"micro_bs": 32, "ngan_sach": 6144, "gc": True}]}

CHAY = r'''import json, os, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, ".")
import kaggle_train as K
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
Path("/kaggle/working/nvidia_smi.txt").write_text(smi, encoding="utf-8")
print(smi, flush=True)
bo_nho = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split("\n")
the = [(int(l.split(",")[0]), int(l.split(",")[1])) for l in bo_nho if l.strip()]
print("cards:", the, flush=True)
if len(the) < 4 or min(m for _, m in the) < 22000:
    raise SystemExit(f"KHONG_PHAI_L4x4: {the}")
K.cai_thu_vien()      # cài một lần ở tiến trình cha, tránh bốn tiến trình cùng pip install
# Lần 1 (18/11) lỗi: ảnh Kaggle có torchao 0.10.0 mà peft đòi >0.16 -> gỡ torchao (gỡ không cần mạng), peft sẽ bỏ qua nhánh torchao
print(subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True).stdout[-300:], flush=True)
dung, dinh = threading.Event(), {}
def lay_mau():
    while not dung.is_set():
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for l in o.strip().splitlines():
            i, m, u = [int(x) for x in l.split(",")]
            d = dinh.setdefault(i, {"mib_dinh": 0, "util_tb_tong": 0, "so_mau": 0})
            d["mib_dinh"] = max(d["mib_dinh"], m); d["util_tb_tong"] += u; d["so_mau"] += 1
        time.sleep(3)
threading.Thread(target=lay_mau, daemon=True).start()
t0, cacs = time.time(), []
for i in range(4):
    cfg = dict(CAU_HINH, ten_trang_thai=f"l4_{i}")
    # Chỉ bản thử: gắn cuộc thi nên /kaggle/input có tệp của cuộc thi, kiem_khoa_test (chỉ có ngoại lệ cho arc-prize) sẽ dừng. Bản thử chỉ đọc pool, không có tệp test của mình.
    ma = ("import sys, json\nsys.path.insert(0, '.')\nimport kaggle_train as K\nK.cai_thu_vien = lambda: None\nK.kiem_khoa_test = lambda: None\nK.main(json.loads(%r))\n" % json.dumps(cfg))
    cacs.append((i, subprocess.Popen([sys.executable, "-c", ma], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)),
                                     stdout=open(f"/kaggle/working/l4_{i}.log", "w"), stderr=subprocess.STDOUT)))
rc = {i: p.wait() for i, p in cacs}
dung.set()
for d in dinh.values():
    d["util_tb"] = round(d.pop("util_tb_tong") / max(d["so_mau"], 1), 1)
Path("/kaggle/working/vram_l4.json").write_text(json.dumps({"rc": rc, "tong_giay": round(time.time() - t0, 1), "dinh_theo_card": dinh}, indent=1), encoding="utf-8")
print("XONG", rc, round(time.time() - t0), "giay", json.dumps(dinh), flush=True)
'''


def dung_notebook(acc, cuoc_thi):
    goc = GOC / "scripts" / "ft_train"
    ghi = "".join(f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n" for ten in ("cham_diem.py", "kaggle_train.py"))
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(CAU_HINH_DO) + "\n"
    o2 = "from pathlib import Path\n" + ghi
    cells = [o1, o2, CHAY]
    THU.mkdir(parents=True, exist_ok=True)
    (THU / f"{SLUG}.ipynb").write_text(json.dumps(DG.tao_notebook(cells, "# Thử Qwen3-4B + LoRA trên Kaggle L4 x4 (mỗi card một tiến trình, gradient checkpointing)"), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    meta = {"id": f"{DG.NGUOI_DUNG}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": acc,
            "dataset_sources": [f"{DG.NGUOI_DUNG}/{DG.SLUG_POOL}"], "competition_sources": [cuoc_thi] if cuoc_thi else [],
            "kernel_sources": [f"{DG.NGUOI_DUNG}/{DG.SLUG_THU_VIEN}", f"{DG.NGUOI_DUNG}/{DG.SLUG_MO_HINH}"], "model_sources": []}
    (THU / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="NvidiaL4")        # đọc từ metadata notebook59a0de1490 sau khi anh chọn "GPU L4 x4" và lưu (18/11/2026): machine_shape = NvidiaL4
    ap.add_argument("--cuoc-thi", default="gemma-4-developer-agent")
    a = ap.parse_args()
    dung_notebook(a.acc, a.cuoc_thi)
    r = subprocess.run([str(KAGGLE), "kernels", "push", "-p", str(THU), "--accelerator", a.acc], capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-800:])


if __name__ == "__main__":
    main()
