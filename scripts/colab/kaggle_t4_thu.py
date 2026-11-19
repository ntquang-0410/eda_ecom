# -*- coding: utf-8 -*-
"""Đo T4 x2 (so với L4 x4) cho huấn luyện Qwen3-4B + LoRA r64 bản cụm, trên tài khoản đang dùng (đặt KAGGLE_NGUOI_DUNG).

Hai card chạy ĐỒNG THỜI, mỗi card một tiến trình: nạp mô hình bf16, huấn luyện thật vài bước ở hai ngân sách token (2048, 4096) trên lô gần độ dài trung vị
và lô dài nhất, ghi giây/bước, token/giây, bộ nhớ đỉnh, loss có hữu hạn không, có tràn bộ nhớ không. Không đọc test, không lưu checkpoint.

  python -I -X utf8 scripts/colab/kaggle_t4_thu.py [--day]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_tlc_v2 as B  # noqa: E402

DO = r'''
import json, os, sys, time, traceback
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, ".")
import torch
import kaggle_train as K
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
card = sys.argv[1]
kq = {"card": card, "gpu": torch.cuda.get_device_name(0), "tong_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
      "bf16_ho_tro": torch.cuda.is_bf16_supported(), "torch": torch.__version__, "ket_qua": []}
try:
    import peft, transformers
    kq["peft"], kq["transformers"] = peft.__version__, transformers.__version__
    thu_mh = K.tim_mot("REVISION.txt").parent
    tok = AutoTokenizer.from_pretrained(str(thu_mh)); tok.padding_side = "right"
    ma, _ = K.xay_mau(K.doc_mau("cum"), tok, 2048)
    ma = sorted(ma, key=lambda x: len(x["ids"]))
    goc = K.nap_goc(thu_mh); goc.config.use_cache = False
    m = get_peft_model(goc, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    m.train()
    kq["mib_sau_nap"] = torch.cuda.memory_allocated() // 2**20
    def lo_gan(vt, ngan_sach):
        # lô gồm các mẫu liền kề quanh vị trí vt (đã xếp theo độ dài) sao cho số mẫu x mẫu dài nhất <= ngan_sach, tối đa 32 mẫu
        k = ma[max(0, vt - 40): vt + 1]
        lo = []
        for x in reversed(k):
            if (len(lo) + 1) * len(ma[vt]["ids"]) > ngan_sach or len(lo) >= 32:
                break
            lo.append(x)
        return lo
    for ngan_sach in (2048, 4096):
        for ten, vt in (("trung_vi", len(ma) // 2), ("dai_nhat", len(ma) - 1)):
            r = {"ngan_sach": ngan_sach, "lo": ten, "buoc": []}
            try:
                lo = lo_gan(vt, ngan_sach)
                r["so_mau_lo"], r["dai_toi_da"] = len(lo), max(len(x["ids"]) for x in lo)
                torch.cuda.reset_peak_memory_stats()
                for i in range(6):
                    t = time.time()
                    ids, lab, mask = K.lo_tensor(lo, tok, "cuda:0")
                    loss = m(input_ids=ids, attention_mask=mask, labels=lab).loss
                    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    r["buoc"].append({"giay": round(time.time() - t, 2), "loss": round(float(loss), 4), "huu_han": bool(torch.isfinite(loss).item())})
                tt = [b["giay"] for b in r["buoc"][1:]]
                r["giay_moi_buoc_tb"] = round(sum(tt) / len(tt), 2)
                r["token_moi_giay"] = round(len(lo) * r["dai_toi_da"] / r["giay_moi_buoc_tb"])
                r["mib_dinh"] = torch.cuda.max_memory_allocated() // 2**20
                r["ket_qua"] = "vua" if r["mib_dinh"] < kq["tong_mib"] * 0.95 else "sat_tran"
            except torch.cuda.OutOfMemoryError:
                r["ket_qua"] = "TRAN_BO_NHO"
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            except Exception:
                r["ket_qua"] = "LOI: " + traceback.format_exc()[-600:]
            kq["ket_qua"].append(r)
            print("DO", card, json.dumps({k: v for k, v in r.items() if k != "buoc"}), flush=True)
    kq["dat"] = True
except Exception:
    kq["dat"] = False
    kq["loi"] = traceback.format_exc()[-1500:]
    print(kq["loi"], flush=True)
open(f"/kaggle/working/do_t4_{card}.json", "w").write(json.dumps(kq, indent=1))
print("DO XONG", card, kq["dat"], flush=True)
'''

CHAY = r'''import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import kaggle_train as K
RA = Path("/kaggle/working")
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
(RA / "nvidia_smi.txt").write_text(smi, encoding="utf-8")
print(smi, flush=True)
K.kiem_khoa_test()
K.cai_thu_vien()
(RA / "do_t4.py").write_text(DO, encoding="utf-8")
t0 = time.time()
ps = []
for c in (0, 1):
    ps.append(subprocess.Popen([sys.executable, "-X", "utf8", str(RA / "do_t4.py"), str(c)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(c)},
                               stdout=open(RA / f"do_t4_{c}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, cwd=os.getcwd()))
while any(p.poll() is None for p in ps):
    time.sleep(60)
    print("dang do", round(time.time() - t0), "giay", flush=True)
rc = [p.returncode for p in ps]
for c in (0, 1):
    print(f"--- card {c} ---")
    print((RA / f"do_t4_{c}.log").read_text(encoding="utf-8")[-3000:], flush=True)
(RA / "tong_ket_t4.json").write_text(json.dumps({"rc": rc, "giay": round(time.time() - t0)}), encoding="utf-8")
print("XONG", rc, round(time.time() - t0), flush=True)
'''


def dung(acc):
    import dong_goi_kaggle as DG
    slug = "nlp-t4-thu"
    thu = GOC / "notebooks" / "kaggle_api" / slug
    goc = GOC / "scripts" / "ft_train"
    ghi = "".join(f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n" for ten in B.FILE_MA)
    cells = ["from pathlib import Path\n" + ghi, "DO = " + repr(DO) + "\n", CHAY]
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(DG.tao_notebook(cells, "# Đo T4 x2 cho Qwen3-4B + LoRA\n\nKhông có dữ liệu test."), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    meta = {"id": f"{DG.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": acc,
            "dataset_sources": [f"{DG.NGUOI_DUNG}/{DG.SLUG_POOL}", f"{DG.NGUOI_DUNG}/{B.SLUG_QUOTA}"], "competition_sources": [],
            "kernel_sources": [f"{DG.NGUOI_DUNG}/{DG.SLUG_THU_VIEN}", f"{DG.NGUOI_DUNG}/{DG.SLUG_MO_HINH}"], "model_sources": []}
    (thu / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return thu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="NvidiaTeslaT4")
    ap.add_argument("--day", action="store_true")
    a = ap.parse_args()
    thu = dung(a.acc)
    print("da dung", thu)
    if a.day:
        r = subprocess.run([str(B.KAGGLE), "kernels", "push", "-p", str(thu), "--accelerator", a.acc], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip()[-800:])


if __name__ == "__main__":
    main()
