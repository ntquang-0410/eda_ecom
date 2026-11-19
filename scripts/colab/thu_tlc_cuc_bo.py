# -*- coding: utf-8 -*-
"""Thử TRÊN MÁY toàn bộ đường chạy của kaggle_train.chay_b78 (cờ bao_ve) bằng mô hình Qwen3 NHỎ NGẪU NHIÊN (tokenizer thật, dữ liệu pool thật), trước khi tốn quota Kaggle.
Mục đích: bắt lỗi MÃ (kiểu TypeError 18/11/2026), không đo chất lượng. Chạy ĐÚNG mã DAU của kaggle_tlc_v2 (chỉ thu nhỏ ngân sách gom lô và max_len cho GPU 4 GB).

Chạy (cần torch CUDA + transformers + peft + pandas + sacrebleu, ghép qua PYTHONPATH, không cài vào venv nào):
  $env:PYTHONPATH = "D:\\nam4_ky1\\_tlc_thu_lib;D:\\LBC-ForWork\\VXB-Intelligence\\venv\\Lib\\site-packages;D:\\nam4_ky1\\e-commerce - EDA\\.venv\\Lib\\site-packages"
  & D:\\LBC-ForWork\\VXB-Intelligence\\venv\\Scripts\\python.exe -X utf8 scripts/colab/thu_tlc_cuc_bo.py [prep] [s1] [s2] [s3] [s4]
Kịch bản: s1 dừng ở bước 5 rồi chạy nối tiếp ở tiến trình mới; s2 loss_nhom lỗi (việc phải đi tiếp); s3 chết ở bước sinh/chấm sau huấn luyện rồi nối tiếp (không huấn luyện lại);
s4 chết giữa epoch 2 rồi nối tiếp từ checkpoint định kỳ."""
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_tlc_v2 as B  # noqa: E402

NGUON = Path(r"D:\nam4_ky1\_tlc_thu_in")
KQ = Path(r"D:\nam4_ky1\_tlc_thu_kq")
HF = Path(r"C:\Users\ADMIN\.cache\huggingface\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c")
FT = GOC / "scripts" / "ft_train"

NHO = {"max_len": 320, "ngan_sach_token": 320, "so_kiem_moi_nhom": 6, "so_sinh_moi_nhom": 2, "vram_toi_thieu_gb": None, "so_train": 200, "luu_moi_buoc": 3, "chrf_toi_thieu": None, "gio_toi_da": 1.0}


def chuan_bi():
    mh = NGUON / "mh"
    shutil.rmtree(NGUON, ignore_errors=True)
    mh.mkdir(parents=True)
    for t in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(HF / t, mh / t)
    (mh / "REVISION.txt").write_text("thu_cuc_bo 0000", encoding="utf-8")
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=4096, tie_word_embeddings=True, eos_token_id=151645, bos_token_id=151643, pad_token_id=151643)
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(mh)
    for n in ("cum", "pair"):
        d = NGUON / "pool"
        d.mkdir(exist_ok=True)
        shutil.copy(GOC / "data" / "processed" / "ft_run" / "goi_pool" / f"pool_{n}.jsonl.gz", d / f"pool_{n}.jsonl.gz")
    print("da chuan bi", NGUON, [p.name for p in mh.iterdir()])


def ma_chay(cfg, tiem=""):
    """Cùng cấu trúc với chay_viec trong notebook: DAU rồi K.main(cfg); thêm: trỏ VAO/RA về máy, thu nhỏ ngân sách."""
    truoc = ("import sys, json\nfrom pathlib import Path\nsys.path.insert(0, '.')\nimport kaggle_train as K\n"
             f"K.VAO = Path({str(NGUON)!r}); K.RA = Path({str(cfg['_ra'])!r})\n")
    sau = "K.loss_tong.__defaults__ = (320,)\nK.loss_kiem.__defaults__ = (320,)\nK.loss_nhom.__defaults__ = (320,)\nK.sinh.__defaults__ = (32, 3000, 'dich', None)\n"
    cfg = {k: v for k, v in cfg.items() if k != "_ra"}
    return truoc + B.DAU.split("import kaggle_train as K\n", 1)[1] + sau + tiem + "K.main(json.loads(%r))\n" % json.dumps(cfg)


def chay(ra, cfg, tiem=""):
    cfg = dict(cfg, _ra=ra)
    ra.mkdir(parents=True, exist_ok=True)
    with open(ra / "log.txt", "a", encoding="utf-8") as lg:
        p = subprocess.run([sys.executable, "-X", "utf8", "-c", ma_chay(cfg, tiem)], cwd=str(FT), stdout=lg, stderr=subprocess.STDOUT)
    return p.returncode


def cfg_viec(ten, p_title, **them):
    ch = B.cfg_chung(1.0)
    ch.update(NHO)
    ch.update(them)
    ch.update(ten_trang_thai="tlc_" + ten, viec=[{"ten": ten, "nhanh": "cum", "fold": 0, "p_title": p_title, "ca_fold": False}])
    return ch


def doc_tt(ra, ten):
    return json.loads((ra / f"tlc_{ten}" / f"tlc_{ten}_trang_thai.json").read_text(encoding="utf-8"))["viec"][ten]


def log(ra):
    return (ra / "log.txt").read_text(encoding="utf-8", errors="replace")


def kiem(ten, dieu_kien, chi_tiet=""):
    print(("  DAT  " if dieu_kien else "  LOI  ") + ten, chi_tiet)
    if not dieu_kien:
        loi.append(ten)


loi = []


def tom_log_loi(ra):
    t = log(ra)
    return t[-1800:] if "Traceback" in t else ""


def xong_het(ra, ten, tong_buoc_ky_vong=None):
    v = doc_tt(ra, ten)
    d = ra / f"tlc_{ten}" / ten
    kiem(f"{ten}: epoch_xong=2", v["epoch_xong"] == 2, str(v["epoch_xong"]))
    kiem(f"{ten}: co ban ep2", (d / "ep2" / "adapter_model.safetensors").exists())
    kiem(f"{ten}: da cham chrF", bool(v.get("cham")), str((v.get("cham") or {}).get("chrf", {}).get("chrf_macro_huong")))
    kiem(f"{ten}: da don checkpoint", not (d / "moi_nhat").exists() and not (d / "moi_nhat_tam").exists())
    kiem(f"{ten}: so buoc = tong buoc", v["buoc"] == v["tong_buoc"], f"{v['buoc']}/{v['tong_buoc']}")
    return v


def s1():
    print("S1: dung o buoc 5 roi chay noi tiep o tien trinh moi")
    ra = KQ / "s1"
    shutil.rmtree(ra, ignore_errors=True)
    cfg = cfg_viec("thu_a", 0.25, dung_sau_buoc=5)
    rc1 = chay(ra, cfg)
    v = doc_tt(ra, "thu_a")
    kiem("lan 1 ket thuc rc=0 (dung co trat tu)", rc1 == 0, f"rc={rc1} " + tom_log_loi(ra))
    kiem("lan 1 buoc=5 va co checkpoint", v["buoc"] == 5 and (ra / "tlc_thu_a" / "thu_a" / "moi_nhat" / "opt.pt").exists(), str(v["buoc"]))
    td = json.loads((ra / "tlc_thu_a" / "thu_a" / "moi_nhat" / "tien_do.json").read_text())
    kiem("tien_do.json khop buoc", td["buoc"] == 5, str(td))
    cfg2 = dict(cfg)
    cfg2.pop("dung_sau_buoc")
    rc2 = chay(ra, cfg2)
    kiem("lan 2 (tien trinh moi) rc=0", rc2 == 0, f"rc={rc2} " + tom_log_loi(ra))
    kiem("lan 2 co 'chay tiep tu checkpoint'", "chay tiep tu checkpoint" in log(ra))
    xong_het(ra, "thu_a")
    kiem("khong co loi trong log", "Traceback" not in log(ra))


def s2():
    print("S2: loss_nhom loi o epoch 1 -> viec van di tiep")
    ra = KQ / "s2"
    shutil.rmtree(ra, ignore_errors=True)
    tiem = ("_o = K.loss_nhom\n_n = {'n': 0}\ndef _bad(*a, **k):\n    _n['n'] += 1\n    if _n['n'] == 1:\n        raise RuntimeError('INJECT loss_nhom')\n    return _o(*a, **k)\nK.loss_nhom = _bad\n")
    rc = chay(ra, cfg_viec("thu_b", 0.1074), tiem)
    kiem("rc=0 du loss_nhom loi", rc == 0, f"rc={rc} " + tom_log_loi(ra)[-600:])
    v = xong_het(ra, "thu_b")
    kiem("co ghi loi_loss_nhom", "1" in (v.get("loi_loss_nhom") or {}), str(list((v.get("loi_loss_nhom") or {}))))
    kiem("loss epoch 2 co so", v["diem"].get("2") is not None, str(v["diem"]))


def s3():
    print("S3: chet o buoc sinh (sau huan luyen) roi chay noi tiep, khong huan luyen lai")
    ra = KQ / "s3"
    shutil.rmtree(ra, ignore_errors=True)
    tiem = "_s = K.sinh\ndef _bad(*a, **k):\n    raise RuntimeError('INJECT sinh')\nK.sinh = _bad\n"
    rc1 = chay(ra, cfg_viec("thu_c", 0.25), tiem)
    kiem("lan 1 loi dung nhu mong doi (rc != 0)", rc1 != 0, f"rc={rc1}")
    v = doc_tt(ra, "thu_c")
    kiem("lan 1 da luu huan luyen xong (epoch_xong=2, ep2)", v["epoch_xong"] == 2 and (ra / "tlc_thu_c" / "thu_c" / "ep2" / "adapter_model.safetensors").exists())
    n1 = log(ra).count("XONG epoch")
    rc2 = chay(ra, cfg_viec("thu_c", 0.25))
    kiem("lan 2 rc=0", rc2 == 0, f"rc={rc2} " + tom_log_loi(ra)[-600:])
    t = log(ra)
    kiem("lan 2 KHONG huan luyen lai", t.count("XONG epoch") == n1, f"{n1} -> {t.count('XONG epoch')}")
    xong_het(ra, "thu_c")


def s4():
    print("S4: chet giua epoch 2 roi chay noi tiep tu checkpoint dinh ky")
    ra = KQ / "s4"
    shutil.rmtree(ra, ignore_errors=True)
    tiem = ("import torch\n_c = torch.nn.utils.clip_grad_norm_\n_n = {'n': 0}\ndef _bad(*a, **k):\n    _n['n'] += 1\n    if _n['n'] == 11:\n        raise RuntimeError('INJECT chet giua epoch 2')\n    return _c(*a, **k)\n"
            "torch.nn.utils.clip_grad_norm_ = _bad\n")
    rc1 = chay(ra, cfg_viec("thu_d", 0.1074), tiem)
    kiem("lan 1 chet dung nhu mong doi (rc != 0)", rc1 != 0, f"rc={rc1}")
    v = doc_tt(ra, "thu_d")
    ck = ra / "tlc_thu_d" / "thu_d" / "moi_nhat" / "tien_do.json"
    kiem("co checkpoint dinh ky", ck.exists(), ck.read_text() if ck.exists() else "")
    rc2 = chay(ra, cfg_viec("thu_d", 0.1074))
    kiem("lan 2 rc=0", rc2 == 0, f"rc={rc2} " + tom_log_loi(ra)[-600:])
    kiem("lan 2 'chay tiep tu checkpoint'", "chay tiep tu checkpoint" in log(ra))
    xong_het(ra, "thu_d")


def s0():
    """Kiểm chứng ngược: DÙNG LẠI đoạn bọc mã cũ gây lỗi 18/11/2026 (không bao_ve) thì bài thử phải bắt được TypeError ngan_sach."""
    print("S0: bo bao cu (loi 18/11) phai bi bat")
    ra = KQ / "s0"
    shutil.rmtree(ra, ignore_errors=True)
    cu = ("_ln, _lt, _s = K.loss_nhom, K.loss_tong, K.sinh\n" "K.loss_nhom = lambda *a, **k: _ln(*a, **{**k, 'ngan_sach': 320})\n"
          "K.loss_tong = lambda *a, **k: _lt(*a, **{**k, 'ngan_sach': 320})\n")
    cfg = cfg_viec("thu_e", 0.25)
    cfg.pop("bao_ve")
    rc = chay(ra, cfg, cu)
    kiem("bo bao cu bi bat: TypeError ngan_sach", rc != 0 and "multiple values for argument 'ngan_sach'" in log(ra), f"rc={rc}")


if __name__ == "__main__":
    cac = sys.argv[1:] or ["prep", "s1", "s2", "s3", "s4"]
    if "prep" in cac:
        chuan_bi()
    for k in cac:
        if k in ("s0", "s1", "s2", "s3", "s4"):
            globals()[k]()
    print("\nKET QUA:", "TAT CA DAT" if not loi else f"CO {len(loi)} LOI: {loi}")
    sys.exit(1 if loi else 0)
