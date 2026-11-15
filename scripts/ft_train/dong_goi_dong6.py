# -*- coding: utf-8 -*-
"""Đẩy notebook dòng thứ sáu (4B công thức cuối học fold 1-4, đo trên bộ eval 1.200 mẫu của bảng năm mô hình) lên Kaggle tài khoản 2.

  $env:KAGGLE_NGUOI_DUNG='honghuynanh'
  python -X utf8 dong_goi_dong6.py --khoi     # 40 hàng mỗi bảng, kiểm luồng và đo tốc độ
  python -X utf8 dong_goi_dong6.py            # đủ 1.200 hàng mỗi bảng

Notebook nlp-dong6: yêu cầu RTX Pro 6000 như các notebook trước (Kaggle đang cấp T4 thay; T4 được chấp nhận nên không chặn theo GPU), Internet tắt, KHÔNG có dữ liệu test.
Nguồn: Dataset pool + Dataset quota (eval_*_sample.csv), thư viện, mô hình Qwen3-4B, và đầu ra của nlp-group3 (cum_f0_p11) và nlp-group3-pair (pair_f0_p11).
Không đẩy lại nlp-group3 và nlp-group3-pair trước khi nlp-dong6 chạy xong thật.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dong_goi_kaggle as G  # noqa: E402

SLUG = "nlp-dong6"
SLUG_QUOTA = "ft-qwen3-4b-quota"
NGUON_ADAPTER = ("nlp-group3", "nlp-group3-pair")
FILE_MA = ("cham_diem.py", "kaggle_train.py", "kaggle_quota.py", "kaggle_cheo.py", "kaggle_dong6.py")


def day(cfg, slug=SLUG):
    if G.NGUOI_DUNG != "honghuynanh":
        raise SystemExit(f"Phải đặt KAGGLE_NGUOI_DUNG=honghuynanh (đang là {G.NGUOI_DUNG}).")
    goc = Path(__file__).resolve().parent
    ghi = "".join(f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n" for ten in FILE_MA)
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cfg) + "\n"
    o2 = "# Ghi mã vào working (phiên tắt Internet)\nfrom pathlib import Path\n" + ghi
    o3 = "import sys\nsys.path.insert(0, '.')\nimport kaggle_dong6\nkaggle_dong6.main(CAU_HINH)\n"
    tieu = "# Dòng thứ sáu: Qwen3-4B công thức cuối (học fold 1-4) trên bộ eval 1.200 mẫu\n\nKhông có dữ liệu test."
    thu = G.THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(G.tao_notebook([o1, o2, o3], tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{G.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "docker_image": G.ANH_RTX, "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_POOL}", f"{G.NGUOI_DUNG}/{SLUG_QUOTA}"], "competition_sources": [G.CUOC_THI],
        "kernel_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_THU_VIEN}", f"{G.NGUOI_DUNG}/{G.SLUG_MO_HINH}"] + [f"{G.NGUOI_DUNG}/{s}" for s in NGUON_ADAPTER],
        "model_sources": []}, indent=1), encoding="utf-8")
    G.goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaRtxPro6000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--khoi", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--gio_toi_da", type=float, default=8.0)
    ap.add_argument("--ngan_sach", type=int, default=12000)
    a = ap.parse_args()
    cfg = {"tap": ["cum", "pair"], "dtype": a.dtype, "bs": 32, "ngan_sach": a.ngan_sach, "gio_toi_da": 0.5 if a.khoi else a.gio_toi_da}
    if a.khoi:
        cfg["so_toi_da"] = 40
    day(cfg)


if __name__ == "__main__":
    main()
