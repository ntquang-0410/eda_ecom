# -*- coding: utf-8 -*-
"""Đẩy notebook thử chéo (bước 2) lên Kaggle tài khoản 2. Chỉ chạy SAU KHI hai adapter fold 0 đã huấn luyện xong ở nlp-group3 (cụm) và nlp-group3-pair (pair).

  $env:KAGGLE_NGUOI_DUNG='honghuynanh'
  python -X utf8 dong_goi_cheo.py [--so_toi_da 600] [--khoi]

Notebook nlp-cheo: GPU RTX Pro 6000 (gắn cuộc thi), tắt Internet, KHÔNG có dữ liệu test. Nguồn: Dataset pool, thư viện, mô hình Qwen3-4B,
và đầu ra của hai notebook huấn luyện (chứa adapter fold 0). Không đẩy lại nlp-group3 và nlp-group3-pair trước khi nlp-cheo chạy xong, vì đầu ra của chúng bị thay.
--khoi: chỉ 60 sản phẩm, để kiểm luồng trước khi chạy đủ.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dong_goi_kaggle as G  # noqa: E402

SLUG = "nlp-cheo"
NGUON_ADAPTER = ("nlp-group3", "nlp-group3-pair")


def day(cfg, slug=SLUG):
    if G.NGUOI_DUNG != "honghuynanh":
        raise SystemExit(f"Phải đặt KAGGLE_NGUOI_DUNG=honghuynanh (đang là {G.NGUOI_DUNG}).")
    goc = Path(__file__).resolve().parent
    ghi = "".join(f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n" for ten in ("kaggle_train.py", "kaggle_cheo.py"))
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cfg) + "\n"
    o2 = "# Ghi mã vào working (phiên tắt Internet)\nfrom pathlib import Path\n" + ghi
    o3 = "import sys\nsys.path.insert(0, '.')\nimport kaggle_cheo\nkaggle_cheo.main(CAU_HINH)\n"
    tieu = "# Thử chéo pair và cụm (Qwen3-4B + adapter fold 0)\n\nGPU RTX Pro 6000, phiên tắt Internet. Chỉ mẫu pool; không có dữ liệu test."
    thu = G.THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(G.tao_notebook([o1, o2, o3], tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{G.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "docker_image": G.ANH_RTX, "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_POOL}"], "competition_sources": [G.CUOC_THI],
        "kernel_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_THU_VIEN}", f"{G.NGUOI_DUNG}/{G.SLUG_MO_HINH}"] + [f"{G.NGUOI_DUNG}/{s}" for s in NGUON_ADAPTER],
        "model_sources": []}, indent=1), encoding="utf-8")
    G.goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaRtxPro6000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so_toi_da", type=int, default=600)
    ap.add_argument("--khoi", action="store_true")
    ap.add_argument("--gio_toi_da", type=float, default=2.0)
    a = ap.parse_args()
    day({"so_toi_da": 60 if a.khoi else a.so_toi_da, "seed": 43, "gio_toi_da": 0.75 if a.khoi else a.gio_toi_da})


if __name__ == "__main__":
    main()
