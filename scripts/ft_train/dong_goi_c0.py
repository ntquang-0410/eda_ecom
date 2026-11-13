# -*- coding: utf-8 -*-
"""Đẩy notebook chấm C0 (cum_f0_p11, pair_f0_p11) lên Kaggle.

Chạy: python -X utf8 scripts/ft_train/dong_goi_c0.py
"""
import json
from pathlib import Path

import dong_goi_kaggle as G
import chung as C

SLUG_QUOTA = "ft-qwen3-4b-quota"
SLUG_TL_LUU = "ft-qwen3-4b-tl-luu"


CY_MAC_DINH = [{"tap": "cum", "c0": "cum_f0_p11"}, {"tap": "pair", "c0": "pair_f0_p11"}]


def day_c0(jobs=None, tieu=None):
    jobs = jobs or CY_MAC_DINH
    goc = Path(__file__).resolve().parent
    ghi = "".join(
        f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
        for ten in ("cham_diem.py", "kaggle_train.py", "kaggle_quota.py", "kaggle_c0.py")
    )
    cfg = {"che_do": "c0", "jobs": jobs}
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cfg)
    o2 = "# Ghi mã vào working (phiên tắt Internet)\nfrom pathlib import Path\n" + ghi
    o3 = "import sys\nsys.path.insert(0, '.')\nimport kaggle_c0\nkaggle_c0.main(CAU_HINH)\n"
    tieu = tieu or ("# Chấm C0 trên dev (" + ", ".join(j["c0"] for j in jobs) + ")\n\nGPU RTX Pro 6000, phiên tắt Internet. Không chứa test.")
    slug = "nlp-group3"
    thu = G.THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(G.tao_notebook([o1, o2, o3], tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{G.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "docker_image": G.ANH_RTX, "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_POOL}", f"{G.NGUOI_DUNG}/{SLUG_QUOTA}"],
        "competition_sources": [G.CUOC_THI],
        "kernel_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_THU_VIEN}", f"{G.NGUOI_DUNG}/{G.SLUG_MO_HINH}", f"{G.NGUOI_DUNG}/{SLUG_TL_LUU}"],
        "model_sources": [],
    }, indent=1), encoding="utf-8")
    G.goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaRtxPro6000")
    print(G.goi("kernels", "status", f"{G.NGUOI_DUNG}/{slug}", check=False)[-1000:])


if __name__ == "__main__":
    import sys
    # ví dụ: python dong_goi_c0.py pair   hoặc   python dong_goi_c0.py cum,pair
    if len(sys.argv) > 1:
        can = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
        jobs = [j for j in CY_MAC_DINH if j["tap"] in can]
    else:
        jobs = None
    day_c0(jobs)
