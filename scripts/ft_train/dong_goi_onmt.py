# -*- coding: utf-8 -*-
"""Day notebook OpenNMT (GPU T4, Internet BAT, KHONG gan cuoc thi) len Kaggle.

  python -X utf8 scripts/ft_train/dong_goi_onmt.py --slug nlp-onmt-khoi --khoi 5000      # chay khoi: 5.000 dong, 200 buoc
  python -X utf8 scripts/ft_train/dong_goi_onmt.py --slug nlp-onmt                        # chay that: 2 huong x (train day du + khop4m_s42)

Dung biendong KAGGLE_NGUOI_DUNG de chon tai khoan (xem dong_goi_kaggle.py). Dataset ft-song-ngu do xuat_song_ngu.py dung ra (khong co test).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dong_goi_kaggle as G  # noqa: E402

SLUG_SONG_NGU = "ft-song-ngu"


def day(cau_hinh, slug):
    goc = Path(__file__).resolve().parent
    ghi = f"Path('kaggle_onmt.py').write_text({(goc / 'kaggle_onmt.py').read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cau_hinh)
    o2 = "# Ghi mã vào working\nfrom pathlib import Path\n" + ghi
    o3 = "import sys\nsys.path.insert(0, '.')\nimport kaggle_onmt\nkaggle_onmt.main(CAU_HINH)\n"
    tieu = "# OpenNMT-py (Transformer nhỏ, huấn luyện từ đầu) trên dữ liệu song ngữ từng dòng\n\nGPU T4, Internet bật (cài OpenNMT-py vào venv riêng). Không chứa test, không gắn cuộc thi."
    thu = G.THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(G.tao_notebook([o1, o2, o3], tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{G.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{G.NGUOI_DUNG}/{SLUG_SONG_NGU}"], "competition_sources": [],
        "kernel_sources": [f"{G.NGUOI_DUNG}/nlp-onmt"] if cau_hinh.get("che_do") == "dich_cum" else [], "model_sources": [],
    }, indent=1), encoding="utf-8")
    G.goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaTeslaT4")
    print(G.goi("kernels", "status", f"{G.NGUOI_DUNG}/{slug}", check=False)[-600:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="nlp-onmt")
    ap.add_argument("--khoi", type=int, default=None, help="chay khoi: chi lay N dong dau, so buoc nho")
    ap.add_argument("--huong", default="zh-vi,vi-zh")
    ap.add_argument("--bien_the", default="train")
    ap.add_argument("--dich_cum", action="store_true", help="chi dich eval ban cum bang mo hinh nlp-onmt da huan luyen (khong huan luyen lai)")
    ap.add_argument("--so_epoch", type=int, default=20, help="tran epoch; dung som theo valid ppl (kien nhan 4 lan valid = 4 epoch)")
    a = ap.parse_args()
    cfg = {"huong": a.huong.split(","), "bien_the": a.bien_the.split(","), "so_epoch": a.so_epoch, "seed": 42}
    if a.dich_cum:
        cfg["che_do"] = "dich_cum"
    if a.khoi:
        cfg["khoi"] = a.khoi
        cfg["khoi_buoc"] = 300
        cfg["huong"] = ["zh-vi"]
        cfg["bien_the"] = ["train"]
    day(cfg, a.slug)


if __name__ == "__main__":
    main()
