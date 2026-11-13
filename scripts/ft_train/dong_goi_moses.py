# -*- coding: utf-8 -*-
"""Day notebook Moses (CPU, Internet BAT, khong GPU, khong test) len Kaggle. Can notebook nlp-moses-build da xong (kernel_sources).

  python -X utf8 scripts/ft_train/dong_goi_moses.py --slug nlp-moses-khoi --khoi 5000 --huong zh-vi --seg char     # khoi: 5.000 dong
  python -X utf8 scripts/ft_train/dong_goi_moses.py --slug nlp-moses-zhvi --huong zh-vi                           # that: seg char + jieba
Dung bien KAGGLE_NGUOI_DUNG de chon tai khoan.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dong_goi_kaggle as G  # noqa: E402

SLUG_SONG_NGU = "ft-song-ngu"
SLUG_BUILD = "nlp-moses-build"


def day(cau_hinh, slug):
    goc = Path(__file__).resolve().parent
    ghi = f"Path('kaggle_moses.py').write_text({(goc / 'kaggle_moses.py').read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cau_hinh)
    o2 = "# Ghi mã vào working\nfrom pathlib import Path\n" + ghi
    o3 = "import sys\nsys.path.insert(0, '.')\nimport kaggle_moses\nkaggle_moses.main(CAU_HINH)\n"
    tieu = "# Moses (phrase-based SMT) trên dữ liệu song ngữ từng dòng\n\nCPU, Internet bật. Không chứa test, không gắn cuộc thi."
    thu = G.THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(G.tao_notebook([o1, o2, o3], tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{G.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{G.NGUOI_DUNG}/{SLUG_SONG_NGU}"], "competition_sources": [], "kernel_sources": [f"{G.NGUOI_DUNG}/{SLUG_BUILD}"], "model_sources": [],
    }, indent=1), encoding="utf-8")
    G.goi("kernels", "push", "-p", str(thu))
    print(G.goi("kernels", "status", f"{G.NGUOI_DUNG}/{slug}", check=False)[-400:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="nlp-moses")
    ap.add_argument("--khoi", type=int, default=None)
    ap.add_argument("--huong", default="zh-vi,vi-zh")
    ap.add_argument("--seg", default="char,jieba")
    ap.add_argument("--mert_lap", type=int, default=10)
    ap.add_argument("--bien_the", default="train")
    ap.add_argument("--eval_cum", action="store_true", help="dich them eval ban cum (tung manh) va ghep lai")
    a = ap.parse_args()
    cfg = {"huong": a.huong.split(","), "seg": a.seg.split(","), "mert_lap": a.mert_lap, "bien_the": a.bien_the.split(","), "eval_cum": a.eval_cum}
    if a.khoi:
        cfg["khoi"] = a.khoi
        cfg["mert_lap"] = 2
    day(cfg, a.slug)


if __name__ == "__main__":
    main()
