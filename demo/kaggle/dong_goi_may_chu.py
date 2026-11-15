# -*- coding: utf-8 -*-
"""Đẩy notebook máy chủ dịch lên Kaggle TÀI KHOẢN 1 (trinhnhatanh; Nhật Anh chỉ định 14/11/2026), GPU T4, Internet bật, KHÔNG gắn cuộc thi, không có dữ liệu test.
Nguồn: mô hình nền (ft-qwen3-4b-mo-hinh) và adapter bản cuối (ft-qwen3-4b-cuoi-luu).

  python -X utf8 demo/kaggle/dong_goi_may_chu.py [--gio 2]
Đọc URL và khóa:  kaggle kernels logs trinhnhatanh/demo-may-chu   (dòng KET_NOI)"""
import argparse
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
import dong_goi_kaggle as G  # noqa: E402

SLUG = "demo-may-chu"
FILE_MA = {"dung_mau.py": GOC / "scripts" / "ft_train" / "dung_mau.py", "chung.py": GOC / "scripts" / "ft_train" / "chung.py",
           **{n: GOC / "demo" / "may_chu" / n for n in ("loi_dan.py", "backend.py", "dich.py", "app.py")},
           "kaggle_may_chu.py": GOC / "demo" / "kaggle" / "kaggle_may_chu.py"}


def day(cfg):
    if G.NGUOI_DUNG != "trinhnhatanh":
        raise SystemExit(f"Phải là tài khoản 1 (KAGGLE_NGUOI_DUNG trống hoặc trinhnhatanh); đang là {G.NGUOI_DUNG}.")
    ghi = "from pathlib import Path\nPath('/kaggle/working/may_chu').mkdir(parents=True, exist_ok=True)\n" + "".join(
        f"Path('/kaggle/working/may_chu/{ten}').write_text({p.read_text(encoding='utf-8')!r}, encoding='utf-8')\n" for ten, p in FILE_MA.items())
    o = ["# Cấu hình\nCAU_HINH = " + repr(cfg) + "\n", ghi, "import sys\nsys.path.insert(0, '/kaggle/working/may_chu')\nimport kaggle_may_chu\nkaggle_may_chu.main(CAU_HINH)\n"]
    thu = G.THU_MUC_API / SLUG
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{SLUG}.ipynb").write_text(json.dumps(G.tao_notebook(o, "# Máy chủ dịch demo (Qwen3-4B + pair_cuoi)\n\nKhông có dữ liệu test."), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{G.NGUOI_DUNG}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_MO_HINH}", f"{G.NGUOI_DUNG}/{G.SLUG_CUOI_LUU}"], "model_sources": []}, indent=1), encoding="utf-8")
    G.goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaTeslaT4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gio", type=float, default=2.0)
    day({"gio": ap.parse_args().gio})
