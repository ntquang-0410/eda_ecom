# -*- coding: utf-8 -*-
"""Đóng gói quota dataset + đẩy notebook smoke/screen lên Kaggle.

Chạy:
  python -X utf8 scripts/ft_train/dong_goi_quota.py pack
  python -X utf8 scripts/ft_train/dong_goi_quota.py smoke
  python -X utf8 scripts/ft_train/dong_goi_quota.py screen
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import dong_goi_kaggle as G
import chung as C

SLUG_QUOTA = "ft-qwen3-4b-quota"
GOI_QUOTA = C.FT_RUN / "goi_quota"
SRC_INTERIM = Path(C.GOC) / "data" / "interim" / "quota_phase0"
SRC_EDA = Path(C.GOC) / "data" / "eda" / "khao_sat_trong_so" / "quota_phase0"
CONFIG_NAMES = {
    "cum": ["natural", "balanced", "title_heavy", "desc_heavy", "tiki_heavy"],
    "pair": ["natural", "balanced", "title_heavy", "attr_heavy", "tiki_heavy"],
}


def pack():
    GOI_QUOTA.mkdir(parents=True, exist_ok=True)
    for tap in ("cum", "pair"):
        shutil.copy2(SRC_INTERIM / f"train_ref_{tap}_flt0.csv", GOI_QUOTA / f"train_ref_{tap}_flt0.csv")
        shutil.copy2(SRC_INTERIM / f"eval_{tap}_sample.csv", GOI_QUOTA / f"eval_{tap}_sample.csv")
        for c in CONFIG_NAMES[tap]:
            shutil.copy2(SRC_INTERIM / f"idx_q_{tap}_{c}_cap960_flt0.npy", GOI_QUOTA / f"idx_q_{tap}_{c}_cap960_flt0.npy")
    shutil.copy2(SRC_EDA / "meta.json", GOI_QUOTA / "meta.json")
    files = sorted(p.name for p in GOI_QUOTA.iterdir() if p.is_file())
    (GOI_QUOTA / "MANIFEST.json").write_text(json.dumps({"files": files, "n_files": len(files)}, ensure_ascii=False, indent=1), encoding="utf-8")
    (GOI_QUOTA / "dataset-metadata.json").write_text(json.dumps({
        "title": SLUG_QUOTA, "id": f"{G.NGUOI_DUNG}/{SLUG_QUOTA}",
        "licenses": [{"name": "other"}],
    }, indent=1), encoding="utf-8")
    st = G.goi("datasets", "create", "-p", str(GOI_QUOTA), check=False)
    if "already exists" in st.lower() or "exists" in st.lower():
        G.goi("datasets", "version", "-p", str(GOI_QUOTA), "-m", "quota phase0 Survey A cap960 flt0")
    elif "error" in st.lower() or "forbidden" in st.lower():
        raise SystemExit("datasets create lỗi: " + st[-1000:])
    for _ in range(20):
        st = G.goi("datasets", "status", f"{G.NGUOI_DUNG}/{SLUG_QUOTA}", check=False)
        if "ready" in st.lower():
            break
        time.sleep(20)
    print("DATASET XONG:", st[-500:])


# Hồ sơ từng mô hình (12/11/2026). ten_thu_muc = thư mục chứa REVISION.txt trong đầu ra notebook đóng gói; khuon: qwen | ecomgpt;
# attn: sdpa | eager (BLOOM không dùng sdpa); gc: bật gradient checkpointing (mô hình lớn, chưa đo bộ nhớ nên bật cho chắc, chỉnh sau khi chạy khói).
# tok_lay_mau_thu_muc: tokenizer dùng để LẤY MẪU, luôn là Qwen để mọi mô hình huấn luyện đúng cùng một danh sách mẫu.
MO_HINH = {
    "qwen3-4b": {"ten": "qwen3-4b", "ten_thu_muc": "Qwen3-4B", "khuon": "qwen", "attn": "sdpa", "gc": False, "slug": G.SLUG_MO_HINH},
    "qwen3-8b": {"ten": "qwen3-8b", "ten_thu_muc": "Qwen3-8B", "khuon": "qwen", "attn": "sdpa", "gc": True, "slug": G.SLUG_MO_HINH_8B},
    "ecomgpt-7b": {"ten": "ecomgpt-7b", "ten_thu_muc": "EcomGPT-7B", "khuon": "ecomgpt", "attn": "eager", "gc": True,
                   "slug": G.SLUG_MO_HINH_ECOM, "tok_lay_mau_thu_muc": "Qwen3-4B", "nguon_them": [G.SLUG_MO_HINH]},
}


def day_quota(cau_hinh, dry=False, slug="nlp-group3", mo_hinh_slug=None, them_nguon=(), gpu=True):
    goc = Path(__file__).resolve().parent
    ghi = "".join(
        f"Path({ten!r}).write_text({(goc / ten).read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
        for ten in ("cham_diem.py", "kaggle_train.py", "kaggle_quota.py")
    )
    o1 = "# ===== Cấu hình =====\nCAU_HINH = " + repr(cau_hinh)
    o2 = "# Ghi mã vào working (phiên tắt Internet)\nfrom pathlib import Path\n" + ghi
    o3 = "import sys\nsys.path.insert(0, '.')\nimport kaggle_quota\nkaggle_quota.main(CAU_HINH)\n"
    tieu = ("# Quota dry run (dựng ngạch theo độ dài thật + kiểm tỉ lệ, KHÔNG train)"
            if dry else "# Quota Survey A bản sửa (5 cấu hình × 2 bộ, 8k token đích/bước)")
    tieu += ("\n\nGPU RTX Pro 6000, phiên tắt Internet. Không chứa test." if gpu else "\n\nCPU, phiên tắt Internet. Không chứa test.")
    thu = G.THU_MUC_API / slug
    thu.mkdir(parents=True, exist_ok=True)
    (thu / f"{slug}.ipynb").write_text(json.dumps(G.tao_notebook([o1, o2, o3], tieu), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    meta = {
        "id": f"{G.NGUOI_DUNG}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": bool(gpu), "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_POOL}", f"{G.NGUOI_DUNG}/{SLUG_QUOTA}"],
        "competition_sources": [G.CUOC_THI] if gpu else [],
        "kernel_sources": [f"{G.NGUOI_DUNG}/{G.SLUG_THU_VIEN}", f"{G.NGUOI_DUNG}/{mo_hinh_slug or G.SLUG_MO_HINH}"]
                          + [f"{G.NGUOI_DUNG}/{s}" for s in them_nguon],
        "model_sources": [],
    }
    if gpu:
        meta.update({"docker_image": G.ANH_RTX, "machine_shape": "NvidiaRtxPro6000"})
    (thu / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if gpu:
        G.goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaRtxPro6000")
    else:
        G.goi("kernels", "push", "-p", str(thu))
    print(G.goi("kernels", "status", f"{G.NGUOI_DUNG}/{slug}", check=False)[-1000:])


def cfg_chung(viec, dry=False):
    return {
        "che_do": "quota_dry" if dry else "quota", "viec": viec,
        "so_eval": 1200,
        "lr": 2e-4, "rank": 16, "micro_bs": 32, "tich_luy": 1,
        "ngan_sach_token": 4096, "max_len": 2048, "seed": 42, "flt": 0,
        "ngan_sach_compute": 4_000_000, "so_token_moi_buoc": 8000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pack", "dry", "screen", "cum", "step1", "check1", "natural", "mh"])
    ap.add_argument("--slug", default="nlp-group3", help="tên notebook đẩy lên (mỗi notebook là một máy GPU riêng, chạy song song được)")
    ap.add_argument("--tap", default="cum,pair")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--mo_hinh", default="qwen3-4b", choices=sorted(MO_HINH), help="chỉ dùng với mode mh")
    ap.add_argument("--config", default="natural")
    ap.add_argument("--dry", action="store_true", help="mode mh: chạy thử trên CPU, không huấn luyện, chỉ kiểm khuôn/độ dài/tỉ lệ")
    ap.add_argument("--dung_sau_buoc", type=int, default=None, help="chạy khói: dừng sau N bước huấn luyện")
    ap.add_argument("--so_eval", type=int, default=None, help="số mẫu đánh giá (mặc định 1.200)")
    ap.add_argument("--gc", choices=["bat", "tat"], default=None, help="ghi đè gradient checkpointing của hồ sơ mô hình")
    ap.add_argument("--ngan_sach_token", type=int, default=None, help="ghi đè ngân sách token mỗi micro-batch")
    a = ap.parse_args()
    if a.mode == "mh":
        # Chạy một mô hình bất kỳ trong MO_HINH với cấu hình natural (title ~11%), cùng danh sách mẫu cho mọi mô hình.
        p = MO_HINH[a.mo_hinh]
        viec = [{"tap": t, "config": a.config, "seed": int(s)} for t in a.tap.split(",") for s in a.seeds.split(",")]
        cfg = cfg_chung(viec, dry=a.dry)
        cfg["mo_hinh"] = {k: v for k, v in p.items() if k not in ("slug", "nguon_them")}
        if a.gc:
            cfg["mo_hinh"]["gc"] = (a.gc == "bat")
        if a.lr is not None:
            cfg["lr"] = a.lr
        if a.dung_sau_buoc:
            cfg["dung_sau_buoc"] = a.dung_sau_buoc
        if a.so_eval:
            cfg["so_eval"] = a.so_eval
        if a.ngan_sach_token:
            cfg["ngan_sach_token"] = a.ngan_sach_token
        day_quota(cfg, dry=a.dry, slug=a.slug, mo_hinh_slug=p["slug"], them_nguon=p.get("nguon_them", []), gpu=not a.dry)
        return
    if a.mode == "natural":
        # Cấu hình "natural" (title ~11%) cho các bộ/seed chọn; mỗi notebook (--slug) chạy độc lập, chia việc cho nhiều notebook để chạy song song
        viec = [{"tap": t, "config": "natural", "seed": int(s)} for t in a.tap.split(",") for s in a.seeds.split(",")]
        cfg = cfg_chung(viec)
        if a.lr is not None:
            cfg["lr"] = a.lr
        day_quota(cfg, slug=a.slug)
        return
    if a.mode == "pack":
        pack()
        return
    if a.mode == "check1":
        # Kiểm môi trường tài khoản 2 (11/11/2026): chạy đúng MỘT việc đã có kết quả ở v55 (cụm, natural, seed 42; chrF macro 72,05)
        # để xem môi trường mới (peft ghim 0.18.1, mô hình đóng gói lại) có tái lập được kết quả cũ không. Khoảng 16 phút.
        day_quota(cfg_chung([{"tap": "cum", "config": "natural", "seed": 42}]))
        return
    if a.mode == "dry":
        chon = {"cum": ["natural", "balanced", "title_heavy"],
                "pair": ["natural", "balanced", "attr_heavy"]}
        viec = [{"tap": t, "config": c, "seed": s} for t in ("cum", "pair") for c in chon[t] for s in (42, 123)]
        day_quota(cfg_chung(viec, dry=True), dry=True)
        return
    if a.mode == "cum":
        viec = [{"tap": "cum", "config": c} for c in CONFIG_NAMES["cum"]]
        day_quota(cfg_chung(viec))
        return
    if a.mode == "step1":
        # Bước 1 đã chốt: 3 cấu hình mỗi bộ × 2 seed (42, 123)
        chon = {"cum": ["natural", "balanced", "title_heavy"],
                "pair": ["natural", "balanced", "attr_heavy"]}
        viec = [{"tap": t, "config": c, "seed": s} for t in ("cum", "pair") for c in chon[t] for s in (42, 123)]
        day_quota(cfg_chung(viec))
        return
    viec = [{"tap": t, "config": c} for t in ("cum", "pair") for c in CONFIG_NAMES[t]]
    day_quota(cfg_chung(viec))


if __name__ == "__main__":
    main()
