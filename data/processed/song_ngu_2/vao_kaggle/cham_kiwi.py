# -*- coding: utf-8 -*-
"""Bước 6a (Kaggle, GPU): chấm điểm CometKiwi (Unbabel/wmt22-cometkiwi-da) cho bản dịch của MỘT mô hình.

Chạy trong môi trường Python riêng có unbabel-comet (notebook tạo bằng uv). Cần biến môi trường HF_TOKEN
(Kaggle Secrets) vì mô hình phải chấp nhận điều khoản trên Hugging Face. Không in, không ghi token.
- Đọc bản dịch ở <ra>/<mo_hinh>/lo_*.parquet, ghi điểm TỪNG LÔ ra <ra>/kiwi_<mo_hinh>/lo_*.parquet.
- Chạy tiếp được như dich.py (bỏ qua đơn vị đã có điểm, đọc thêm --cu); hết --gio_toi_da thì dừng êm.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", required=True, help="don_vi_can_dich.parquet")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--mo_hinh", required=True)
    ap.add_argument("--gio_toi_da", type=float, default=2.0)
    ap.add_argument("--lo", type=int, default=5000)
    ap.add_argument("--cu", action="append", default=[])
    ap.add_argument("--gia", action="store_true", help="điểm giả, CHỈ để thử trên máy không có GPU và không có COMET")
    a = ap.parse_args()
    t0 = time.time()
    ra = Path(a.ra) / f"kiwi_{a.mo_hinh}"
    ra.mkdir(parents=True, exist_ok=True)
    for k, cu in enumerate(a.cu):
        for f in sorted(Path(cu).glob(f"kiwi_{a.mo_hinh}/lo_*.parquet")):
            if not (ra / f"lo_cu{k}_{f.name}").exists():
                shutil.copy2(f, ra / f"lo_cu{k}_{f.name}")
    # cột câu nguồn: van_ban_zh (bộ 1, Trung -> Việt) hoặc van_ban_nguon (bộ 2, Việt -> Trung); CometKiwi nhận cả hai cặp ngôn ngữ
    cot_nguon = "van_ban_nguon" if "van_ban_nguon" in pd.read_parquet(a.vao).columns else "van_ban_zh"
    DV = pd.read_parquet(a.vao, columns=["ma_don_vi", cot_nguon]).rename(columns={cot_nguon: "van_ban_zh"})
    files = sorted((Path(a.ra) / a.mo_hinh).glob("lo_*.parquet"))
    if not files:
        log("Chưa có bản dịch của", a.mo_hinh, "- bỏ qua")
        return
    BD = pd.concat([pd.read_parquet(f, columns=["ma_don_vi", "ban_dich"]) for f in files]).drop_duplicates("ma_don_vi", keep="last")
    da = set()
    for f in ra.glob("lo_*.parquet"):
        da |= set(pd.read_parquet(f, columns=["ma_don_vi"]).ma_don_vi)
    X = BD[~BD.ma_don_vi.isin(da)].merge(DV, on="ma_don_vi", how="left")
    X = X[X.ban_dich.fillna("").str.strip() != ""].reset_index(drop=True)
    log(f"kiwi {a.mo_hinh}: cần chấm {len(X):,} (đã có {len(da):,})")
    the = {"mo_hinh_cham": "Unbabel/wmt22-cometkiwi-da", "ban_dich_cua": a.mo_hinh, "bat_dau": time.strftime("%Y-%m-%d %H:%M:%S"),
           "can_cham": int(len(X)), "da_co_truoc": len(da)}
    f_the = Path(a.ra) / f"the_kiwi_{a.mo_hinh}.json"
    if len(X):
        if a.gia:
            import hashlib

            class _Gia:
                def predict(self, data, **k):
                    return type("O", (), {"scores": [int(hashlib.md5(d["mt"].encode()).hexdigest()[:4], 16) / 65535 for d in data]})()
            model = _Gia()
        else:
            from comet import download_model, load_from_checkpoint
            model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
        so = [int(m.group(1)) for f in ra.glob("lo_*.parquet") for m in [re.fullmatch(r"lo_(\d+)", f.stem)] if m]
        stt = max(so) + 1 if so else 0
        xong = 0
        for i in range(0, len(X), a.lo):
            if time.time() - t0 > a.gio_toi_da * 3600:
                the["dung_vi_het_gio"] = True
                log("Hết giờ cho phép, dừng êm")
                break
            lo = X.iloc[i:i + a.lo]
            out = model.predict([{"src": s, "mt": m} for s, m in zip(lo.van_ban_zh, lo.ban_dich)], batch_size=64, gpus=1,
                                progress_bar=False)
            tam = ra / f".tam_{stt:05d}.parquet"
            pd.DataFrame({"ma_don_vi": lo.ma_don_vi.values, "diem": [round(float(x), 5) for x in out.scores]}).to_parquet(tam, index=False)
            tam.rename(ra / f"lo_{stt:05d}.parquet")
            stt += 1
            xong += len(lo)
            log(f"kiwi {a.mo_hinh}: {xong:,}/{len(X):,}")
        the["da_cham_phien_nay"] = xong
    the.update(ket_thuc=time.strftime("%Y-%m-%d %H:%M:%S"), giay=round(time.time() - t0))
    f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
