# -*- coding: utf-8 -*-
"""P2-C: chenh lech kep cua TITLE zh->vi (nhom tham chieu Qwen / Gemma) bang zero-shot Qwen3-4B, ban CUM, eval fold 0. Ke hoach khoa: docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P2.md.

Loai 1 (thien vi san co cua ho mo hinh): (zero-shot - moc) o nhom Qwen tru (zero-shot - moc) o nhom Gemma, moc = Moses (hoc du), OpenNMT, EcomGPT-7B.
Loai 2 (phan do fine-tune): (Qwen3-4B tinh chinh - zero-shot) o nhom Qwen tru cung dai luong o nhom Gemma.
Dung lai ham cua p1_phan_tich (cham_bo, nhom_tham_chieu, _kep, _corpus; bootstrap theo san pham 2.000 lan seed 42). Kiem tai lap truoc: kep cua Qwen3-4B so voi cac moc
phai khop vong_tham_chieu_cum.json cua P1 (title) tung so mot. Chi eval fold 0, khong doc test. Ban pair khong dung duoc, khong phan tich.

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p6_kep_zero_shot.py   (OPENBLAS_NUM_THREADS=1)
Ra:   data/processed/ft_run/phan_bien_v2/kep_zero_shot_cum.json
"""
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "scripts" / "ft_train"))
sys.path.insert(0, str(GOC / "scripts" / "phan_bien_v2"))
import p1_phan_tich as P1  # noqa: E402

R = GOC / "data" / "processed" / "ft_run"
RA = R / "phan_bien_v2"
ZS = R / "kaggle_tlc_song_song" / "zero_shot" / "sinh_zero_shot_cum.jsonl"
TEN_ZS = "Qwen3-4B zero-shot"


def diem_title(nhanh, m, refs, tap):
    """Diem tung san pham cho title zh->vi: chrF corpus cua title cua san pham, trung binh qua cac lan chay (seed), nhu p1_phan_tich.vong_tham_chieu."""
    per = []
    for lan, files in m["lan_chay"].items():
        df, _ = P1.cham_bo(nhanh, files, refs, tap)
        acc = {}
        for r in df[(df.huong == "zh->vi") & (df.loai == "title")].to_dict("records"):
            pid = str(r["id"]).split("_")[0]
            acc.setdefault(pid, []).append((r["hyp_cham"], r["_ref"]))
        per.append({p: P1._corpus("zh->vi", [x[0] for x in v], [x[1] for x in v]) for p, v in acc.items()})
    pid = sorted(set.intersection(*[set(x) for x in per]))        # sap xep: bootstrap xac dinh, khong phu thuoc thu tu bam cua set
    return {p: float(np.mean([x[p] for x in per])) for p in pid}


def main():
    nhanh = "cum"
    refs = P1.refs_bo(nhanh)
    cm = P1.map_category(nhanh)
    refs["category"] = refs["id"].str.split("_").str[0].map(cm).fillna("")
    tap = P1.tap_da_thay(nhanh)
    nhom = P1.nhom_tham_chieu(nhanh)["title"]
    models = {m["ten"]: m for m in P1.mo_hinh(nhanh)}
    chon = {k: v for k, v in models.items() if k == "Qwen3-4B" or "Moses (train" in k or "OpenNMT" in k or "EcomGPT" in k}
    assert ZS.exists(), ZS
    chon[TEN_ZS] = {"ten": TEN_ZS, "loai": "LLM_zero_shot", "lan_chay": {"1": [ZS]}}
    diem = {ten: diem_title(nhanh, m, refs, tap) for ten, m in chon.items()}
    pids = set(diem["Qwen3-4B"])
    ra = {"ke_hoach": "docs/phan_bien_vong2/KE_HOACH_PHAN_TICH_P2.md", "n_boot": P1.N_BOOT, "seed_boot": P1.SEED,
          "nhom_trong_eval": {g: sum(1 for p in pids if nhom.get(p) == g) for g in sorted({nhom.get(p) for p in pids})},
          "diem_tb_theo_nhom": {t: {g: float(np.mean([d[p] for p in d if nhom.get(p) == g])) for g in ("qwen", "gemma")} for t, d in diem.items()},
          "kiem_tai_lap_P1": {}, "loai1_zero_shot_so_moc": {}, "loai2_tinh_chinh_tru_zero_shot": None}
    bs = [t for t in diem if "Moses (train" in t or "OpenNMT" in t or "EcomGPT" in t]
    # kiem tai lap: kep cua 4B so voi cac moc phai trung P1
    cu = json.loads((RA / "vong_tham_chieu_cum.json").read_text(encoding="utf-8"))["title"]
    for B in bs:
        moi = P1._kep(diem["Qwen3-4B"], diem[B], nhom, "Qwen3-4B", B)
        k = f"Qwen3-4B − {B}"
        assert k in cu, (k, list(cu))
        # diem uoc (chenh) phai trung tung so; hai dau khoang bootstrap co the lech nho vi P1 rut theo thu tu bam cua set (khong xac dinh), P2 sap xep nen xac dinh
        sai_diem = abs(moi["kep"]["chenh"] - cu[k]["kep"]["chenh"])
        sai_ci = max(abs(moi["kep"][x] - cu[k]["kep"][x]) for x in ("thap", "cao"))
        ra["kiem_tai_lap_P1"][k] = {"moi": moi["kep"], "P1": cu[k]["kep"], "lech_diem": float(sai_diem), "lech_hai_dau_ci": float(sai_ci)}
        assert sai_diem <= 1e-9 and sai_ci <= 0.3, f"TAI LAP LECH {k}: diem {sai_diem}, ci {sai_ci}"
    # loai 1
    for B in bs:
        ra["loai1_zero_shot_so_moc"][f"{TEN_ZS} − {B}"] = P1._kep(diem[TEN_ZS], diem[B], nhom, TEN_ZS, B)
    # loai 2
    ra["loai2_tinh_chinh_tru_zero_shot"] = P1._kep(diem["Qwen3-4B"], diem[TEN_ZS], nhom, "Qwen3-4B", TEN_ZS)
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "kep_zero_shot_cum.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    print("nhom trong eval:", ra["nhom_trong_eval"])
    print("diem TB theo nhom:", {t: {g: round(v, 2) for g, v in d.items()} for t, d in ra["diem_tb_theo_nhom"].items()})
    print("tai lap P1: lech diem lon nhat", max(v["lech_diem"] for v in ra["kiem_tai_lap_P1"].values()), "| lech hai dau CI lon nhat", round(max(v["lech_hai_dau_ci"] for v in ra["kiem_tai_lap_P1"].values()), 3))
    for k, v in ra["loai1_zero_shot_so_moc"].items():
        print(f"loai1 {k}: Qwen {v['qwen']['chenh']:+.2f} (n={v['qwen']['n_sp']}), Gemma {v['gemma']['chenh']:+.2f} (n={v['gemma']['n_sp']}), kep {v['kep']['chenh']:+.2f} [{v['kep']['thap']:+.2f}; {v['kep']['cao']:+.2f}]")
    v = ra["loai2_tinh_chinh_tru_zero_shot"]
    print(f"loai2 4B-tinh chinh tru zero-shot: Qwen {v['qwen']['chenh']:+.2f}, Gemma {v['gemma']['chenh']:+.2f}, kep {v['kep']['chenh']:+.2f} [{v['kep']['thap']:+.2f}; {v['kep']['cao']:+.2f}]")


if __name__ == "__main__":
    main()
