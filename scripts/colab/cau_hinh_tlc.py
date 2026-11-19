# -*- coding: utf-8 -*-
"""Cấu hình job Colab cho khảo sát tỉ trọng title mức 25% và 35% (trả lời phản biện vòng 2, mục 3.3.5), trên fold 0, bản cụm và bản pair.

Cùng cách chạy như phiên Kaggle đã chạy p11 và p75 (docs/ket_qua_kaggle, phiên TL): kaggle_train.chay_b78, trọng số loss w_title, mọi title thấy 2 epoch,
micro_bs 32, ngân sách 4.096 token, max_len 2.048, seed 42, lr 2e-4, rank 64, 2 epoch, 750 mẫu kiểm và 250 mẫu sinh mỗi nhóm. Khác phiên cũ: máy Colab (cùng loại card),
chạy từng lượt 30 phút, thêm vram_toi_thieu_gb. Hai việc p11 (cụm, pair) chạy lại làm ĐỐI CHỨNG: xem môi trường/lượt chạy tiếp có làm lệch điểm so với phiên Kaggle không.
"""
import json
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
GOI_POOL = GOC / "data" / "processed" / "ft_run" / "goi_pool"
MO_HINH = {"repo": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c"}   # đúng revision của phiên Kaggle (bao_cao_tl.json)

# (tên việc, nhánh, p_title). p_title của p11 giữ đúng giá trị phiên cũ (cụm 0,1074; pair 0,1086) để đối chứng so được.
VIEC_THAT = [("cum_f0_p11", "cum", 0.1074), ("pair_f0_p11", "pair", 0.1086),
             ("cum_f0_p25", "cum", 0.25), ("cum_f0_p35", "cum", 0.35),
             ("pair_f0_p25", "pair", 0.25), ("pair_f0_p35", "pair", 0.35)]
VIEC_KHOI = [("cum_f0_p25", "cum", 0.25), ("pair_f0_p35", "pair", 0.35)]


def viec_cfg(ds):
    return [{"ten": t, "nhanh": n, "fold": 0, "p_title": p, "ca_fold": False} for t, n, p in ds]


def cfg_chung():
    return {"che_do": "b78", "micro_bs": 32, "tich_luy": 1, "ngan_sach_token": 4096, "gc": False, "max_len": 2048, "seed": 42,
            "so_kiem_moi_nhom": 750, "so_sinh_moi_nhom": 250, "chrf_toi_thieu": 65.0, "lap_toi_da": 0.05, "vram_toi_thieu_gb": 80,
            "gio_toi_da": 0.5}


def pool_sha256():
    mf = json.loads((GOI_POOL / "MANIFEST.json").read_text(encoding="utf-8"))
    return {f"pool_{nh}.jsonl.gz": mf["nhanh"][nh]["sha256_gz"] for nh in ("cum", "pair")}


def tao_job(che_do):
    """che_do: 'that' (6 việc, ~8–9 giờ) hoặc 'khoi' (2 việc rất ngắn để kiểm toàn bộ đường ống, kể cả chạy tiếp sau khi dừng)."""
    if che_do == "that":
        ds, tt = VIEC_THAT, "tlc"
        cfg = {**cfg_chung(), "ten_trang_thai": tt, "viec": viec_cfg(ds)}
        extra = {"chunk_gio": 0.5, "tran_gio": 10.0, "cho_tai_phut": 180}
    elif che_do == "khoi":
        ds, tt = VIEC_KHOI, "thu_tlc"
        # so_train 320 + rut_gon_sinh 3: chạy vài phút; dung_sau_buoc 3: mỗi lượt dừng sớm sau 3 bước để thử luôn đường chạy tiếp và đóng gói checkpoint
        cfg = {**cfg_chung(), "ten_trang_thai": tt, "viec": viec_cfg(ds), "so_kiem_moi_nhom": 10, "so_sinh_moi_nhom": 250, "rut_gon_sinh": 3,
               "so_train": 320, "dung_sau_buoc": 3, "chrf_toi_thieu": None}
        extra = {"chunk_gio": 0.5, "tran_gio": 1.0, "cho_tai_phut": 30, "giu_adapter": False}      # khói: không giữ adapter để tải nhanh; vẫn kiểm tải checkpoint 1,6 GB
    else:
        raise ValueError(che_do)
    return {"phien": tt, "ten_trang_thai": tt, "cfg": cfg, "viec": [t for t, _, _ in ds], "mo_hinh": MO_HINH, "pool_sha256": pool_sha256(),
            "ghim_phien_ban": True, "phien_ban": {"torch": "2.10.0", "transformers": "5.0.0", "peft": "0.18.1"}, "giu_adapter": True, **extra}
