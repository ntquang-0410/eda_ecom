# -*- coding: utf-8 -*-
"""Chạy TRÊN MÁY COLAB: OpenNMT-py huấn luyện từ đầu với MỘT seed (cùng cấu hình lần D3, chỉ đổi seed), rồi dịch eval bản cụm bằng chính mô hình đó.
Dùng nguyên kaggle_onmt.py (không sửa), chỉ đổi đường dẫn và bỏ bước cài (venv /content/onmt_env cài một lần cho mọi seed, tránh hai tiến trình giẫm lên nhau).
Dùng:  python onmt_seed_vm.py <seed>      Dữ liệu: /content/kg/data.tar.gz (chỉ các tệp song ngữ cần dùng, không có test)
Ra:    /content/kq/s<seed>/ket_qua.tar.gz (sinh.jsonl, sinh_cum.jsonl, báo cáo, log), /content/kq/s<seed>.DONE (json) khi xong, kể cả khi lỗi."""
import json
import sys
import tarfile
import time
import traceback
from pathlib import Path

seed = int(sys.argv[1])
W = Path(f"/content/kg/s{seed}")
KQ = Path(f"/content/kq/s{seed}")
KQ.mkdir(parents=True, exist_ok=True)
t0 = time.time()
tt = {"seed": seed, "loi": None}
try:
    W.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/content/kg/data.tar.gz") as t:
        t.extractall(W)                # tệp tự dựng của mình, đã kiểm hash trước khi tải lên
    sys.path.insert(0, "/content/code")
    import kaggle_onmt as O
    O.VAO, O.RA = W, W / "out"
    (W / "out").mkdir(exist_ok=True)
    O.VENV = Path("/content/onmt_env")
    O.PY = str(O.VENV / "bin" / "python")
    O.cai = lambda: None
    huong = ["zh-vi", "vi-zh"]
    O.main({"huong": huong, "bien_the": ["train"], "so_epoch": 20, "seed": seed})
    rp = json.loads((W / "out" / "onmt_report.json").read_text(encoding="utf-8"))
    if any("loi" in v for v in rp["viec"]):
        raise RuntimeError("huan_luyen_loi:" + json.dumps([v.get("loi") for v in rp["viec"]]))
    O.main({"che_do": "dich_cum", "huong": huong, "so_epoch": 20, "seed": seed})
    tt["giay"] = round(time.time() - t0)
    with tarfile.open(KQ / "ket_qua.tar.gz", "w:gz") as t:
        for p in sorted((W / "out").rglob("*")):
            if p.is_file() and p.suffix != ".pt" and "spm_data" not in p.parts:
                t.add(p, arcname=str(p.relative_to(W / "out")))
except BaseException:
    tt["loi"] = traceback.format_exc()[-3000:]
    print(tt["loi"], flush=True)
tt["tong_giay"] = round(time.time() - t0)
(KQ.parent / f"s{seed}.DONE").write_text(json.dumps(tt), encoding="utf-8")
print("DONE", json.dumps(tt), flush=True)
