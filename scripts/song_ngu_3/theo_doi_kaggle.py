# -*- coding: utf-8 -*-
"""Theo dõi một notebook Kaggle của bộ 3 đến khi xong rồi tải đầu ra về. Chịu được mất mạng: chỉ dừng khi Kaggle báo rõ COMPLETE, ERROR
hoặc CANCEL; lỗi mạng hoặc câu trả lời lạ thì chờ rồi hỏi lại (lần 24/10/2026 một lần timeout làm vòng lặp cũ tưởng notebook đã xong
và tải về thư mục rỗng). Tải về xong kiểm có file lô; nếu rỗng thì báo lỗi, không coi là xong.

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\theo_doi_kaggle.py <gemma|qwen> <thư mục con đích> [số giờ tối đa, mặc định 8]
"""
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
PY = str(Path(sys.executable))
KAGGLE = str(Path(sys.executable).parent / "kaggle")
SLUG = {"gemma": "song-ngu-3-cum", "qwen": "song-ngu-3-cum-qwen"}


def trang_thai(kid):
    try:
        r = subprocess.run([KAGGLE, "kernels", "status", kid], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"KernelWorkerStatus\.(\w+)", r.stdout + r.stderr)
    return m.group(1) if m else None


def main():
    bt, dich = sys.argv[1], sys.argv[2]
    gio = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    kid = f"trinhnhatanh/{SLUG[bt]}"
    t0 = time.time()
    loi_lien = 0
    while time.time() - t0 < gio * 3600:
        s = trang_thai(kid)
        print(f"{time.strftime('%H:%M')} ({(time.time() - t0) / 60:.0f} phút) {bt}: {s or 'không đọc được (mạng?), hỏi lại'}", flush=True)
        if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
            break
        loi_lien = loi_lien + 1 if s is None else 0
        time.sleep(180 if s else 60)
    else:
        print("Hết số giờ theo dõi mà notebook chưa xong.")
        return 2
    if s != "COMPLETE":
        print(f"Notebook kết thúc với trạng thái {s}: vẫn tải đầu ra về để xem log.")
    for lan in range(5):
        r = subprocess.run([PY, "-X", "utf8", str(ROOT / "scripts" / "song_ngu_3" / "kaggle_api.py"), "tai_ve", bt, dich],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        d = ROOT / "data" / "processed" / "song_ngu_3" / dich
        lo = list(d.rglob("lo_*.parquet"))
        print(f"Tải về lần {lan + 1}: {len(lo)} file lô, {sum(f.stat().st_size for f in lo):,} byte", flush=True)
        if lo:
            return 0 if s == "COMPLETE" else 1
        time.sleep(60)
    print("Tải về nhưng không thấy file lô nào: kiểm tra log.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
