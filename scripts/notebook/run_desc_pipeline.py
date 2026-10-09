# -*- coding: utf-8 -*-
r"""Chạy tuần tự các ô code của notebook pipeline description (P0 đến P13) trong một kernel Jupyter thật,
dùng đúng Python đang chạy script này (nên dùng .venv của repo), rồi ghi kết quả vào một BẢN SAO notebook.

Cách dùng (từ gốc repo):
    .venv\Scripts\python.exe scripts\notebook\run_desc_pipeline.py [notebook] [file_ra]

Mặc định: notebooks\desc-pipeline.ipynb  ->  notebooks\desc-pipeline.da_chay.ipynb
(notebook gốc không bị sửa; bản đã chạy chứa các dòng dữ liệu mẫu nên KHÔNG commit, .gitignore đã loại).
Dừng ở ô lỗi đầu tiên, in tên lỗi, thoát mã 1. Chạy xong hết thì in ALL OK, thoát mã 0.
"""
import json
import sys
import time
from pathlib import Path
from queue import Empty

from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[2]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "desc-pipeline.ipynb"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + ".da_chay.ipynb")
nb = json.loads(src.read_text(encoding="utf-8"))

km = KernelManager(kernel_name="python3")
km.start_kernel(cwd=str(src.resolve().parent))
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=180)
n, failed = 0, False
try:
    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n += 1
        code = "".join(cell["source"])
        t = time.time()
        msg_id = kc.execute(code)
        outputs = []
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=3600)
            except Empty:
                outputs.append({"output_type": "stream", "name": "stderr", "text": "TIMEOUT\n"})
                failed = True
                break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mt, c = msg["msg_type"], msg["content"]
            if mt == "status" and c["execution_state"] == "idle":
                break
            if mt == "stream":
                if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                    outputs[-1]["text"] += c["text"]
                else:
                    outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
            elif mt in ("display_data", "execute_result"):
                o = {"output_type": mt, "data": c["data"], "metadata": c.get("metadata", {})}
                if mt == "execute_result":
                    o["execution_count"] = c["execution_count"]
                outputs.append(o)
            elif mt == "error":
                outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"], "traceback": c["traceback"]})
                failed = True
        for o in outputs:
            if o["output_type"] == "stream":
                o["text"] = o["text"].splitlines(keepends=True)
        cell["outputs"] = outputs
        cell["execution_count"] = n
        err = [o for o in outputs if o["output_type"] == "error"]
        print(f"ô {ci:2d} (code #{n}) {'LOI' if err else 'ok'} {time.time() - t:7.1f}s", flush=True)
        if err:
            print("   ", err[0]["ename"], err[0]["evalue"][:600], flush=True)
            break
        if failed:
            break
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(("THAT BAI, kết quả dở dang ở " if failed else "ALL OK, kết quả ở ") + str(dst))
sys.exit(1 if failed else 0)
