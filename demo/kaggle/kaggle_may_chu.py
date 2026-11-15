# -*- coding: utf-8 -*-
"""Chạy máy chủ dịch trên Kaggle (GPU T4, Internet bật) và mở ra qua đường hầm Cloudflare để extension gọi.
Dùng đúng cấu hình đã đo ở dòng thứ sáu (Qwen3-4B fp16 + LoRA), adapter pair_cuoi (demo), một GPU (GPU thứ hai không cần).
In ra log các dòng 'KET_NOI url=... key=...' để đọc bằng `kaggle kernels logs`; khóa truy cập sinh ngẫu nhiên mỗi phiên.
Hết giờ (cfg['gio']) thì tự tắt."""
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

GOC = Path("/kaggle/working/may_chu")
CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def sh(c, **kw):
    return subprocess.run(c, shell=True, text=True, capture_output=True, **kw)


def goi_json(url, body, khoa, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", "X-Demo-Key": khoa})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")), time.time() - t0


def main(cfg):
    r = sh("pip install -q 'peft==0.18.1' flask 2>&1 | tail -2")
    print("pip:", r.stdout.strip()[-300:], flush=True)
    sys.path.insert(0, str(GOC))
    os.chdir(GOC)
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpu", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    co_so = next(Path("/kaggle/input").rglob("REVISION.txt")).parent
    adapter = next(p for p in Path("/kaggle/input").rglob("pair_cuoi/ep2/adapter_model.safetensors")).parent
    print("co_so", co_so, "| revision", co_so.joinpath("REVISION.txt").read_text().split(), "| adapter", adapter, flush=True)

    from backend import HF
    import app as A
    t0 = time.time()
    backend = HF(str(co_so), str(adapter), ten=f"kaggle-fp16-pair_cuoi")
    print(f"nạp mô hình {time.time() - t0:.0f}s trên {backend.gpu}", flush=True)
    khoa = secrets.token_urlsafe(24)
    app = A.tao_app(backend, A.BoDem(str(GOC / "bo_dem.sqlite")), khoa)
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", 8765, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    sh(f"curl -fsSL -o cloudflared {CLOUDFLARED} && chmod +x cloudflared")
    cf = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8765", "--no-autoupdate"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(GOC))
    url, dong = None, []
    t0 = time.time()
    while url is None and time.time() - t0 < 120:
        l = cf.stdout.readline()
        if not l:
            break
        dong.append(l.strip())
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", l)
        if m:
            url = m.group(0)
    threading.Thread(target=lambda: [None for _ in cf.stdout], daemon=True).start()       # xả đệm để cloudflared không nghẽn
    if not url:
        print("KHONG_MO_DUOC_DUONG_HAM", dong[-8:], flush=True)
        return
    print(f"KET_NOI url={url} key={khoa}", flush=True)

    # tự kiểm qua đường hầm công khai (như extension sẽ gọi)
    time.sleep(8)
    kq = {"url": url, "gpu": backend.gpu}
    try:
        kq["title"], kq["giay_title"] = goi_json(url + "/dich", {"loai": "title", "zh": "源头厂家高级感立体短袖T恤男夏季薄款新款潮牌圆领半袖男士体恤"}, khoa)
        cap = [["品牌", "Elmich"], ["材质", "棉麻"], ["颜色", "黑色"], ["是否含遥控", "否"], ["容量", "500ml"]] * 4
        kq["khoi20"], kq["giay_khoi20"] = goi_json(url + "/dich", {"loai": "khoi", "title": "Elmich 不锈钢保温杯 500ml 黑色", "cap": cap}, khoa)
        kq["khoi20_lech_dong"] = kq["khoi20"].get("lech_dong")
        kq["khong_khoa"] = None
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/dich", data=b"{}", headers={"Content-Type": "application/json"}), timeout=30)
        except urllib.error.HTTPError as e:
            kq["khong_khoa"] = e.code
    except Exception as e:      # noqa: BLE001
        kq["loi_tu_kiem"] = f"{type(e).__name__}: {e}"
    print("TU_KIEM", json.dumps(kq, ensure_ascii=False), flush=True)

    het = time.time() + cfg.get("gio", 2.0) * 3600
    while time.time() < het:
        time.sleep(300)
        print(f"song {int((het - time.time()) / 60)} phut nua", flush=True)
    print("HET_GIO", flush=True)
