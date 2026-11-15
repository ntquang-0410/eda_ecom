# -*- coding: utf-8 -*-
"""Thử content.js trên các trang 1688 đã lưu (demo/mau_trang). Trang dựng từ file lưu (mọi yêu cầu mạng khác bị chặn),
chrome.runtime.sendMessage được thay bằng cầu nối Python gọi máy chủ dịch.

  Chế độ giả (không GPU, kiểm logic DOM):   demo\\.venv\\Scripts\\python.exe -X utf8 demo/extension/test_extension.py
  Chế độ thật (máy chủ Kaggle/laptop):       ... test_extension.py --that DUONG_DAN_LOG_KAGGLE [--ra ket_qua.json]
    (đọc dòng 'KET_NOI url=... key=...' trong log; khóa không in ra màn hình)
Kiểm: tên và số dòng thuộc tính đổi, giá trị không hỏng cấu trúc, bấm lại trả ĐÚNG bản gốc, bấm nữa trở lại bản dịch."""
import argparse
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

AQ = Path(__file__).resolve().parent
sys.path.insert(0, str(AQ.parent / "may_chu"))

STUB = """
window.chrome = { runtime: { sendMessage: async (m) => JSON.parse(await window.py_goi(JSON.stringify(m))) } };
"""
SNAP = """
() => {
  const q = (s) => [...document.querySelectorAll(s)].map(e => e.textContent.trim());
  return { title: document.querySelector('.module-od-title h1').textContent.trim(),
           cpv_ten: q('.cpv-attr-label'), cpv_gt: q('.cpv-attr-value'),
           bang_ten: q('.attr-descriptions th.ant-descriptions-item-label'), bang_gt: q('.attr-descriptions td.ant-descriptions-item-content') };
}
"""
NUT = "document.documentElement.lastElementChild.shadowRoot.getElementById('nut')"
TT = "document.documentElement.lastElementChild.shadowRoot.getElementById('tt')"


def cau_noi_gia():
    import app as A
    from test_may_chu import Gia
    d = tempfile.TemporaryDirectory()
    bd = A.BoDem(str(Path(d.name) / "x.sqlite"))
    cli = A.tao_app(Gia(), bd, "k").test_client()

    def goi(s):
        m = json.loads(s)
        r = cli.post("/dich", json=m["body"], headers={"X-Demo-Key": "k"}) if m["loai_yeu_cau"] == "dich" else cli.get("/suc_khoe", headers={"X-Demo-Key": "k"})
        return json.dumps({"ok": r.status_code == 200, "data": r.get_json(), "loi": None if r.status_code == 200 else f"HTTP {r.status_code}"})
    return goi, (lambda: (bd.dong(), d.cleanup()))


def cau_noi_that(duong_log=None, url=None):
    if url:                       # máy chủ cục bộ không khóa (laptop / chế độ chỉ-bộ-đệm)
        khoa = ""
    else:
        txt = Path(duong_log).read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"KET_NOI url=(\S+) key=(\S+)", txt)
        if not m:
            raise SystemExit("Chưa thấy dòng KET_NOI trong log (máy chủ chưa lên).")
        url, khoa = m[-1]

    def goi(s):
        m = json.loads(s)
        req = urllib.request.Request(url + ("/dich" if m["loai_yeu_cau"] == "dich" else "/suc_khoe"),
                                     data=json.dumps(m["body"]).encode("utf-8") if m["loai_yeu_cau"] == "dich" else None,
                                     headers={"Content-Type": "application/json", "X-Demo-Key": khoa})
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.dumps({"ok": True, "data": json.loads(r.read().decode("utf-8"))})
        except urllib.error.HTTPError as e:
            return json.dumps({"ok": False, "loi": f"HTTP {e.code}"})
        except Exception as e:      # noqa: BLE001
            return json.dumps({"ok": False, "loi": f"{type(e).__name__}: {e}"})
    return goi, (lambda: None)


def chay(goi, gia, ra, chi=None):
    loi, kq = 0, []
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        for f in sorted((AQ.parent / "mau_trang").glob("*.html")):
            if chi and f.stem not in chi:
                continue
            oid, html = f.stem, f.read_text(encoding="utf-8")
            pg = br.new_page()

            def chan(route, request, h=html, o=oid):
                if request.url.startswith(f"https://detail.1688.com/offer/{o}"):
                    route.fulfill(status=200, content_type="text/html; charset=utf-8", body=h)
                else:
                    route.abort()
            pg.route("**/*", chan)
            pg.expose_function("py_goi", goi)
            pg.add_init_script(STUB)
            pg.goto(f"https://detail.1688.com/offer/{oid}.html", wait_until="domcontentloaded")
            pg.add_script_tag(path=str(AQ / "content.js"))
            g = pg.evaluate(SNAP)
            t0 = time.time()
            pg.evaluate(NUT + ".click()")
            try:
                pg.wait_for_function(NUT + ".textContent.includes('bản gốc')", timeout=600000)
            except Exception:
                print("LOI", oid, "không xong:", pg.evaluate(TT + ".textContent"))
                loi += 1
                pg.close()
                continue
            giay = time.time() - t0
            v = pg.evaluate(SNAP)
            tt = pg.evaluate(TT + ".textContent")
            n = len(g["cpv_ten"]) + len(g["bang_ten"])
            ten_g, ten_v = g["cpv_ten"] + g["bang_ten"], v["cpv_ten"] + v["bang_ten"]
            ok = [("title đổi", v["title"] != g["title"] and (not gia or v["title"] == "vi:" + g["title"])),
                  ("hầu hết nhãn được đổi", sum(a != b for a, b in zip(ten_g, ten_v)) >= (n if gia else 0.8 * n)),
                  ("số phần tử giữ nguyên", len(ten_v) == len(ten_g) and len(v["cpv_gt"]) == len(g["cpv_gt"]) and len(v["bang_gt"]) == len(g["bang_gt"]))]
            pg.evaluate(NUT + ".click()")
            ok.append(("bấm lần 2 trả đúng bản gốc", pg.evaluate(SNAP) == g))
            pg.evaluate(NUT + ".click()")
            ok.append(("bấm lần 3 trở lại bản dịch", pg.evaluate(SNAP) == v))
            bad = [n_ for n_, x in ok if not x]
            loi += len(bad)
            print("DAT " if not bad else "LOI ", oid, f"| nhãn {n} | {giay:.0f} s |", tt, "|", bad or "")
            kq.append({"offer": oid, "giay": round(giay, 1), "trang_thai": tt, "goc": g, "dich": v})
            pg.close()
        br.close()
    if ra:
        Path(ra).write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    return loi


def chay_ep_tieng_trung(goi_goc):
    """Trang đang là bản 1688 tự dịch (tên tiếng Anh): nút đề nghị chuyển sang tiếng Trung -> cookie -> tải lại -> tự dịch.
    Ca 2: 1688 vẫn trả trang tiếng Anh sau khi đặt cookie -> báo hướng dẫn rõ, không lặp vô hạn."""
    loi = 0
    f = next(iter(sorted((AQ.parent / "mau_trang").glob("*.html"))))
    html_zh = f.read_text(encoding="utf-8")
    m = re.search(r'(class="module-od-title".*?<h1>)(.*?)(</h1>)', html_zh, re.S)
    html_en = html_zh[:m.start(2)] + "Super Popular Bucket Bag for Women" + html_zh[m.end(2):]
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        for ten, giu_anh in (("ép được tiếng Trung", False), ("1688 vẫn trả tiếng Anh", True)):
            st = {"zh": False, "so_lan_ep": 0}

            def goi(s, st=st, giu_anh=giu_anh):
                if json.loads(s)["loai_yeu_cau"] == "ep_zh":
                    st["so_lan_ep"] += 1
                    st["zh"] = not giu_anh
                    return json.dumps({"ok": True})
                return goi_goc(s)
            pg = br.new_page()
            pg.route("**/*", lambda route, request, st=st: route.fulfill(status=200, content_type="text/html; charset=utf-8", body=html_zh if st["zh"] else html_en)
                     if request.url.startswith("https://detail.1688.com/offer/") else route.abort())
            pg.expose_function("py_goi", goi)
            pg.add_init_script(STUB)
            # chạy lại sau mỗi lần tải trang như extension thật (document_idle: đợi DOM xong)
            pg.add_init_script("document.addEventListener('DOMContentLoaded', () => {\nif (window.top !== window) return;       // extension thật chỉ chạy ở khung chính (all_frames=false)\n" + (AQ / "content.js").read_text(encoding="utf-8") + "\n});")
            pg.goto("https://detail.1688.com/offer/1.html", wait_until="domcontentloaded")
            pg.wait_for_function("!!document.documentElement.lastElementChild.shadowRoot", timeout=10000)
            pg.evaluate(NUT + ".click()")
            pg.wait_for_function(NUT + ".textContent.includes('Chuyển trang sang tiếng Trung')", timeout=10000)
            pg.evaluate(NUT + ".click()")
            ok = []
            if not giu_anh:
                try:
                    pg.wait_for_function(NUT + ".textContent.includes('bản gốc')", timeout=60000)
                    ok.append(("tự dịch sau khi tải lại", pg.evaluate("document.querySelector('.module-od-title h1').textContent").startswith("vi:")))
                except Exception:
                    ok.append(("tự dịch sau khi tải lại", False))
            else:
                try:
                    pg.wait_for_function(TT + ".textContent.includes('vẫn trả trang')", timeout=30000)
                    ok.append(("báo hướng dẫn khi vẫn tiếng Anh", True))
                except Exception:
                    ok.append(("báo hướng dẫn khi vẫn tiếng Anh", False))
                ok.append(("chỉ ép đúng 1 lần (không lặp)", st["so_lan_ep"] == 1))
            bad = [n for n, x in ok if not x]
            loi += len(bad)
            print(("DAT " if not bad else "LOI "), "ép tiếng Trung:", ten, bad or "")
            pg.close()
        br.close()
    return loi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--that", help="đường dẫn log Kaggle có dòng KET_NOI (bỏ trống = mô hình giả)")
    ap.add_argument("--url", help="địa chỉ máy chủ cục bộ không khóa, vd http://127.0.0.1:8765")
    ap.add_argument("--ra", help="ghi kết quả (gốc/dịch từng trang) ra JSON")
    ap.add_argument("--chi", nargs="*", help="chỉ chạy các mã offer này")
    a = ap.parse_args()
    thuc = bool(a.that or a.url)
    goi, dong = (cau_noi_that(a.that, a.url) if thuc else cau_noi_gia())
    try:
        loi = chay(goi, gia=not thuc, ra=a.ra, chi=a.chi)
        if not thuc:
            loi += chay_ep_tieng_trung(goi)
    finally:
        dong()
    print("TAT CA DAT" if not loi else f"{loi} LỖI")
    sys.exit(1 if loi else 0)


if __name__ == "__main__":
    main()
