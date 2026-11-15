# -*- coding: utf-8 -*-
"""Máy chủ dịch cho extension: POST /dich, GET /suc_khoe. Một lượt sinh tại một thời điểm (khóa), bộ đệm SQLite.

  laptop:  python app.py --llama http://127.0.0.1:8770
  Kaggle:  python app.py --hf <thư mục nền> <thư mục adapter>      (đặt DEMO_KEY để bật khóa truy cập khi mở ra Internet)
Nghe 127.0.0.1; ra Internet (nếu cần) qua đường hầm. DEMO_KEY đặt thì mọi yêu cầu phải có header X-Demo-Key đúng."""
import argparse
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time

from flask import Flask, jsonify, request

import dich as D
import loi_dan as L
from backend import ChuaDichSan


class BoDem:
    def __init__(self, duong_dan):
        self.db = sqlite3.connect(duong_dan, check_same_thread=False)
        self.db.execute("create table if not exists dem (k text primary key, v text, t real)")
        self.khoa = threading.Lock()

    def dong(self):
        self.db.close()

    def lay(self, k):
        with self.khoa:
            r = self.db.execute("select v from dem where k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def ghi(self, k, v):
        with self.khoa:
            self.db.execute("insert or replace into dem values (?,?,?)", (k, json.dumps(v, ensure_ascii=False), time.time()))
            self.db.commit()


def tao_app(backend, bo_dem, khoa_truy_cap=None):
    app = Flask(__name__)
    gpu = threading.Lock()

    def khoa_dem(loai, *phan):
        return hashlib.sha256(json.dumps([backend.ten, loai, *phan], ensure_ascii=False).encode("utf-8")).hexdigest()

    @app.before_request
    def kiem_khoa():
        if khoa_truy_cap and request.method != "OPTIONS" and not hmac.compare_digest(request.headers.get("X-Demo-Key", ""), khoa_truy_cap):
            return jsonify({"loi": "khong_co_quyen"}), 401

    @app.after_request
    def cors(r):
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Demo-Key"
        return r

    @app.route("/suc_khoe")
    def suc_khoe():
        try:
            return jsonify({"mo_hinh": backend.ten, **backend.suc_khoe()})
        except Exception as e:      # noqa: BLE001
            return jsonify({"loi": f"{type(e).__name__}: {e}"}), 503

    @app.route("/dich", methods=["POST", "OPTIONS"])
    def dich():
        if request.method == "OPTIONS":
            return ("", 204)
        j = request.get_json(force=True)
        loai = j.get("loai")
        try:
            if loai == "title":
                k = khoa_dem("title", j["zh"])
                v = bo_dem.lay(k)
                if v is None:
                    t0 = time.time()
                    with gpu:
                        v = {"vi": D.dich_title(backend.sinh, j["zh"]), "giay": round(time.time() - t0, 2)}
                    bo_dem.ghi(k, v)
                    v["dem"] = False
                else:
                    v["dem"] = True
                return jsonify(v)
            if loai == "khoi":
                cap = [(t, g) for t, g in j["cap"]]
                k = khoa_dem("khoi", j["title"], cap)
                v = bo_dem.lay(k)
                if v is None:
                    t0 = time.time()
                    with gpu:
                        ra, tk = D.dich_khoi(backend.sinh, j["title"], cap)
                    v = {"cap": ra, **tk, "giay": round(time.time() - t0, 2)}
                    bo_dem.ghi(k, v)
                    v["dem"] = False
                else:
                    v["dem"] = True
                return jsonify(v)
        except (KeyError, ValueError) as e:
            return jsonify({"loi": f"yeu_cau_sai:{e}"}), 400
        except ChuaDichSan:
            return jsonify({"loi": "chua_dich_san: trang này chưa có trong bộ đệm (chế độ không mô hình)"}), 503
        return jsonify({"loi": "loai_la"}), 400

    return app


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--llama")
    g.add_argument("--hf", nargs=2, metavar=("CO_SO", "ADAPTER"))
    g.add_argument("--chi-bo-dem", metavar="TEN_MO_HINH", help="không mô hình, chỉ trả bản dịch đã lưu trong --dem (TEN_MO_HINH trùng tên lúc dịch, vd kaggle-fp16-pair_cuoi)")
    ap.add_argument("--cong", type=int, default=8765)
    ap.add_argument("--dem", default="bo_dem.sqlite")
    ap.add_argument("--ten", default=None)
    a = ap.parse_args()
    from backend import HF, ChiBoDem, LlamaServer
    if a.chi_bo_dem:
        backend = ChiBoDem(a.chi_bo_dem)
    else:
        backend = LlamaServer(a.llama, a.ten or "llama-q5km-pair_cuoi") if a.llama else HF(a.hf[0], a.hf[1], a.ten or "hf-fp16-pair_cuoi")
    tao_app(backend, BoDem(a.dem), os.environ.get("DEMO_KEY")).run(host="127.0.0.1", port=a.cong, threaded=True)


if __name__ == "__main__":
    main()
