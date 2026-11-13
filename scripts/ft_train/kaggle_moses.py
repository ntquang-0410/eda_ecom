# -*- coding: utf-8 -*-
"""Moses (dich may thong ke theo cum tu) tren du lieu song ngu tung dong cua minh. Notebook Kaggle CPU, Internet BAT (apt + pip), khong test.

Quy trinh (theo config/moses_pipeline.sh cua Quang, chon them tham so hop ly):
  tach tu (vi: tach dau cau, giu so thap phan; zh: "char" = moi chu Han la mot token nhu Quang, hoac "jieba" = tach tu) -> loc cap (<=80 token, ti le <=9)
  -> can chinh eflomal (model 3, 4-4-4 lan lap) 2 chieu -> doi xung grow-diag-final-and (viet bang Python) -> mo hinh ngon ngu KenLM 5-gram (Quang: 3-gram)
  -> train-model.perl (buoc 4-9, cum toi da 7 tu, reordering msd-bidirectional-fe) -> MERT tren dev 1.000 dong (200 title + 800 dong thuoc tinh, fold 0, ngoai mau eval)
  -> loc bang cum theo dau vao -> giai ma eval (cung 1.200 mau eval cua LLM, tung dong) -> ghep lai chu -> sinh.jsonl {id, nhom, hyp}.
Moses duoc bien dich san o notebook nlp-moses-build (dau ra moses_bin_scripts.tgz gan qua kernel_sources). Moi (huong, seg) la mot mo hinh;
chon seg theo chrF tren DEV (khong phai eval), bao cao ca hai.
"""
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

VAO = Path("/kaggle/input")
RA = Path("/kaggle/working")
MOSES = Path("/tmp/moses")
B, S = MOSES / "bin", MOSES / "scripts"

# ---------- tach tu / ghep lai (ham thuan, co test) ----------
RE_VI = re.compile(r"\d+(?:[.,]\d+)*|\w+|[^\w\s]", re.UNICODE)
RE_ZH_CHAR = re.compile(r"[A-Za-z0-9]+(?:[.,][0-9]+)*|[㐀-鿿]|[^\sA-Za-z0-9㐀-鿿]")
_jieba = None


def tok_vi(s):
    return RE_VI.findall(s)


def tok_zh_char(s):
    return RE_ZH_CHAR.findall(s)


def tok_zh_jieba(s):
    global _jieba
    if _jieba is None:
        import jieba
        jieba.setLogLevel(60)
        _jieba = jieba
    return [t.strip() for t in _jieba.lcut(s) if t.strip()]


def lay_tach(ngon_ngu, seg):
    if ngon_ngu == "vi":
        return tok_vi
    return tok_zh_char if seg == "char" else tok_zh_jieba


def detok_vi(toks):
    s = " ".join(toks)
    s = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"(?<=\w)\s*([/-])\s*(?=\w)", r"\1", s)
    return s.strip()


def detok_zh(toks):
    out = ""
    for t in toks:
        if out and re.match(r"[A-Za-z0-9]", out[-1]) and re.match(r"[A-Za-z0-9]", t[0]):
            out += " "
        out += t
    return out.strip()


def ghep(ngon_ngu, dong):
    toks = dong.split()
    return detok_vi(toks) if ngon_ngu == "vi" else detok_zh(toks)


# ---------- doi xung hoa can chinh: grow-diag-final-and ----------
def doc_cap(dong):
    return {tuple(int(x) for x in p.split("-")) for p in dong.split()}


def grow_diag_final_and(fwd, rev):
    """fwd, rev: tap (i, j) = (chi so nguon, chi so dich). Tra ve tap cap sau grow-diag-final-and (Koehn et al. 2003)."""
    union = fwd | rev
    A = set(fwd & rev)
    ns = {i for i, _ in A}
    nt = {j for _, j in A}
    lang_gieng = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    doi = True
    while doi:
        doi = False
        for (i, j) in sorted(A):
            for di, dj in lang_gieng:
                p = (i + di, j + dj)
                if p in union and p not in A and (p[0] not in ns or p[1] not in nt):
                    A.add(p)
                    ns.add(p[0])
                    nt.add(p[1])
                    doi = True
    for p in sorted(union):              # final-and: them diem noi ca hai dau deu chua co can chinh
        if p not in A and p[0] not in ns and p[1] not in nt:
            A.add(p)
            ns.add(p[0])
            nt.add(p[1])
    return A


def doi_xung(tep_fwd, tep_rev, tep_ra, n_nguon, n_dich):
    """Doc hai tep 'i-j' (moi dong mot cau), ghi tep doi xung. Tu kiem huong tep rev (neu chi so vuot do dai thi dao)."""
    lat = None
    n = 0
    with open(tep_fwd, encoding="utf-8") as f, open(tep_rev, encoding="utf-8") as r, open(tep_ra, "w", encoding="utf-8") as o:
        for k, (a, b) in enumerate(zip(f, r)):
            fa, rb = doc_cap(a), doc_cap(b)
            if lat is None and k < 2000:
                sai = sum(1 for (i, j) in rb if i >= n_nguon[k] or j >= n_dich[k])
                sai_dao = sum(1 for (i, j) in rb if j >= n_nguon[k] or i >= n_dich[k])
                if sai > 0 and sai_dao == 0:
                    lat = True
            if lat:
                rb = {(j, i) for (i, j) in rb}
            o.write(" ".join(f"{i}-{j}" for i, j in sorted(grow_diag_final_and(fa, rb))) + "\n")
            n += 1
    return n, bool(lat)


# ---------- he thong ----------
def sh(cmd, tail=1500, check=True, log=None):
    t = time.time()
    r = subprocess.run(["bash", "-lc", "set -o pipefail; " + cmd], text=True, capture_output=True)
    out = (r.stdout or "") + (r.stderr or "")
    if log:
        Path(log).write_text(out, encoding="utf-8")
    print(">>", cmd[:260], "| ma", r.returncode, "| %.0fs" % (time.time() - t), flush=True)
    print(out[-tail:], flush=True)
    if check and r.returncode:
        raise RuntimeError(f"lenh_loi:{r.returncode}:{cmd[:140]}")
    return r.returncode, out


def tim(ten):
    ds = [p for p in VAO.rglob(ten) if p.is_file()]
    if not ds:
        raise RuntimeError(f"khong_thay:{ten}")
    return ds[0]


def doc(ten, toi_da=None):
    L = tim(ten).read_text(encoding="utf-8").split("\n")
    if L and L[-1] == "":
        L = L[:-1]
    return L[:toi_da] if toi_da else L


def cai():
    sh("apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev libgoogle-perftools-dev libsparsehash-dev > /tmp/apt.log 2>&1; tail -2 /tmp/apt.log")
    tgz = [p for p in VAO.rglob("moses_bin_scripts.tgz")]
    if not tgz:
        raise RuntimeError("khong_thay_moses_bin_scripts.tgz (gan notebook nlp-moses-build qua kernel_sources)")
    MOSES.mkdir(parents=True, exist_ok=True)
    sh(f"tar xzf {tgz[0]} -C {MOSES} && chmod -R +x {B} && ls {B} | wc -l && {B}/moses --version 2>&1 | head -2")
    sh("pip install -q eflomal jieba sacrebleu 2>&1 | tail -3")


def chrf_dev(hyp, ref, ngon_ngu):
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2 if ngon_ngu == "vi" else 0).corpus_score(hyp, [ref]).score


def chay_mot(hg, seg, bt, cfg, W0):
    sl, tl = hg.split("-")
    t0 = time.time()
    k = cfg.get("khoi")
    W = W0 / f"{hg}_{seg}_{bt}"
    W.mkdir(parents=True, exist_ok=True)
    tok_s, tok_t = lay_tach(sl, seg), lay_tach(tl, seg)
    tr_s, tr_t = doc(f"{hg}.{bt}.src", k), doc(f"{hg}.{bt}.tgt", k)
    dv_s, dv_t = doc(f"{hg}.dev.src", 200 if k else None), doc(f"{hg}.dev.tgt", 200 if k else None)
    ev_s, ev_id, ev_nh = doc(f"{hg}.eval.src", k), doc(f"{hg}.eval.id", k), doc(f"{hg}.eval.nhom", k)
    # 1) tach tu + loc cap
    A, Bt, ns, nt = [], [], [], []
    for s, t in zip(tr_s, tr_t):
        a, b = tok_s(s), tok_t(t)
        if 1 <= len(a) <= 80 and 1 <= len(b) <= 80 and max(len(a), len(b)) / max(1, min(len(a), len(b))) <= 9:
            A.append(" ".join(a)); Bt.append(" ".join(b)); ns.append(len(a)); nt.append(len(b))
    (W / f"train.{sl}").write_text("\n".join(A) + "\n", encoding="utf-8")
    (W / f"train.{tl}").write_text("\n".join(Bt) + "\n", encoding="utf-8")
    (W / f"dev.{sl}").write_text("\n".join(" ".join(tok_s(x)) for x in dv_s) + "\n", encoding="utf-8")
    (W / f"dev.{tl}").write_text("\n".join(" ".join(tok_t(x)) for x in dv_t) + "\n", encoding="utf-8")
    (W / f"eval.{sl}").write_text("\n".join(" ".join(tok_s(x)) for x in ev_s) + "\n", encoding="utf-8")
    # eval ban cum (tung manh) dich cung luot: ghi chung vao evalall de loc bang cum mot lan
    cum_s = doc(f"{hg}.evalcum.src", k) if cfg.get("eval_cum") else []
    cum_id = doc(f"{hg}.evalcum.id", k) if cfg.get("eval_cum") else []
    (W / f"evalall.{sl}").write_text("\n".join(" ".join(tok_s(x)) for x in ev_s + cum_s) + "\n", encoding="utf-8")
    print(f"[{hg}/{seg}] cap huan luyen {len(tr_s)} -> {len(A)} sau loc; token nguon tb {sum(ns)/len(ns):.1f}, dich tb {sum(nt)/len(nt):.1f}; {time.time()-t0:.0f}s", flush=True)
    # 2) can chinh eflomal 2 chieu + doi xung
    from eflomal import Aligner
    al = Aligner(model=3, n_iterations=(4, 4, 4), n_samplers=4)
    with open(W / f"train.{sl}", encoding="utf-8") as fs, open(W / f"train.{tl}", encoding="utf-8") as ft:
        al.align(fs, ft, links_filename_fwd=str(W / "fwd"), links_filename_rev=str(W / "rev"), quiet=True)
    n_cap, da_lat = doi_xung(W / "fwd", W / "rev", W / "sym.grow-diag-final-and", ns, nt)
    print(f"[{hg}/{seg}] can chinh xong {n_cap} cau (dao huong rev: {da_lat}); {time.time()-t0:.0f}s", flush=True)
    # 3) mo hinh ngon ngu 5-gram KenLM (chi tren phia dich cua train)
    (W / "lmtmp").mkdir(exist_ok=True)
    sh(f"{B}/lmplz -o 5 -S 8G -T {W}/lmtmp --discount_fallback < {W}/train.{tl} > {W}/lm.arpa && {B}/build_binary trie {W}/lm.arpa {W}/lm.bin && rm -f {W}/lm.arpa", tail=600)
    # 4) bang cum + reordering
    (W / "tm" / "model").mkdir(parents=True, exist_ok=True)       # bat dau tu buoc 4 thi script khong tu tao thu muc
    sh(f"perl {S}/training/train-model.perl -root-dir {W}/tm -corpus {W}/train -f {sl} -e {tl} -alignment grow-diag-final-and "
       f"-alignment-file {W}/sym -first-step 4 -last-step 9 -lm 0:5:{W}/lm.bin:8 -max-phrase-length 7 -reordering msd-bidirectional-fe "
       f"-cores 4 -sort-buffer-size 4G", tail=2500, log=W / "train_model.log")
    ini = W / "tm" / "model" / "moses.ini"
    # 5) MERT tren dev
    tuned = False
    try:
        sh(f"perl {S}/training/mert-moses.pl {W}/dev.{sl} {W}/dev.{tl} {B}/moses {ini} --working-dir {W}/mert --mertdir {B} --rootdir {S} "
           f"--decoder-flags='-threads 4 -v 0' --nbest 100 --maximum-iterations {cfg.get('mert_lap', 10)}", tail=1500, log=W / "mert.log")
        if (W / "mert" / "moses.ini").exists():
            ini, tuned = W / "mert" / "moses.ini", True
    except Exception as e:
        print("MERT loi, dung trong so mac dinh:", e, flush=True)
    # 6) giai ma dev (de chon seg) va eval, deu qua loc bang cum theo dau vao
    ket = {}
    for ten, n_ in (("dev", None), ("evalall", None)):
        sh(f"perl {S}/training/filter-model-given-input.pl {W}/f_{ten} {ini} {W}/{ten}.{sl} > /dev/null 2>&1; ls {W}/f_{ten}/moses.ini", tail=200)
        sh(f"{B}/moses -f {W}/f_{ten}/moses.ini -threads 4 -v 0 < {W}/{ten}.{sl} > {W}/{ten}.hyp.tok", tail=300, log=W / f"giai_ma_{ten}.log")
    dev_hyp = [ghep(tl, x) for x in (W / "dev.hyp.tok").read_text(encoding="utf-8").split("\n")[:len(dv_t)]]
    dev_chrf = round(chrf_dev(dev_hyp, dv_t, tl), 2)
    tat_ca = [ghep(tl, x) for x in (W / "evalall.hyp.tok").read_text(encoding="utf-8").split("\n")[:len(ev_s) + len(cum_s)]]
    ev_hyp, cum_hyp = tat_ca[:len(ev_s)], tat_ca[len(ev_s):]
    ra = RA / "moses" / f"{hg}_{seg}_{bt}"
    ra.mkdir(parents=True, exist_ok=True)
    (ra / "sinh.jsonl").write_text("\n".join(json.dumps({"id": i, "nhom": g, "hyp": h}, ensure_ascii=False) for i, g, h in zip(ev_id, ev_nh, ev_hyp)), encoding="utf-8")
    n_cum = 0
    if cum_s:      # ghep cac manh cua cung mot mau cum bang "; " (title chi co mot manh)
        gom = {}
        for i, h in zip(cum_id, cum_hyp):
            gom.setdefault(i.rsplit("#", 1)[0], []).append(h)
        nhom_cum = {json.loads(l)["id"]: json.loads(l)["nhom"] for l in tim(f"{hg}.evalcum_items.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()}
        (ra / "sinh_cum.jsonl").write_text("\n".join(json.dumps({"id": i, "nhom": nhom_cum.get(i, ""), "hyp": "; ".join(v)}, ensure_ascii=False) for i, v in gom.items()), encoding="utf-8")
        n_cum = len(gom)
    ve = {"huong": hg, "seg": seg, "bien_the": bt, "n_eval_cum": n_cum, "n_manh_cum": len(cum_s), "dong_train": len(tr_s), "dong_sau_loc": len(A), "mert": tuned, "dev_chrf": dev_chrf, "n_eval": len(ev_s),
          "so_hyp_rong": sum(1 for x in ev_hyp if not x.strip()), "giay": round(time.time() - t0, 1)}
    sh(f"cp {W}/train_model.log {W}/mert.log {ra}/ 2>/dev/null; cp {ini} {ra}/moses_tuned.ini 2>/dev/null; rm -rf {W}", check=False, tail=100)
    return ve


def main(cfg):
    t0 = time.time()
    cai()
    W0 = Path("/tmp/mw")
    W0.mkdir(exist_ok=True)
    bc = {"cau_hinh": cfg, "viec": []}
    for hg in cfg["huong"]:
        for seg in cfg["seg"]:
            for bt in cfg.get("bien_the", ["train"]):
                try:
                    bc["viec"].append(chay_mot(hg, seg, bt, cfg, W0))
                except Exception as e:
                    import traceback
                    bc["viec"].append({"huong": hg, "seg": seg, "bien_the": bt, "loi": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2500:]})
                    print("LOI", hg, seg, bt, e, flush=True)
                (RA / "moses_report.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1), encoding="utf-8")
    bc["giay_tong"] = round(time.time() - t0, 1)
    (RA / "moses_report.json").write_text(json.dumps(bc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("XONG", len(bc["viec"]), flush=True)


if __name__ == "__main__":
    raise SystemExit("Nhung vao notebook, goi main(CAU_HINH)")

