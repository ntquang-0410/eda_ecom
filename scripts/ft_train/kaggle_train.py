"""Huấn luyện Qwen3-4B + LoRA trên Kaggle (RTX Pro 6000, phiên tắt Internet). Notebook nlp-group3 nhúng file này và gọi main(CAU_HINH).

che_do="thu_b5" (B5): chạy so_buoc bước để kiểm môi trường: loss giảm, lưu/nạp lại checkpoint cho cùng loss, đo token/giây và bộ nhớ, sinh thử
vài mẫu của fold kiểm bằng mô hình fine-tune và zero-shot. che_do="b6" (B6): dò loại dần trên 1 fold, xem chay_b6.
che_do="b78" (B7, B8): cấu hình đã chốt chạy trên pair fold 0 và cụm fold 1, 2, chấm thêm LoRA attention, xem chay_b78.
KHÓA TEST: chỉ đọc pool_*.jsonl.gz, dừng nếu thấy file test trong /kaggle/input.
Các hàm thuần (xay_mau, chia_lo, lich_lr, chon_vong2, ...) không cần torch để test được trên máy (test_kaggle_train.py).
"""
import gzip
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

VAO = Path("/kaggle/input")
RA = Path("/kaggle/working")
NHAC_EOS = "<|im_end|>"


# ---------------------------------------------------------------- hàm thuần
def xay_mau(mau, tok, max_len):
    """Mã hóa mỗi mẫu: prompt theo khuôn chat (tắt suy luận), đích kết thúc bằng <|im_end|>. Loss chỉ trên phần dịch (labels -100 cho prompt).
    Trả (danh sách {id, ids, labels, fold5}, số mẫu vượt max_len bị bỏ)."""
    ra, bo = [], 0
    for m in mau:
        p = tok.apply_chat_template(m["messages"][:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        a = m["messages"][1]["content"] + NHAC_EOS + "\n"
        ip, ia = tok(p, add_special_tokens=False)["input_ids"], tok(a, add_special_tokens=False)["input_ids"]
        if len(ip) + len(ia) > max_len:
            bo += 1
            continue
        ra.append({"id": m["id"], "ids": ip + ia, "labels": [-100] * len(ip) + ia, "fold5": m["fold5"],
                   "huong": m.get("huong"), "loai": m.get("loai"), "category": m.get("category")})
    return ra, bo


def cat_ngan_sach(k, micro_bs, ngan_sach):
    """k đã xếp theo độ dài: cắt thành lô <= micro_bs mẫu và (số mẫu x mẫu dài nhất) <= ngan_sach token, tránh tràn GPU ở lô dài."""
    lo, hien = [], []
    for m in k:
        if hien and (len(hien) >= micro_bs or (len(hien) + 1) * len(m["ids"]) > ngan_sach):
            lo.append(hien)
            hien = []
        hien.append(m)
    return lo + [hien] if hien else lo


def chia_lo(mau, micro_bs, seed, ngan_sach=8192, he_so_xo=50):
    """Lô gần cùng độ dài để ít đệm: xáo, cắt khúc he_so_xo*micro_bs, xếp theo độ dài trong khúc, chia lô theo ngân sách, rồi xáo thứ tự lô."""
    rng = random.Random(seed)
    x = list(mau)
    rng.shuffle(x)
    lo = []
    khuc = he_so_xo * micro_bs
    for i in range(0, len(x), khuc):
        lo += cat_ngan_sach(sorted(x[i:i + khuc], key=lambda m: len(m["ids"])), micro_bs, ngan_sach)
    rng.shuffle(lo)
    return lo


def gom_buoc(lo, co_batch):
    """Gom micro-batch thành bước optimizer: mỗi bước gom tới khi đủ co_batch mẫu. Đếm trước để lịch lr có đúng tổng số bước (số mẫu mỗi bước có thể hơi vượt)."""
    buoc, hien, dem = [], [], 0
    for k in lo:
        hien.append(k)
        dem += len(k)
        if dem >= co_batch:
            buoc.append(hien)
            hien, dem = [], 0
    return buoc + [hien] if hien else buoc


def lich_lr(buoc, tong, ti_le_warmup=0.03):
    """Hệ số lr: tăng tuyến tính trong warmup rồi giảm theo cosine về 0."""
    w = max(1, int(tong * ti_le_warmup))
    if buoc < w:
        return (buoc + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (buoc - w) / max(1, tong - w)))


# B6: 6 cấu hình của lưới + 1 cấu hình LoRA chỉ attention (cùng lr, rank với lr1e-4_r16, để biết có dùng được kiểu nhanh hơn không)
CAU_HINH_B6 = [{"ten": f"lr{lr}_r{r}", "lr": float(lr), "rank": r} for r in (16, 64) for lr in ("5e-5", "1e-4", "2e-4")]
CAU_HINH_B6.append({"ten": "lr1e-4_r16_attn", "lr": 1e-4, "rank": 16, "muc_tieu": ["q_proj", "k_proj", "v_proj", "o_proj"]})
SO_EPOCH_B6 = 2       # tối đa 2 epoch (05/11/2026), các bước sau cũng dùng 2 epoch


class DungHetGio(Exception):
    """Dừng có trật tự: đã lưu trạng thái, phiên sau chạy tiếp được."""


def diem_macro(loss_nhom):
    """loss_nhom {'huong|loai': loss}. Mỗi hướng = trung bình đều title và mo_ta; điểm = trung bình đều các hướng (không theo token)."""
    theo_huong = {}
    for k, v in loss_nhom.items():
        theo_huong.setdefault(k.split("|")[0], []).append(v)
    return sum(sum(x) / len(x) for x in theo_huong.values()) / len(theo_huong)


def chon_vong2(diem, moi=0.01, toi_thieu=2, toi_da=3):
    """diem {tên: loss, thấp là tốt}. Lấy toi_thieu cấu hình đầu, thêm cấu hình lệch không quá moi so với đầu bảng, trần toi_da.
    Bỏ cấu hình loss không hữu hạn."""
    x = sorted((v, k) for k, v in diem.items() if math.isfinite(v))
    if not x:
        raise RuntimeError("khong_con_cau_hinh_hop_le")
    chon = [k for _, k in x[:toi_thieu]]
    chon += [k for v, k in x[toi_thieu:toi_da] if v <= x[0][0] * (1 + moi)]
    return chon


def lay_phan_tang(mau, moi_nhom, seed):
    """Mẫu cố định theo (hướng, loại), dùng chung cho mọi cấu hình để so sánh theo cặp."""
    rng = random.Random(seed)
    nhom = {}
    for m in sorted(mau, key=lambda m: m["id"]):
        nhom.setdefault(f"{m['huong']}|{m['loai']}", []).append(m)
    ra = []
    for k in sorted(nhom):
        ra += rng.sample(nhom[k], min(moi_nhom, len(nhom[k])))
    return ra


def bootstrap_chenh(a, b, nhom, so_lan=2000, seed=42):
    """a, b: điểm từng mục của hai cấu hình trên cùng mục; nhom: 'huong|loai' từng mục. Khoảng 95% của chênh (a - b) theo điểm macro, lấy mẫu lại trong từng nhóm."""
    rng = random.Random(seed)
    theo = {}
    for x, y, n in zip(a, b, nhom):
        theo.setdefault(n, []).append(x - y)
    kq = []
    for _ in range(so_lan):
        d = {n: sum(rng.choices(v, k=len(v))) / len(v) for n, v in theo.items()}
        kq.append(diem_macro(d))
    kq.sort()
    return {"chenh": diem_macro({n: sum(v) / len(v) for n, v in theo.items()}), "thap": kq[int(0.025 * so_lan)], "cao": kq[int(0.975 * so_lan) - 1]}


# B7, B8 (Nhật Anh duyệt 05/11/2026): cấu hình chốt từ B6; (tên việc, nhánh, fold kiểm). cum_f0 không huấn luyện, dùng lại adapter B6
CAU_HINH_CHOT = {"ten": "lr2e-4_r64", "lr": 2e-4, "rank": 64}
VIEC_B78 = [("cum_f0", "cum", 0), ("pair_f0", "pair", 0), ("cum_f1", "cum", 1), ("cum_f2", "cum", 2)]
CAP_ATTN = ("lr1e-4_r16_attn", "lr1e-4_r16")


def chon_mau_sinh(mau_cum, mau_nhanh, fold, moi_nhom, seed, rut_gon=None):
    """Mẫu sinh cố định của một fold: chọn trên nhánh cụm đúng như B6 (seed + 1), rồi lấy cùng id ở nhánh cần chấm, để hai nhánh chấm trên cùng sản phẩm.
    rut_gon: giữ vài mẫu đầu mỗi nhóm khi chạy khói, vẫn là tập con của mẫu thật nên dùng lại được bản dịch đã có. Trả (mẫu, số id thiếu ở nhánh đó)."""
    chon = lay_phan_tang([m for m in mau_cum if m["fold5"] == fold and m["phan"] == 0], moi_nhom, seed + 1)
    if rut_gon:
        dem, gon = {}, []
        for m in chon:
            k = f"{m['huong']}|{m['loai']}"
            dem[k] = dem.get(k, 0) + 1
            if dem[k] <= rut_gon:
                gon.append(m)
        chon = gon
    theo = {m["id"]: m for m in mau_nhanh}
    return [theo[m["id"]] for m in chon if m["id"] in theo], sum(m["id"] not in theo for m in chon)


def gop_loss(tung_cat):
    """tung_cat {'huong|loai|category': (tổng loss, số token)}. Trả (loss {'huong|loai'} cộng theo token, điểm macro theo hướng)."""
    gop = {}
    for k, (t, n) in tung_cat.items():
        a = gop.setdefault(k.rsplit("|", 1)[0], [0.0, 0])
        a[0] += t
        a[1] += n
    nhom = {k: t / max(n, 1) for k, (t, n) in sorted(gop.items())}
    return nhom, diem_macro(nhom)


# Tỉ lệ title : mô tả (mục tiêu d, plan mục 0). Trọng số đặt theo mẫu: mô tả = 1, title = w. Một micro-batch: token nhân trọng số mẫu, chia tổng trọng số token.
# Một bước: micro-batch nặng theo (tổng trọng số mẫu / số mẫu của bước) nhân hệ số chung he_chuan, KHÔNG chia cho tổng trọng số của bước: nhiều bước chỉ toàn title
# hoặc toàn mô tả, chia theo bước sẽ triệt tiêu trọng số ở đó (đo thật: tỉ trọng title kẹt trong 14%–71%). w = 1 trùng đúng cách tính không trọng số.
def tom_tat_buoc(ds_buoc):
    """Rút gọn để tính nhanh: mỗi bước là danh sách (số mẫu title, số mẫu mô tả, token đích title, token đích mô tả) từng micro-batch."""
    ra = []
    for buoc in ds_buoc:
        hang = []
        for k in buoc:
            n_t = n_m = t_t = t_m = 0
            for m in k:
                so = sum(1 for x in m["labels"] if x != -100)
                if m["loai"] == "title":
                    n_t, t_t = n_t + 1, t_t + so
                else:
                    n_m, t_m = n_m + 1, t_m + so
            hang.append((n_t, n_m, t_t, t_m))
        ra.append(hang)
    return ra


def ti_trong_title(tom_tat, w):
    """Phần trọng số title trong loss, cộng trên mọi bước; là tỉ lệ danh nghĩa khi đặt w, chưa tính tác động của clip gradient."""
    title = tong = 0.0
    for hang in tom_tat:
        so_mau = sum(a + b for a, b, _, _ in hang)
        for a, b, c, d in hang:
            nang = (w * a + b) / so_mau
            mau = w * c + d
            tong += nang
            if mau:
                title += nang * (w * c / mau)
    return title / tong


def he_chuan(mau, w):
    """Giữ độ lớn loss trung bình như khi không trọng số, để lr vẫn so được với mốc."""
    return len(mau) / sum(trong_so_lo(mau, w))


def tim_w_title(tom_tat, p, so_lan=60):
    """Tìm w để tỉ trọng title đạt p, chia đôi trên thang log vì ti_trong_title tăng theo w."""
    if not 0 < p < 1:
        raise ValueError(f"p_ngoai_khoang:{p}")
    thap, cao = math.log(1e-4), math.log(1e4)
    for _ in range(so_lan):
        giua = (thap + cao) / 2
        if ti_trong_title(tom_tat, math.exp(giua)) < p:
            thap = giua
        else:
            cao = giua
    return math.exp((thap + cao) / 2)


def trong_so_lo(k, w):
    """Trọng số từng mẫu của micro-batch: title = w, mô tả = 1."""
    return [w if m["loai"] == "title" else 1.0 for m in k]


def kiem_bat_thuong(cham, chrf_toi_thieu, lap_toi_da):
    """Dấu hiệu lần chạy hỏng, để dừng sớm thay vì đốt tiếp quota khi không ai theo dõi. Trả lý do hoặc None. Ngưỡng là để bắt hỏng, không phải để chọn tỉ lệ."""
    diem = cham["chrf"]["chrf_macro_huong"]
    if diem is None or diem < chrf_toi_thieu:
        return f"chrF {diem} duoi nguong {chrf_toi_thieu}"
    ti_le = cham["sinh_lap"] / max(cham["so_mau"], 1)
    if ti_le > lap_toi_da:
        return f"sinh lap {ti_le:.3f} vuot nguong {lap_toi_da}"
    return None


def chon_trang_thai(ds):
    """ds: [(đường dẫn, trạng thái)]. Khi gắn cả bản lưu và đầu ra chính notebook, lấy bản tiến xa hơn (nhiều bước và nhiều phần chấm hơn), tránh nối nhầm bản cũ."""
    def tien_do(x):
        viec = x[1].get("viec", {}).values()
        return (sum(1 for v in viec if v.get("cham")), sum(v.get("buoc", 0) for v in viec))
    return max(ds, key=tien_do)


# ---------------------------------------------------------------- môi trường
def tim_mot(mau_ten):
    ds = sorted(VAO.rglob(mau_ten))
    if not ds:
        raise RuntimeError(f"khong_thay:{mau_ten}")
    return ds[0]


def kiem_khoa_test():
    la = [str(p) for p in VAO.rglob("*") if p.is_file() and ("test_" in p.name or p.name in {"split.parquet", "ds_test.txt"})
          and "site-packages" not in str(p) and "arc-prize" not in str(p)]
    if la:
        raise RuntimeError(f"CO_FILE_TEST_TRONG_INPUT:{la[:5]}")


def cai_thu_vien():
    wh = tim_mot("peft-*.whl").parent
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wh), "peft", "sacrebleu"],
                       capture_output=True, text=True)
    print((r.stdout or "")[-1500:], (r.stderr or "")[-1500:], flush=True)
    if r.returncode:
        raise RuntimeError("cai_thu_vien_loi")


def doc_mau(nhanh):
    """Kaggle có thể tự giải nén .gz khi nhận Dataset, nên đọc được cả hai dạng (nhận theo chữ ký file, không theo đuôi)."""
    ds = [p for p in sorted(VAO.rglob(f"pool_{nhanh}.jsonl*")) if p.is_file()]
    if not ds:
        raise RuntimeError(f"khong_thay:pool_{nhanh}.jsonl")
    with open(ds[0], "rb") as f:
        nen = f.read(2) == b"\x1f\x8b"
    print("doc mau:", ds[0], "nen" if nen else "khong nen", flush=True)
    mo = gzip.open(ds[0], "rt", encoding="utf-8") if nen else open(ds[0], encoding="utf-8")
    with mo as f:
        return [json.loads(x) for x in f]


# ---------------------------------------------------------------- phần cần torch
def lo_tensor(lo, tok, thiet_bi):
    import torch
    dai = max(len(m["ids"]) for m in lo)
    pad = tok.pad_token_id
    ids = torch.tensor([m["ids"] + [pad] * (dai - len(m["ids"])) for m in lo], device=thiet_bi)
    lab = torch.tensor([m["labels"] + [-100] * (dai - len(m["labels"])) for m in lo], device=thiet_bi)
    mask = torch.tensor([[1] * len(m["ids"]) + [0] * (dai - len(m["ids"])) for m in lo], device=thiet_bi)
    return ids, lab, mask


def loss_kiem(model, tok, mau, micro_bs, thiet_bi, ngan_sach=8192):
    """Loss trung bình theo token đích trên tập kiểm cố định."""
    tong, dem = loss_tong(model, tok, mau, micro_bs, thiet_bi, ngan_sach)
    return tong / max(dem, 1)


def loss_tong(model, tok, mau, micro_bs, thiet_bi, ngan_sach=8192):
    """Tổng loss và số token đích, để cộng dồn đúng trọng số giữa các nhóm."""
    import torch
    import torch.nn.functional as F
    model.eval()
    tong, dem = 0.0, 0
    with torch.no_grad():
        for k in cat_ngan_sach(sorted(mau, key=lambda m: len(m["ids"])), micro_bs, ngan_sach):
            ids, lab, mask = lo_tensor(k, tok, thiet_bi)
            logit = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            nhan = lab[:, 1:]
            tong += F.cross_entropy(logit.reshape(-1, logit.size(-1)).float(), nhan.reshape(-1), ignore_index=-100, reduction="sum").item()
            dem += (nhan != -100).sum().item()
    model.train()
    return tong, dem


def loss_co_trong_so(model, ids, lab, mask, ts, thiet_bi):
    """Loss một micro-batch: token nhân trọng số mẫu, chia tổng trọng số token. ts toàn 1 thì bằng out.loss của mô hình (có phép thử trước khi chạy)."""
    import torch
    import torch.nn.functional as F
    logit = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    nhan = lab[:, 1:]
    ce = F.cross_entropy(logit.reshape(-1, logit.size(-1)), nhan.reshape(-1), ignore_index=-100, reduction="none").view_as(nhan)
    tw = torch.tensor(ts, dtype=torch.float32, device=thiet_bi)[:, None] * (nhan != -100)
    return (ce * tw).sum() / tw.sum()


def nap_goc(duong_dan):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(duong_dan), dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")


def sinh(model, tok, mau, thiet_bi, bs=32, ngan_sach=32768):
    """Giải mã greedy cố định; trần max_new_tokens theo độ dài đích tham chiếu. Xếp theo độ dài và cắt lô theo ngân sách cho nhanh; trả đúng thứ tự mau."""
    import torch
    tok.padding_side = "left"
    model.eval()
    mo = []
    for m in mau:
        p = tok.apply_chat_template(m["messages"][:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tran = min(1500, len(tok(m["dich"], add_special_tokens=False)["input_ids"]) * 2 + 64)
        mo.append({"m": m, "p": p, "tran": tran, "ids": [0] * (len(tok(p, add_special_tokens=False)["input_ids"]) + tran)})
    kq = {}
    for k in cat_ngan_sach(sorted(mo, key=lambda x: len(x["ids"])), bs, ngan_sach):
        enc = tok([x["p"] for x in k], return_tensors="pt", padding=True, add_special_tokens=False).to(thiet_bi)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max(x["tran"] for x in k), do_sample=False, pad_token_id=tok.pad_token_id)
        for x, o in zip(k, out):
            kq[x["m"]["id"]] = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    return [{"id": m["id"], "hyp_tho": kq[m["id"]], "hyp": kq[m["id"]]} for m in mau]


def loss_nhom(model, tok, mau, thiet_bi, ngan_sach=8192):
    """Loss theo token đích của từng nhóm 'huong|loai' trên mẫu kiểm cố định; trả dict và điểm macro."""
    theo = {}
    for m in mau:
        theo.setdefault(f"{m['huong']}|{m['loai']}", []).append(m)
    ra = {k: loss_kiem(model, tok, v, 32, thiet_bi, ngan_sach) for k, v in sorted(theo.items())}
    return ra, diem_macro(ra)


def chay_b6(cfg, bc):
    """B6 dò loại dần trên 1 fold (Nhật Anh chốt 05/11/2026). Vòng 1: mọi cấu hình 1 epoch, so loss. Vòng 2: cấu hình được chọn chạy nốt tới epoch cuối (SO_EPOCH_B6).
    Vòng 3: cấu hình được chọn, ở epoch loss tốt nhất, sinh và chấm chrF trên mẫu cố định. Lịch lr cosine tính cho toàn bộ SO_EPOCH_B6 epoch nên vòng 2 nối tiếp đúng lịch.
    Trạng thái ghi vào b6/b6_trang_thai.json sau mỗi mốc; hết giờ thì lưu rồi dừng có trật tự, phiên sau chạy tiếp."""
    import shutil
    import torch
    import transformers
    import peft
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    thiet_bi = "cuda:0"
    t_bat_dau = cfg["_t_bat_dau"]
    het_giay = cfg.get("gio_toi_da", 8.5) * 3600
    thu_ra = RA / "b6"
    tep = thu_ra / "b6_trang_thai.json"
    if not tep.exists():                               # phiên sau: gắn trạng thái phiên trước làm input
        cu = sorted(VAO.rglob("b6_trang_thai.json"))
        if cu:
            shutil.copytree(cu[0].parent, thu_ra)
            print("chay tiep tu:", cu[0], flush=True)
    thu_ra.mkdir(exist_ok=True)
    st = json.loads(tep.read_text(encoding="utf-8")) if tep.exists() else {"cau_hinh": {}, "chon": None, "cham": {}}

    def ghi():
        tam = tep.with_suffix(".tmp")
        tam.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tam.replace(tep)
        bc["b6"] = st

    bc["moi_truong"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)}
    thu_mh = tim_mot("REVISION.txt").parent
    bc["mo_hinh"] = thu_mh.joinpath("REVISION.txt").read_text().split()
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    goc_mau = doc_mau(cfg["nhanh"])
    ma, bo = xay_mau(goc_mau, tok, cfg["max_len"])
    fk = cfg["fold_kiem"]
    train = [m for m in ma if m["fold5"] != fk]
    kiem = lay_phan_tang([m for m in ma if m["fold5"] == fk], cfg["so_kiem_moi_nhom"], cfg["seed"])
    if cfg.get("so_train"):
        train = random.Random(cfg["seed"]).sample(train, cfg["so_train"])
    if {m["id"] for m in train} & {m["id"] for m in kiem}:
        raise RuntimeError("giao_train_kiem")
    st.update(so_train=len(train), so_kiem=len(kiem), so_mau_bo_max_len=bo, so_token_train=sum(len(m["ids"]) for m in train), fold_kiem=fk)
    co_batch = cfg["micro_bs"] * cfg["tich_luy"]
    # Chống nối nhầm trạng thái: phiên sau phải cùng dữ liệu, cùng cách chia lô, cùng số epoch với phiên đã tạo trạng thái
    van_tay = {"so_train": len(train), "so_kiem": len(kiem), "so_token_train": st["so_token_train"], "fold_kiem": fk, "seed": cfg["seed"], "so_epoch": SO_EPOCH_B6,
               "micro_bs": cfg["micro_bs"], "ngan_sach": cfg["ngan_sach_token"], "nhanh": cfg["nhanh"]}
    if st.get("van_tay") and st["van_tay"] != van_tay:
        raise RuntimeError(f"TRANG_THAI_KHONG_KHOP: cu {st['van_tay']} moi {van_tay}")
    st["van_tay"] = van_tay
    # Bước của từng epoch được dựng trước, cùng seed cho mọi cấu hình; lịch lr dùng đúng tổng số bước của mọi epoch
    buoc_theo_epoch = [gom_buoc(chia_lo(train, cfg["micro_bs"], cfg["seed"] + e, cfg["ngan_sach_token"]), co_batch) for e in range(SO_EPOCH_B6)]
    tong_buoc = sum(len(x) for x in buoc_theo_epoch)
    st["tong_buoc"] = tong_buoc
    ds_cau_hinh = {c["ten"]: c for c in CAU_HINH_B6 if not cfg.get("chi_cau_hinh") or c["ten"] in cfg["chi_cau_hinh"]}
    for ten in ds_cau_hinh:
        st["cau_hinh"].setdefault(ten, {"epoch_xong": 0, "buoc": 0, "vi_tri_lo": 0, "loss": {}, "diem": {}, "token_moi_giay": {}})
    ghi()
    goc = nap_goc(thu_mh)
    goc.config.use_cache = False
    dem_dung = {"con": cfg.get("dung_sau_buoc")}       # chỉ dùng khi chạy khói, để thử chạy tiếp

    def het_gio():
        return time.time() - t_bat_dau > het_giay

    def huan_luyen_toi(ten, den_epoch):
        nonlocal goc
        c, s = ds_cau_hinh[ten], st["cau_hinh"][ten]
        if s["epoch_xong"] >= den_epoch or s.get("hong"):
            return
        d, moi_nhat = thu_ra / ten, thu_ra / ten / "moi_nhat"
        d.mkdir(exist_ok=True)
        torch.manual_seed(cfg["seed"])
        if (moi_nhat / "opt.pt").exists():
            m = PeftModel.from_pretrained(goc, str(moi_nhat), is_trainable=True)
        else:
            m = get_peft_model(goc, LoraConfig(r=c["rank"], lora_alpha=2 * c["rank"], lora_dropout=0.05, target_modules=c.get("muc_tieu", "all-linear"), task_type="CAUSAL_LM"))
        tham_so = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(tham_so, lr=c["lr"], weight_decay=0.0)
        if (moi_nhat / "opt.pt").exists():
            opt.load_state_dict(torch.load(moi_nhat / "opt.pt", map_location=thiet_bi))
        s["so_tham_so"] = sum(p.numel() for p in tham_so)

        def luu():
            if moi_nhat.exists():
                shutil.rmtree(moi_nhat)
            m.save_pretrained(str(moi_nhat))
            torch.save(opt.state_dict(), moi_nhat / "opt.pt")
            ghi()

        m.train()
        hong = False
        for e in range(s["epoch_xong"], den_epoch):
            ds_buoc = buoc_theo_epoch[e]
            i, tok_do, t0 = s["vi_tri_lo"], 0, time.time()
            while i < len(ds_buoc) and not hong:
                cac_lo = ds_buoc[i]
                i += 1
                so_mau = sum(len(k) for k in cac_lo)
                tong = 0.0
                for k in cac_lo:
                    ids, lab, mask = lo_tensor(k, tok, thiet_bi)
                    out = m(input_ids=ids, attention_mask=mask, labels=lab)
                    (out.loss * len(k) / so_mau).backward()         # trọng số theo số mẫu thật của bước
                    tong += out.loss.item() * len(k) / so_mau
                    tok_do += int(mask.sum().item())
                for g in opt.param_groups:
                    g["lr"] = c["lr"] * lich_lr(s["buoc"], tong_buoc)
                torch.nn.utils.clip_grad_norm_(tham_so, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                s["buoc"] += 1
                s["vi_tri_lo"] = i
                if not math.isfinite(tong):
                    hong = True
                if s["buoc"] % 100 == 0:
                    print(ten, "epoch", e + 1, "buoc", s["buoc"], "/", tong_buoc, "loss", round(tong, 4), "lr", opt.param_groups[0]["lr"], flush=True)
                if dem_dung["con"] is not None:
                    dem_dung["con"] -= 1
                dung = dem_dung["con"] == 0
                if not hong and (het_gio() or dung):
                    if dung:
                        dem_dung["con"] = None
                    luu()
                    print("DUNG, da luu", ten, "epoch", e + 1, "vi tri lo", i, flush=True)
                    goc = m.unload()           # gỡ LoRA khỏi mô hình gốc, tránh chồng hai lớp khi chạy tiếp trong cùng phiên
                    del m, opt
                    torch.cuda.empty_cache()
                    raise DungHetGio(ten)
            if hong:
                s.update(hong=True)
                s["diem"][str(e + 1)] = float("inf")
                print("LOSS KHONG HUU HAN, bo cau hinh", ten, flush=True)
                break
            s["token_moi_giay"][str(e + 1)] = round(tok_do / max(time.time() - t0, 1e-6), 1)
            s["epoch_xong"], s["vi_tri_lo"] = e + 1, 0
            ln, dm = loss_nhom(m, tok, kiem, thiet_bi)
            m.train()
            s["loss"][str(e + 1)], s["diem"][str(e + 1)] = ln, dm
            m.save_pretrained(str(d / f"ep{e + 1}"))
            luu()
            print(ten, "XONG epoch", e + 1, "diem loss", round(dm, 5), ln, "token/giay", s["token_moi_giay"][str(e + 1)], "gio da chay", round((time.time() - t_bat_dau) / 3600, 2), flush=True)
        goc = m.unload()
        del m, opt
        torch.cuda.empty_cache()
        ghi()

    # Vòng 1
    for ten in ds_cau_hinh:
        huan_luyen_toi(ten, 1)
    if st["chon"] is None:
        st["chon"] = chon_vong2({t: s["diem"].get("1", float("inf")) for t, s in st["cau_hinh"].items()})
        ghi()
    print("CHON VONG 2:", st["chon"], flush=True)
    # Vòng 2
    for ten in st["chon"]:
        huan_luyen_toi(ten, SO_EPOCH_B6)
    # Vòng 3: sinh và chấm trên mẫu cố định, ở epoch loss tốt nhất của từng cấu hình
    import pandas as pd
    import cham_diem as D
    mau_sinh = lay_phan_tang([m for m in goc_mau if m["fold5"] == fk and m["phan"] == 0], cfg["so_sinh_moi_nhom"], cfg["seed"] + 1)
    tc = pd.DataFrame(mau_sinh)
    diem_muc = {}
    for ten in st["chon"]:
        s = st["cau_hinh"][ten]
        if s.get("hong"):
            continue
        ep = min(s["diem"], key=lambda e: s["diem"][e])
        tep_sinh = thu_ra / ten / f"sinh_ep{ep}.jsonl"
        if ten not in st["cham"]:
            if het_gio():
                raise DungHetGio("sinh")
            m = PeftModel.from_pretrained(goc, str(thu_ra / ten / f"ep{ep}"))
            t = time.time()
            du = sinh(m, tok, mau_sinh, thiet_bi)
            giay = time.time() - t
            tep_sinh.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in du) + "\n", encoding="utf-8")
            goc = m.unload()
            del m
            torch.cuda.empty_cache()
            tm, th = D.cham(tc, pd.DataFrame(du), nhanh=cfg["nhanh"])
            st["cham"][ten] = {"epoch": ep, "giay_moi_mau": round(giay / len(du), 3), "chrf": th["diem_chinh"], "theo_nhom": th.get("theo_nhom"),
                               "ti_le_cat_loi_dan": float(tm.da_cat_loi_dan.mean()), "sinh_lap": int(tm.sinh_lap.sum()) if "sinh_lap" in tm else None}
            ghi()
            print(ten, "epoch", ep, "chrF", th["diem_chinh"].get("chrf_macro_huong"), "giay/mau", st["cham"][ten]["giay_moi_mau"], flush=True)
        du = [json.loads(x) for x in tep_sinh.read_text(encoding="utf-8").splitlines()]
        tm, _ = D.cham(tc, pd.DataFrame(du), nhanh=cfg["nhanh"])
        diem_muc[ten] = tm.set_index("id").chrf
    # So theo cặp: cấu hình đầu bảng theo chrF với từng cấu hình còn lại, khoảng tin cậy bootstrap (chứa 0 nghĩa là hòa)
    xep = sorted(st["cham"], key=lambda t: -st["cham"][t]["chrf"]["chrf_macro_huong"])
    nhom = [f"{m['huong']}|{m['loai']}" for m in mau_sinh]
    ids = [m["id"] for m in mau_sinh]
    st["so_cap_chrf"] = {f"{xep[0]} - {t}": bootstrap_chenh(diem_muc[xep[0]][ids].tolist(), diem_muc[t][ids].tolist(), nhom) for t in xep[1:]}
    st["xep_hang_chrf"] = xep
    st["xong"] = True
    ghi()
    bc["dat"] = True
    print("XEP HANG CHRF:", xep, st["so_cap_chrf"], flush=True)


def chay_b78(cfg, bc):
    """B7, B8 (D18): cấu hình đã chốt chạy trên pair fold 0 và cụm fold 1, 2; cụm fold 0 dùng lại adapter và bản dịch B6. Mỗi việc: huấn luyện, loss cả fold theo
    hướng, loại, category, rồi sinh và chấm chrF trên 1.000 mẫu cố định. Trước đó chấm chrF LoRA attention so với LoRA mọi lớp cùng lr, rank ở epoch 1.
    Không tự chạy thêm khi chốt chặn bật: tong_hop_b78.py tính chốt chặn trên máy, người quyết. Không dùng chung vòng huấn luyện với chay_b6 để giữ nguyên mã đã tạo ra kết quả B6.
    Mỗi pha tự bỏ qua nếu đã có trong trạng thái; hết giờ thì lưu rồi dừng có trật tự, phiên sau chạy tiếp."""
    import shutil
    import pandas as pd
    import torch
    import transformers
    import peft
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    import cham_diem as D

    thiet_bi = "cuda:0"
    t_bat_dau = cfg["_t_bat_dau"]
    het_giay = cfg["gio_toi_da"] * 3600
    ten_tt = cfg.get("ten_trang_thai", "b78")      # khảo sát tỉ lệ dùng trạng thái riêng "tl", không lẫn với B7, B8
    thu_ra = RA / ten_tt
    tep = thu_ra / f"{ten_tt}_trang_thai.json"
    if not tep.exists():
        cu = sorted(VAO.rglob(tep.name))
        if cu:
            duong_cu, _ = chon_trang_thai([(p, json.loads(p.read_text(encoding="utf-8"))) for p in cu])
            shutil.copytree(duong_cu.parent, thu_ra)
            print("chay tiep tu:", duong_cu, flush=True)
    thu_ra.mkdir(exist_ok=True)
    st = json.loads(tep.read_text(encoding="utf-8")) if tep.exists() else {"viec": {}, "attn": {}, "xong": None}
    ds_viec = cfg.get("viec") or [{"ten": t, "nhanh": n, "fold": f, "ca_fold": True} for t, n, f in VIEC_B78]

    def ghi():
        tam = tep.with_suffix(".tmp")
        tam.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tam.replace(tep)
        bc["b78"] = st

    def kiem_gio(o_dau):
        if time.time() - t_bat_dau > het_giay:
            raise DungHetGio(o_dau)

    # Chống nối nhầm trạng thái: phiên sau phải cùng cách chạy với phiên đã tạo trạng thái
    van_tay = {k: cfg.get(k) for k in ("seed", "micro_bs", "ngan_sach_token", "max_len", "so_kiem_moi_nhom", "so_sinh_moi_nhom", "rut_gon_sinh", "so_train", "so_mau_ca_fold")}
    van_tay.update(so_epoch=SO_EPOCH_B6, cau_hinh=CAU_HINH_CHOT["ten"])
    if st.get("van_tay") and st["van_tay"] != van_tay:
        raise RuntimeError(f"TRANG_THAI_KHONG_KHOP: cu {st['van_tay']} moi {van_tay}")
    st["van_tay"] = van_tay

    kho, st_b6 = None, None
    if not cfg.get("viec"):
        ds_b6 = sorted(VAO.rglob("b6_trang_thai.json"))     # kho lưu B6, notebook riêng nên không bị thay khi đẩy lại nlp-group3
        if len(ds_b6) != 1:
            raise RuntimeError(f"can_dung_mot_kho_luu_b6:{ds_b6}")
        kho = ds_b6[0].parent
        st_b6 = json.loads(ds_b6[0].read_text(encoding="utf-8"))
        print("kho luu B6:", kho, flush=True)
    bc["moi_truong"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)}
    thu_mh = tim_mot("REVISION.txt").parent
    bc["mo_hinh"] = thu_mh.joinpath("REVISION.txt").read_text().split()
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    goc_mau = {n: doc_mau(n) for n in ("cum", "pair")}
    ma = {}
    for n, g in goc_mau.items():
        ma[n], bo = xay_mau(g, tok, cfg["max_len"])
        st[f"so_mau_bo_max_len_{n}"] = bo
    ghi()
    goc = nap_goc(thu_mh)
    goc.config.use_cache = False
    dem_dung = {"con": cfg.get("dung_sau_buoc")}       # chỉ dùng khi chạy khói, để thử chạy tiếp

    if any(j.get("p_title") for j in ds_viec):         # phép thử trước khi tốn giờ: trọng số 1:1 phải đúng bằng loss của mô hình, trên lô trộn title và mô tả
        n_thu = ds_viec[0]["nhanh"]
        tit = min((m for m in ma[n_thu] if m["loai"] == "title"), key=lambda m: len(m["ids"]))
        mt = min((m for m in ma[n_thu] if m["loai"] == "mo_ta"), key=lambda m: len(m["ids"]))
        ids, lab, mask = lo_tensor([tit, mt], tok, thiet_bi)
        goc.eval()
        with torch.no_grad():
            a = goc(input_ids=ids, attention_mask=mask, labels=lab).loss.item()
            b = loss_co_trong_so(goc, ids, lab, mask, [1.0, 1.0], thiet_bi).item()
        goc.train()
        st["thu_loss_1_1"] = {"out_loss": a, "co_trong_so": b}
        print("thu loss 1:1", a, b, flush=True)
        if not abs(a - b) <= 1e-3:         # viết thế này để NaN cũng bị bắt
            raise RuntimeError(f"LOSS_TRONG_SO_KHONG_KHOP:{a}:{b}")

    def dung_bat_thuong(ten, ly_do):
        st["bat_thuong"] = f"{ten}: {ly_do}"
        ghi()
        print("BAT THUONG, dung phien de khong dot quota:", st["bat_thuong"], flush=True)
        raise DungHetGio("bat_thuong")

    def lay_mau_sinh(nhanh, fold):
        mau, thieu = chon_mau_sinh(goc_mau["cum"], goc_mau[nhanh], fold, cfg["so_sinh_moi_nhom"], cfg["seed"], cfg.get("rut_gon_sinh"))
        return mau, thieu

    def tap_da_thay(nhanh, fold):
        return D.xay_tap_da_thay(pd.DataFrame(goc_mau[nhanh])[["loai", "category", "nguon", "nhanh", "fold5"]], fold, nhanh)

    def doc_sinh(duong_dan, mau):
        ids = {m["id"] for m in mau}
        du = [r for r in map(json.loads, duong_dan.read_text(encoding="utf-8").splitlines()) if r["id"] in ids]
        if len(du) != len(mau):
            raise RuntimeError(f"ban_dich_thieu_muc:{duong_dan}:{len(du)}/{len(mau)}")
        return du

    def ghi_sinh(duong_dan, du):
        tam = duong_dan.with_suffix(".tmp")
        tam.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in du) + "\n", encoding="utf-8")
        tam.replace(duong_dan)

    def cham_mot(nhanh, mau, du, tap, epoch, giay_moi_mau):
        tm, th = D.cham(pd.DataFrame(mau), pd.DataFrame(du), nhanh=nhanh, tap_da_thay=tap)
        return tm, D._sach({"epoch": str(epoch), "giay_moi_mau": giay_moi_mau, "so_mau": len(mau), "sinh_lap": int(tm.sinh_lap.sum()), "ti_le_cat_loi_dan": float(tm.da_cat_loi_dan.mean()),
                            "chrf": th["diem_chinh"], "theo_huong": th["theo_huong"], "theo_nhom": th["theo_nhom"], "theo_category": th["theo_category"], "dong_da_thay_moi": th.get("dong_da_thay_moi")})

    def huan_luyen(ten, v, nhanh, fold, p_title=None, lr=None):
        nonlocal goc
        if v["epoch_xong"] >= SO_EPOCH_B6 or v.get("hong"):
            return
        train = [m for m in ma[nhanh] if m["fold5"] != fold]
        if cfg.get("so_train"):
            train = random.Random(cfg["seed"]).sample(train, cfg["so_train"])
        kiem = lay_phan_tang([m for m in ma[nhanh] if m["fold5"] == fold], cfg["so_kiem_moi_nhom"], cfg["seed"])
        if {m["id"] for m in train} & {m["id"] for m in kiem}:
            raise RuntimeError("giao_train_kiem")
        co_batch = cfg["micro_bs"] * cfg["tich_luy"]
        buoc_theo_epoch = [gom_buoc(chia_lo(train, cfg["micro_bs"], cfg["seed"] + e, cfg["ngan_sach_token"]), co_batch) for e in range(SO_EPOCH_B6)]
        tong_buoc = sum(len(x) for x in buoc_theo_epoch)
        if v.get("tong_buoc") not in (None, tong_buoc):
            raise RuntimeError(f"TONG_BUOC_KHONG_KHOP:{ten}:{v['tong_buoc']}:{tong_buoc}")
        v.update(so_train=len(train), so_kiem=len(kiem), so_token_train=sum(len(m["ids"]) for m in train), tong_buoc=tong_buoc)
        w = None
        if p_title:
            tat = tom_tat_buoc([b for e in buoc_theo_epoch for b in e])
            w = tim_w_title(tat, p_title)
            if v.get("w_title") is not None and abs(v["w_title"] - w) > 1e-6:
                raise RuntimeError(f"W_TITLE_KHONG_KHOP:{ten}:{v['w_title']}:{w}")
            chuan = he_chuan(train, w)
            v.update(p_muc_tieu=p_title, w_title=w, he_chuan=chuan, p_thuc=ti_trong_title(tat, w), p_khi_w_1=ti_trong_title(tat, 1.0))
            print(ten, "p muc tieu", p_title, "p thuc", round(v["p_thuc"], 5), "w_title", round(w, 5), "he_chuan", round(chuan, 4), "p khi w=1", round(v["p_khi_w_1"], 5), flush=True)
            if abs(v["p_thuc"] - p_title) > 1e-3:
                raise RuntimeError(f"KHONG_DAT_TI_LE:{ten}:{v['p_thuc']}:{p_title}")
        ghi()
        c = {**CAU_HINH_CHOT, "lr": lr or CAU_HINH_CHOT["lr"]}
        v["lr"] = c["lr"]
        d, moi_nhat = thu_ra / ten, thu_ra / ten / "moi_nhat"
        d.mkdir(exist_ok=True)
        torch.manual_seed(cfg["seed"])
        if v["buoc"] > 0 and not (moi_nhat / "opt.pt").exists():     # đã chạy dở mà mất checkpoint: dừng, không âm thầm tạo LoRA mới rồi chạy tiếp từ giữa
            raise RuntimeError(f"MAT_CHECKPOINT:{ten}:buoc={v['buoc']}")
        if (moi_nhat / "opt.pt").exists():
            m = PeftModel.from_pretrained(goc, str(moi_nhat), is_trainable=True)
        else:
            m = get_peft_model(goc, LoraConfig(r=c["rank"], lora_alpha=2 * c["rank"], lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
        tham_so = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(tham_so, lr=c["lr"], weight_decay=0.0)
        if (moi_nhat / "opt.pt").exists():
            opt.load_state_dict(torch.load(moi_nhat / "opt.pt", map_location=thiet_bi))
        v["so_tham_so"] = sum(p.numel() for p in tham_so)

        def luu():
            if moi_nhat.exists():
                shutil.rmtree(moi_nhat)
            m.save_pretrained(str(moi_nhat))
            torch.save(opt.state_dict(), moi_nhat / "opt.pt")
            ghi()

        m.train()
        for e in range(v["epoch_xong"], SO_EPOCH_B6):
            ds_buoc = buoc_theo_epoch[e]
            i, tok_do, t0 = v["vi_tri_lo"], 0, time.time()
            while i < len(ds_buoc):
                cac_lo = ds_buoc[i]
                i += 1
                so_mau = sum(len(k) for k in cac_lo)
                tong = 0.0
                for k in cac_lo:
                    ids, lab, mask = lo_tensor(k, tok, thiet_bi)
                    if w:
                        ts = trong_so_lo(k, w)
                        loss_k, he = loss_co_trong_so(m, ids, lab, mask, ts, thiet_bi), chuan * sum(ts) / so_mau
                    else:
                        loss_k, he = m(input_ids=ids, attention_mask=mask, labels=lab).loss, len(k) / so_mau
                    (loss_k * he).backward()
                    tong += loss_k.item() * he
                    tok_do += int(mask.sum().item())
                for g in opt.param_groups:
                    g["lr"] = c["lr"] * lich_lr(v["buoc"], tong_buoc)
                chuan_grad = float(torch.nn.utils.clip_grad_norm_(tham_so, 1.0))
                # Clip có thể làm giảm tác dụng của trọng số: ghi tỉ lệ bước bị cắt theo loại để đánh giá, tách bước toàn title, toàn mô tả, trộn
                nhom_buoc = {x["loai"] for k in cac_lo for x in k}
                loai_buoc = "tron" if len(nhom_buoc) > 1 else next(iter(nhom_buoc))
                tk = v.setdefault("thong_ke_grad", {}).setdefault(loai_buoc, {"so_buoc": 0, "tong_chuan": 0.0, "so_bi_cat": 0})
                tk["so_buoc"] += 1
                tk["tong_chuan"] += chuan_grad if math.isfinite(chuan_grad) else 0.0
                tk["so_bi_cat"] += chuan_grad > 1.0
                opt.step()
                opt.zero_grad(set_to_none=True)
                v["buoc"] += 1
                v["vi_tri_lo"] = i
                if not math.isfinite(tong):
                    v["hong"] = True
                    print("LOSS KHONG HUU HAN, bo viec", ten, "buoc", v["buoc"], flush=True)
                    goc = m.unload()
                    del m, opt
                    torch.cuda.empty_cache()
                    ghi()
                    return
                if v["buoc"] % 100 == 0:
                    print(ten, "epoch", e + 1, "buoc", v["buoc"], "/", tong_buoc, "loss", round(tong, 4), "lr", opt.param_groups[0]["lr"],
                          "grad", {a: (x["so_buoc"], round(x["tong_chuan"] / x["so_buoc"], 3), x["so_bi_cat"]) for a, x in v["thong_ke_grad"].items()}, flush=True)
                if dem_dung["con"] is not None:
                    dem_dung["con"] -= 1
                dung = dem_dung["con"] == 0
                if dung or time.time() - t_bat_dau > het_giay:
                    if dung:
                        dem_dung["con"] = None
                    luu()
                    print("DUNG, da luu", ten, "epoch", e + 1, "vi tri lo", i, flush=True)
                    raise DungHetGio(ten)
            v["token_moi_giay"][str(e + 1)] = round(tok_do / max(time.time() - t0, 1e-6), 1)
            ln, dm = loss_nhom(m, tok, kiem, thiet_bi)
            m.train()
            v["loss"][str(e + 1)], v["diem"][str(e + 1)] = ln, dm
            v["epoch_xong"], v["vi_tri_lo"] = e + 1, 0
            if e + 1 == SO_EPOCH_B6:
                m.save_pretrained(str(d / f"ep{e + 1}"))
                ghi()
                shutil.rmtree(moi_nhat, ignore_errors=True)    # xong việc thì bỏ, mỗi bản kèm opt.pt ~1,6 GB
            else:
                luu()
            print(ten, "XONG epoch", e + 1, "diem loss", round(dm, 5), ln, "token/giay", v["token_moi_giay"][str(e + 1)], "gio da chay", round((time.time() - t_bat_dau) / 3600, 2), flush=True)
        goc = m.unload()
        del m, opt
        torch.cuda.empty_cache()

    def danh_gia(ten, v, nhanh, fold, thu_adapter, tep_sinh, ca_fold=True):
        nonlocal goc
        if v.get("hong") or ((v.get("loss_ca_fold") or not ca_fold) and v.get("cham")):
            return
        kiem_gio(ten)
        m = PeftModel.from_pretrained(goc, str(thu_adapter))
        if ca_fold and not v.get("loss_ca_fold"):
            ca_fold = [x for x in ma[nhanh] if x["fold5"] == fold]
            if cfg.get("so_mau_ca_fold"):
                ca_fold = random.Random(cfg["seed"]).sample(ca_fold, cfg["so_mau_ca_fold"])
            theo = {}
            for x in ca_fold:
                theo.setdefault(f"{x['huong']}|{x['loai']}|{x['category']}", []).append(x)
            t = time.time()
            tung = {k: loss_tong(m, tok, a, 32, thiet_bi) for k, a in sorted(theo.items())}
            nhom, dm = gop_loss(tung)
            v["loss_ca_fold"] = {"so_mau": len(ca_fold), "diem_macro": dm, "theo_nhom": nhom, "theo_category": {k: {"loss": a / max(n, 1), "so_token": n} for k, (a, n) in tung.items()}}
            ghi()
            print(ten, "loss ca fold", round(dm, 5), nhom, "so mau", len(ca_fold), "giay", round(time.time() - t), flush=True)
        if not v.get("cham"):
            kiem_gio(ten)
            mau, thieu = lay_mau_sinh(nhanh, fold)
            v["so_id_sinh_thieu"] = thieu
            if not tep_sinh.exists():
                t = time.time()
                du = sinh(m, tok, mau, thiet_bi)
                v["giay_moi_mau"] = round((time.time() - t) / len(du), 3)
                ghi_sinh(tep_sinh, du)
            _, v["cham"] = cham_mot(nhanh, mau, doc_sinh(tep_sinh, mau), tap_da_thay(nhanh, fold), SO_EPOCH_B6, v.get("giay_moi_mau"))
            ghi()
            print(ten, "chrF", v["cham"]["chrf"]["chrf_macro_huong"], "so mau", len(mau), "thieu", thieu, "giay/mau", v.get("giay_moi_mau"), "gio da chay", round((time.time() - t_bat_dau) / 3600, 2), flush=True)
        goc = m.unload()
        del m
        torch.cuda.empty_cache()

    # Việc 1: LoRA attention so với LoRA mọi lớp cùng lr, rank, epoch 1, cụm fold 0, cùng mẫu sinh của B6. Khảo sát tỉ lệ không cần phần này
    mau_a, tm_a = None, {}
    if not cfg.get("viec"):
        mau_a, _ = lay_mau_sinh("cum", 0)
        (thu_ra / "attn").mkdir(exist_ok=True)
    for ten in (() if cfg.get("viec") else CAP_ATTN):
        tep_a = thu_ra / "attn" / f"{ten}_sinh.jsonl"
        if ten not in st["attn"]:
            kiem_gio(f"attn {ten}")
            m = PeftModel.from_pretrained(goc, str(kho / ten / "ep1"))
            t = time.time()
            du = sinh(m, tok, mau_a, thiet_bi)
            giay = round((time.time() - t) / len(du), 3)
            ghi_sinh(tep_a, du)
            goc = m.unload()
            del m
            torch.cuda.empty_cache()
            _, st["attn"][ten] = cham_mot("cum", mau_a, du, None, 1, giay)
            ghi()
            print(ten, "epoch 1 chrF", st["attn"][ten]["chrf"]["chrf_macro_huong"], "giay/mau", giay, flush=True)
        tm_a[ten], _ = cham_mot("cum", mau_a, doc_sinh(tep_a, mau_a), None, 1, None)
    if not cfg.get("viec") and "so_cap" not in st["attn"]:
        ids_a = [m["id"] for m in mau_a]
        a, b = (tm_a[t].set_index("id").chrf[ids_a].tolist() for t in CAP_ATTN)
        st["attn"]["so_cap"] = bootstrap_chenh(a, b, [f"{m['huong']}|{m['loai']}" for m in mau_a])
        ghi()
        print("ATTN - MOI LOP:", st["attn"]["so_cap"], flush=True)

    # Việc 2 đến 5
    ten_chot = CAU_HINH_CHOT["ten"]
    for j in ds_viec:
        ten, nhanh, fold = j["ten"], j["nhanh"], j["fold"]
        v = st["viec"].setdefault(ten, {"nhanh": nhanh, "fold": fold, "cau_hinh": ten_chot, "epoch_xong": 0, "buoc": 0, "vi_tri_lo": 0, "loss": {}, "diem": {}, "token_moi_giay": {}, "loss_ca_fold": None, "cham": None})
        if ten == "cum_f0":
            b = st_b6["cau_hinh"][ten_chot]
            v.update(nguon_adapter="b6", epoch_xong=b["epoch_xong"], loss=b["loss"], diem=b["diem"], token_moi_giay=b["token_moi_giay"], giay_moi_mau=st_b6["cham"][ten_chot]["giay_moi_mau"])
            danh_gia(ten, v, nhanh, fold, kho / ten_chot / "ep2", kho / ten_chot / "sinh_ep2.jsonl")
            if not cfg.get("rut_gon_sinh") and v.get("cham"):      # chấm lại bản dịch B6 phải ra đúng điểm B6
                goc_b6 = st_b6["cham"][ten_chot]["chrf"]["chrf_macro_huong"]
                v["khop_b6"] = abs(v["cham"]["chrf"]["chrf_macro_huong"] - goc_b6) < 1e-6
                ghi()
                if not v["khop_b6"]:
                    raise RuntimeError(f"CHAM_LAI_KHONG_KHOP_B6:{v['cham']['chrf']['chrf_macro_huong']}:{goc_b6}")
            continue
        huan_luyen(ten, v, nhanh, fold, j.get("p_title"), j.get("lr"))
        if cfg.get("chrf_toi_thieu") is not None and v.get("hong"):
            dung_bat_thuong(ten, "loss khong huu han")
        danh_gia(ten, v, nhanh, fold, thu_ra / ten / f"ep{SO_EPOCH_B6}", thu_ra / ten / f"sinh_ep{SO_EPOCH_B6}.jsonl", j.get("ca_fold", True))
        if cfg.get("chrf_toi_thieu") is not None and v.get("cham"):
            ly_do = kiem_bat_thuong(v["cham"], cfg["chrf_toi_thieu"], cfg["lap_toi_da"])
            if ly_do:
                dung_bat_thuong(ten, ly_do)
    st["xong"] = True
    ghi()
    bc["dat"] = not any(v.get("hong") for v in st["viec"].values())
    print("B78 XONG", {t: (v["cham"] or {}).get("chrf", {}).get("chrf_macro_huong") for t, v in st["viec"].items()}, flush=True)


def thu_b5(cfg, bc):
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer
    import peft

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    thiet_bi = "cuda:0"
    bc["moi_truong"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                        "gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                        "cuda_khả_dụng": torch.cuda.is_available()}
    print(bc["moi_truong"], flush=True)
    thu_mh = tim_mot("REVISION.txt").parent
    bc["mo_hinh"] = thu_mh.joinpath("REVISION.txt").read_text().split()
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    goc_mau = doc_mau(cfg["nhanh"])
    bc["so_mau_pool"] = len(goc_mau)
    ma, bo = xay_mau(goc_mau, tok, cfg["max_len"])
    bc["so_mau_vuot_max_len_bi_bo"] = bo
    bc["so_token_toan_bo_1_epoch"] = sum(len(m["ids"]) for m in ma)
    v0 = ma[0]
    print("MAU MA HOA:\n", tok.decode(v0["ids"])[:600], "\nPHAN TINH LOSS:\n", tok.decode([x for x in v0["labels"] if x != -100])[:300], flush=True)
    train = [m for m in ma if m["fold5"] != cfg["fold_kiem"]]
    kiem_all = [m for m in ma if m["fold5"] == cfg["fold_kiem"]]
    bc["so_mau_train"], bc["so_mau_kiem"] = len(train), len(kiem_all)
    if set(m["id"] for m in train) & set(m["id"] for m in kiem_all):
        raise RuntimeError("giao_train_kiem")
    rng = random.Random(cfg["seed"])
    kiem = rng.sample(kiem_all, cfg["so_mau_kiem"])

    t = time.time()
    model = nap_goc(thu_mh)
    bc["giay_nap_mo_hinh"] = round(time.time() - t, 1)
    model = get_peft_model(model, LoraConfig(r=cfg["rank"], lora_alpha=2 * cfg["rank"], lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    # Đo 05/11/2026: không checkpoint + lô 4096 token nhanh nhất (5.873 token/giây, 52 GB); 8192 token không checkpoint tràn 95 GB
    if cfg.get("gc", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    tham_so = [p for p in model.parameters() if p.requires_grad]
    bc["so_tham_so_lora"] = sum(p.numel() for p in tham_so)
    print("tham so LoRA:", bc["so_tham_so_lora"], flush=True)

    bc["loss_kiem_truoc"] = loss_kiem(model, tok, kiem, cfg["micro_bs"] * 2, thiet_bi)
    print("loss kiem truoc:", bc["loss_kiem_truoc"], flush=True)

    opt = torch.optim.AdamW(tham_so, lr=cfg["lr"], weight_decay=0.0)
    ngan_sach = cfg.get("ngan_sach_token", 4096)
    lo = chia_lo(train, cfg["micro_bs"], cfg["seed"], ngan_sach)
    bc["ngan_sach_token"] = ngan_sach
    co_batch = cfg["micro_bs"] * cfg["tich_luy"]        # batch hiệu dụng tính theo số mẫu
    tong_buoc = cfg["so_buoc"]
    model.train()
    nhat_ky, tok_dem, tok_do, t_do = [], 0, 0, None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    it = iter(lo)
    for buoc in range(tong_buoc):
        if buoc == 5:                    # bỏ 5 bước đầu khỏi phép đo tốc độ
            t_do, tok_do = time.time(), 0
        tong, da_co = 0.0, 0
        while da_co < co_batch:          # gom đủ batch hiệu dụng; lô dài có ít mẫu hơn nên số micro-batch thay đổi
            k = next(it)
            ids, lab, mask = lo_tensor(k, tok, thiet_bi)
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            (out.loss * len(k) / co_batch).backward()
            tong += out.loss.item() * len(k) / co_batch
            da_co += len(k)
            n = int(mask.sum().item())
            tok_dem += n
            tok_do += n
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * lich_lr(buoc, tong_buoc)
        torch.nn.utils.clip_grad_norm_(tham_so, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if buoc % 10 == 0 or buoc == tong_buoc - 1:
            nhat_ky.append({"buoc": buoc, "loss": round(tong, 4), "giay": round(time.time() - t0, 1)})
            print(nhat_ky[-1], flush=True)
    bc["nhat_ky_loss"] = nhat_ky
    bc["token_moi_giay"] = round(tok_do / max(time.time() - t_do, 1e-6), 1)
    bc["tong_token_da_chay"] = tok_dem
    bc["vram_dinh_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print("token/giay:", bc["token_moi_giay"], "VRAM dinh GB:", bc["vram_dinh_gb"], flush=True)

    bc["loss_kiem_sau"] = loss_kiem(model, tok, kiem, cfg["micro_bs"] * 2, thiet_bi)
    print("loss kiem sau:", bc["loss_kiem_sau"], flush=True)
    bc["loss_giam"] = bc["loss_kiem_sau"] < bc["loss_kiem_truoc"]

    ck = RA / "ck_b5"
    model.save_pretrained(str(ck))
    bc["ck_mb"] = round(sum(p.stat().st_size for p in ck.rglob("*") if p.is_file()) / 2**20, 1)
    del model, opt
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(nap_goc(thu_mh), str(ck))
    bc["loss_kiem_sau_nap_lai"] = loss_kiem(model, tok, kiem, cfg["micro_bs"] * 2, thiet_bi)
    bc["nap_lai_khop"] = abs(bc["loss_kiem_sau_nap_lai"] - bc["loss_kiem_sau"]) < 1e-3
    print("loss sau nap lai:", bc["loss_kiem_sau_nap_lai"], "khop:", bc["nap_lai_khop"], flush=True)

    # Sinh thử trên fold kiểm (không dùng test): chia đều theo hướng x loại
    chon, n_moi = [], max(1, cfg["so_mau_sinh"] // 4)
    id_co = {m["id"] for m in kiem}
    for h in ("zh->vi", "vi->zh"):
        for l in ("title", "mo_ta"):
            ung = [m for m in goc_mau if m["fold5"] == cfg["fold_kiem"] and m["huong"] == h and m["loai"] == l and m["phan"] == 0 and m["id"] in id_co]
            chon += rng.sample(ung, min(n_moi, len(ung)))
    t = time.time()
    du_ft = sinh(model, tok, chon, thiet_bi)
    bc["giay_sinh_moi_mau_ft"] = round((time.time() - t) / max(1, len(chon)), 2)
    t = time.time()
    with model.disable_adapter():
        du_zs = sinh(model, tok, chon, thiet_bi)
    bc["giay_sinh_moi_mau_zs"] = round((time.time() - t) / max(1, len(chon)), 2)

    import pandas as pd
    import cham_diem as D
    tc = pd.DataFrame(chon)
    for ten, du in (("fine_tune", du_ft), ("zero_shot", du_zs)):
        (RA / f"sinh_{ten}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in du) + "\n", encoding="utf-8")
        tm, th = D.cham(tc, pd.DataFrame(du), nhanh=cfg["nhanh"])
        bc[f"cham_{ten}"] = th.get("diem_chinh")
        bc[f"cham_{ten}_ti_le_cat_loi_dan"] = float(tm.da_cat_loi_dan.mean())
    bc["vi_du_sinh"] = [{"id": m["id"], "huong": m["huong"], "loai": m["loai"], "nguon": m["nguon"][:200], "dich": m["dich"][:200],
                         "ft": a["hyp_tho"][:200], "zs": b["hyp_tho"][:300]} for m, a, b in zip(chon, du_ft, du_zs)]


def do_toc_do(cfg, bc):
    """B5: so token/giây và VRAM giữa các cách chia lô (kể cả lô dài nhất để biết có tràn không). Chỉ đo tốc độ, không lưu mô hình."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    thiet_bi = "cuda:0"
    thu_mh = tim_mot("REVISION.txt").parent
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    ma, _ = xay_mau(doc_mau(cfg["nhanh"]), tok, cfg["max_len"])
    train = [m for m in ma if m["fold5"] != cfg["fold_kiem"]]
    model = get_peft_model(nap_goc(thu_mh), LoraConfig(r=cfg["rank"], lora_alpha=2 * cfg["rank"], lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    tham_so = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tham_so, lr=cfg["lr"], weight_decay=0.0)
    dai_nhat = sorted(train, key=lambda m: -len(m["ids"]))
    bc["do_toc_do"] = []
    for c in cfg["cac_cau_hinh"]:
        kq = dict(c)
        if c["gc"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            xau = cat_ngan_sach(dai_nhat[:64], c["micro_bs"], c["ngan_sach"])[0]     # lô tệ nhất
            ids, lab, mask = lo_tensor(xau, tok, thiet_bi)
            model(input_ids=ids, attention_mask=mask, labels=lab).loss.backward()
            opt.zero_grad(set_to_none=True)
            kq["vram_lo_dai_nhat_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            it = iter(chia_lo(train, c["micro_bs"], cfg["seed"], c["ngan_sach"]))
            dem, t0 = 0, None
            for buoc in range(cfg["so_buoc_do"]):
                if buoc == 3:
                    dem, t0 = 0, time.time()
                da_co = 0
                while da_co < 32:
                    k = next(it)
                    ids, lab, mask = lo_tensor(k, tok, thiet_bi)
                    (model(input_ids=ids, attention_mask=mask, labels=lab).loss * len(k) / 32).backward()
                    da_co += len(k)
                    dem += int(mask.sum().item())
                opt.step()
                opt.zero_grad(set_to_none=True)
            kq["token_moi_giay"] = round(dem / (time.time() - t0), 1)
            kq["vram_dinh_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        except torch.OutOfMemoryError:
            kq["tran_bo_nho"] = True
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        bc["do_toc_do"].append(kq)
        print(kq, flush=True)
    bc["dat"] = True


def do_nguyen_nhan(cfg, bc):
    """Tìm chỗ chậm trong một bước huấn luyện: thời gian từng phần trên cùng một bộ lô cố định, GPU có chạy hết công suất không, và vài biến thể."""
    import os
    import threading
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    thiet_bi = "cuda:0"
    thu_mh = tim_mot("REVISION.txt").parent
    tok = AutoTokenizer.from_pretrained(str(thu_mh))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    ma, _ = xay_mau(doc_mau(cfg["nhanh"]), tok, cfg["max_len"])
    train = [m for m in ma if m["fold5"] != cfg["fold_kiem"]]
    lo = chia_lo(train, cfg["micro_bs"], cfg["seed"], cfg["ngan_sach_token"])[:cfg["so_lo_do"]]
    goc = nap_goc(thu_mh)
    c = goc.config
    tong_tok = sum(len(m["ids"]) for k in lo for m in k)
    o_dem = sum(len(k) * max(len(m["ids"]) for m in k) for k in lo)
    tinh_loss = sum(sum(x != -100 for x in m["labels"]) for k in lo for m in k)
    bc["mo_hinh_cau_truc"] = {"so_lop": c.num_hidden_layers, "hidden": c.hidden_size, "tu_vung": c.vocab_size, "attn": getattr(c, "_attn_implementation", None),
                              "so_cpu": os.cpu_count(), "dtype": str(next(goc.parameters()).dtype)}
    bc["lo_do"] = {"so_lo": len(lo), "token_that": tong_tok, "o_tinh_gom_dem": o_dem, "ti_le_dem": round(1 - tong_tok / o_dem, 3),
                   "token_tinh_loss": tinh_loss, "ti_le_token_tinh_loss": round(tinh_loss / tong_tok, 3)}
    print(bc["mo_hinh_cau_truc"], bc["lo_do"], flush=True)

    def dong_ho():
        torch.cuda.synchronize()
        return time.perf_counter()

    def forward(model, ids, lab, mask, chi_dich):
        if not chi_dich:
            return model(input_ids=ids, attention_mask=mask, labels=lab).loss
        trong = model.base_model.model                       # chỉ tính lm_head ở vị trí có loss, tránh ma trận logits toàn bộ
        h = trong.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
        nhan = lab[:, 1:]
        chon = nhan != -100
        return F.cross_entropy(trong.lm_head(h[chon]).float(), nhan[chon])

    def do_bien_the(model, ten, chi_dich=False, lay_gpu=False):
        tham_so = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(tham_so, lr=1e-4, weight_decay=0.0)
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tg = {"tensor": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        mau_gpu, dung = [], threading.Event()

        def lay():
            while not dung.is_set():
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                mau_gpu.append(r.stdout.strip())
                time.sleep(1)

        if lay_gpu:
            threading.Thread(target=lay, daemon=True).start()
        dem = 0
        for i, k in enumerate(lo):
            t = dong_ho()
            ids, lab, mask = lo_tensor(k, tok, thiet_bi)
            t1 = dong_ho()
            loss = forward(model, ids, lab, mask, chi_dich)
            t2 = dong_ho()
            loss.backward()
            t3 = dong_ho()
            torch.nn.utils.clip_grad_norm_(tham_so, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            t4 = dong_ho()
            if i >= 3:                    # bỏ vài lô đầu
                tg["tensor"] += t1 - t
                tg["forward"] += t2 - t1
                tg["backward"] += t3 - t2
                tg["optimizer"] += t4 - t3
                dem += int(mask.sum().item())
        dung.set()
        tong = sum(tg.values())
        kq = {"ten": ten, "token_moi_giay": round(dem / tong, 1), "vram_dinh_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
              "ti_trong_%": {a: round(100 * b / tong, 1) for a, b in tg.items()}, "so_tham_so_lora": sum(p.numel() for p in tham_so)}
        if mau_gpu:
            kq["nvidia_smi_mau"] = mau_gpu[len(mau_gpu) // 2:len(mau_gpu) // 2 + 4]
        bc["bien_the"].append(kq)
        print(kq, flush=True)
        del opt

    # Mốc: chỉ chạy suy luận mô hình gốc, không LoRA, không gradient
    goc.eval()
    t = dong_ho()
    dem = 0
    with torch.no_grad():
        for k in lo:
            ids, lab, mask = lo_tensor(k, tok, thiet_bi)
            goc(input_ids=ids, attention_mask=mask)
            dem += int(mask.sum().item())
    bc["suy_luan_goc_token_moi_giay"] = round(dem / (dong_ho() - t), 1)
    print("suy luan goc token/giay:", bc["suy_luan_goc_token_moi_giay"], flush=True)
    bc["bien_the"] = []

    def lora(model, r=16, dropout=0.05, muc_tieu="all-linear", **kw):
        return get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, target_modules=muc_tieu, task_type="CAUSAL_LM"), **kw)

    for ten, args, chi_dich, lay_gpu in [
            ("hien_tai_all_linear_r16", {}, False, True),
            ("khong_dropout", {"dropout": 0.0}, False, False),
            ("chi_attention_qkvo", {"muc_tieu": ["q_proj", "k_proj", "v_proj", "o_proj"]}, False, False),
            ("r64", {"r": 64}, False, False),
            ("lm_head_chi_vi_tri_dich", {}, True, False),
            ("lora_cung_bf16", {"autocast_adapter_dtype": False}, False, False)]:
        m = lora(goc, **args)
        try:
            do_bien_the(m, ten, chi_dich, lay_gpu)
        except Exception:
            bc["bien_the"].append({"ten": ten, "loi": traceback.format_exc()[-500:]})
            print(bc["bien_the"][-1], flush=True)
        goc = m.unload()
        del m
    bc["dat"] = True


def main(cfg):
    import os
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    bc = {"cau_hinh": cfg, "dat": False}
    t = time.time()
    try:
        kiem_khoa_test()
        bc["khoa_test"] = "khong_co_file_test_trong_input"
        cai_thu_vien()
        if cfg["che_do"] in ("b6", "b78"):
            cfg["_t_bat_dau"] = t
            try:
                (chay_b6 if cfg["che_do"] == "b6" else chay_b78)(cfg, bc)
            except DungHetGio as e:
                bc["dung_het_gio"] = str(e)            # kết thúc phiên bình thường để Kaggle lưu đầu ra; phiên sau chạy tiếp
                print("KET THUC PHIEN CO TRAT TU, dung o", e, flush=True)
        elif cfg["che_do"] == "do_toc_do":
            do_toc_do(cfg, bc)
        elif cfg["che_do"] == "do_nguyen_nhan":
            do_nguyen_nhan(cfg, bc)
        elif cfg["che_do"] == "thu_b5":
            thu_b5(cfg, bc)
            bc["dat"] = bool(bc.get("loss_giam") and bc.get("nap_lai_khop"))
        else:
            raise RuntimeError(f"che_do_chua_ho_tro:{cfg['che_do']}")
    except Exception:
        bc["loi"] = traceback.format_exc()
        print(bc["loi"], flush=True)
    bc["tong_giay"] = round(time.time() - t, 1)
    ten_bc = f"bao_cao_{cfg.get('ten_trang_thai', cfg['che_do'])}.json"
    (RA / ten_bc).write_text(json.dumps(bc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("KET QUA:", json.dumps({k: v for k, v in bc.items() if k not in ("vi_du_sinh", "cau_hinh", "nhat_ky_loss", "b6", "b78")}, ensure_ascii=False, default=str)[:3000], flush=True)
    if "loi" in bc:
        raise RuntimeError(f"loi, xem {ten_bc}")
