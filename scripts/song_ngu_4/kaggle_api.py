# -*- coding: utf-8 -*-
"""Điều khiển Kaggle bằng API cho bộ song ngữ 4 (1688 Trung -> Việt, hiệu đính cả cụm). Token đọc từ C:\\Users\\<user>\\.kaggle\\access_token
(anh tự lưu; script không in, không ghi token). Chạy từ gốc repo bằng python có gói kaggle:

    kaggle_api.py tao_dataset                      # lần đầu: Dataset RIÊNG TƯ từ data/processed/song_ngu_4/vao_kaggle
    kaggle_api.py cap_nhat_dataset "<ghi chú>"     # các lần sau: phiên bản mới của Dataset
    kaggle_api.py day_phan <K>                      # chạy song song: đẩy K notebook Gemma (that + tiep), mỗi cái một phần product_id (hash)
    kaggle_api.py tao_dataset_ra_cu <thư mục con> ["<ghi chú>"]  # Dataset riêng tư song-ngu-4-cum-ra-cu từ các lô đã dịch (phiên trước)
    kaggle_api.py day <gemma|qwen> [thu|that] [tiep]  # lưu notebook bản mới, KHÔNG chạy; tiep: gắn Dataset ra-cu (nạp các lô đã dịch)
    kaggle_api.py chay <qwen|qwen_rtx> [thu|that] [tiep]   # đẩy và chạy nền; qwen = GPU T4 x2, qwen_rtx = notebook dl-group2 trên RTX Pro 6000; tiep: nạp đầu ra phiên trước, dịch phần còn thiếu
        (Gemma KHÔNG chạy được bằng API: Kaggle Secrets không đọc được. Gemma: `day gemma <che_do>` rồi anh Attach secret (lần đầu) và
         bấm Save & Run All (Commit) trên web; đầu ra vẫn tải về bằng `tai_ve`)
    kaggle_api.py trang_thai <gemma|qwen>
    kaggle_api.py tai_ve <gemma|qwen> <thư mục con trong data/processed/song_ngu_4>
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")
KAGGLE = str(Path(sys.executable).parent / "kaggle")
NGUOI_DUNG = "trinhnhatanh"
DATASET = f"{NGUOI_DUNG}/song-ngu-4-cum-vao"
DATASET_RA_CU = f"{NGUOI_DUNG}/song-ngu-4-cum-ra-cu"
VAO = ROOT / "data" / "processed" / "song_ngu_4" / "vao_kaggle"
BIEN_THE = {"gemma": {"slug": "song-ngu-4-cum-gemma", "file": "song-ngu-4-cum-gemma-kaggle.ipynb", "gpu": False},
            "qwen": {"slug": "song-ngu-4-cum-qwen", "file": "song-ngu-4-cum-qwen-kaggle.ipynb", "gpu": True}}
# Qwen trên GPU RTX Pro 6000 (96 GB): chạy trong notebook dl-group2 (anh cho ghi đè 26/10/2026, bản cũ lưu ở notebooks/_sao_luu/dl-group2_20261026_truoc_ghi_de)
# competition_sources: GPU RTX Pro 6000 chỉ hiện khi notebook gắn cuộc thi ARC Prize 2026 (gỡ cuộc thi là mất lựa chọn GPU, 26/10/2026)
# docker_image: ảnh môi trường sẵn có của notebook này (GPU RTX Pro 6000 đi kèm ảnh riêng; chỉ đặt --accelerator thì kernel song-ngu-4-cum-qwen vẫn chạy T4 x2)
BIEN_THE["qwen_rtx"] = dict(BIEN_THE["qwen"], slug="dl-group2", competition_sources=["arc-prize-2026-arc-agi-3"], enable_internet=False, docker_image="gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")
# notebook nlp-group3: anh cho ghi đè 26/10/2026 (bản đọc được lưu ở notebooks/_sao_luu/nlp-group3_20261026_chi_doc; 5 nguồn dữ liệu riêng tư cũ sẽ mất)
BIEN_THE["qwen_nlp"] = dict(BIEN_THE["qwen"], slug="nlp-group3", competition_sources=["arc-prize-2026-arc-agi-3"], enable_internet=False,
                            kernel_sources=["trinhnhatanh/song-ngu-4-dong-goi-thu-vien", "trinhnhatanh/song-ngu-4-dong-goi-mo-hinh"], docker_image="gcr.io/kaggle-private-byod/python@sha256:00377cd1b3d470a605bc5b0ceca79969e369644e9b36802242a1c70e627372f9")
GPU_CUA = {"qwen": "NvidiaTeslaT4", "qwen_rtx": "NvidiaRtxPro6000", "qwen_nlp": "NvidiaRtxPro6000"}
for _i in range(1, 6):      # chạy song song: phần i của Gemma là notebook riêng gemma_p<i> (phần 0 là notebook gemma)
    BIEN_THE[f"gemma_p{_i}"] = dict(BIEN_THE["gemma"], slug=f"song-ngu-4-cum-gemma-p{_i}")
FILE_DATASET = {"cum_can_dich_zv.parquet", "cau_hinh_kaggle.json", "dich.py", "dich_cum_zv.py", "dataset-metadata.json"}


def goi(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    print(out)
    if check and r.returncode != 0:
        raise SystemExit(f"Lệnh kaggle {args[0]} {args[1] if len(args) > 1 else ''} lỗi (mã {r.returncode}).")
    return out


def meta_dataset():
    # luôn chép bản code mới nhất vào thư mục tải lên; thư mục chỉ được có đúng các file dự kiến (không lỡ tải dữ liệu khác lên)
    shutil.copyfile(ROOT / "src" / "song_ngu_1" / "kaggle" / "dich.py", VAO / "dich.py")
    shutil.copyfile(ROOT / "src" / "song_ngu_4" / "kaggle" / "dich_cum_zv.py", VAO / "dich_cum_zv.py")
    (VAO / "dataset-metadata.json").write_text(json.dumps({"title": "song-ngu-4-cum-vao", "id": DATASET,
                                                          "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    co = {p.name for p in VAO.iterdir()}
    assert co == FILE_DATASET, f"thư mục tải lên có file lạ hoặc thiếu file: {sorted(co ^ FILE_DATASET)}"


def tao_dataset_ra_cu(nguon, ghi_chu):
    """Dataset RIÊNG TƯ chứa các lô đã dịch (ra_cum_zv/api_tuy_chon/lo_*.parquet) của phiên trước. Cách gắn kernel_sources vào chính
    notebook đó KHÔNG hoạt động khi bấm Commit trên web (26/10/2026: phiên 2 báo 'đầu ra cũ: không có', dịch lại từ đầu, mất 11 giờ + 8 USD)."""
    ra = ROOT / "data" / "processed" / "song_ngu_4" / "ra_cu_kaggle"
    src = ROOT / "data" / "processed" / "song_ngu_4" / nguon / "ra_cum_zv" / "api_tuy_chon"
    lo = sorted(src.glob("lo_*.parquet"))
    assert lo, f"không có lô nào trong {src}"
    if ra.exists():
        shutil.rmtree(ra)
    dich = ra / "ra_cum_zv" / "api_tuy_chon"
    dich.mkdir(parents=True)
    for f in lo:
        shutil.copy2(f, dich / f.name)
    (ra / "dataset-metadata.json").write_text(json.dumps({"title": "song-ngu-4-cum-ra-cu", "id": DATASET_RA_CU,
                                                         "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
    print(f"{len(lo)} lô từ {nguon}")
    co = goi("datasets", "status", DATASET_RA_CU, check=False)
    if "ready" in co:
        goi("datasets", "version", "-p", str(ra), "-m", ghi_chu, "-r", "zip")
    else:
        goi("datasets", "create", "-p", str(ra), "-r", "zip")      # không có --public: riêng tư


def thu_muc_notebook(bt, che_do, tiep=False, phan=(0, 1), cum=None):
    """tiep=True: gắn Dataset DATASET_RA_CU (các lô đã dịch của phiên trước) làm dữ liệu vào; notebook tự nhận các lô cũ
    (thư mục ra_cum_zv) qua --cu và chỉ dịch phần còn thiếu. Dùng khi phiên trước dừng vì hết giờ."""
    v = BIEN_THE[bt]
    kid = f"{NGUOI_DUNG}/{v['slug']}"
    thu = ROOT / "notebooks" / "kaggle_api" / v["slug"]
    thu.mkdir(parents=True, exist_ok=True)
    nb = json.loads((ROOT / "notebooks" / v["file"]).read_text(encoding="utf-8"))
    s = "".join(nb["cells"][1]["source"])
    s2, n = re.subn(r'^CHE_DO = "(thu|that)"', f'CHE_DO = "{che_do}"', s, count=1, flags=re.M)
    assert n == 1, "không thấy dòng CHE_DO trong ô Cấu hình"
    s2, n = re.subn(r'^PHIEN_TIEP = (True|False)', f'PHIEN_TIEP = {tiep}', s2, count=1, flags=re.M)
    assert n == 1, "không thấy dòng PHIEN_TIEP trong ô Cấu hình"
    s2, n = re.subn(r"^SO_PHAN = \d+", f"SO_PHAN = {phan[1]}", s2, count=1, flags=re.M)
    s2, m = re.subn(r"^CHI_SO_PHAN = \d+", f"CHI_SO_PHAN = {phan[0]}", s2, count=1, flags=re.M)
    assert n == 1 and m == 1, "không thấy dòng SO_PHAN / CHI_SO_PHAN trong ô Cấu hình"
    if cum:          # cum = file JSON là danh sách ma_cum; nhúng vào notebook dạng zlib + base64
        import base64
        import zlib
        ds = json.loads(Path(cum).read_text(encoding="utf-8"))
        assert isinstance(ds, list) and ds and all(isinstance(x, str) for x in ds) and len(set(ds)) == len(ds), "file cụm phải là danh sách ma_cum duy nhất"
        b64 = base64.b64encode(zlib.compress(json.dumps(sorted(ds), ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")
        s2, n = re.subn(r'^CHI_CAC_CUM = ".*?"', lambda _: f'CHI_CAC_CUM = "{b64}"', s2, count=1, flags=re.M)
        assert n == 1, "không thấy dòng CHI_CAC_CUM trong ô Cấu hình"
    nb["cells"][1]["source"] = s2.splitlines(keepends=True)
    (thu / v["file"]).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": v["slug"], "code_file": v["file"], "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": v["gpu"], "enable_tpu": False, "enable_internet": v.get("enable_internet", True),
        **({"docker_image": v["docker_image"], "machine_shape": GPU_CUA[bt]} if "docker_image" in v else {}),
        "dataset_sources": [DATASET] + ([DATASET_RA_CU] if tiep else []), "competition_sources": v.get("competition_sources", []), "kernel_sources": v.get("kernel_sources", []),
        "model_sources": []},
        indent=1), encoding="utf-8")
    return thu, kid


def main():
    lenh = sys.argv[1] if len(sys.argv) > 1 else ""
    if lenh == "tao_dataset":
        meta_dataset()
        goi("datasets", "create", "-p", str(VAO))          # không có --public: Dataset riêng tư
    elif lenh == "cap_nhat_dataset":
        meta_dataset()
        goi("datasets", "version", "-p", str(VAO), "-m", sys.argv[2] if len(sys.argv) > 2 else "cập nhật")
    elif lenh == "trang_thai_dataset":
        goi("datasets", "status", DATASET)
    elif lenh == "tao_dataset_ra_cu":
        tao_dataset_ra_cu(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cập nhật")
    elif lenh == "day":
        cum = next((a.split("=", 1)[1] for a in sys.argv[4:] if a.startswith("cum=")), None)
        thu, kid = thu_muc_notebook(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "thu", tiep="tiep" in sys.argv[4:], cum=cum)
        goi("kernels", "push", "-p", str(thu), "--no-run")
    elif lenh == "day_phan":      # day_phan <K>: đẩy K notebook Gemma (gemma, gemma_p1..), mỗi cái dịch một phần product_id, chế độ that + tiep
        so = int(sys.argv[2])
        assert 2 <= so <= 6, "K từ 2 đến 6"
        for i in range(so):
            thu, kid = thu_muc_notebook("gemma" if i == 0 else f"gemma_p{i}", "that", tiep=True, phan=(i, so))
            goi("kernels", "push", "-p", str(thu), "--no-run")
    elif lenh == "chay":
        assert sys.argv[2] in GPU_CUA, "Gemma phải bấm Save & Run All (Commit) trên web (Kaggle Secrets không đọc được khi chạy bằng API)"
        thu, kid = thu_muc_notebook(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "thu", tiep="tiep" in sys.argv[4:])
        goi("kernels", "push", "-p", str(thu), "--accelerator", GPU_CUA[sys.argv[2]])
    elif lenh == "trang_thai":
        goi("kernels", "status", f"{NGUOI_DUNG}/{BIEN_THE[sys.argv[2]]['slug']}")
    elif lenh == "tai_ve":
        kid = f"{NGUOI_DUNG}/{BIEN_THE[sys.argv[2]]['slug']}"
        dich = ROOT / "data" / "processed" / "song_ngu_4" / sys.argv[3]
        if dich.exists() and any(dich.iterdir()):
            cu = dich.with_name(dich.name + "_cu")
            if cu.exists():
                shutil.rmtree(cu)
            dich.rename(cu)
            print("Thư mục đích đã có dữ liệu, đổi tên thành", cu.name)
        dich.mkdir(parents=True, exist_ok=True)
        goi("kernels", "output", kid, "-p", str(dich))
        print("Đã tải về", dich, ":", sorted(str(p.relative_to(dich)) for p in dich.rglob("*") if p.is_file())[:40])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
