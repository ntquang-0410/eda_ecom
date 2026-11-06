# -*- coding: utf-8 -*-
"""Điều khiển Kaggle bằng API cho bộ song ngữ 3 (thay cho tạo Dataset, import notebook, tải đầu ra bằng tay). Token đọc từ
C:\\Users\\<user>\\.kaggle\\access_token (anh tự lưu; script không in, không ghi token). Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\kaggle_api.py tao_dataset        # lần đầu: Dataset RIÊNG TƯ từ data/processed/song_ngu_3/vao_kaggle
    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\kaggle_api.py cap_nhat_dataset   # các lần sau: phiên bản mới của Dataset
    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\kaggle_api.py day <gemma|qwen> [thu|that]   # lưu notebook bản mới, KHÔNG chạy
    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\kaggle_api.py chay qwen [thu|that]          # đẩy và chạy nền trên GPU T4 x2
        (Gemma KHÔNG chạy được bằng API: Kaggle Secrets không đọc được, ConnectionError, thử 23 và 24/10/2026. Gemma: `day gemma that` rồi
         anh bấm Save & Run All (Commit) trên web; đầu ra vẫn tải về bằng `tai_ve`)
    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\kaggle_api.py trang_thai <gemma|qwen>
    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\kaggle_api.py tai_ve <gemma|qwen> <thư mục con trong data/processed/song_ngu_3>
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
DATASET = f"{NGUOI_DUNG}/song-ngu-3-cum-vao"
VAO = ROOT / "data" / "processed" / "song_ngu_3" / "vao_kaggle"
# hai notebook cho hai mô hình (24/10/2026). Gemma giữ kernel cũ song-ngu-3-cum vì secret khoá Gemini đã Attach ở đó.
BIEN_THE = {"gemma": {"slug": "song-ngu-3-cum", "file": "song-ngu-3-cum-gemma-kaggle.ipynb", "gpu": False},
            "qwen": {"slug": "song-ngu-3-cum-qwen", "file": "song-ngu-3-cum-qwen-kaggle.ipynb", "gpu": True}}


def goi(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    print(out)
    if check and r.returncode != 0:
        raise SystemExit(f"Lệnh kaggle {args[0]} {args[1] if len(args) > 1 else ''} lỗi (mã {r.returncode}).")
    return out


def meta_dataset():
    # luôn chép bản code mới nhất vào thư mục tải lên (tránh đẩy dich_cum.py cũ)
    shutil.copyfile(ROOT / "src" / "song_ngu_1" / "kaggle" / "dich.py", VAO / "dich.py")
    shutil.copyfile(ROOT / "src" / "song_ngu_3" / "kaggle" / "dich_cum.py", VAO / "dich_cum.py")
    (VAO / "dataset-metadata.json").write_text(json.dumps({"title": "song-ngu-3-cum-vao", "id": DATASET,
                                                          "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")


def thu_muc_notebook(bt, che_do):
    """Chép notebook của biến thể bt (gemma | qwen) vào thư mục đẩy, đặt CHE_DO, ghi kernel-metadata.json (riêng tư, Internet, gắn Dataset;
    GPU chỉ bật cho Qwen). Trả về (thư mục, mã kernel)."""
    v = BIEN_THE[bt]
    kid = f"{NGUOI_DUNG}/{v['slug']}"
    thu = ROOT / "notebooks" / "kaggle_api" / v["slug"]
    thu.mkdir(parents=True, exist_ok=True)
    nb = json.loads((ROOT / "notebooks" / v["file"]).read_text(encoding="utf-8"))
    s = "".join(nb["cells"][1]["source"])
    s2, n = re.subn(r'^CHE_DO = "(thu|that)"', f'CHE_DO = "{che_do}"', s, count=1, flags=re.M)
    assert n == 1, "không thấy dòng CHE_DO trong ô Cấu hình"
    nb["cells"][1]["source"] = s2.splitlines(keepends=True)
    (thu / v["file"]).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (thu / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": v["slug"], "code_file": v["file"], "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": v["gpu"], "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1), encoding="utf-8")
    return thu, kid


def main():
    lenh = sys.argv[1]
    if lenh == "tao_dataset":
        meta_dataset()
        goi("datasets", "create", "-p", str(VAO))          # không có --public: Dataset riêng tư
    elif lenh == "cap_nhat_dataset":
        meta_dataset()
        goi("datasets", "version", "-p", str(VAO), "-m", sys.argv[2] if len(sys.argv) > 2 else "cập nhật")
    elif lenh == "trang_thai_dataset":
        goi("datasets", "status", DATASET)
    elif lenh == "day":           # day <gemma|qwen> [thu|that]: lưu bản mới, KHÔNG chạy
        thu, kid = thu_muc_notebook(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "thu")
        goi("kernels", "push", "-p", str(thu), "--no-run")
    elif lenh == "chay":          # chay <qwen> [thu|that]: đẩy và chạy nền (Gemma không chạy được bằng API: secret không đọc được)
        assert sys.argv[2] == "qwen", "Gemma phải bấm Save & Run All (Commit) trên web (Kaggle Secrets không đọc được khi chạy bằng API)"
        thu, kid = thu_muc_notebook("qwen", sys.argv[3] if len(sys.argv) > 3 else "thu")
        goi("kernels", "push", "-p", str(thu), "--accelerator", "NvidiaTeslaT4")
    elif lenh == "trang_thai":    # trang_thai <gemma|qwen>
        goi("kernels", "status", f"{NGUOI_DUNG}/{BIEN_THE[sys.argv[2]]['slug']}")
    elif lenh == "tai_ve":        # tai_ve <gemma|qwen> <thư mục con trong data/processed/song_ngu_3>
        kid = f"{NGUOI_DUNG}/{BIEN_THE[sys.argv[2]]['slug']}"
        dich = ROOT / "data" / "processed" / "song_ngu_3" / sys.argv[3]
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
