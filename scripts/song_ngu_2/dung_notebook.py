# -*- coding: utf-8 -*-
"""Dựng hai notebook Kaggle của bộ song ngữ 2 (Tiki Việt -> Trung) từ hai notebook của bộ 1 (khuôn):

    notebooks/song-ngu-1-gemma-kaggle.ipynb -> notebooks/song-ngu-2-gemma-kaggle.ipynb   (mô hình A, gọi API, không GPU)
    notebooks/song-ngu-1-kaggle.ipynb       -> notebooks/song-ngu-2-kaggle.ipynb         (mô hình B + CometKiwi, GPU T4 x2)

Chỉ đổi: chiều dịch (--huong vi_zh), cách chọn Dataset (đúng Dataset Tiki), tên thư mục đầu ra (ra_gemma_tiki, ra_tiki) và chữ giải thích.
Mỗi chỗ thay phải khớp đúng 1 lần, không thì dừng (khuôn đã đổi). Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_2\\dung_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks"
sys.stdout.reconfigure(encoding="utf-8")

CHON_DATASET_CU = '''ds = glob.glob("/kaggle/input/**/don_vi_can_dich.parquet", recursive=True)
assert ds, "Chưa gắn Dataset có don_vi_can_dich.parquet (Add Data)."
VAO = Path(ds[0]).parent'''
CHON_DATASET_MOI = '''def _huong(p):
    try:
        return json.loads((Path(p).parent / "cau_hinh_kaggle.json").read_text(encoding="utf-8")).get("huong")
    except Exception:
        return None

ds = [p for p in glob.glob("/kaggle/input/**/don_vi_can_dich.parquet", recursive=True) if _huong(p) == "vi_zh"]
assert ds, "Chưa gắn Dataset của bộ Tiki (don_vi_can_dich.parquet cùng cau_hinh_kaggle.json có huong = vi_zh) (Add Data)."
VAO = Path(ds[0]).parent'''


def thay(s, cu, moi, so_lan=1):
    assert s.count(cu) == so_lan, f"khuôn đã đổi, không thấy đúng {so_lan} chỗ: {cu[:70]!r} (thấy {s.count(cu)})"
    return s.replace(cu, moi)


def doc(ten):
    return json.loads((NB / ten).read_text(encoding="utf-8"))


def ghi(nb, ten):
    (NB / ten).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def nguon(c):
    return "".join(c["source"])


def dat(c, s):
    c["source"] = s.splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None


# ---------------- notebook Gemma (mô hình A, không GPU) ----------------
nb = doc("song-ngu-1-gemma-kaggle.ipynb")
C = nb["cells"]
dat(C[0], nguon(C[0]).replace(
    "# Bộ song ngữ thứ nhất: mô hình A (Gemma 4) dịch trên Kaggle, không cần GPU",
    "# Bộ song ngữ thứ hai (Tiki): mô hình A (Gemma 4) dịch Việt sang Trung trên Kaggle, không cần GPU"))
dat(C[0], thay(nguon(C[0]), "để dịch chữ Trung sang tiếng Việt", "để dịch **tiếng Việt sang tiếng Trung giản thể** (thuộc tính sản phẩm Tiki)"))
dat(C[0], thay(nguon(C[0]), "thư mục `ra_gemma/`", "thư mục `ra_gemma_tiki/`"))
dat(C[0], thay(nguon(C[0]), "Add Data**: gắn Dataset riêng tư chứa thư mục `vao_kaggle` **bản mới nhất** (4 file;",
               "Add Data**: gắn Dataset riêng tư của bộ Tiki, chứa thư mục `vao_kaggle` của `data/processed/song_ngu_2/` **bản mới nhất** (4 file, "
               "`cau_hinh_kaggle.json` phải có `huong = vi_zh`;"))
dat(C[1], thay(nguon(C[1]), 'CHE_DO = "thu"', 'HUONG = "vi_zh"                   # chiều dịch: Việt sang Trung (bộ Tiki); không đổi\nCHE_DO = "thu"'))
dat(C[2], thay(nguon(C[2]), CHON_DATASET_CU, CHON_DATASET_MOI))
dat(C[2], thay(nguon(C[2]), 'assert "--gop" in _dich and "KHOA_HONG" in _dich, "dich.py trong Dataset là bản CŨ (thiếu chế độ gộp hoặc nhiều khoá)',
               'assert "--gop" in _dich and "KHOA_HONG" in _dich and "--huong" in _dich, "dich.py trong Dataset là bản CŨ (thiếu chế độ gộp, nhiều khoá hoặc --huong vi_zh)'))
dat(C[2], thay(nguon(C[2]), '"/kaggle/working/ra_gemma" if CHE_DO == "that" else "/kaggle/working/ra_gemma_thu"',
               '"/kaggle/working/ra_gemma_tiki" if CHE_DO == "that" else "/kaggle/working/ra_gemma_tiki_thu"'))
dat(C[2], thay(nguon(C[2]), 'thư mục ra_gemma chứa api_tuy_chon/lo_*.parquet', 'thư mục ra_gemma_tiki chứa api_tuy_chon/lo_*.parquet'))
dat(C[2], thay(nguon(C[2]), 'if Path(p).parent.parent.name == "ra_gemma"', 'if Path(p).parent.parent.name == "ra_gemma_tiki"'))
dat(C[4], thay(nguon(C[4]), '"--mo_hinh", "api_tuy_chon",', '"--mo_hinh", "api_tuy_chon", "--huong", HUONG,'))
ghi(nb, "song-ngu-2-gemma-kaggle.ipynb")

# ---------------- notebook GPU (mô hình B + CometKiwi) ----------------
nb = doc("song-ngu-1-kaggle.ipynb")
C = nb["cells"]
dat(C[0], thay(nguon(C[0]), "# Bộ song ngữ thứ nhất: mô hình B dịch và CometKiwi chấm điểm trên Kaggle",
               "# Bộ song ngữ thứ hai (Tiki): mô hình B dịch Việt sang Trung và CometKiwi chấm điểm trên Kaggle"))
dat(C[0], thay(nguon(C[0]), "**dịch** chữ Trung sang tiếng Việt bằng **mô hình B**", "**dịch tiếng Việt sang tiếng Trung giản thể** (thuộc tính sản phẩm Tiki) bằng **mô hình B**"))
dat(C[0], thay(nguon(C[0]), "ở notebook riêng `song-ngu-1-gemma-kaggle`", "ở notebook riêng `song-ngu-2-gemma-kaggle`"))
dat(C[0], thay(nguon(C[0]), "Các bước khác chạy trên laptop (`scripts/song_ngu_1/`).", "Các bước khác chạy trên laptop (`scripts/song_ngu_2/`)."))
dat(C[0], thay(nguon(C[0]), "thư mục `ra/` hoặc `ra_thu/`", "thư mục `ra_tiki/` hoặc `ra_tiki_thu/`"))
dat(C[0], thay(nguon(C[0]), "Add Data: Dataset riêng tư chứa thư mục `vao_kaggle` (có `don_vi_can_dich.parquet`, `dich.py`, `cham_kiwi.py`).",
               "Add Data: Dataset riêng tư của bộ Tiki, chứa thư mục `vao_kaggle` của `data/processed/song_ngu_2/` (có `don_vi_can_dich.parquet`, `dich.py`, "
               "`cham_kiwi.py`, `cau_hinh_kaggle.json` với `huong = vi_zh`). Không gắn Dataset của bộ 1 cùng lúc."))
dat(C[1], thay(nguon(C[1]), 'CHE_DO = "thu"', 'HUONG = "vi_zh"           # chiều dịch: Việt sang Trung (bộ Tiki); không đổi\nCHE_DO = "thu"'))
dat(C[1], thay(nguon(C[1]), 'MO_HINH_B = "qwen3_30b_a3b"', 'MO_HINH_B = "qwen3_14b"'))
dat(C[2], thay(nguon(C[2]), CHON_DATASET_CU, CHON_DATASET_MOI))
dat(C[2], thay(nguon(C[2]), 'RA = Path("/kaggle/working/ra" if CHE_DO == "that" else "/kaggle/working/ra_thu")',
               'RA = Path("/kaggle/working/ra_tiki" if CHE_DO == "that" else "/kaggle/working/ra_tiki_thu")'))
dat(C[2], thay(nguon(C[2]), 'if Path(p).parent.parent.name == "ra"})   # chỉ lấy đầu ra chạy thật, không lẫn ra_thu',
               'if Path(p).parent.parent.name == "ra_tiki"})   # chỉ lấy đầu ra chạy thật, không lẫn ra_tiki_thu'))
dat(C[2], thay(nguon(C[2]), 'if Path(p).parent.parent.name in ("ra_gemma", "ra")})', 'if Path(p).parent.parent.name in ("ra_gemma_tiki", "ra_tiki")})'))
dat(C[2], thay(nguon(C[2]), "(thư mục ra_gemma)", "(thư mục ra_gemma_tiki)"))
dat(C[5], thay(nguon(C[5]), '"--mo_hinh", MO_HINH[k],\n             "--backend", BACKEND,', '"--mo_hinh", MO_HINH[k], "--huong", HUONG,\n             "--backend", BACKEND,'))
ghi(nb, "song-ngu-2-kaggle.ipynb")

for ten in ["song-ngu-2-gemma-kaggle.ipynb", "song-ngu-2-kaggle.ipynb"]:
    n = doc(ten)
    for i, c in enumerate(n["cells"]):
        if c["cell_type"] == "code":
            s = nguon(c)
            # cú pháp hợp lệ: lệnh shell (!...) thay bằng pass giữ nguyên thụt lề
            compile("\n".join((l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("!", "%")) else l for l in s.splitlines()),
                    f"{ten}[{i}]", "exec")
    print(ten, (NB / ten).stat().st_size, "byte,", len(n["cells"]), "ô")
