# -*- coding: utf-8 -*-
"""Qwen3-8B BẢN CUỐI trên toàn pool (13/11/2026, Nhật Anh duyệt, Prompt 974-975): cùng công thức bản cuối của Qwen3-4B (07/11, Prompt 808):
rank 64, alpha 128, lr 2e-4, 2 epoch, title ~11% (cụm 10,74%, pair 10,86%, w_title tính lại), cách 1 = loại mẫu title của sản phẩm bị loc_title.py gắn cờ, giữ mẫu mô tả.
Khác 4B ở một điểm: bật gradient checkpointing (cfg "gc"), vì 8B không có chỗ cho lô 4096 token không checkpoint trên 95 GB. Mọi thứ còn lại giữ nguyên để so cân.
Không còn fold giữ lại nên không chấm chất lượng; mỗi việc chỉ sinh thử vài mẫu của tập huấn luyện để bắt hỏng. Chất lượng đo sau, trên tập test khóa.

  $env:KAGGLE_NGUOI_DUNG='honghuynanh'
  python -X utf8 chay_cuoi_8b.py khoi    # khói: hai việc nối nhau trên 320 mẫu, ở notebook nlp-8b-cuoi-cum, trạng thái thu_cuoi8b
  python -X utf8 chay_cuoi_8b.py that    # thật: cụm ở nlp-8b-cuoi-cum (trạng thái cuoi8b_cum), pair ở nlp-8b-cuoi-pair (cuoi8b_pair), song song

Không chờ xong ở đây: đẩy rồi thoát; xem bằng `kaggle kernels status`. Máy tắt thì phiên vẫn chạy. Không đẩy lại hai notebook này khi chưa tải đầu ra của lần chạy trước.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dong_goi_kaggle as G  # noqa: E402

SLUG_CUM, SLUG_PAIR = "nlp-8b-cuoi-cum", "nlp-8b-cuoi-pair"
VIEC_CUM, VIEC_PAIR = G.VIEC_CUOI[0], G.VIEC_CUOI[1]
assert VIEC_CUM["ten"] == "cum_cuoi" and VIEC_PAIR["ten"] == "pair_cuoi"


def cau_hinh(viec, ten_tt, khoi=False):
    """Dựng từ cau_hinh_cuoi của 4B (cùng ds title bị cờ, cùng ngưỡng bắt hỏng) rồi đổi danh sách việc, tên trạng thái và bật gc."""
    c = G.cau_hinh_cuoi(khoi=khoi)
    c.update(viec=viec, ten_trang_thai=ten_tt, gc=True, vram_toi_thieu_gb=80)   # RTX Pro 6000 có 95 GB; T4 (14,5 GB) thì dừng ngay
    if not khoi:
        c["gio_toi_da"] = 5.5       # ~3 giờ mỗi việc [ƯỚC]; trần chỉ để lần chạy lỗi không đốt quota
    else:
        c["gio_toi_da"] = 0.75
    return c


def main():
    if G.NGUOI_DUNG != "honghuynanh":
        raise SystemExit(f"Phải đặt KAGGLE_NGUOI_DUNG=honghuynanh (đang là {G.NGUOI_DUNG}).")
    cd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cd == "khoi":
        G.chay_b5(cau_hinh([VIEC_CUM, VIEC_PAIR], "thu_cuoi8b", khoi=True), slug=SLUG_CUM, mo_hinh_slug=G.SLUG_MO_HINH_8B)
    elif cd == "that":
        G.chay_b5(cau_hinh([VIEC_CUM], "cuoi8b_cum"), slug=SLUG_CUM, mo_hinh_slug=G.SLUG_MO_HINH_8B)
        G.chay_b5(cau_hinh([VIEC_PAIR], "cuoi8b_pair"), slug=SLUG_PAIR, mo_hinh_slug=G.SLUG_MO_HINH_8B)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
