# -*- coding: utf-8 -*-
"""Dựng lời dẫn đúng chữ lúc huấn luyện (tái dùng dung_mau.chi_dan, không chép lại) và khuôn chat Qwen3 tắt suy luận.
dung_mau.py và chung.py nằm ở scripts/ft_train trong repo; trên Kaggle/Colab đặt cạnh file này."""
import math
import sys
from pathlib import Path

_QUA = Path(__file__).resolve()
for _p in (_QUA.parent, _QUA.parents[2] / "scripts" / "ft_train"):
    if (_p / "dung_mau.py").exists():
        sys.path.insert(0, str(_p))
        break
from dung_mau import chi_dan  # noqa: E402

KHUON = "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"     # trùng apply_chat_template(enable_thinking=False), đã kiểm
DUNG = ("<|im_end|>", "<|endoftext|>")


def khuon(u):
    return KHUON.format(u=u)


def loi_dan_title(zh):
    return chi_dan("title", "pair", "zh->vi", zh, zh)


def loi_dan_khoi(title_zh, cap):
    """cap: [(ten, gia_tri), ...] -> khối pair, mỗi thuộc tính một dòng 'tên: giá trị'."""
    khoi = "\n".join(f"{t}: {g}" for t, g in cap)
    return chi_dan("mo_ta", "pair", "zh->vi", khoi, title_zh), khoi


def chon_tran(nguon):
    """Trần token mới: 3,5 x độ dài nguồn + 64, tối đa 2048 (như dich_bang_adapter.chon_tran; số ký tự >= số token nên không chặt hơn)."""
    return min(2048, math.ceil(3.5 * len(nguon)) + 64)
