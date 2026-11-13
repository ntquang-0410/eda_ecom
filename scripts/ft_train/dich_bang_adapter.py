"""Dịch title hoặc danh sách thuộc tính bằng adapter LoRA bản cuối (Qwen3-4B + adapter), đúng cách đã dùng khi huấn luyện và đo (kaggle_train.sinh).
Cần: transformers, peft, torch có GPU (bf16 cần khoảng 9–10 GB bộ nhớ GPU). CHƯA chạy thử trên máy Nhật Anh (GPU yếu); cách dựng lời dẫn và trần sinh đã được kiểm bằng test, phần nạp
mô hình giống hệt `kaggle_train.nap_goc` + `PeftModel.from_pretrained` đã chạy trên Kaggle.

Ví dụ:
  python scripts/ft_train/dich_bang_adapter.py --co-so Qwen/Qwen3-4B --adapter data/processed/ft_run/ban_cuoi/cuoi/pair_cuoi/ep2 \
      --huong zh->vi --loai title --nguon "Elmich 不锈钢保温杯 500ml 黑色"
  python scripts/ft_train/dich_bang_adapter.py --co-so Qwen/Qwen3-4B --adapter .../pair_cuoi/ep2 --nhanh pair --huong zh->vi --loai mo_ta \
      --title-nguon "Elmich 不锈钢保温杯 500ml 黑色" --nguon $'品牌: Elmich\n容量: 500ml\n颜色: 黑色'
Quy ước: --nhanh pair = mỗi thuộc tính một dòng 'tên: giá trị', ngăn bằng xuống dòng; --nhanh cum = các thuộc tính ngăn bằng '; '. Dùng adapter cùng nhánh với định dạng nguồn."""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dung_mau import chi_dan  # noqa: E402  (đúng chữ lời dẫn đã dùng khi huấn luyện)

REVISION_CO_SO = "1cfa9a7208912126459214e8b04321603b3df60c"      # Qwen/Qwen3-4B đã dùng


def chon_tran(n_tok_nguon):
    """Giống kaggle_train.chon_tran_sinh: 3,5 lần độ dài nguồn cộng 64, tối đa 2048 token mới."""
    return min(2048, math.ceil(3.5 * n_tok_nguon) + 64)


def dung_loi_dan(huong, loai, nhanh, nguon, title_nguon=None):
    """Nội dung 'user' của khuôn chat. Với mô tả phải có title nguồn (chỉ làm ngữ cảnh, không dịch), như khi huấn luyện."""
    if loai == "mo_ta" and not title_nguon:
        raise ValueError("loai=mo_ta can --title-nguon (title nguon, chi lam ngu canh)")
    return chi_dan(loai, nhanh, huong, nguon, title_nguon)


def nap(co_so, adapter, hop_nhat=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(co_so)
    model = AutoModelForCausalLM.from_pretrained(co_so, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, adapter)
    if hop_nhat:                                   # gộp LoRA vào trọng số để suy luận nhanh hơn; không cần peft khi dùng
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def dich(model, tok, huong, loai, nhanh, nguon, title_nguon=None):
    import torch
    msg = [{"role": "user", "content": dung_loi_dan(huong, loai, nhanh, nguon, title_nguon)}]
    p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
    tran = chon_tran(len(tok(nguon, add_special_tokens=False)["input_ids"]))
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=tran, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--co-so", default="Qwen/Qwen3-4B", help="mô hình nền (đã dùng revision " + REVISION_CO_SO + ")")
    ap.add_argument("--adapter", required=True, help="thư mục adapter, ví dụ .../pair_cuoi/ep2")
    ap.add_argument("--huong", required=True, choices=["zh->vi", "vi->zh"])
    ap.add_argument("--loai", required=True, choices=["title", "mo_ta"])
    ap.add_argument("--nhanh", default="pair", choices=["cum", "pair"])
    ap.add_argument("--nguon", required=True)
    ap.add_argument("--title-nguon")
    ap.add_argument("--hop-nhat", action="store_true")
    a = ap.parse_args()
    model, tok = nap(a.co_so, a.adapter, a.hop_nhat)
    print(dich(model, tok, a.huong, a.loai, a.nhanh, a.nguon, a.title_nguon))


if __name__ == "__main__":
    main()
