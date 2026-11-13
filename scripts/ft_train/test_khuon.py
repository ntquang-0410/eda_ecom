# -*- coding: utf-8 -*-
"""Kiem thu phan khuon loi dan / ma EOS / che nhan cho nhieu mo hinh (kaggle_quota.tao_khuon, ma_hoa, thong_ke_dai). Khong can GPU, khong can mo hinh that:
dung tokenizer gia (moi ky tu la mot token, '</s>' va '<|im_end|>' la token dac biet). Chay: python test_khuon.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_quota as Q  # noqa: E402

DAC_BIET = {"</s>": 2, "<|im_end|>": 3}


class TokGia:
    eos_token = "</s>"
    unk_token_id = 0
    pad_token_id = 1

    def __call__(self, chuoi, add_special_tokens=False):
        ra = []
        for s in chuoi:
            ids, i = [], 0
            while i < len(s):
                hit = next((t for t in DAC_BIET if s.startswith(t, i)), None)
                if hit:
                    ids.append(DAC_BIET[hit])
                    i += len(hit)
                else:
                    ids.append(1000 + ord(s[i]))
                    i += 1
            ra.append(ids)
        return {"input_ids": ra}

    def convert_tokens_to_ids(self, t):
        return DAC_BIET.get(t, 0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def mau_gia(loai="title"):
    chi_dan = "Translate this product title from Chinese to Vietnamese. Output only the translation."
    nguon, dich = "耳机", "Tai nghe"
    return {"id": "x1", "chi_dan": chi_dan, "nguon": nguon, "loai": loai, "group": "1688_title", "fold5": 1,
            "messages": [{"role": "user", "content": chi_dan + "\n\n" + nguon}, {"role": "assistant", "content": dich}]}


def main():
    tok = TokGia()
    m = mau_gia()
    # 1) khuon qwen: giu nguyen khuon cu
    kq = Q.tao_khuon(tok, "qwen")
    assert kq["duoi"] == "<|im_end|>\n", kq["duoi"]
    assert kq["lam"](m).startswith("<|im_start|>user\n") and kq["lam"](m).endswith("</think>\n\n")
    # 2) khuon ecomgpt: Alpaca co muc Input, bat dau bang cau chuan, ket thuc bang '### Response:', ban dich ket thuc bang </s>
    ke = Q.tao_khuon(tok, "ecomgpt")
    p = ke["lam"](m)
    assert p.startswith("Below is an instruction that describes a task, paired with an input"), p[:80]
    assert "### Instruction:\n" + m["chi_dan"] + "\n\n### Input:\n耳机\n\n### Response:" in p, p
    assert p.endswith("### Response:")
    assert ke["duoi"] == "</s>"
    # 3) ma hoa: nhan che dung phan loi dan, phan dich giu nguyen, ket thuc dung EOS
    ra = Q.ma_hoa([m], tok, cap=4096, khuon=ke)[0]
    n_p = len(tok([p])["input_ids"][0])
    assert ra["labels"][:n_p] == [-100] * n_p
    assert ra["labels"][n_p:] == ra["ids"][n_p:]
    assert ra["ids"][-1] == 2, ra["ids"][-3:]
    assert ra["n_tgt"] == len("Tai nghe") + 1
    # 4) khuon Qwen mac dinh (khuon=None) giong khuon qwen tuong minh
    r0 = Q.ma_hoa([m], tok, cap=4096)[0]
    r1 = Q.ma_hoa([m], tok, cap=4096, khuon=kq)[0]
    assert r0 == r1
    # 5) vuot cap: ma_hoa bao loi ro; thong_ke_dai chi dem, khong loi
    try:
        Q.ma_hoa([m], tok, cap=5, khuon=ke)
        raise AssertionError("phai bao loi VUOT_CAP")
    except RuntimeError as e:
        assert "VUOT_CAP_KHI_MA_HOA" in str(e)
    tk = Q.thong_ke_dai([m, m], tok, 5, ke, "pair")
    assert tk["n"] == 2 and tk["n_vuot_cap"] == 2 and tk["ti_le_mau_ket_thuc_dung_eos"] == 1.0, tk
    assert "title" in tk["vi_du"]
    # 6) khuon la phai bao loi
    try:
        Q.tao_khuon(tok, "khong_co")
        raise AssertionError("phai bao loi KHUON_LA")
    except RuntimeError as e:
        assert "KHUON_LA" in str(e)
    print("TAT CA DAT (test_khuon)")


if __name__ == "__main__":
    main()
