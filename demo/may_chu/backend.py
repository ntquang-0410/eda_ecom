# -*- coding: utf-8 -*-
"""Hai cách sinh, cùng giao diện sinh(prompt_chat, tran) -> văn bản, greedy:
- LlamaServer: llama-server của llama.cpp (laptop; nền GGUF Q5_K_M + --lora).
- HF: transformers + peft fp16 (Kaggle/Colab T4; đúng cấu hình đã đo ở dòng thứ sáu)."""
import json
import urllib.request

from loi_dan import DUNG


class LlamaServer:
    def __init__(self, url="http://127.0.0.1:8770", ten="llama-server"):
        self.url, self.ten = url.rstrip("/"), ten

    def sinh(self, prompt, tran):
        body = {"prompt": prompt, "n_predict": int(tran), "temperature": 0, "top_k": 1, "seed": 42, "repeat_penalty": 1.0, "cache_prompt": False, "stop": list(DUNG)}
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read().decode("utf-8")).get("content", "")

    def suc_khoe(self):
        with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))


class ChuaDichSan(Exception):
    """Yêu cầu không có trong bộ đệm (chế độ chỉ-bộ-đệm không có mô hình để dịch)."""


class ChiBoDem:
    """Không có mô hình: chỉ trả bản dịch đã lưu trong bộ đệm SQLite (lấy từ phiên Kaggle trước).
    `ten` PHẢI trùng tên backend lúc dịch (khóa bộ đệm có tên mô hình), ví dụ 'kaggle-fp16-pair_cuoi'."""
    def __init__(self, ten):
        self.ten = ten

    def sinh(self, prompt, tran):
        raise ChuaDichSan()

    def suc_khoe(self):
        return {"status": "ok", "che_do": "chi_bo_dem"}


class HF:
    def __init__(self, co_so, adapter, ten="hf-fp16", dtype="fp16"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.ten = torch, ten
        self.tok = AutoTokenizer.from_pretrained(co_so)
        m = AutoModelForCausalLM.from_pretrained(co_so, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype], device_map={"": 0}, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(m, adapter).merge_and_unload().eval()       # gộp LoRA vào trọng số: sinh nhanh hơn, cùng kết quả (sai khác làm tròn fp16)
        self.gpu = torch.cuda.get_device_name(0)

    def sinh(self, prompt, tran):
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(tran), do_sample=False, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def suc_khoe(self):
        return {"status": "ok", "gpu": self.gpu}
