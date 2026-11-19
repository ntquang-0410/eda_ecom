# -*- coding: utf-8 -*-
"""3.6 Bang compute 5 mo hinh + bang hop nhat OpenNMT. Doc report, khong train.

Chay: .venv\\Scripts\\python.exe -I -X utf8 scripts/phan_bien_v2/p1_compute.py
"""
import json
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
R = GOC / "data" / "processed" / "ft_run"
RA = R / "phan_bien_v2"


def doc(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


def job(p, tap="cum", config="natural"):
    d = doc(p)
    return [x for x in d["viec"] if x.get("tap") == tap and x.get("config") == config and "loi" not in x][0]


def main():
    q4 = job("kaggle_quota_step1/quota_report.json")
    q8 = job("kaggle_quota_nlp-8b/quota_report.json")
    qe = job("kaggle_quota_nlp-ecom/quota_report.json")
    on = doc("kaggle_onmt_that/onmt_report.json")
    mz = doc("kaggle_moses_v2_zhvi/moses_report.json")
    mv = doc("kaggle_moses_v2_vizh/moses_report.json")

    def onmt_row(v):
        return {"huong": v["huong"], "dong": v["dong_train"], "token": v["token_sp_train"],
                "buoc": v["steps"], "giay": v["giay_train"]}

    rows = [
        {"mo_hinh": "Qwen3-4B", "tham_so": "4,0B (LoRA 132.120.576)",
         "pretrain": "~36.000B token [CHƯA KIỂM, báo cáo kỹ thuật Qwen3]",
         "finetune": f"{q4['token_compute']/1e6:.1f}M token (sàng lọc, 2 seed)".replace(".", ","),
         "luot": "1 lượt 4M token (2 seed)", "buoc": q4["so_buoc"], "giay": q4["giay"], "ghi_chu": "max_len 2.048, rank 16"},
        {"mo_hinh": "Qwen3-8B", "tham_so": "8,2B",
         "pretrain": "~36.000B token [CHƯA KIỂM, báo cáo kỹ thuật Qwen3]",
         "finetune": f"{q8['token_compute']/1e6:.1f}M token (sàng lọc, 2 seed)".replace(".", ","),
         "luot": "1 lượt 4M token (2 seed)", "buoc": q8["so_buoc"], "giay": q8["giay"], "ghi_chu": "cùng mẫu với 4B"},
        {"mo_hinh": "EcomGPT-7B", "tham_so": "7B (BLOOMZ-7b1-mt)",
         "pretrain": "BLOOM ~366B token [CHƯA KIỂM, báo cáo BLOOM]",
         "finetune": f"{qe['token_compute']/1e6:.1f}M token (sàng lọc, 2 seed)".replace(".", ","),
         "luot": "1 lượt 3,7M token (2 seed)", "buoc": qe["so_buoc"], "giay": qe["giay"], "ghi_chu": "tokenizer BLOOM cho ít token hơn"},
        {"mo_hinh": "OpenNMT (từ đầu)", "tham_so": "37M [theo biên bản vòng 2, chưa đối chiếu report]",
         "pretrain": "0",
         "finetune": "4,71M token SP (3,78M zh→vi + 0,93M vi→zh); 269.895 + 61.401 dòng",
         "luot": "≤20 epoch; zh→vi dừng sớm, vi→zh đủ 20", "buoc": "22.740 + 5.620", "giay": on["viec"][0]["giay_train"] + on["viec"][1]["giay_train"],
         "ghi_chu": "2 hướng, 1 seed"},
        {"mo_hinh": "Moses (phrase-based)", "tham_so": "không có tham số học sâu",
         "pretrain": "0",
         "finetune": "269.895 + 61.401 dòng; 2 biến thể (học đủ, khớp 4M)",
         "luot": "MERT 10 lượt", "buoc": "—", "giay": mz.get("giay_tong", 0) + mv.get("giay_tong", 0),
         "ghi_chu": "char-level, MERT 10 lần"},
    ]
    for x in rows:
        x["gpu_gio"] = round(x["giay"] / 3600, 2)

    hop_nhat = [
        {"cot": "Nhánh Nhật Anh (D3)", "tham_so": "37M", "lop/hidden": "4 lớp / 256", "tokenizer": "SentencePiece unigram 16k mỗi ngôn ngữ",
         "hoc": "Noam 2e-3, ≤20 epoch, dừng sớm, trung bình 5 checkpoint", "decode": "beam 5",
         "du_lieu": "dòng của bản pair", "seed": "1", "tap_do": "eval fold 0", "thuoc": "chrF"},
        {"cot": "Nhánh Quang (theo biên bản)", "tham_so": "37M", "lop/hidden": "4 lớp / 256", "tokenizer": "char-level 32k",
         "hoc": "30k step A / 22.990 B", "decode": "beam 512 + replace_unk",
         "du_lieu": "A/B (block)", "seed": "không ghi", "tap_do": "eval_F", "thuoc": "COMET-da/QE 0,349"},
    ]
    ra = {"compute": rows,
          "hop_nhat_opennmt": hop_nhat,
          "ghi_chu": "Hai tập đo và hai thước đo khác nhau, không so điểm trực tiếp; chỉ so cấu hình. "
                     "Pretraining của Qwen3/BLOOM là số công bố, chưa kiểm ngoại tuyến."}
    RA.mkdir(parents=True, exist_ok=True)
    (RA / "compute.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ra, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
