# -*- coding: utf-8 -*-
"""Metric của bài tổng kết VLSP (arXiv 2501.08621) và bài JNLP (2308.10482): SacreBLEU, BLEU kiểu mteval-v13a, TER. Cạnh chrF của cham_diem.py, không thay nó.

Cách tính (Nhật Anh chốt 13/11/2026): corpus score theo từng nhóm hướng/loại, phân biệt hoa thường; đích tiếng Việt tách `13a`, đích tiếng Trung tách `zh` (từng chữ Hán),
đúng chữ ký của bài 2308.10482 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.zh / tok.13a).
- sacrebleu: BLEU mặc định (làm mượt exp).
- bleu_mteval: BLEU không làm mượt, như `mteval-v13a.pl -c` (ở mức corpus gần như trùng sacrebleu).
- ter: TER phân biệt hoa thường. Đích Trung: tách từng chữ Hán thành một từ trước khi chấm (cat_chu_han), vì TER tách từ theo khoảng trắng và bản dịch/tham chiếu
  có thể khác nhau chỗ khoảng trắng (đã gặp: Moses viết "保修？:否", tham chiếu "保修？: 否" làm TER tính sai 100%).
Nhận bảng từng mẫu của cham_diem.cham(): cột huong, loai, hyp_cham (bản dịch đã cắt lời dẫn, cùng bản dịch dùng cho chrF), _ref (tham chiếu).
"""
import sacrebleu
import re

from sacrebleu.metrics import BLEU, TER

NHOM = ["zh->vi/title", "zh->vi/mo_ta", "vi->zh/title", "vi->zh/mo_ta"]
METRIC = ("sacrebleu", "bleu_mteval", "ter")
W_TITLE_ZHVI = 0.10        # cùng hằng với chrF (cham_co_dien.W_TITLE_ZHVI)


def _tach(huong):
    return "13a" if huong.endswith("vi") else "zh"


_HAN = re.compile(r"([㐀-䶿一-鿿豈-﫿])")


def cat_chu_han(s):
    """Tách từng chữ Hán thành một từ (khoảng trắng hai bên), giữ nguyên các chuỗi không phải chữ Hán; gộp khoảng trắng thừa.
    Bài VLSP làm đúng như vậy ('Chinese texts were segmented at character level, keeping non-Chinese strings as they are') trước khi chấm.
    Cần cho TER vì TER tách từ theo khoảng trắng: bản dịch viết liền ('保修？:否') và tham chiếu có khoảng trắng ('保修？: 否') sẽ bị tính sai nếu không tách chữ.
    sacrebleu BLEU đã tự tách chữ Hán (tok zh) nên không cần bước này."""
    return " ".join(_HAN.sub(r" \1 ", s).split())


def tao_metric(huong):
    """Ba bộ chấm cho một hướng (hướng quyết định bộ tách chữ của BLEU)."""
    tk = _tach(huong)
    return {"sacrebleu": BLEU(tokenize=tk), "bleu_mteval": BLEU(tokenize=tk, smooth_method="none"),
            "ter": TER(case_sensitive=True)}       # TER mặc định của sacrebleu không phân biệt hoa thường; đây phân biệt như BLEU. Đích Trung: đã tách chữ Hán ở diem_nhom


def diem_nhom(huong, hyps, refs):
    """Điểm corpus của một nhóm; None nếu nhóm rỗng. hyp rỗng vẫn được tính (như chrF). Đích Trung: TER chấm trên văn bản đã tách từng chữ Hán."""
    if len(hyps) == 0:
        return None
    hyps, refs = [str(h) for h in hyps], [str(r) for r in refs]
    ra = {}
    for k, m in tao_metric(huong).items():
        h, r = (([cat_chu_han(x) for x in hyps], [cat_chu_han(x) for x in refs]) if k == "ter" and huong.endswith("zh") else (hyps, refs))
        ra[k] = float(m.corpus_score(h, [r]).score)
    return ra


def chu_ky():
    """Chữ ký sacrebleu của từng cấu hình, để ghi vào kết quả và tái lập."""
    ra = {"sacrebleu_phien_ban": sacrebleu.__version__}
    for huong in ("zh->vi", "vi->zh"):
        for k, m in tao_metric(huong).items():
            m.corpus_score(["a b c d"], [["a b c d"]])       # sacrebleu chỉ lập chữ ký sau khi đã chấm ít nhất một lần (cần số tham chiếu)
            ra[f"{huong}/{k}"] = str(m.get_signature())
    return ra


def diem_bleu_ter(df):
    """df: bảng từng mẫu của cham_diem.cham(). Trả {nhom: {sacrebleu, bleu_mteval, ter} | None} cho 4 nhóm."""
    ra = {}
    for nhom in NHOM:
        huong, loai = nhom.split("/")
        d = df[(df["huong"] == huong) & (df["loai"] == loai)]
        ra[nhom] = diem_nhom(huong, d["hyp_cham"].tolist(), d["_ref"].tolist())
    return ra


def chinh_macro(theo_nhom, metric, w=W_TITLE_ZHVI):
    """(chinh, macro) của một metric: chinh = w·title + (1-w)·mô tả ở zh→vi; macro = trung bình đều hai hướng (mỗi hướng = trung bình đều title và mô tả).
    Thiếu nhóm nào thì trả None cho số cần nhóm đó."""
    g = {k: (v or {}).get(metric) for k, v in theo_nhom.items()}
    t, m = g.get("zh->vi/title"), g.get("zh->vi/mo_ta")
    chinh = None if t is None or m is None else w * t + (1 - w) * m
    if any(g.get(k) is None for k in NHOM):
        return chinh, None
    return chinh, ((g["zh->vi/title"] + g["zh->vi/mo_ta"]) / 2 + (g["vi->zh/title"] + g["vi->zh/mo_ta"]) / 2) / 2


# ---------------------------------------------------------------- khoảng tin cậy (bootstrap theo sản phẩm)
# BLEU và TER cộng dồn được: điểm corpus tính từ tổng thống kê từng câu. Nên chỉ tính thống kê từng câu MỘT lần, gộp theo sản phẩm, rồi mỗi lần rút mẫu
# chỉ cần nhân trọng số rút (số lần mỗi sản phẩm được chọn) với ma trận (sản phẩm x thống kê). Cùng trọng số cho mọi mô hình nên so theo cặp được.
_BT = ("sacrebleu", "ter")        # bleu_mteval trùng sacrebleu ở mức corpus nên không tính khoảng tin cậy riêng


def thong_ke_df(df, pids):
    """df: bảng từng mẫu của cham_diem.cham() (cột id, huong, loai, hyp_cham, _ref). pids: danh sách sản phẩm chung cho mọi mô hình (cùng thứ tự).
    Trả {nhom: {metric: ma trận (so_san_pham x so_thong_ke)}}; sản phẩm không có mặt trong nhóm thì hàng toàn 0."""
    import numpy as np
    pid = df["id"].astype(str).str.split("_").str[0]
    idx = {p: i for i, p in enumerate(pids)}
    ra = {}
    for nhom in NHOM:
        huong, loai = nhom.split("/")
        d = df[(df["huong"] == huong) & (df["loai"] == loai)]
        hyps, refs = d["hyp_cham"].astype(str).tolist(), d["_ref"].astype(str).tolist()
        ms = tao_metric(huong)
        th, tr = ([cat_chu_han(x) for x in hyps], [cat_chu_han(x) for x in refs]) if huong.endswith("zh") else (hyps, refs)
        st = {"sacrebleu": np.array(ms["sacrebleu"]._extract_corpus_statistics(hyps, [refs]), dtype=float),
              "ter": np.array(ms["ter"]._extract_corpus_statistics(th, [tr]), dtype=float)}
        hang = [idx[p] for p in pid.loc[d.index]]
        P = {}
        for k, v in st.items():
            m = np.zeros((len(pids), v.shape[1]))
            np.add.at(m, hang, v)
            P[k] = m
        ra[nhom] = P
    return ra


def diem_boot(tk_cac_lan, w):
    """tk_cac_lan: list thong_ke_df của các seed của MỘT mô hình; w: (B, so_san_pham) số lần rút. Trả {(nhom, metric): mảng (B,)} = trung bình qua seed,
    đúng cách bảng chính lấy trung bình điểm corpus qua seed (w toàn 1 thì ra đúng điểm bảng chính)."""
    import numpy as np
    ra = {}
    for nhom in NHOM:
        ms = tao_metric(nhom.split("/")[0])
        for k in _BT:
            acc = 0.0
            for tk in tk_cac_lan:
                S = w @ tk[nhom][k]
                acc = acc + np.array([float(ms[k]._compute_score_from_stats(list(r)).score) for r in S])
            ra[(nhom, k)] = acc / len(tk_cac_lan)
    return ra


def chinh_macro_boot(ra, metric, w_title=W_TITLE_ZHVI):
    """Từ diem_boot: (chinh, macro) mỗi cái là mảng (B,), cùng công thức chinh_macro."""
    g = lambda n: ra[(n, metric)]
    chinh = w_title * g("zh->vi/title") + (1 - w_title) * g("zh->vi/mo_ta")
    macro = ((g("zh->vi/title") + g("zh->vi/mo_ta")) / 2 + (g("vi->zh/title") + g("vi->zh/mo_ta")) / 2) / 2
    return chinh, macro


def khoang(x, q=(2.5, 97.5)):
    import numpy as np
    return [float(np.percentile(x, q[0])), float(np.percentile(x, q[1]))]


def trung_binh_lan(cac_lan):
    """Trung bình điểm corpus qua các lần chạy (seed); cac_lan: list các dict theo_nhom."""
    ra = {}
    for nhom in NHOM:
        vs = [l[nhom] for l in cac_lan if l.get(nhom)]
        ra[nhom] = {k: sum(v[k] for v in vs) / len(vs) for k in METRIC} if vs else None
    return ra
