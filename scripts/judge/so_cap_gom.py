"""CHẾ ĐỘ GOM: nhiều mục trong MỘT lời gọi (rubric v2, chế độ một lượt) để tiết kiệm token.

Lý do (31/10/2026): 85% token vào của title/pair là phần quy tắc lặp lại ở MỖI lời gọi (~727 token quy tắc so với ~127 token nội dung).
Gom K mục vào một lời gọi thì quy tắc chỉ gửi một lần. Mọi nguyên tắc chống thiên vị giữ nguyên:
 - MÙ: lời nhắc chỉ có hướng dịch, loại văn bản, nhóm hàng, (ngữ cảnh nguồn), nguồn, Translation 1/2 của từng mục. Không có tên mô hình,
   điểm Kiwi, item_id (số thứ tự trong lời gọi chỉ là 1..K).
 - Thứ tự hai bản của TỪNG mục ngẫu nhiên theo băm cố định của item_id (SR.thu_tu_bang_bam), nên không mô hình nào được lợi vì vị trí.
 - Judge phải xét TỪNG mục độc lập, buộc chọn 1 trong 2, khai equivalent khi tương đương (chọn bản 1).
Đầu ra gọn: {"results": [{"i": 1, "d1": [...], "d2": [...], "eq": false, "w": 1, "c": "W"}, ...]}."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rubric as R  # noqa: E402
import so_cap_rubric as SR  # noqa: E402

PHIEN_BAN_GOM = "so_cap_v2+gom"
MA_LOI = {"W": "meaning", "M": "completeness", "A": "completeness", "B": "completeness", "T": "terminology", "N": "none"}   # c -> tiêu chí cũ
_CAU_MO = "You will see one SOURCE text and two candidate translations, labelled Translation 1 and Translation 2. You must decide which one is better."
_CAU_MO_GOM = ("You will see several numbered ITEMS. Each ITEM has one SOURCE text and two candidate translations, labelled Translation 1 and "
               "Translation 2. For EACH item you must decide which translation is better. Judge every ITEM independently of the others.")
_DUOI = "Reply with ONE JSON object and nothing else, with the fields in exactly this order:"


def _thay_doan_dau_va_duoi(he_thong_v2):
    """Từ HE_THONG_V2 (hoặc bản ngữ cảnh) tạo bản GOM: đổi câu mở đầu, thay khối trả lời JSON bằng định dạng nhiều mục."""
    dau = he_thong_v2.index(_CAU_MO)
    s = he_thong_v2.replace(_CAU_MO, _CAU_MO_GOM, 1).replace('set "equivalent" to true', 'set "eq" to true').replace('set "equivalent" to false', 'set "eq" to false')
    cuoi = s.index("Reply with ONE JSON object")
    khoi = (s[:cuoi] +
            'Reply with ONE JSON object and nothing else: {"results": [ ... ]} with exactly one entry per ITEM, in the same order. '
            'Each entry has the fields in exactly this order: {"i": the ITEM number, "d1": ["defects of Translation 1, at most 2, at most 6 words each"], '
            '"d2": ["defects of Translation 2, at most 2, at most 6 words each"], "eq": true or false (equivalent), "w": 1 or 2 (the winner), '
            '"c": "W|M|A|B|T|N" (the kind of the most serious defect of the losing translation: W wrong, M missing, A added, B broken, '
            'T term, N none)}.')
    assert dau >= 0 and "ITEMS" in khoi
    return khoi


HE_THONG_GOM = _thay_doan_dau_va_duoi(SR.HE_THONG_V2)
HE_THONG_GOM_NGU_CANH = _thay_doan_dau_va_duoi(SR.HE_THONG_V2_NGU_CANH)


def he_thong_gom(co_ngu_canh):
    return HE_THONG_GOM_NGU_CANH if co_ngu_canh else HE_THONG_GOM


def dung_lan_goi(nhom):
    """nhom: các mục CÙNG hướng dịch và CÙNG chế độ ngữ cảnh. Trả (lời nhắc người dùng, [ánh xạ vị trí->mô hình của từng mục])."""
    huong, ten_nguon, ten_dich = SR.HUONG[nhom[0]["huong"]]
    dong = [huong, ""]
    anh_xa = []
    for k, muc in enumerate(nhom, 1):
        tt = SR.thu_tu_bang_bam(muc["item_id"])
        v1, v2 = SR.THU_TU[tt]
        dong.append(f"ITEM {k} | text type: {SR.LOAI[muc['kind']]} | product category: {muc['category']}")
        if SR.co_ngu_canh(muc):
            dong.append(f"PRODUCT CONTEXT ({ten_nguon}, for understanding only):\n{muc['ngu_canh']}")
        dong += [f"SOURCE ({ten_nguon}): {muc['nguon']}", f"Translation 1 ({ten_dich}): {muc[v1]}",
                 f"Translation 2 ({ten_dich}): {muc[v2]}", ""]
        anh_xa.append({1: "gemma" if v1 == "cand_gemma" else "qwen", 2: "gemma" if v2 == "cand_gemma" else "qwen", "thu_tu": tt})
    return "\n".join(dong).rstrip(), anh_xa


def doc_lan_goi(van_ban):
    """Đọc câu trả lời nhiều mục. Trả {số mục: {w, eq, c, d1, d2}} chỉ gồm các mục HỢP LỆ (w là 1 hoặc 2)."""
    d = R._lay_json(van_ban)
    ds = d.get("results")
    if not isinstance(ds, list):
        raise ValueError("thieu_results")
    out = {}
    for e in ds:
        if not isinstance(e, dict):
            continue
        try:
            i = int(e.get("i"))
        except (TypeError, ValueError):
            continue
        w = e.get("w")
        if isinstance(w, str):
            w = w.strip()
            w = int(w) if w in ("1", "2") else w
        if w not in (1, 2) or i in out:
            continue
        eq = e.get("eq", False)
        if isinstance(eq, str):
            eq = eq.strip().lower() == "true"
        c = e.get("c", "N")
        out[i] = {"w": w, "eq": bool(eq), "c": c if c in MA_LOI else "N",
                  "d1": [str(x)[:60] for x in (e.get("d1") or [])][:3] if isinstance(e.get("d1"), list) else [],
                  "d2": [str(x)[:60] for x in (e.get("d2") or [])][:3] if isinstance(e.get("d2"), list) else []}
    return out


def chia_gom(muc_list, k):
    """Chia các mục (giữ nguyên thứ tự ưu tiên) thành nhóm <= k mục, mỗi nhóm cùng hướng dịch và cùng chế độ ngữ cảnh.
    Mục có ngữ cảnh dài nên nhóm nhỏ hơn: tối đa max(3, k // 2)."""
    nhom = {}
    for m in muc_list:
        nhom.setdefault((m["huong"], SR.co_ngu_canh(m)), []).append(m)
    out = []
    for (_, co_nc), ds in nhom.items():
        kk = max(3, k // 2) if co_nc else k
        out += [ds[i:i + kk] for i in range(0, len(ds), kk)]
    out.sort(key=lambda g: min(m.get("_thu_tu", 0) for m in g))            # giữ thứ tự ưu tiên giữa các nhóm
    return out


def cham_gom(nhom, khoa, goi, max_token=None):
    """Chấm một nhóm bằng MỘT lời gọi. Trả [(mục, kết quả)] theo thứ tự của nhóm; mục nào thiếu/hỏng thì trạng thái 'loi' (chạy tiếp sẽ làm lại)."""
    co_nc = SR.co_ngu_canh(nhom[0])
    prompt, anh_xa = dung_lan_goi(nhom)
    tn = [{"role": "system", "content": he_thong_gom(co_nc)}, {"role": "user", "content": prompt}]
    nd, v, r, kt, phu = goi(khoa, tn, max_token or min(4000, 90 * len(nhom) + 300))
    n = len(nhom)
    chung = {"tokens_vao": v / n, "tokens_ra": r / n, "giay_goi": round((phu.get("giay") or 0) / n, 2),
             "nguon_tinh_phi": phu.get("nguon"), "chi_phi_usd": (phu.get("chi_phi") or 0.0) / n,
             "list_cost_usd": (phu.get("list_cost") or 0.0) / n, "allowance_usd": (phu.get("allowance") or 0.0) / n}
    if kt == "length":
        return [(m, {"trang_thai": "loi", "ly_do": "bi_cat_het_token", **chung}) for m in nhom]
    try:
        kq = doc_lan_goi(nd)
    except (ValueError, json.JSONDecodeError) as e:
        return [(m, {"trang_thai": "loi", "ly_do": f"doc_loi:{e}", "tra_loi_tho": nd[:200], **chung}) for m in nhom]
    out = []
    for k, m in enumerate(nhom, 1):
        e = kq.get(k)
        if e is None:
            out.append((m, {"trang_thai": "loi", "ly_do": "thieu_trong_lan_goi", **chung}))
            continue
        chon = anh_xa[k - 1][e["w"]]
        ban = {"thu_tu": anh_xa[k - 1]["thu_tu"], "vi_tri_chon": e["w"], "chon": chon, "tieu_chi": MA_LOI[e["c"]],
               "ly_do": f"d1={e['d1']}; d2={e['d2']}", "tuong_duong": e["eq"], "so_loi_1": len(e["d1"]), "so_loi_2": len(e["d2"])}
        out.append((m, {"trang_thai": "ok", "ket_qua": chon, "luot": [ban], "gom": n, **chung}))
    return out
