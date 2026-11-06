# -*- coding: utf-8 -*-
"""Thử pipeline bộ song ngữ 3 (dịch cả cụm thuộc tính theo sản phẩm) bằng dữ liệu giả: chia phần, rút mẫu, câu lệnh, bóc JSON,
hỏi lại, chạy dich_cum.py với backend giả, tách cụm về cặp và so với bộ 2. Không đọc dữ liệu thật, không cần GPU. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\test_song_ngu_3.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_1" / "kaggle"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_3" / "kaggle"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_3 import chuan_bi, ket_qua  # noqa: E402
import dich_cum as DC  # noqa: E402

DAT, TONG = 0, 0


def kiem(ten, dung):
    global DAT, TONG
    TONG += 1
    DAT += bool(dung)
    print(("OK   " if dung else "SAI  ") + ten)


# --- chia phần ---
muc = [(1, "Màu", "Đỏ"), (2, "Mô tả", "x" * 30), (3, "Size", "M"), (4, "HDSD", "y" * 100), (5, "Lưu ý", "z")]
p = chuan_bi.chia_phan(muc, max_ky_tu=40)
kiem("chia phần: không mất mục, giữ thứ tự", [m for ph in p for m in ph] == muc)
kiem("chia phần: mỗi phần không quá giới hạn, trừ mục một mình đã quá", all(sum(len(t) + len(v) for _, t, v in ph) <= 40 or len(ph) == 1 for ph in p))
kiem("chia phần: mục dài hơn giới hạn đi riêng một phần", [len(ph) for ph in p] == [2, 1, 1, 1] or any(ph == [muc[3]] for ph in p))
kiem("chia phần: sản phẩm ngắn là một phần", chuan_bi.chia_phan(muc[:2]) == [muc[:2]])
kiem("tên ở phần khác: không lấy tên của chính phần đó, một phần thì rỗng",
     "Màu" not in chuan_bi.ten_khac(p, 0) and "HDSD" in chuan_bi.ten_khac(p, 0) and chuan_bi.ten_khac([muc], 0) == "")

# --- rút mẫu ---
sp = pd.DataFrame({"product_id": [f"p{i}" for i in range(60)], "category": ["a"] * 20 + ["b"] * 20 + ["c"] * 20})
m1, m2 = chuan_bi.rut_mau(sp, n=10, hat=1), chuan_bi.rut_mau(sp, n=10, hat=1)
dem = sp[sp.product_id.isin(m1)].category.value_counts().to_dict()
kiem("rút mẫu: đúng số, chia đều danh mục (dư cho danh mục đầu), cùng hạt giống ra cùng mẫu",
     len(m1) == 10 and dem == {"a": 4, "b": 3, "c": 3} and m1 == m2)

# --- câu lệnh ---
mj = json.dumps([{"i": 1, "n": "Màu", "v": "Đỏ"}, {"i": 2, "n": "Size", "v": "M"}], ensure_ascii=False)
msgs = DC.cau_lenh_cum("food", mj)
kiem("câu lệnh: hệ thống + người dùng, có danh mục Trung, đủ JSON và số mục",
     msgs[0]["role"] == "system" and "食品" in msgs[1]["content"] and '"v": "Đỏ"' in msgs[1]["content"] and "exactly 2 items" in msgs[1]["content"])
kiem("câu lệnh: sản phẩm nhiều phần có ghi phần mấy và tên thuộc tính phần khác",
     "part 2 of 3" in DC.cau_lenh_cum("home", mj, "Chất liệu; Kích thước", 1, 3)[1]["content"])
kiem("câu lệnh: bắt dịch từ tiếng Anh thường, giữ tên Việt khi không chắc", "Free Size -> 均码" in DC.HE_THONG_CUM and "instead of guessing" in DC.HE_THONG_CUM)

# --- bóc JSON ---
ok = '[{"i": 1, "n": "颜色", "v": "红色"}, {"i": 2, "n": "尺码", "v": "M"}]'
kiem("bóc: JSON đúng", DC.phan_tich(ok, 2)[0] == [{"i": 1, "n": "颜色", "v": "红色"}, {"i": 2, "n": "尺码", "v": "M"}])
kiem("bóc: có rào ```json và chữ thừa quanh mảng vẫn đọc được", DC.phan_tich("```json\n" + ok + "\n```", 2)[0] is not None
     and DC.phan_tich("Đây là bản dịch: " + ok + " xong", 2)[0] is not None)
kiem("bóc: số thứ tự là chuỗi vẫn nhận", DC.phan_tich('[{"i": "1", "n": "颜色", "v": "红色"}]', 1)[0] is not None)
kiem("bóc: thiếu mục -> lỗi lệch số mục", DC.phan_tich('[{"i": 1, "n": "颜色", "v": "红色"}]', 2)[1].startswith("lech_so_muc"))
kiem("bóc: trùng số thứ tự -> lỗi", DC.phan_tich('[{"i": 1, "n": "a", "v": "b"}, {"i": 1, "n": "c", "v": "d"}]', 2)[1] == "trung_muc_1")
kiem("bóc: thiếu v hoặc v rỗng -> lỗi", DC.phan_tich('[{"i": 1, "n": "a"}]', 1)[0] is None and DC.phan_tich('[{"i": 1, "n": "a", "v": " "}]', 1)[0] is None)
kiem("bóc: không phải JSON -> lỗi", DC.phan_tich("颜色: 红色", 1)[1] == "khong_co_mang_json" and DC.phan_tich("[1, 2", 2)[0] is None)

# --- sót tiếng Việt ---
kiem("sót Việt: tên riêng Việt viết hoa kèm chữ Hán được giữ (Cầu Đất)", not DC.gia_tri_sot_viet("80% Cầu Đất 罗布斯塔咖啡"))
kiem("sót Việt: chữ Việt viết thường là sót", DC.gia_tri_sot_viet("红色, đỏ"))
kiem("sót Việt: không có chữ Hán là sót (ĐẶT CỌC)", DC.gia_tri_sot_viet("ĐẶT CỌC"))
kiem("sót Việt: quá 4 từ Việt là sót", DC.gia_tri_sot_viet("Một Bốn Năm Sáu Bảy 中"))
kiem("sót Việt: tên thuộc tính có bất kỳ chữ Việt nào là sót", DC.muc_sot_viet([{"i": 1, "n": "Màu 颜色", "v": "红色"}]) == [1])


# --- dịch một lô: hỏi lại ---
class Gia:
    """Backend giả: trả lời lần 1 theo bảng, lần 2 (hội thoại có 4 tin) theo bảng khác."""
    def __init__(self, lan1, lan2):
        self.lan1, self.lan2, self.goi = lan1, lan2, []

    def sinh(self, msgs, max_new, lap):
        self.goi.append(len(msgs))
        bang = self.lan2 if len(msgs[0]) == 4 else self.lan1
        return [bang[json.loads(m[1]["content"].split("\n")[-1])[0]["v"]] for m in msgs]


lo = pd.DataFrame({"category": ["food"] * 4, "ten_khac": [""] * 4, "phan": [0] * 4, "so_phan": [1] * 4, "so_ky_tu": [10] * 4, "so_muc": [1] * 4,
                   "muc_json": [json.dumps([{"i": 1, "n": "Màu", "v": v}], ensure_ascii=False) for v in ["Đỏ", "Xanh", "Vàng", "Tím"]]})
g = Gia({"Đỏ": ('[{"i": 1, "n": "颜色", "v": "红色"}]', "stop"), "Xanh": ("không phải json", "stop"),
         "Vàng": ('[{"i": 1, "n": "颜色", "v": "vàng"}]', "stop"), "Tím": ("", "loi_api")},
        {"Xanh": ('[{"i": 1, "n": "颜色", "v": "蓝色"}]', "stop"), "Vàng": ('[{"i": 1, "n": "颜色", "v": "黄色"}]', "stop")})
kq = DC.dich_lo(g, lo, 1.0, 8192)
kiem("dịch lô: bản đúng ngay lần đầu không hỏi lại", kq[0]["muc"][0]["v"] == "红色" and not kq[0]["hoi_lai"])
kiem("dịch lô: sai cấu trúc -> hỏi lại, bản sau đúng thì dùng (sửa được)", kq[1]["muc"][0]["v"] == "蓝色" and kq[1]["sua_duoc"])
kiem("dịch lô: còn sót Việt -> hỏi lại, bản sau ít sót hơn thì dùng", kq[2]["muc"][0]["v"] == "黄色" and kq[2]["sua_duoc"] and kq[2]["sot"] == [])
kiem("dịch lô: lỗi API không hỏi lại, giữ mã lỗi để lần sau dịch lại", kq[3]["ly_do_dung"] == "loi_api" and kq[3]["muc"] is None and not kq[3]["hoi_lai"])
kiem("dịch lô: chỉ một lượt hỏi lại gom các cụm cần hỏi", g.goi == [4, 2])

# --- chạy thật bước chuẩn bị + dich_cum.py (backend giả của dich.py) + tách cụm về cặp ---
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    B = pd.DataFrame({
        "pair_id": [1, 2, 3, 4, 5, 6, 7],
        "product_id": ["p1", "p1", "p1", "p2", "p2", "p3", "p3"],
        "category": ["food", "food", "food", "home", "home", "auto", "auto"],
        "vi_tri": [1, 0, 2, 0, 1, 0, 1],
        "name_vi": ["Thành phần", "Thương hiệu", "Nhà sản xuất", "Màu", "Lưu ý", "Màu", "Thương hiệu"],
        "gia_tri_vi": ["Gạo", "Vinamilk", "Công ty X, 12 Lê Lợi", "Đỏ, Xanh", "Liên hệ shopquyetnam xin cảm ơn", "Đen", "ABC Store"],
        "trang_thai_vao": ["dung", "dung", "van_loai", "dung", "dung", "dung", "dung"]})
    f_bc = td / "bang_cap.parquet"
    B.to_parquet(f_bc, index=False)
    CUM, the = chuan_bi.chay(td / "ra", f_bc, so_mau=2, in_ra=lambda *a: None)
    kiem("chuẩn bị: bỏ cặp định tuyến cất riêng và cặp lọc riêng tư (shop, thương hiệu Store) trước khi đưa vào mô hình",
         the["cap_bo_vi_van_loai"] == 1 and the["cap_bo_vi_rieng_tu"] == 2 and the["cap_dua_vao"] == 4)
    m_p1 = json.loads(CUM.set_index("product_id").at["p1", "muc_json"])
    kiem("chuẩn bị: cụm theo đúng thứ tự trên trang (vi_tri) và pair_ids khớp mục",
         [m["n"] for m in m_p1] == ["Thương hiệu", "Thành phần"] and json.loads(CUM.set_index("product_id").at["p1", "pair_ids"]) == [2, 1])
    kiem("chuẩn bị: mẫu thử đúng số sản phẩm", CUM[CUM.mau_thu].product_id.nunique() == 2)
    vao = td / "ra" / "vao_kaggle"
    kiem("chuẩn bị: thư mục Kaggle có đủ file", all((vao / f).exists() for f in ["cum_can_dich.parquet", "cau_hinh_kaggle.json", "dich.py", "dich_cum.py"]))
    lenh = [sys.executable, str(vao / "dich_cum.py"), "--vao", str(vao / "cum_can_dich.parquet"), "--ra", str(td / "kq"),
            "--mo_hinh", "qwen3_14b", "--backend", "gia", "--lo", "2"]
    r = subprocess.run(lenh, capture_output=True, text=True, encoding="utf-8")
    lo_ra = sorted((td / "kq" / "qwen3_14b").glob("lo_*.parquet"))
    KQ = pd.concat([pd.read_parquet(f) for f in lo_ra]) if lo_ra else pd.DataFrame()
    kiem("dich_cum.py chạy được với backend giả, ghi lô, đủ cụm (bản giả không phải JSON -> loi_cau_truc, có lưu câu thô)",
         r.returncode == 0 and len(KQ) == len(CUM) and set(KQ.trang_thai) == {"loi_cau_truc"} and (KQ.tho != "").all())
    r2 = subprocess.run(lenh, capture_output=True, text=True, encoding="utf-8")
    kiem("dich_cum.py chạy lại: không dịch lại cụm đã có", r2.returncode == 0 and len(list((td / "kq" / "qwen3_14b").glob("lo_*.parquet"))) == len(lo_ra))
    # tách cụm về cặp với kết quả giả đúng cấu trúc
    kq_a = pd.DataFrame({"ma_cum": CUM.ma_cum, "trang_thai": "ok",
                         "ket_qua": [json.dumps([{"i": m["i"], "n": "名" + str(m["i"]), "v": "值" if m["v"] != "Vinamilk" else "Vinamilk"}
                                                 for m in json.loads(x)], ensure_ascii=False) for x in CUM.muc_json]})
    (td / "kg" / "api_tuy_chon").mkdir(parents=True)
    kq_a.to_parquet(td / "kg" / "api_tuy_chon" / "lo_00000.parquet", index=False)
    pd.DataFrame({"pair_id": [1, 2, 4], "name_zh": ["成分", "品牌", "颜色"], "gia_tri_zh": ["值", "Vinamilk", "红色, 蓝色"],
                  "gia_tri_zh_A": [None] * 3, "gia_tri_zh_B": [None] * 3, "cach_chon": ["giong_het", "quy_tac_ten_hang", "giong_het"],
                  "do_tin_cay": ["cao"] * 3}).to_parquet(td / "bo2.parquet", index=False)
    P = ket_qua.lap_bang(CUM, td / "kg", td / "bo2.parquet")
    kiem("so sánh: mỗi cặp đưa vào một dòng, B chưa dịch thì trạng thái chua_dich", len(P) == 4 and set(P.trang_thai_B) == {"chua_dich"})
    kiem("so sánh: ghép đúng bản đơn vị bộ 2 theo pair_id; cặp không có ở bộ 2 để trống",
         P.set_index("pair_id").at[4, "gia_tri_zh_don_vi"] == "红色, 蓝色" and pd.isna(P.set_index("pair_id").at[6, "gia_tri_zh_don_vi"]))
    kiem("so sánh: đánh dấu giá trị giống bản đơn vị sau chuẩn hoá", P.set_index("pair_id").at[2, "gt_giong_don_vi_A"]
         and not P.set_index("pair_id").at[4, "gt_giong_don_vi_A"])
    kiem("so sánh: luật bộ 2 chạy trên từng cặp ('Đỏ, Xanh' -> '值' lệch số mục trượt; B chưa có -> thieu)",
         "lech_so_muc" in P.set_index("pair_id").at[4, "truot_gt_A"] and P.set_index("pair_id").at[4, "truot_gt_B"] == ["thieu"])
    tt = ket_qua.tom_tat(CUM, P)
    kiem("tóm tắt: đếm cụm theo trạng thái và cặp có bản dịch", tt["A"]["cap_co_ban_dich"] == 4 and tt["B"]["cap_co_ban_dich"] == 0
         and tt["cap_khong_co_o_bo_2"] == 1)

print(f"\n{DAT}/{TONG} ca đạt")
