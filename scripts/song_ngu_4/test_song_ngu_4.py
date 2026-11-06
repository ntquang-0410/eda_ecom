# -*- coding: utf-8 -*-
"""Thử bộ 4 (1688 Trung -> Việt, hiệu đính cả cụm) bằng dữ liệu giả. Không cần GPU. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe -X utf8 scripts\\song_ngu_4\\test_song_ngu_4.py
"""
import json
import subprocess
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_1" / "kaggle"))
sys.path.insert(0, str(ROOT / "src" / "song_ngu_4" / "kaggle"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_4 import chuan_bi as CB  # noqa: E402
from song_ngu_4 import ket_qua as KQ  # noqa: E402
import dich_cum_zv as DC  # noqa: E402

DAT, TONG = 0, 0


def kiem(ten, dung):
    global DAT, TONG
    TONG += 1
    DAT += bool(dung)
    print(("OK   " if dung else "SAI  ") + ten)


# ---------- tên sản phẩm ----------
kiem("tên sản phẩm: xoá tên shop, số điện thoại, URL", CB.lam_tieu_de("温州栩栩家居 收纳箱 13812345678 www.abc.com 大容量", "温州栩栩家居")
     == "收纳箱 大容量")
kiem("tên sản phẩm: xoá số dài (QQ, mã liên hệ)", CB.lam_tieu_de("收纳箱QQ12345678", None) == "收纳箱QQ")
kiem("tên sản phẩm: cắt ở MAX_TIEU_DE", len(CB.lam_tieu_de("箱" * 500, None)) == CB.cau_hinh.MAX_TIEU_DE if hasattr(CB, "cau_hinh")
     else len(CB.lam_tieu_de("箱" * 500, None)) == CB.C.MAX_TIEU_DE)

# ---------- mục và chia phần ----------
R = namedtuple("R", "n_gui v_gui n_khoa n_nhap v_khoa v_nhap")
m, c = CB.muc_cua(R("品牌", "贝来", "Thương hiệu", None, "Beilai", None), 1)
kiem("mục: tên và giá trị khoá thì không phải xuất gì", m == {"i": 1, "n": "品牌", "v": "贝来", "n_vi": "Thương hiệu", "v_vi": "Beilai"} and c == [])
m, c = CB.muc_cua(R("颜色", "透明", None, "Màu sắc", None, "Minh bạch"), 2)
kiem("mục: nháp tên và giá trị, phải xuất n và v", m["draft_n"] == "Màu sắc" and m["draft_v"] == "Minh bạch" and c == ["n", "v"])
m, c = CB.muc_cua(R("颜色", "透明", "Màu sắc", None, None, None), 3)
kiem("mục: không có bản giá trị thì dịch mới (không draft_v, không v_vi)", "draft_v" not in m and "v_vi" not in m and c == ["v"])
m, c = CB.muc_cua(R("颜色", "红" * 400, "Màu sắc", None, "Đỏ " * 400, None), 4)
kiem("mục: giá trị khoá quá dài chỉ hiện phần đầu kèm ' …' (ngữ cảnh), không phải xuất",
     m["v"] == "红" * CB.C.MAX_KHOA_HIEN + " …" and m["v_vi"].endswith(" …") and len(m["v_vi"]) == 2 * CB.C.MAX_KHOA_HIEN + 2 and c == [])
m, _ = CB.muc_cua(R("颜色", "红色", "Màu sắc", None, "Đỏ", None), 5)
kiem("mục: giá trị khoá ngắn giữ nguyên", m["v"] == "红色" and m["v_vi"] == "Đỏ")
muc = [{"n": "名" * 10, "v": "值" * 500, "draft_v": "x" * 1000} for _ in range(5)]      # mỗi mục trọng số 10 + 500 + 500 = 1010
p = CB.chia_phan(muc, 2000)
kiem("chia phần: không quá trọng số, không đổi thứ tự, không bỏ mục", [len(x) for x in p] == [1, 1, 1, 1, 1] and sum(p, []) == muc)
p = CB.chia_phan(muc[:1] + [{"n": "a", "v": "b"}] * 3, 2000)
kiem("chia phần: mục nhỏ gom chung", [len(x) for x in p] == [4])
kiem("tên phần khác: bỏ trùng, chỉ phần khác", CB.ten_khac([[{"n": "颜色"}], [{"n": "尺码"}, {"n": "颜色"}]], 1) == "颜色"
     and CB.ten_khac([[{"n": "a"}]], 0) == "")

# ---------- câu lệnh, bóc JSON ----------
mj = json.dumps([{"i": 1, "n": "品牌", "v": "贝来", "n_vi": "Thương hiệu", "v_vi": "Beilai"},
                 {"i": 2, "n": "颜色", "v": "透明", "n_vi": "Màu sắc", "draft_v": "Minh bạch"}], ensure_ascii=False)
msgs = DC.cau_lenh("mother_baby", "婴儿硅胶奶嘴", mj, json.dumps({"2": ["v"]}))
kiem("câu lệnh: có tên sản phẩm, danh mục, đủ mục, nói rõ phải xuất mục nào trường nào",
     "婴儿硅胶奶嘴" in msgs[1]["content"] and "mẹ và bé" in msgs[1]["content"] and '"draft_v": "Minh bạch"' in msgs[1]["content"]
     and "item 2: v" in msgs[1]["content"] and "Output exactly 1 objects" in msgs[1]["content"])
kiem("câu lệnh: dặn giữ nguyên bản nháp đúng, 斤 đổi kg, giữ dấu che", "Keep the draft EXACTLY" in DC.HE_THONG
     and "Never rewrite a correct draft" in DC.HE_THONG and "0.5 kg" in DC.HE_THONG and "[SĐT]" in DC.HE_THONG)
kiem("câu lệnh v3: có bảng thuật ngữ (拉链 khóa kéo, 防水台, 跨境, 揭阳) và ví dụ cấm sửa vì văn phong",
     all(s in DC.HE_THONG for s in ("拉链 = khóa kéo", "防水台", "跨境不出国内", "揭阳 = Yết Dương", "\"Ba năm\" to \"3 năm\""))
     and DC.PHIEN_BAN_CAU_LENH.startswith("cum-zh_vi-v4"))
_tn = DC.HE_THONG.split("Glossary")[1].split("Example input")[0]
kiem("câu lệnh v4: thêm 独立站, 低帮, 镜头, 小白鞋, 莓茶, 无印风",
     all(s in _tn for s in ("独立站 (a sales platform) = Website độc lập", "低帮 / 中帮 / 高帮", "单镜头 -> Một ống kính", "小白鞋 = Giày trắng",
                            "莓茶 = Trà dây", "无印风 = Phong cách tối giản Nhật")))
kiem("câu lệnh v4: cách dịch 跨境不出国内 không có dấu phẩy (lần thử v3 trượt lech_so_muc vì dấu phẩy) và cấm \", \" trong một mục",
     "跨境不出国内 -> Chỉ bán xuất khẩu.\n" in _tn and "không bán trong nước" not in _tn
     and "Never put \", \" inside one item" in DC.HE_THONG)
kiem("câu lệnh: phần thứ k của sản phẩm dài có tên các phần khác",
     "part 2 of 3" in DC.cau_lenh("home", "", mj, "{}", "尺码; 颜色", 1, 3)[1]["content"])
can = {"2": ["v"], "5": ["n", "v"]}
ok, ly = DC.phan_tich('```json\n[{"i": 2, "v": "Trong suốt", "c": "colour"}, {"i": 5, "n": "Cỡ", "v": "M", "c": ""}]\n```', can)
kiem("bóc: đúng, bỏ rào markdown", ok["2"] == {"c": "colour", "v": "Trong suốt"} and ok["5"]["n"] == "Cỡ" and ok["_thua"] == 0)
ok, _ = DC.phan_tich('[{"i": 1, "v": "Beilai"}, {"i": 2, "v": "Trong suốt"}, {"i": 5, "n": "Cỡ", "v": "M"}]', can)
kiem("bóc: mục khoá xuất thừa bị bỏ qua và đếm; thiếu c thì c rỗng", ok["_thua"] == 1 and ok["2"]["c"] == "")
kiem("bóc: thiếu trường -> lỗi", DC.phan_tich('[{"i": 2, "v": "a"}, {"i": 5, "v": "M"}]', can)[1] == "muc_5_thieu_n")
kiem("bóc: trường rỗng -> lỗi", DC.phan_tich('[{"i": 2, "v": "  "}, {"i": 5, "n": "a", "v": "M"}]', can)[1] == "muc_2_thieu_v")
kiem("bóc: thiếu mục -> lỗi", DC.phan_tich('[{"i": 2, "v": "a"}]', can)[1] == "thieu_muc_5")
kiem("bóc: trùng mục -> lỗi", DC.phan_tich('[{"i": 2, "v": "a"}, {"i": 2, "v": "b"}, {"i": 5, "n": "c", "v": "d"}]', can)[1] == "trung_muc_2")
kiem("bóc: không có mảng / JSON hỏng", DC.phan_tich("xin chào", can)[1] == "khong_co_mang_json" and DC.phan_tich("[{,}]", can)[1] == "json_hong")
kiem("sót Hán: chỉ xét trường phải xuất", DC.muc_sot_han({"2": {"v": "Màu 透明"}, "5": {"n": "Cỡ", "v": "M"}}, can) == ["2"])


# ---------- dịch lô với backend giả: hỏi lại một lần ----------
class Gia:
    def __init__(self, lan1, lan2):
        self.lan1, self.lan2, self.goi = lan1, lan2, []

    def sinh(self, msgs, max_new, lap):
        self.goi.append(len(msgs))
        bang = self.lan2 if len(msgs[0]) == 4 else self.lan1
        return [(bang[m[1]["content"]], "stop") for m in msgs]


lo = pd.DataFrame({"category": ["home", "home", "home"], "tieu_de": ["a", "b", "c"], "ten_khac": ["", "", ""], "phan": [0, 0, 0],
                   "so_phan": [1, 1, 1], "muc_json": [mj, mj, mj], "can_json": [json.dumps({"2": ["v"]}), json.dumps({"2": ["v"]}), "{}"],
                   "so_ky_tu_nhap": [9, 9, 0], "so_ky_tu_can": [2, 2, 0], "so_truong_can": [1, 1, 0]})
u = [DC.cau_lenh(*x)[1]["content"] for x in zip(lo.category, lo.tieu_de, lo.muc_json, lo.can_json)]
be = Gia({u[0]: '[{"i": 2, "v": "Trong suốt", "c": ""}]', u[1]: '[{"i": 2, "v": "透明", "c": ""}]'},
         {u[1]: '[{"i": 2, "v": "Trong suốt", "c": "fix"}]'})
r = DC.dich_lo(be, lo, 1.0, 4096)
kiem("dịch lô: cụm đúng không hỏi lại; cụm sót Hán hỏi lại và sửa được; cụm toàn khoá không gọi",
     r[0]["muc"]["2"]["v"] == "Trong suốt" and not r[0]["hoi_lai"] and r[1]["hoi_lai"] and r[1]["sua_duoc"]
     and r[1]["muc"]["2"]["v"] == "Trong suốt" and r[2]["ly_do_dung"] == "khong_can" and be.goi == [2, 1])
be = Gia({u[0]: "xin lỗi", u[1]: '[{"i": 2, "v": "Trong suốt"}]'}, {u[0]: "vẫn không"})
r = DC.dich_lo(be, lo, 1.0, 4096)
kiem("dịch lô: hỏi lại vẫn hỏng thì giữ trạng thái sai cấu trúc", r[0]["muc"] is None and r[0]["hoi_lai"] and not r[0]["sua_duoc"])

# ---------- cột công bố đồng bộ các bộ trước (anh chốt 25/10/2026) ----------
import pyarrow.parquet as _pq  # noqa: E402
from song_ngu_4 import hoan_tat as HT  # noqa: E402
_REL = ROOT / "data" / "release"
_c1 = _pq.ParquetFile(_REL / "description_zh_to_vi" / "description_pairs_zh_vi.parquet").schema_arrow.names
_c3 = _pq.ParquetFile(_REL / "description_vi_to_zh_cum" / "description_products_vi_zh.parquet").schema_arrow.names
kiem("cột công bố: bảng cặp bộ 4 = cột bộ 1 + nguon_ban_dich", HT.COT_CAP_CONG_BO == _c1 + ["nguon_ban_dich"])
kiem("cột công bố: bảng sản phẩm bộ 4 cùng cột bộ 3 (có so_cap_thap ở cuối), chỉ đổi thứ tự hai cột description (nguồn trước)",
     _c3[-1] == "so_cap_thap" and HT.COT_SP_CONG_BO[-1] == "so_cap_thap" and sorted(HT.COT_SP_CONG_BO) == sorted(_c3)
     and [c for c in HT.COT_SP_CONG_BO if not c.startswith("description")] == [c for c in _c3 if not c.startswith("description")]
     and HT.COT_SP_CONG_BO.index("description_zh") < HT.COT_SP_CONG_BO.index("description_vi"))

# ---------- gói phát hành bộ 4 đã dựng: schema thật khớp bộ 1 và bộ 3 đã công bố ----------
_R4 = _REL / "description_zh_to_vi_cum"
if (_R4 / "description_pairs_zh_vi_ctx.parquet").exists():
    _p4 = _pq.ParquetFile(_R4 / "description_pairs_zh_vi_ctx.parquet").schema_arrow
    _s4 = _pq.ParquetFile(_R4 / "description_products_zh_vi.parquet").schema_arrow
    _p1 = _pq.ParquetFile(_REL / "description_zh_to_vi" / "description_pairs_zh_vi.parquet").schema_arrow
    _s3 = _pq.ParquetFile(_REL / "description_vi_to_zh_cum" / "description_products_vi_zh.parquet").schema_arrow
    kiem("gói bộ 4: bảng cặp = đúng tên và kiểu cột của bộ 1 + nguon_ban_dich (string) ở cuối",
         _p4.names == _p1.names + ["nguon_ban_dich"] and all(_p4.field(n).type == _p1.field(n).type for n in _p1.names)
         and str(_p4.field("nguon_ban_dich").type) in ("string", "large_string"))
    kiem("gói bộ 4: bảng sản phẩm = cùng cột và kiểu với bộ 3 (chỉ khác thứ tự: description_zh trước description_vi)",
         sorted(_s4.names) == sorted(_s3.names) and all(_s4.field(n).type == _s3.field(n).type for n in _s3.names) and _s4.names[-1] == "so_cap_thap")
    _tom = (_R4 / "SHA256SUMS.txt").read_text(encoding="ascii").split()
    kiem("gói bộ 4: SHA256SUMS ghi đúng 3 file và khớp bản trên máy",
         len(_tom) == 6 and all(__import__("hashlib").sha256((_R4 / n).read_bytes()).hexdigest() == h for h, n in zip(_tom[0::2], _tom[1::2])))
else:
    print("(bỏ qua test gói bộ 4: chưa dựng thư mục phát hành)")

# ---------- Qwen dự phòng (anh chốt 26/10/2026): Gemma ưu tiên, Qwen chỉ thay chỗ Gemma trượt luật VÀ có nháp ----------
_G4 = ROOT / "data" / "processed" / "song_ngu_4" / "tu_kaggle_thu_gemma_v4"
_Q4 = ROOT / "data" / "processed" / "song_ngu_4" / "thu_qwen_rtx6000_offline"
if _G4.exists() and _Q4.exists():
    from song_ngu_4 import cau_hinh as _C  # noqa: E402
    _CUM = pd.read_parquet(_C.THU_MUC_RA / "vao_kaggle" / "cum_can_dich_zv.parquet", columns=["ma_cum", "mau_thu"])
    _P = pd.read_parquet(_C.THU_MUC_RA / "cap_vao.parquet")
    _P = _P[_P.ma_cum.isin(_CUM[_CUM.mau_thu].ma_cum)]
    _GA, _SA, _TA = HT.ghep(_P, _G4, cho_phep_thieu=True)
    _GB, _SB, _TB = HT.ghep(_P, _G4, cho_phep_thieu=True, thu_muc_qwen=_Q4)
    kiem("Qwen dự phòng: cách chọn AB, không dùng bản dịch mới của Qwen",
         _TB["cach_chon"] == "AB" and "dich_moi_B" not in _TB["nguon_gia_tri"] and "dich_moi_B" not in _TB["nguon_ten"])
    kiem("Qwen dự phòng: mọi cặp lấy từ Qwen có nhãn thấp và nguồn qwen_sua",
         set(_GB[_GB.nguon_ban_dich == "qwen_sua"].do_tin_cay) <= {"thap"}
         and _TB["qwen_thay_gemma_gia_tri"] <= (_GB.nguon_ban_dich == "qwen_sua").sum() <= _TB["qwen_thay_gemma_gia_tri"] + _TB["qwen_thay_gemma_ten"])
    kiem("Qwen dự phòng: Qwen giữ nháp khi Gemma trượt luật thì KHÔNG nâng lên giữ-nháp (nhãn cao)",
         _TB["nguon_gia_tri"].get("giu_nhap", 0) == _TA["nguon_gia_tri"].get("giu_nhap", 0)
         and _TB["nguon_ten"].get("giu_nhap", 0) == _TA["nguon_ten"].get("giu_nhap", 0))
    _kb = _GB[_GB.nguon_ban_dich != "qwen_sua"].set_index("pair_id")
    _ka = _GA.set_index("pair_id")
    _chung = _kb.index.intersection(_ka.index)
    kiem("Qwen dự phòng: chỗ không dùng Qwen giống hệt kết quả chỉ Gemma (Gemma luôn ưu tiên)",
         (_kb.loc[_chung].gia_tri_vi == _ka.loc[_chung].gia_tri_vi).all() and (_kb.loc[_chung].do_tin_cay == _ka.loc[_chung].do_tin_cay).all())
    kiem("Qwen dự phòng: không mất cặp nào so với chỉ Gemma", len(_GB) >= len(_GA) and set(_GA.pair_id) <= set(_GB.pair_id))
else:
    print("(bỏ qua test Qwen dự phòng: thiếu thư mục lần thử)")

# ---------- nhất quán và sửa lỗi hệ thống (kiểm toàn bộ 26/10/2026) ----------
def _dong(name_zh="颜色", n="Màu sắc", n_ng="giu_nhap", zh="黑色", v="Đen", v_ng="giu_nhap", n_nhap="Màu sắc", v_nhap="Đen"):
    return {"name_zh": name_zh, "n_cuoi_A": n, "n_nguon_A": n_ng, "n_nhap": n_nhap, "gia_tri_zh": zh, "v_gui": zh, "v_cuoi_A": v, "v_nguon_A": v_ng,
            "v_nhap": v_nhap, "nguon_gt_vao": "giu"}


_NQ = pd.DataFrame([_dong() for _ in range(60)] + [
    _dong(n="Kích thước", n_ng="sua_A"),                                                                            # 60: tên bị đổi theo giá trị
    _dong(name_zh="主要下游平台", n="Nền tảng", n_nhap="Nền tảng", zh="ebay, 独立站", v="ebay, Trạm độc lập", v_nhap="ebay, Trạm độc lập"),  # 61
    _dong(zh="厚底", v="Đế độn mũi", v_ng="sua_A", v_nhap="Đế dày"),                                                  # 62: áp nhầm bảng thuật ngữ
    _dong(zh="拉链袋, 抽屉", v="Ngăn kéo, Ngăn kéo", v_nhap="Ngăn kéo, Ngăn kéo"),                                      # 63: có 抽屉 thì giữ
    _dong(zh="独立站", v="Trạm độc lập", v_ng="khoa", v_nhap=None),                                                    # 64: bản khoá không đụng
    _dong(name_zh="货号", n="Mã hàng", n_nhap="Mã hàng", zh="日式", v="Phong cách tối giản Nhật", v_ng="sua_A", v_nhap="Kiểu Nhật"),   # 65
    _dong(zh="休闲, 无印风, 慵懒风", v="Thường ngày, Phong cách không nhãn hiệu, Phong cách lười biếng", v_ng="khoa",
          v_nhap="Thường ngày, Phong cách không nhãn hiệu, Phong cách lười biếng")])                                         # 66: giá trị DÀI giữ bộ 1
_NQ = pd.concat([_NQ, pd.DataFrame([_dong(name_zh="筒高", n="Chiều cao cổ", n_nhap="Chiều cao cổ", zh="中筒", v="Thân trung", v_nhap="Thân trung"),
                                     _dong(name_zh="筒高", n="Chiều cao cổ", n_nhap="Chiều cao cổ", zh="短筒", v="Thân ngắn", v_nhap="Thân ngắn")])], ignore_index=True)  # 67, 68
_NQ.at[66, "nguon_gt_vao"] = HT.DAI
_kq = HT.nhat_quan(_NQ, "A", KQ.doc_khoang())
kiem("nhất quán: 中筒 'Thân trung' -> 'Cổ vừa', 短筒 'Thân ngắn' -> 'Cổ ngắn' (cùng họ lỗi 'Thân thấp')",
     _NQ.at[67, "v_cuoi_A"] == "Cổ vừa" and _NQ.at[68, "v_cuoi_A"] == "Cổ ngắn")
kiem("nhất quán: giá trị dài giữ nguyên bản bộ 1 (đánh dấu khoá) vẫn được sửa lỗi 无印风 sát chữ",
     _NQ.at[66, "v_cuoi_A"] == "Thường ngày, Phong cách tối giản Nhật, Phong cách lười biếng" and _NQ.at[66, "nguon_gt_vao"] == HT.NGUON_SUA_LOI)
kiem("nhất quán: tên bị đổi theo giá trị (颜色 -> Kích thước) được trả về tên chuẩn", _NQ.at[60, "n_cuoi_A"] == "Màu sắc" and _NQ.at[60, "n_nguon_A"] == "giu_nhap"
     and _kq["ten"] == 1)
kiem("nhất quán: 独立站 'Trạm độc lập' -> 'Website độc lập', nguồn quy tắc, nhãn theo bản khoá",
     _NQ.at[61, "v_cuoi_A"] == "ebay, Website độc lập" and _NQ.at[61, "v_nguon_A"] == "khoa" and _NQ.at[61, "nguon_gt_vao"] == HT.NGUON_SUA_LOI)
kiem("nhất quán: 厚底 'Đế độn mũi' -> 'Đế dày'; 日式 'Phong cách tối giản Nhật' -> 'Kiểu Nhật'",
     _NQ.at[62, "v_cuoi_A"] == "Đế dày" and _NQ.at[65, "v_cuoi_A"] == "Kiểu Nhật")
kiem("nhất quán: 拉链 kèm 抽屉 (ngăn kéo thật) và bản khoá không bị đụng", _NQ.at[63, "v_cuoi_A"] == "Ngăn kéo, Ngăn kéo" and _NQ.at[64, "v_cuoi_A"] == "Trạm độc lập")

# ---------- luật từng trường ----------
K = {0: (1.0, 35.4), 1: (1.0, 9.0), 2: (1.17, 7.7), 3: (1.14, 5.5), 4: (1.02, 4.6)}
L = lambda zh, vi, nhap=None: KQ.luat_truong(zh, zh, vi, nhap, K)  # noqa: E731
kiem("luật: bản đúng qua", L("透明", "Trong suốt", "Minh bạch") == [])
# Mọi cách dịch trong bảng thuật ngữ phải qua đủ luật khi sửa đúng bản nháp sai của bộ 1 (lần thử v3: 跨境不出国内 có dấu phẩy trượt
# lech_so_muc; lần thử v4: "Phong cách Muji" trượt latin_moi + mat_viet). Dùng khoảng độ dài thật của bộ 1.
_KT = KQ.doc_khoang()
THUAT_NGU = [("黑色-2.2寸（跨境不出国内）, 黑色-2.4寸（跨境不出国内）", None, "Đen-2.2 inch (Chỉ bán xuất khẩu), Đen-2.4 inch (Chỉ bán xuất khẩu)"),
             ("ebay, 亚马逊, wish, 速卖通, 独立站, LAZADA", "ebay, Amazon, wish, AliExpress, Trạm độc lập, LAZADA",
              "ebay, Amazon, wish, AliExpress, Website độc lập, LAZADA"),
             ("独立站", "Trạm độc lập", "Website độc lập"), ("低帮(0-3cm)", "Thân thấp (0-3cm)", "Cổ thấp (0-3cm)"),
             ("中帮", "Thân trung", "Cổ vừa"), ("高帮", None, "Cổ cao"), ("单镜头", "Đơn thấu kính", "Một ống kính"),
             ("双镜头", "Song thấu kính", "Ống kính kép"), ("小白鞋", "Giày trắng nhỏ", "Giày trắng"), ("莓茶", "Trà dâu", "Trà dây"),
             ("无印风", "Phong cách không nhãn hiệu", "Phong cách tối giản Nhật"),
             ("拉链暗袋, 手机袋, 夹层拉链袋", "Ngăn kéo khóa ẩn, Ngăn đựng điện thoại, Ngăn kéo khóa giữa",
              "Túi khóa kéo ẩn, Ngăn đựng điện thoại, Ngăn phụ có khóa kéo"), ("拉链", "Ngăn kéo", "Khóa kéo"),
             ("松糕跟, 交叉绑带, 防水台", None, "Gót bánh mì, Dây buộc chéo, Đế độn mũi"), ("广东揭阳", "Khiết Dương, Quảng Đông", "Yết Dương, Quảng Đông"),
             ("纯色", "Màu thuần", "Màu trơn"), ("单排扣", None, "Một hàng khuy"), ("中腰", None, "Cạp vừa")]
_truot = [(z, KQ.luat_truong(z, z, v, n, _KT)) for z, n, v in THUAT_NGU if KQ.luat_truong(z, z, v, n, _KT)]
kiem(f"luật: mọi cách dịch của bảng thuật ngữ qua luật khi sửa nháp sai ({len(THUAT_NGU)} trường hợp) {_truot}", not _truot)
kiem("luật: cách dịch trong bộ kiểm có mặt trong câu lệnh", all(s in DC.HE_THONG for s in (
    "Chỉ bán xuất khẩu", "Website độc lập", "Cổ thấp / Cổ vừa / Cổ cao", "Một ống kính", "Ống kính kép", "Giày trắng (", "Trà dây",
    "Phong cách tối giản Nhật", "Túi khóa kéo ẩn", "Ngăn phụ có khóa kéo", "đế độn mũi", "Yết Dương", "Màu trơn", "Một hàng khuy",
    "Cạp vừa")))
# từ quen thêm 26/10/2026 (nhóm A+B) không bị bắt oan; cụm tiếng Anh nhóm C vẫn bị bắt
kiem("luật latin_moi: từ Việt/vay quen và tên nền tảng/hãng (nhóm A+B) không bị bắt",
     all("latin_moi" not in L(zh, vi, nhap) for zh, vi, nhap in [
         ("20寸登机箱", "20 inch Vali xách tay", "20 inch hòm xách tay"), ("双肩包", "Balo hai vai", "Túi hai vai"),
         ("藏青背心", "Áo gile xanh navy", "Áo ba lỗ xanh tím than"), ("咨询客服", "Liên hệ CSKH", "Tư vấn chăm sóc khách hàng"),
         ("马卡龙色", "Màu Macaron", "Màu bánh"), ("豆豆鞋", "Giày loafer", "Giày đậu"), ("魔术贴", "Miếng dán velcro", "Miếng dán ma thuật"),
         ("天猫", "Tmall", "Thiên Miêu"), ("速卖通", "AliExpress", "Tốc bán thông"), ("亚马逊", "Amazon", "A Mã Tốn"),
         ("络达", "Airoha", "Lạc Đạt"), ("熊果苷", "Arbutin", "Hùng quả cam")]))
kiem("luật latin_moi: cụm tiếng Anh nhóm C vẫn bị bắt (dropshipping, flagship, livestream, memory foam)",
     all("latin_moi" in L(zh, vi, nhap) for zh, vi, nhap in [
         ("一件代发", "Dropshipping", "Giao hàng một món"), ("旗舰版", "Phiên bản Flagship", "Phiên bản hàng đầu"),
         ("直播货源", "Nguồn hàng livestream", "Nguồn hàng phát trực tiếp"), ("记忆棉", "Memory foam", "Bông ghi nhớ")]))
kiem("luật: sót chữ Hán", "han_sot" in L("透明", "Trong 透明", "Minh bạch"))
kiem("luật: lệch số", "lech_so" in L("0-6个月", "0-12 tháng", "0-6 tháng"))
kiem("luật: 斤 viết cân", "jin_can" in L("2斤装", "Gói 2 cân", "Gói 1 kg"))
kiem("luật: 斤 đổi đúng ra kg thì qua", L("2斤装", "Gói 1 kg", "Gói 2 cân") == [])
kiem("luật: mất mã model", "mat_ma" in L("XL-2026型", "Mẫu", "Mẫu XL-2026"))
kiem("luật: lệch số mục danh sách", "lech_so_muc" in L("红色, 黑色, 白色", "Đỏ, Đen", "Đỏ, Đen, Trắng"))
kiem("luật: mất dấu che", "mat_dau_che" in L("联系[SĐT]", "Liên hệ", "Liên hệ [SĐT]"))
kiem("luật: lời thừa", "loi_thua" in L("透明", "Bản dịch: Trong suốt", "Trong suốt"))
kiem("luật: chèn từ tiếng Anh lạ mà nháp không có -> latin_moi", "latin_moi" in L("米白色", "Ivory", "Trắng ngà"))
kiem("luật: từ Latin đã có ở nháp hoặc ở chữ Trung thì không bắt", L("超细纤维", "Sợi microfiber siêu mịn", "Sợi microfiber") == []
     and L("Nike同款", "Kiểu dáng Nike", "Cùng kiểu Nike") == [])
kiem("luật: từ quen (cotton, kg) và chữ Việt không dấu không bắt", L("纯棉", "Cotton nguyên chất", "Bông nguyên chất") == []
     and L("透明", "Trong", "Minh bạch") == [])
kiem("luật: mất quá nửa chữ Việt có dấu so với nháp -> mat_viet", "mat_viet" in L("墨染青筠", "Mo Ran Qing Yun", "Mặc Nhiễm Thanh Quân"))
kiem("luật: dịch mới (không nháp) chèn tiếng Anh -> latin_moi", "latin_moi" in L("墨染青筠", "Ink bamboo", None))
kiem("luật: từ mượn quen (camera, kaki, socola, axit) không bắt", L("前后双录行车记录仪", "Camera hành trình ghi hình trước sau") == []
     and L("卡其色", "Kaki", None) == [] and L("巧克力味", "Vị socola", None) == [])
kiem("luật: phiên âm ĐÚNG tên riêng của nguồn không bắt, phiên âm SAI vẫn bắt",
     L("碧然堂胶原蛋白面膜", "Mặt nạ collagen Birantang", None) == [] and "latin_moi" in L("富安娜家纺四件套", "Bộ 4 món Fuganna", None))
kiem("luật: cụm tiếng Anh vẫn bắt dù có từ mượn quen", "latin_moi" in L("记忆棉", "Memory foam", "Cao su non")
     and "latin_moi" in L("旗舰店新款", "Bản flagship mới", "Mẫu mới cửa hàng chính hãng"))
kiem("luật: nháp NaN (đọc từ parquet) coi như không có nháp", L("墨染青筠", "Ink bamboo", float("nan")) == L("墨染青筠", "Ink bamboo", None)
     and KQ.chon_truong(None, float("nan"), [("A", "Trong suốt")], "透明", "透明", K)[:2] == ("Trong suốt", "dich_moi_A"))
kiem("luật: độ dài bất thường", "do_dai" in L("红", "Màu đỏ tươi rực rỡ như lửa mùa hè rất đẹp và sang trọng", "Đỏ"))

# ---------- chọn trường ----------
kiem("chọn: khoá luôn thắng", KQ.chon_truong("Beilai", None, [("A", "Bei Lai")], "贝来", "贝来", K)[:2] == ("Beilai", "khoa"))
kiem("chọn: A qua luật và khác nháp -> sua_A", KQ.chon_truong(None, "Minh bạch", [("A", "Trong suốt"), ("B", "Trong")], "透明", "透明", K)[:2]
     == ("Trong suốt", "sua_A"))
kiem("chọn: A giống nháp sau chuẩn hoá -> giữ NGUYÊN CHỮ nháp", KQ.chon_truong(None, "Trong suốt", [("A", "trong suốt ")], "透明", "透明", K)[:2]
     == ("Trong suốt", "giu_nhap"))
kiem("chọn: A trượt, B qua -> sua_B", KQ.chon_truong(None, "Minh bạch", [("A", "Trong 透明"), ("B", "Trong suốt")], "透明", "透明", K)[:2]
     == ("Trong suốt", "sua_B"))
x = KQ.chon_truong(None, "Minh bạch", [("A", "Trong 透明"), ("B", "Ivory")], "透明", "透明", K)
kiem("chọn: cả hai trượt -> lùi về nháp, ghi luật của A", x[:2] == ("Minh bạch", "lui_nhap") and x[2] == ["han_sot"])
kiem("chọn: không có bản mô hình -> giữ nháp", KQ.chon_truong(None, "Minh bạch", [("A", None)], "透明", "透明", K)[:2]
     == ("Minh bạch", "nhap_khong_co_ban_mo_hinh"))
kiem("chọn: không nháp, A qua -> dich_moi_A", KQ.chon_truong(None, None, [("A", "Trong suốt")], "透明", "透明", K)[:2]
     == ("Trong suốt", "dich_moi_A"))
kiem("chọn: không nháp, không ứng viên qua -> khong_co", KQ.chon_truong(None, None, [("A", "透明")], "透明", "透明", K)[:2] == (None, "khong_co"))

# ---------- ghép lô Kaggle giả về cặp ----------
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    P = pd.DataFrame({"pair_id": [1, 2, 3, 4], "product_id": ["p", "p", "q", "q"], "category": ["home"] * 4, "ma_cum": ["p#0", "p#0", "q#0", "q#0"],
                      "i": [1, 2, 1, 2], "name_zh": ["品牌", "颜色", "材质", "尺码"], "gia_tri_zh": ["贝来", "透明", "纯棉", "M"],
                      "n_gui": ["品牌", "颜色", "材质", "尺码"], "v_gui": ["贝来", "透明", "纯棉", "M"],
                      "n_khoa": ["Thương hiệu", "Màu sắc", None, "Kích cỡ"], "n_nhap": [None, None, "Chất liệu", None],
                      "v_khoa": ["Beilai", None, None, "M"], "v_nhap": [None, "Minh bạch", "Bông", None]})
    for k, m, kq in (("A", "api_tuy_chon", {"p#0": {"2": {"v": "Trong suốt", "c": "colour"}}, "q#0": None}),
                     ("B", "qwen3_14b", {"p#0": {"2": {"v": "Trong suốt", "c": ""}},
                                         "q#0": {"1": {"n": "Chất liệu", "v": "Cotton", "c": "x"}}})):
        (td / k / m).mkdir(parents=True)
        pd.DataFrame({"ma_cum": list(kq), "ket_qua": [json.dumps(v, ensure_ascii=False) if v else "" for v in kq.values()],
                      "trang_thai": ["ok" if v else "loi_cau_truc" for v in kq.values()]}).to_parquet(td / k / m / "lo_00000.parquet")
    X = KQ.lap_bang(P, {"A": td / "A", "B": td / "B"}, K).set_index("pair_id")
    kiem("ghép: khoá giữ khoá", X.at[1, "v_cuoi_AB"] == "Beilai" and X.at[1, "v_nguon_AB"] == "khoa")
    kiem("ghép: A sửa qua luật -> chọn A", X.at[2, "v_cuoi_AB"] == "Trong suốt" and X.at[2, "v_nguon_AB"] == "sua_A")
    kiem("ghép: cụm A sai cấu trúc -> dùng B nếu qua luật", X.at[3, "v_cuoi_AB"] == "Cotton" and X.at[3, "v_nguon_AB"] == "sua_B"
         and X.at[3, "tt_A"] == "loi_cau_truc" and X.at[3, "n_nguon_AB"] == "giu_nhap")
    kiem("ghép: chỉ A thì cụm sai cấu trúc lùi về nháp", X.at[3, "v_cuoi_A"] == "Bông" and X.at[3, "v_nguon_A"] == "nhap_khong_co_ban_mo_hinh")
    kiem("ghép: luật riêng từng mô hình có ghi", X.at[2, "v_luat_A"] == "" and X.at[1, "v_luat_A"] == "")
    t = KQ.tom_tat(X.reset_index())
    kiem("tóm tắt: đếm đúng trạng thái cụm", t["A"]["cum_theo_trang_thai"] == {"ok": 1, "loi_cau_truc": 1})

# ---------- chạy thật dich_cum_zv.py bằng backend giả (ghi lô, chạy tiếp) ----------
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    cum = pd.DataFrame({"ma_cum": ["p#0", "q#0"], "product_id": ["p", "q"], "category": ["home", "home"], "phan": [0, 0], "so_phan": [1, 1],
                        "tieu_de": ["a", "b"], "muc_json": [mj, mj], "can_json": [json.dumps({"2": ["v"]}), "{}"],
                        "pair_ids": ["[1, 2]", "[3, 4]"], "ten_khac": ["", ""], "so_muc": [2, 2], "so_truong_can": [1, 0],
                        "so_ky_tu": [6, 6], "so_ky_tu_nhap": [9, 0], "so_ky_tu_can": [2, 0], "trong_so": [10.0, 5.0], "mau_thu": [True, False]})
    cum.to_parquet(td / "cum.parquet")
    import shutil
    shutil.copyfile(ROOT / "src" / "song_ngu_1" / "kaggle" / "dich.py", td / "dich.py")      # như trong Dataset Kaggle: hai file cùng thư mục
    shutil.copyfile(ROOT / "src" / "song_ngu_4" / "kaggle" / "dich_cum_zv.py", td / "dich_cum_zv.py")
    lenh = [sys.executable, str(td / "dich_cum_zv.py"), "--vao", str(td / "cum.parquet"), "--ra",
            str(td / "ra"), "--mo_hinh", "qwen3_14b", "--backend", "gia"]
    r1 = subprocess.run(lenh, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lo1 = pd.concat([pd.read_parquet(f) for f in (td / "ra" / "qwen3_14b").glob("lo_*.parquet")])
    r2 = subprocess.run(lenh, capture_output=True, text=True, encoding="utf-8", errors="replace")
    the = json.loads((td / "ra" / "the_qwen3_14b.json").read_text(encoding="utf-8"))
    kiem("chạy script: ghi lô đủ 2 cụm (backend giả trả chữ thường -> sai cấu trúc), cụm toàn khoá ok",
         r1.returncode == 0 and len(lo1) == 2 and set(lo1.trang_thai) == {"loi_cau_truc", "ok"})
    kiem("chạy script: lần hai không dịch lại", r2.returncode == 0 and the.get("ghi_chu") == "đã đủ")
    r3 = subprocess.run(lenh + ["--chi_mau_thu", "--ra", str(td / "ra2")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    lo3 = pd.concat([pd.read_parquet(f) for f in (td / "ra2" / "qwen3_14b").glob("lo_*.parquet")])
    kiem("chạy script: --chi_mau_thu chỉ dịch cụm mẫu", r3.returncode == 0 and list(lo3.ma_cum) == ["p#0"])

print(f"\n{DAT}/{TONG} đạt")
sys.exit(0 if DAT == TONG else 1)
