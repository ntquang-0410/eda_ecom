# -*- coding: utf-8 -*-
"""Thử bản v2 của bộ 3 (tên sản phẩm làm ngữ cảnh + mục khoá + hiệu đính bản nháp) bằng dữ liệu giả. Không cần GPU. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_3\\test_song_ngu_3_v2.py
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
sys.path.insert(0, str(ROOT / "src" / "song_ngu_3" / "kaggle"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_3 import chuan_bi_v2 as CB  # noqa: E402
from song_ngu_3 import ket_qua_v2 as KQ2  # noqa: E402
import dich_cum as DC  # noqa: E402

DAT, TONG = 0, 0


def kiem(ten, dung):
    global DAT, TONG
    TONG += 1
    DAT += bool(dung)
    print(("OK   " if dung else "SAI  ") + ten)


# --- chuẩn hoá gam, tên sản phẩm ---
kiem("gr/gram sau số -> g", CB.chuan_gr("30gr") == "30g" and CB.chuan_gr("1Kg, 500 GRAM") == "1Kg, 500g" and CB.chuan_gr("约 100-110 gr/盒") == "约 100-110g/盒")
kiem("gr không đứng sau số thì giữ (grape, gr đơn lẻ)", CB.chuan_gr("grape 5") == "grape 5" and CB.chuan_gr("Màu gr") == "Màu gr")
kiem("tên sản phẩm: xoá tên shop trong ngoặc vuông (không phân biệt hoa thường) và số điện thoại",
     CB.lam_tieu_de("[KoSuyTu] Giá Đỡ Điện Thoại 0912345678 3 Trong 1", "kosuytu") == "Giá Đỡ Điện Thoại 3 Trong 1")
kiem("tên sản phẩm: dài thì cắt ở ranh giới từ", len(CB.lam_tieu_de("Tã " * 200, None)) <= CB.MAX_TIEU_DE + 4)

# --- mục v2: khoá và nháp ---
R = namedtuple("R", "name_vi gia_tri_vi cach_ten ten_zh_don_vi gia_tri_zh cach_chon")
m, c = CB.muc_v2(R("Thương hiệu", "Andes", "tu_dien_p6", "品牌", "Andes", "quy_tac_ten_hang"))
kiem("mục: tên trong từ điển P6 và thương hiệu đều khoá, không phải xuất gì", m.get("n_zh") == "品牌" and m.get("v_zh") == "Andes" and c == [])
m, c = CB.muc_v2(R("Màu", "Hồng", "giong_het", "颜色", "红色", "khac_chu_cung_nghia_A"))
kiem("mục: tên và giá trị máy dịch thành bản nháp, phải xuất n và v", m.get("draft_n") == "颜色" and m.get("draft_v") == "红色" and c == ["n", "v"])
m, c = CB.muc_v2(R("Màu", "Hồng", "tu_dien_p6", "颜色", None, None))
kiem("mục: cặp bộ 2 đã cất (không có bản dịch) thì không có nháp, dịch mới", "draft_v" not in m and "v_zh" not in m and c == ["v"])
m, _ = CB.muc_v2(R("Trọng lượng", "30gr", "tu_dien_p6", "重量", "30gr", "quy_tac_do_luong"))
kiem("mục: giá trị khoá tuyến số đo đổi gr -> g", m["v_zh"] == "30g")
m, _ = CB.muc_v2(R("Mã", "GR-30gr", "tu_dien_p6", "型号", "GR-30gr", "quy_tac_ma_model"))
kiem("mục: mã model khoá giữ nguyên (không đổi gr trong mã)", m["v_zh"] == "GR-30gr")

# --- câu lệnh, bóc JSON ---
mj = json.dumps([{"i": 1, "n": "Thương hiệu", "v": "Andes", "n_zh": "品牌", "v_zh": "Andes"},
                 {"i": 2, "n": "Màu", "v": "Hồng", "n_zh": "颜色", "draft_v": "红色"}], ensure_ascii=False)
msgs = DC.cau_lenh_cum_v2("auto", "Mũ bảo hiểm 3/4 Andes", mj, json.dumps({"2": ["v"]}))
kiem("câu lệnh v2: có tên sản phẩm, đủ mục, nói rõ phải xuất mục nào trường nào",
     "Mũ bảo hiểm 3/4 Andes" in msgs[1]["content"] and '"draft_v": "红色"' in msgs[1]["content"] and "item 2: v" in msgs[1]["content"])
kiem("câu lệnh v2: dặn giữ nguyên bản nháp đúng, không sửa vì văn phong", "Keep the draft EXACTLY" in DC.HE_THONG_CUM_V2
     and "Never rewrite a correct draft" in DC.HE_THONG_CUM_V2)
can = {"2": ["v"], "5": ["n", "v"]}
ok, ly = DC.phan_tich_v2('[{"i": 2, "v": "粉色", "c": "pink"}, {"i": 5, "n": "尺码", "v": "M", "c": ""}]', can)
kiem("bóc v2: đúng", ok["2"] == {"c": "pink", "v": "粉色"} and ok["5"]["n"] == "尺码" and ok["_thua"] == 0)
ok, _ = DC.phan_tich_v2('[{"i": 1, "v": "Andes"}, {"i": 2, "v": "粉色"}, {"i": 5, "n": "尺码", "v": "M"}]', can)
kiem("bóc v2: mô hình xuất thừa mục khoá -> bỏ qua, đếm thừa; thiếu c -> c rỗng", ok["_thua"] == 1 and ok["2"]["c"] == "")
kiem("bóc v2: thiếu trường bắt buộc -> lỗi", DC.phan_tich_v2('[{"i": 2, "v": "粉色"}, {"i": 5, "v": "M"}]', can)[1] == "muc_5_thieu_n")
kiem("bóc v2: thiếu mục -> lỗi", DC.phan_tich_v2('[{"i": 2, "v": "粉色"}]', can)[1] == "thieu_muc_5")
kiem("bóc v2: trùng mục -> lỗi", DC.phan_tich_v2('[{"i": 2, "v": "a"}, {"i": 2, "v": "b"}, {"i": 5, "n": "c", "v": "d"}]', can)[1] == "trung_muc_2")
kiem("sót Việt v2: chỉ xét trường phải xuất", DC.muc_sot_viet_v2({"2": {"v": "hồng"}, "5": {"n": "尺码", "v": "M"}}, can) == ["2"])


# --- dịch lô v2 với backend giả ---
class Gia:
    def __init__(self, lan1, lan2):
        self.lan1, self.lan2, self.goi = lan1, lan2, []

    def sinh(self, msgs, max_new, lap):
        self.goi.append(len(msgs))
        bang = self.lan2 if len(msgs[0]) == 4 else self.lan1
        return [bang[m[1]["content"].split("Product title (context only, do not translate): ")[1].split("\n")[0]] for m in msgs]


def mjv(v):
    return json.dumps([{"i": 1, "n": "Màu", "v": v, "n_zh": "颜色", "draft_v": "?"}], ensure_ascii=False)


lo = pd.DataFrame({"category": ["auto"] * 4, "tieu_de": ["t1", "t2", "t3", "t4"], "ten_khac": [""] * 4, "phan": [0] * 4, "so_phan": [1] * 4,
                   "muc_json": [mjv("Hồng"), mjv("Đen"), mjv("Xanh"), json.dumps([{"i": 1, "n": "x", "v": "y", "n_zh": "a", "v_zh": "b"}])],
                   "can_json": [json.dumps({"1": ["v"]})] * 3 + ["{}"], "so_ky_tu": [10] * 4, "so_ky_tu_nhap": [2] * 4, "so_truong_can": [1, 1, 1, 0]})
g = Gia({"t1": ('[{"i": 1, "v": "粉色", "c": "pink"}]', "stop"), "t2": ("hỏng", "stop"), "t3": ('[{"i": 1, "v": "vàng"}]', "stop")},
        {"t2": ('[{"i": 1, "v": "黑色", "c": ""}]', "stop"), "t3": ('[{"i": 1, "v": "蓝色", "c": "untranslated"}]', "stop")})
kq = DC.dich_lo_v2(g, lo, 1.0, 8192)
kiem("dịch lô v2: cụm toàn khoá không gọi mô hình", kq[3]["ly_do_dung"] == "khong_can" and g.goi[0] == 3)
kiem("dịch lô v2: đúng ngay thì giữ, sai cấu trúc hỏi lại sửa được, sót Việt hỏi lại sửa được",
     kq[0]["muc"]["1"]["v"] == "粉色" and not kq[0]["hoi_lai"] and kq[1]["muc"]["1"]["v"] == "黑色" and kq[1]["sua_duoc"]
     and kq[2]["muc"]["1"]["v"] == "蓝色" and kq[2]["sot"] == [])

# --- chọn bản cuối: luật chặt ---
kiem("chọn: mục khoá luôn lấy bản khoá dù mô hình xuất khác", KQ2.chon_truong("品牌", None, [("A", "牌子")], "Thương hiệu", "ten")[:2] == ("品牌", "khoa"))
kiem("chọn: mô hình giữ đúng nháp -> giu_nhap", KQ2.chon_truong(None, "红色, 蓝色", [("A", "红色，蓝色")], "Đỏ, Xanh", "gia_tri")[1] == "giu_nhap")
kiem("chọn: mô hình sửa và qua luật -> sua_A", KQ2.chon_truong(None, "红色", [("A", "粉色")], "Hồng", "gia_tri")[:2] == ("粉色", "sua_A"))
kiem("chọn: bản sửa trượt luật (mất số) -> lùi về nháp",
     KQ2.chon_truong(None, "30g 袋装", [("A", "袋装")], "Túi 30g", "gia_tri")[:2] == ("30g 袋装", "lui_nhap"))
kiem("chọn: A trượt, B qua luật -> lấy B", KQ2.chon_truong(None, "30g 袋装", [("A", "袋装"), ("B", "30g 包装袋")], "Túi 30g", "gia_tri")[:2]
     == ("30g 包装袋", "sua_B"))
kiem("chọn: bản sửa quay về tiếng Anh nguyên văn (mất quá nửa chữ Hán của nháp) -> lùi về nháp",
     KQ2.chon_truong(None, "银色", [("A", "Silver")], "Silver", "gia_tri")[:2] == ("银色", "lui_nhap")
     and KQ2.chon_truong(None, "水, 甘油, 香精", [("A", "Water, Glycerin, Fragrance")], "Water, Glycerin, Fragrance", "gia_tri")[1] == "lui_nhap")
kiem("chọn: bản sửa giữ đủ chữ Hán (thêm chút Latin) vẫn nhận", KQ2.chon_truong(None, "LifePO4锂电池", [("A", "磷酸铁锂电池 (LifePO4)")],
                                                                        "Pin Lithium LifePO4", "gia_tri")[1] == "sua_A")
kiem("chọn: không có nháp, mô hình trượt -> không có bản", KQ2.chon_truong(None, None, [("A", "hồng")], "Hồng", "gia_tri")[:2] == (None, "khong_co"))
kiem("chọn: cụm sai cấu trúc (không có bản mô hình) -> dùng nháp", KQ2.chon_truong(None, "红色", [("A", None)], "Đỏ", "gia_tri")[:2]
     == ("红色", "nhap_khong_co_ban_mo_hinh"))

# --- đầu cuối trên dữ liệu giả ---
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    B = pd.DataFrame({"pair_id": [1, 2, 3, 4, 5], "product_id": ["p1", "p1", "p1", "p2", "p2"], "category": ["auto"] * 3 + ["food"] * 2,
                      "vi_tri": [0, 1, 2, 0, 1], "name_vi": ["Thương hiệu", "Màu", "Nhà nhập khẩu", "Thương hiệu", "Trọng lượng"],
                      "gia_tri_vi": ["Andes", "Hồng", "Công ty X", "Ông Cua", "30gr"],
                      "trang_thai_vao": ["dung", "dung", "van_loai", "dung", "dung"],
                      "ma_don_vi_ten": ["T1", "T2", "T3", "T1", "T4"]})
    B.to_parquet(td / "bc.parquet", index=False)
    pd.DataFrame({"ma_don_vi": ["T1", "T2", "T4"], "loai": ["ten"] * 3, "cach_chon": ["tu_dien_p6", "giong_het", "tu_dien_p6"],
                  "ban": ["品牌", "颜色", "重量"]}).to_parquet(td / "dv.parquet", index=False)
    pd.DataFrame({"pair_id": [1, 2, 4, 5], "name_zh": ["品牌", "颜色", "品牌", "重量"], "gia_tri_zh": ["Andes", "红色", "Ông Cua", "30gr"],
                  "cach_chon": ["quy_tac_ten_hang", "khac_chu_cung_nghia_A", "quy_tac_ten_hang", "quy_tac_do_luong"],
                  "do_tin_cay": ["cao", "vua", "cao", "cao"]}).to_parquet(td / "b2.parquet", index=False)
    pd.DataFrame({"product_id": ["p1", "p2"], "title_vi_cleaned": ["Mũ bảo hiểm 3/4 Andes màu hồng", "Gạo Ông Cua 10kg"],
                  "shop": ["andes official", None]}).to_parquet(td / "tk.parquet", index=False)
    CUM, the = CB.chay(td / "ra", td / "bc.parquet", td / "b2.parquet", td / "dv.parquet", td / "tk.parquet", so_mau=2, in_ra=lambda *a: None)
    kiem("đầu cuối: bỏ cặp định tuyến cất riêng; p2 toàn khoá (thương hiệu + số đo đã đổi g)", the["cap"] == 4
         and CUM.set_index("product_id").at["p2", "can_json"] == "{}" and '"v_zh": "30g"' in CUM.set_index("product_id").at["p2", "muc_json"])
    kiem("đầu cuối: p1 chỉ phải xuất mục Màu (tên không có trong từ điển nên cả tên và giá trị là nháp)",
         json.loads(CUM.set_index("product_id").at["p1", "can_json"]) == {"2": ["n", "v"]})
    (td / "ra" / "vao_kaggle" / "dich.py").write_bytes((ROOT / "src" / "song_ngu_1" / "kaggle" / "dich.py").read_bytes())
    (td / "ra" / "vao_kaggle" / "dich_cum.py").write_bytes((ROOT / "src" / "song_ngu_3" / "kaggle" / "dich_cum.py").read_bytes())
    r = subprocess.run([sys.executable, str(td / "ra" / "vao_kaggle" / "dich_cum.py"), "--vao", str(td / "ra" / "vao_kaggle" / "cum_can_dich_v2.parquet"),
                        "--ra", str(td / "kq"), "--mo_hinh", "qwen3_14b", "--backend", "gia"], capture_output=True, text=True, encoding="utf-8")
    K = pd.concat([pd.read_parquet(f) for f in (td / "kq" / "qwen3_14b").glob("lo_*.parquet")])
    kiem("đầu cuối: dich_cum.py nhận v2, cụm toàn khoá không gọi mô hình, bản giả không phải JSON -> loi_cau_truc có lưu câu thô",
         r.returncode == 0 and set(zip(K.ma_cum, K.trang_thai)) == {("p1#0", "loi_cau_truc"), ("p2#0", "ok")} and "v2" in r.stdout)
    (td / "kg" / "api_tuy_chon").mkdir(parents=True)
    pd.DataFrame({"ma_cum": ["p1#0", "p2#0"], "trang_thai": ["ok", "ok"],
                  "ket_qua": [json.dumps({"2": {"n": "颜色", "v": "粉色", "c": "pink"}, "_thua": 0}), json.dumps({"_thua": 0})]}
                 ).to_parquet(td / "kg" / "api_tuy_chon" / "lo_00000.parquet", index=False)
    P = KQ2.lap_bang_v2(CUM, td / "kg", td / "b2.parquet").set_index("pair_id")
    kiem("đầu cuối: bản cuối A = khoá cho thương hiệu/tên, sửa theo ngữ cảnh cho Màu (粉色), số đo 30g",
         P.at[1, "v_cuoi_A"] == "Andes" and P.at[2, "v_cuoi_A"] == "粉色" and P.at[2, "v_nguon_A"] == "sua_A" and P.at[5, "v_cuoi_A"] == "30g")
    kiem("đầu cuối: B sai cấu trúc -> AB vẫn lấy A; tóm tắt đếm 2 giá trị khác bản đơn vị (粉色, 30g)",
         P.at[2, "v_cuoi_AB"] == "粉色" and KQ2.tom_tat_v2(CUM, P.reset_index())["chon_A"]["gia_tri_khac_don_vi"] == 2)

print(f"\n{DAT}/{TONG} ca đạt")
