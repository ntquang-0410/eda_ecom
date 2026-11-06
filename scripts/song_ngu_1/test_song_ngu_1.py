# -*- coding: utf-8 -*-
"""Thử pipeline bộ song ngữ thứ nhất bằng câu tổng hợp (không đọc dữ liệu thật, không cần GPU). Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_1\\test_song_ngu_1.py
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from song_ngu_1 import chuan_bi, hoan_tat  # noqa: E402
from song_ngu_1 import cau_hinh as C  # noqa: E402
from song_ngu_1.chon_ban import chon  # noqa: E402
from song_ngu_1.dinh_tuyen import chu_thich_ma, dinh_tuyen, doc_tu_chung, doi_ngay, ten_hang, van_ban_gui  # noqa: E402
from song_ngu_1.luat_kiem import do_dai_bat_thuong, khoang_do_dai, kiem_mot  # noqa: E402
from song_ngu_1.van_ban import chrf, chuan_hoa  # noqa: E402

DAT, TONG = 0, 0


def kiem(ten, dung):
    global DAT, TONG
    TONG += 1
    DAT += bool(dung)
    print(("OK   " if dung else "SAI  ") + ten)


# --- chuẩn hoá và chrF ---
kiem("dấu thanh cũ và mới như nhau", chuan_hoa("Hòa bình, thủy tinh") == chuan_hoa("hoà bình, thuỷ tinh"))
kiem("7,5 = 7.5; 30 kg = 30kg", chuan_hoa("7,5 cm 30 kg") == chuan_hoa("7.5cm 30kg"))
kiem("chrF giống hệt = 100", chrf(chuan_hoa("Nhựa mềm"), chuan_hoa("nhựa mềm")) == 100.0)
kiem("chrF hai cách dịch khác chữ thấp (khoảng 12)", 10 < chrf(chuan_hoa("Keo mềm làm sạch xe hơi"), chuan_hoa("Gel mềm vệ sinh ô tô")) < 15)

# --- phân loại (quy tắc lấy lại từ pipeline làm sạch) ---
tc = doc_tu_chung(C.FILE_TU_CHUNG)
kiem("mã thật có chú thích tỉnh", chu_thich_ma("粤妆20161271") == "粤妆20161271 (Quảng Đông)")
kiem("mã cấp quốc gia", chu_thich_ma("国妆网备进字（沪）2024007457") == "国妆网备进字（沪）2024007457 (quốc gia, Thượng Hải)")
kiem("ngày đổi kiểu Việt", doi_ngay("2023年05月24日") == "24/05/2023")
kiem("tên hãng pinyin", ten_hang("和洁士", tc) == ("Hejieshi", "pinyin"))
kiem("tên hãng từ chung", ten_hang("其他", tc)[0] == "Khác")
kiem("tên hãng Latin chép nguyên", ten_hang("LYLYSHOP", tc) == ("LYLYSHOP", "latin"))
kiem("tên công ty không đoán", ten_hang("广州市某某贸易有限公司", tc)[0] is None)
kiem("giá trị không có chữ Hán chép nguyên", dinh_tuyen("货号", "hjs200g", "", tc) == ("chep_nguyen", "hjs200g"))
kiem("cặp mã thật theo nhãn chep_ma", dinh_tuyen("生产许可证号", "粤妆20161271", "chep_ma;", tc)[0] == "ma_that")
kiem("giá trị thường gửi mô hình", dinh_tuyen("种类", "汽车清洁软胶", "", tc) == ("mo_hinh", None))
kiem("dấu che đổi sang dạng Việt", van_ban_gui("联系[电话]") == "联系[SĐT]")


# --- luật kiểm ---
def tr(zh, vi, gui=None):
    return kiem_mot(zh, gui or van_ban_gui(zh), vi)


kiem("rỗng thì trượt", tr("软胶", "  ")[0] == ["rong"])
kiem("còn chữ Hán thì trượt", "han_sot" in tr("宝蓝色", "Màu 宝蓝色")[0])
kiem("lời giải thích thừa thì trượt", "loi_thua" in tr("宝蓝色", "Dịch: Màu xanh hoàng gia")[0])
kiem("dịch luôn phần ngữ cảnh thì trượt (ca thật M0)",
     "lo_ngu_canh" in tr("商务, 复古风, ins风", "Thể loại sản phẩm: Điện tử kỹ thuật số Tên thuộc tính: Phong cách")[0])
kiem("gắn nhãn 'Xxx: ' trước bản dịch thì trượt (ca thật M0)",
     "lo_ngu_canh" in tr("哑光, 金闪, 柔雾", "Kết cấu: Mờ, Lấp lánh ánh vàng, Sương mịn")[0])
kiem("bản Trung có dấu hai chấm thì không bắt nhãn", "lo_ngu_canh" not in tr("颜色：红色", "Màu sắc: Đỏ")[0])
kiem("tên thuộc tính 产品类型 dịch 'Loại sản phẩm' vẫn qua", tr("产品类型", "Loại sản phẩm")[0] == [])
kiem("lặp từ thì trượt", "lap_tu" in tr("软胶", "nhựa nhựa nhựa nhựa nhựa nhựa nhựa nhựa mềm")[0])
# luật lặp từ sửa 21/10/2026: danh sách khớp số mục mà mỗi mục lặp phần đuôi như nguồn thì không bị bắt (ca thật 1.746 bản Gemma)
_mau = ["橙", "蓝", "绿", "咖", "粉", "灰", "黑", "白"]
_vi_mau = ["Cam", "Lam", "Lục", "Nâu", "Hồng", "Xám", "Đen", "Trắng"]
_zh_ds = ", ".join(f"原野-{m}【全棉磨毛优质 更保暖更亲肤】" for m in _mau)
_vi_ds = ", ".join(f"Nguyên Dã-{m}【cotton chải lông chất lượng tốt hơn ấm áp hơn thân thiện với làn da】" for m in _vi_mau)
kiem("lặp từ: danh sách 8 mục khớp nguồn, mỗi mục lặp phần đuôi như nguồn: KHÔNG bắt", "lap_tu" not in tr(_zh_ds, _vi_ds)[0])
kiem("lặp từ: cùng bản Việt nhưng nguồn chỉ có 4 mục (lệch số mục): vẫn bắt",
     "lap_tu" in tr(", ".join(f"原野-{m}【全棉磨毛优质 更保暖更亲肤】" for m in _mau[:4]), _vi_ds)[0])
kiem("lặp từ: không phải danh sách, ít từ khác nhau: vẫn bắt",
     "lap_tu" in tr("全棉磨毛优质更保暖更亲肤", " ".join(["cotton chải lông ấm áp hơn"] * 5))[0])
kiem("lặp từ: danh sách khớp số mục nhưng có cụm lặp liên tiếp (ca thật 'Ren ren ren ren...'): vẫn bắt",
     "lap_tu" in tr("蕾丝, 白色", "Ren ren ren ren ren ren ren ren ren, Trắng")[0])
kiem("đổi số thì trượt", "lech_so" in tr("200g/罐", "Hộp 20g")[0])
kiem("giữ số thì qua", tr("200g/罐", "200g/hộp")[0] == [])
kiem("万 đổi ra số đầy đủ thì qua", "lech_so" not in tr("5万次", "50.000 lần")[0])
kiem("斤 chia đôi đúng thì qua", tr("承重150斤", "Tải trọng 75 kg")[0] == [])
kiem("斤 giữ nguyên số thành kg thì trượt", "jin_sai" in tr("承重150斤", "Tải trọng 150 kg")[0])
# --- 斤 (sửa 22/10/2026): "cân" bị bắt; khoảng 斤 đã đổi đúng ra kg không còn bị lech_so oan; tự đổi "cân" ra kg khi chắc chắn ---
from doi_chieu import doi_can  # noqa: E402
_zh_size = "M【建议90-100斤】, L【建议100-120斤】"
kiem("斤: khoảng 90-100斤 đổi đúng ra kg (ca thật dòng 38) thì QUA, không còn lech_so oan",
     tr(_zh_size, "M【Gợi ý 45-50kg】, L【Gợi ý 50-60kg】")[0] == [])
kiem("斤: giữ số mà viết 'cân' thì trượt jin_can", "jin_can" in tr(_zh_size, "M【khuyến nghị 90-100 cân】, L【khuyến nghị 100-120 cân】")[0])
kiem("斤: giá '25 tệ/cân' cũng trượt jin_can (người Việt đọc là mỗi kg)", "jin_can" in tr("25元/斤", "25 tệ/cân")[0])
kiem("斤: '半斤' viết 'nửa cân' thì trượt jin_can", "jin_can" in tr("半斤真空包装*2", "Đóng gói chân không nửa cân*2")[0])
kiem("斤: '130元一斤' viết '130 tệ một cân' thì trượt jin_can", "jin_can" in tr("130元一斤", "130 tệ một cân")[0])
kiem("斤: '(斤/43元)' viết '(cân/43 tệ)' thì trượt jin_can", "jin_can" in tr("大红袍（斤/43元）", "Đại Hồng Bào (cân/43 tệ)")[0])
kiem("斤: 'M (100-120)斤' viết 'M (100-120) cân' thì trượt jin_can", "jin_can" in tr("M (100-120)斤", "M (100-120) cân")[0])
kiem("斤: 'cân nặng' (thể trọng) không bị bắt", "jin_can" not in tr("S建议体重80-95斤", "S gợi ý cân nặng 40-47.5kg")[0])
kiem("斤: '按斤散装' viết 'bán lẻ theo cân' không bị bắt", "jin_can" not in tr("按斤散装", "Bán lẻ theo cân")[0])
kiem("doi_can: '1 cân, 2 cân, 3 cân, 5 cân' thành '0.5 kg, 1 kg, 1.5 kg, 2.5 kg'",
     doi_can("1斤, 2斤, 3斤, 5斤", "1 cân, 2 cân, 3 cân, 5 cân") == "0.5 kg, 1 kg, 1.5 kg, 2.5 kg")
kiem("doi_can: khoảng 'M【khuyến nghị 90-100 cân】' thành '45-50 kg'",
     doi_can("M【建议90-100斤】", "M【khuyến nghị 90-100 cân】") == "M【khuyến nghị 45-50 kg】")
kiem("doi_can: giá tính theo 斤 thì KHÔNG tự đổi (đổi sai là sai tiền)", doi_can("25元/斤, 32斤/箱", "25 tệ/cân, 32 cân/thùng") is None)
kiem("doi_can: số lượng 斤 và 'cân' không khớp một đối một thì không đổi", doi_can("2斤装", "gói 2 cân, 3 cân") is None)
kiem("doi_can: không có 'cân' thì không đổi", doi_can("10斤", "5 kg") is None)
kiem("mất mã cỡ XL thì trượt", "mat_ma" in tr("XL码", "Cỡ lớn")[0])
kiem("giữ mã cỡ XL thì qua", tr("XL码", "Cỡ XL")[0] == [])
kiem("danh sách đủ 4 mục thì qua", tr("蓝色, 白色, 米黄色, 粉色", "Xanh lam, Trắng, Vàng be, Hồng")[0] == [])
kiem("danh sách thiếu mục thì trượt", "lech_so_muc" in tr("蓝色, 白色, 米黄色, 粉色", "Xanh lam, Trắng, Hồng")[0])
kiem("số thập phân kiểu Việt không tính là thêm mục", tr("7.5cm, 8.5cm", "7,5cm, 8,5cm")[0] == [])
kiem("mất dấu che thì trượt", "mat_dau_che" in tr("联系[电话]", "Liên hệ")[0])
kiem("chữ Latin lạ thì nhắc", "latin_la" in tr("维生素C安瓶精华", "Tinh chất vitamin C amping")[1])
kiem("chữ Việt không dấu không bị nhắc", "latin_la" not in tr("红色", "Mau do cho con")[1])
kiem("寸 viết cm thì nhắc CB_cun", "CB_cun" in tr("20寸", "20 cm")[1])
kh = khoang_do_dai([4] * 100, [10] * 98 + [1, 60])
kiem("độ dài bất thường bị bắt", do_dai_bat_thuong(4, 60, kh) and not do_dai_bat_thuong(4, 10, kh))

# --- chọn bản (5 trường hợp; bản sửa 15/10/2026: ưu tiên A, giống hệt giữ luôn, sàn riêng cho chuỗi Trung ngắn) ---
T, K, D, KN = 60, 0.5, 0.25, 0.35
r = chon("Nhựa mềm", "nhựa mềm", [], [], [], [], 0.8, 0.7, T, K, D)
kiem("TH1 giống hệt: giữ, tin cậy cao", r["ban"] is not None and r["do_tin_cay"] == "cao")
r = chon("Túi đeo chéo", "Túi đeo chéo", [], [], [], [], 0.44, 0.44, T, K, D, dai_zh=3, kappa_ngan=KN)
kiem("TH1 giống hệt, chuỗi Trung ngắn, điểm 0,44 trên sàn ngắn (ca thật 斜挎包): giữ, tin cậy cao", r["tu"] == "A" and r["do_tin_cay"] == "cao")
r = chon("Túi đeo chéo", "Túi đeo chéo", [], [], [], [], 0.44, 0.44, T, K, D, dai_zh=30, kappa_ngan=KN)
kiem("TH1 giống hệt dù cả hai điểm dưới sàn 0,5: vẫn giữ, tin cậy vừa", r["tu"] == "A" and r["do_tin_cay"] == "vua")
r = chon("Túi đeo chéo", "Túi đeo chéo", [], [], [], [], None, None, T, K, D)
kiem("TH1 giống hệt, chưa có điểm CometKiwi: vẫn giữ, tin cậy vừa", r["ban"] is not None and r["do_tin_cay"] == "vua")
r = chon("Hộp quà gỗ nguyên khối 120g", "Hộp quà bằng gỗ nguyên khối 120g", [], [], [], [], 0.70, 0.82, T, K, D)
kiem("TH2 gần giống, chênh điểm dưới delta: ưu tiên A", r["tu"] == "A" and r["cach_chon"].startswith("gan_giong") and r["do_tin_cay"] == "vua")
r = chon("Hộp quà gỗ nguyên khối 120g", "Hộp quà bằng gỗ nguyên khối 120g", [], [], [], [], 0.55, 0.82, T, K, D)
kiem("TH2 gần giống, B vượt từ delta: bỏ ưu tiên, lấy B", r["tu"] == "B")
r = chon("Keo mềm làm sạch xe hơi", "Gel mềm vệ sinh ô tô", [], [], [], [], 0.86, 0.88, T, K, D)
kiem("TH3 khác chữ cùng nghĩa, điểm gần nhau: ưu tiên A", r["tu"] == "A" and r["cach_chon"].startswith("khac_chu_cung_nghia"))
r = chon("Màu bàn chải", "Màu sắc của chiếc bàn chải", [], [], [], [], 0.40, 0.71, T, K, D, dai_zh=4, kappa_ngan=KN)
kiem("TH3 chuỗi ngắn, B hơn 0,31 điểm (ca thật 刷子颜色): theo điểm, lấy B, tin cậy thấp", r["tu"] == "B" and r["do_tin_cay"] == "thap")
r = chon("Chiều dài lòng giày", "Phần bên trong giày", [], [], [], [], 0.45, 0.37, T, K, D, dai_zh=3, kappa_ngan=KN)
kiem("TH3 chuỗi ngắn, cả hai dưới 0,5 nhưng trên sàn ngắn 0,35 (ca thật 鞋内长): giữ A", r["tu"] == "A")
r = chon("Chiều dài lòng giày", "Phần bên trong giày", [], [], [], [], 0.45, 0.37, T, K, D, dai_zh=30, kappa_ngan=KN)
kiem("TH3 cùng điểm nhưng chuỗi Trung dài: sàn 0,5, cất riêng", r["ban"] is None)
r = chon("Bóng giác hút", "Quả cầu có các miếng hút", [], [], [], [], 0.29, 0.33, T, K, D, dai_zh=3, kappa_ngan=KN)
kiem("TH3 chuỗi ngắn, cả hai dưới sàn ngắn (ca thật 吸盘球): cất riêng", r["ban"] is None)
r = chon("Từ 2800 trở lên", "2800 vụ", [], [], [], [], 0.75, 0.43, T, K, D, dai_zh=5, kappa_ngan=KN)
kiem("TH3 ca thật 2800起: lấy A, tin cậy vừa", r["tu"] == "A" and r["do_tin_cay"] == "vua")
r = chon("Keo mềm làm sạch xe hơi", "Gel mềm vệ sinh ô tô", [], [], [], [], 0.86, 0.88, T, K, D, uu_tien="B")
kiem("uu_tien = B: khi điểm gần nhau thì lấy B (đổi được bằng tham số)", r["tu"] == "B")
r = chon("Từ 2800 trở lên", "2800 vụ", [], [], [], [], 0.75, 0.43, T, K, D, dai_zh=5, kappa_ngan=KN, uu_tien="B")
kiem("uu_tien = B nhưng A vượt từ delta: bỏ ưu tiên, lấy A", r["tu"] == "A")
r = chon("Ý A", "Ý B khác hẳn", [], [], ["CB_cun"], [], 0.70, 0.70, T, K, D)
kiem("hai bản điểm gần nhau, A nhiều cờ nhắc hơn: lấy bản ít cờ (B)", r["tu"] == "B")
r = chon("Keo mềm làm sạch xe hơi", "Gel mềm vệ sinh ô tô", [], [], [], [], 0.86, 0.41, T, K, D)
kiem("TH3 một bản vượt rõ: lấy A, tin cậy thấp", r["tu"] == "A" and r["do_tin_cay"] == "thap")
r = chon("Keo mềm làm sạch xe hơi", "Gel mềm vệ sinh ô tô", [], [], [], [], 0.42, 0.40, T, K, D)
kiem("TH3 cả hai điểm thấp: cất riêng", r["ban"] is None)
r = chon("Tải trọng 150 kg", "Tải trọng 75 kg", ["jin_sai"], [], [], [], 0.9, 0.8, T, K, D)
kiem("TH4 chỉ B qua luật: lấy B", r["tu"] == "B" and r["do_tin_cay"] == "thap")
r = chon("", None, ["rong"], ["thieu"], [], [], None, None, T, K, D)
kiem("TH5 cả hai hỏng: cất riêng", r["ban"] is None and r["ly_do_cat"] == "ca_hai_truot")
# luật 22/10/2026 (chi_doi_khi_truot): hai bản cùng qua luật thì luôn lấy bản ưu tiên, Kiwi chỉ quyết định độ tin cậy
LM = {"chi_doi_khi_truot": True}
r = chon("Thường ngày", "Thư giãn", [], [], [], [], 0.43, 0.76, T, K, D, dai_zh=2, kappa_ngan=KN, **LM)
kiem("luật mới, ca thật 休闲: Qwen hơn Gemma 0,33 điểm nhưng vẫn lấy Gemma, tin cậy vừa", r["tu"] == "A" and r["do_tin_cay"] == "vua")
r = chon("Sán Đầu", "Thâm Quyền", [], [], [], [], 0.31, 0.44, T, K, D, dai_zh=2, kappa_ngan=KN, **LM)
kiem("luật mới, ca thật 汕头: Gemma dưới sàn, Qwen trên sàn: vẫn lấy Gemma, tin cậy thấp, cách chọn uu_tien_duoi_san_A",
     r["tu"] == "A" and r["do_tin_cay"] == "thap" and r["cach_chon"] == "uu_tien_duoi_san_A")
r = chon("Áo khoác cardigan", "Áo khoác tay ngắn", [], [], ["latin_la"], [], 0.66, 0.44, T, K, D, dai_zh=2, kappa_ngan=KN, **LM)
kiem("luật mới, ca thật 开衫: Gemma nhiều cờ nhắc hơn vẫn lấy Gemma", r["tu"] == "A")
r = chon("Thành phần vải chính: Cotton", "Bông", ["lo_ngu_canh"], [], [], [], 0.76, 0.46, T, K, D, dai_zh=1, kappa_ngan=KN, **LM)
kiem("luật mới: Gemma trượt luật thì vẫn dùng bản Qwen (chi_B)", r["tu"] == "B" and r["cach_chon"] == "chi_B")
r = chon("Keo mềm làm sạch xe hơi", "Gel mềm vệ sinh ô tô", [], [], [], [], 0.42, 0.40, T, K, D, **LM)
kiem("luật mới: cả hai dưới sàn thì vẫn cất riêng", r["ban"] is None)
r = chon("Hộp quà gỗ nguyên khối 120g", "Hộp quà bằng gỗ nguyên khối 120g", [], [], [], [], 0.40, 0.82, T, K, D, **LM)
kiem("luật mới, gần giống: lấy Gemma dù dưới sàn, tin cậy thấp", r["tu"] == "A" and r["do_tin_cay"] == "thap")

# --- chạy suốt trên dữ liệu tổng hợp: chuẩn bị -> giả đầu ra Kaggle -> hoàn tất ---
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    cap = pd.DataFrame([
        # pair_id, product_id, category, vi_tri, name_zh, gia_tri_zh, vai_tro, nhan, ly_do, gia_tri_vi (bản cũ, chỉ để đối chiếu phần quy tắc)
        (0, "p1", "auto", 0, "材质", "软胶", "mo_ta_hang", "", "", "x"),
        (1, "p1", "auto", 1, "品牌", "和洁士", "thuong_hieu", "", "", "Hejieshi"),
        (2, "p1", "auto", 2, "货号", "hjs200g", "ma_giay_to", "chep_ma;", "", "hjs200g"),
        (3, "p1", "auto", 3, "种类", "汽车清洁软胶", "mo_ta_hang", "", "T2_CHUA_KIEM;", "x"),
        (4, "p1", "auto", 4, "颜色", "蓝色, 白色", "lua_chon_sku", "", "", "x"),
        (5, "p2", "home", 0, "材质", "软胶", "mo_ta_hang", "", "", "x"),
        (6, "p2", "home", 1, "规格", "/", "mo_ta_hang", "", "RONG;", "/"),
        (7, "p2", "home", 2, "品牌", "某某某某某某某某有限公司", "thuong_hieu", "", "DS09_dai;", "x"),
        (8, "p2", "home", 3, "品牌", "是", "thuong_hieu", "", "", "x"),      # thương hiệu vô nghĩa: cất riêng (21/10/2026)
    ], columns=["pair_id", "product_id", "category", "vi_tri", "name_zh", "gia_tri_zh", "vai_tro", "nhan", "ly_do", "gia_tri_vi"])
    f_cap = td / "cap.parquet"
    cap.to_parquet(f_cap, index=False)
    B, DV, the = chuan_bi.chay(f_cap, None, td / "ra", in_ra=lambda *a: None)
    kiem("chuẩn bị: 2 cặp bị loại hoặc chờ xem (RONG, DS09_dai)", int((B.trang_thai_vao != "dung").sum()) == 2)
    kiem("chuẩn bị: gộp trùng 软胶 thành 1 đơn vị", int((DV.van_ban_zh == "软胶").sum()) == 1)
    kiem("chuẩn bị: đủ file tải lên Kaggle", all((td / "ra" / "vao_kaggle" / f).exists() for f in ["don_vi_can_dich.parquet", "dich.py", "cham_kiwi.py"]))
    # giả đầu ra Kaggle
    dich_a = {"软胶": "Nhựa mềm", "汽车清洁软胶": "Keo mềm làm sạch xe hơi", "蓝色, 白色": "Xanh lam, Trắng",
              "材质": "Chất liệu", "品牌": "Thương hiệu", "货号": "Mã hàng", "种类": "Loại", "颜色": "Màu sắc", "规格": "Quy cách"}
    dich_b = {"软胶": "nhựa mềm", "汽车清洁软胶": "Gel mềm vệ sinh ô tô", "蓝色, 白色": "Xanh lam",
              "材质": "Chất liệu", "品牌": "Thương hiệu", "货号": "Mã sản phẩm", "种类": "Loại", "颜色": "Màu sắc", "规格": "Quy cách"}
    kk = td / "tu_kaggle" / "ra"
    for ten, bd in [(C.MO_HINH["A"], dich_a), (C.MO_HINH["B"], dich_b)]:
        (kk / ten).mkdir(parents=True)
        (kk / f"kiwi_{ten}").mkdir(parents=True)
        pd.DataFrame({"ma_don_vi": DV.ma_don_vi, "ban_dich": DV.van_ban_zh.map(bd), "ly_do_dung": "stop", "backend": "gia"}).to_parquet(kk / ten / "lo_00000.parquet")
        pd.DataFrame({"ma_don_vi": DV.ma_don_vi, "diem": 0.85}).to_parquet(kk / f"kiwi_{ten}" / "lo_00000.parquet")
    Dd, CR, U, the2 = hoan_tat.chay(td / "tu_kaggle", td / "ra", in_ra=lambda *a: None, tu_dien={"ten": {}, "gia_tri": {}})
    kiem("hoàn tất: bản đầy đủ đúng 18 cột", list(Dd.columns) == hoan_tat.COT_DAY_DU and len(Dd.columns) == 18)
    cb = pd.read_parquet(td / "ra" / the2["file"]["cong_bo"])
    kiem("hoàn tất: bản công bố đúng 9 cột", list(cb.columns) == C.COT_CONG_BO)
    kiem("hoàn tất: giữ 6 cặp (p1 đủ 5 cặp, p2 cặp 材质), cất riêng 3", len(Dd) == 6 and len(CR) == 3)
    kiem("hoàn tất: thương hiệu '是' bị cất riêng với lý do ten_hang_vo_nghia, không thành 'Shi'",
         CR.set_index("pair_id").loc[8, "ly_do_cat"] == "ten_hang_vo_nghia" and 8 not in set(Dd.pair_id))
    kiem("hoàn tất: danh sách thiếu mục ở bản B nên lấy bản A", Dd.set_index("pair_id").loc[4, "gia_tri_vi"] == "Xanh lam, Trắng")
    kiem("hoàn tất: tên hãng theo quy tắc", Dd.set_index("pair_id").loc[1, "gia_tri_vi"] == "Hejieshi")
    kiem("hoàn tất: cặp T2 cũ được nhận lại", 3 in set(Dd.pair_id))
    # bước 6c (21/10/2026): từ điển đã duyệt và bảng có/không đè bản máy
    Dd2, CR2, U2, the3 = hoan_tat.chay(td / "tu_kaggle", td / "ra", in_ra=lambda *a: None,
                                       tu_dien={"ten": {"颜色": "Màu"}, "gia_tri": {"软胶": "Keo mềm"}})
    d2 = Dd2.set_index("pair_id")
    kiem("6c: tên có trong từ điển P6 dùng bản P6 (颜色 thành 'Màu', bản máy là 'Màu sắc')", d2.loc[4, "name_vi"] == "Màu")
    kiem("6c: giá trị có trong P7 dùng bản P7, cách chọn tu_dien_p7, tin cậy cao, cả hai sản phẩm",
         d2.loc[0, "gia_tri_vi"] == "Keo mềm" and d2.loc[0, "cach_chon"] == "tu_dien_p7" and d2.loc[0, "do_tin_cay"] == "cao"
         and d2.loc[5, "gia_tri_vi"] == "Keo mềm")
    kiem("6c: giá trị không có trong từ điển vẫn theo bước chọn bản", d2.loc[4, "gia_tri_vi"] == "Xanh lam, Trắng")
    kiem("6c: thẻ nguồn gốc ghi số đơn vị theo cách cố định và bảng có/không",
         the3["co_dinh"]["don_vi"].get("tu_dien_p7", {}).get("don_vi") == 1 and the3["co_dinh"]["co_khong"]["否"] == "Không")
kiem("6c: bảng có/không đứng trước P7 (anh chốt 21/10/2026): 不支持 thành 'Không hỗ trợ' dù P7 ghi 'Không được hỗ trợ'",
     hoan_tat.ban_co_dinh("gia_tri", "不支持", {}, {"不支持": "Không được hỗ trợ"}) == ("Không hỗ trợ", "quy_tac_co_khong"))
kiem("6c: tên có trong file bổ sung P6b dùng bản P6b (nhãn tu_dien_p6b); P6 đứng trước khi trùng",
     hoan_tat.ban_co_dinh("ten", "裤门襟", {}, {}, tu_dien_ten_bs={"裤门襟": "Khóa kéo quần"}) == ("Khóa kéo quần", "tu_dien_p6b")
     and hoan_tat.ban_co_dinh("ten", "货号", {"货号": "Mã sản phẩm"}, {}, tu_dien_ten_bs={"货号": "khác"}) == ("Mã sản phẩm", "tu_dien_p6"))
kiem("6c: giá trị không nằm trong bảng có/không vẫn lấy P7", hoan_tat.ban_co_dinh("gia_tri", "整单", {}, {"整单": "Nguyên lô"}) == ("Nguyên lô", "tu_dien_p7"))
kiem("6c: 否 không có trong P7 thì lấy bảng có/không", hoan_tat.ban_co_dinh("gia_tri", "否", {}, {}) == ("Không", "quy_tac_co_khong"))
kiem("6c: 无 thành 'Không có', 不支持 thành 'Không hỗ trợ'",
     hoan_tat.ban_co_dinh("gia_tri", "无", {}, {})[0] == "Không có" and hoan_tat.ban_co_dinh("gia_tri", "不支持", {}, {})[0] == "Không hỗ trợ")
kiem("6c: 可以 không nằm trong bảng có/không (tuỳ ngữ cảnh), để mô hình dịch", hoan_tat.ban_co_dinh("gia_tri", "可以", {}, {}) is None)
kiem("6c: bảng có/không chỉ áp cho giá trị, không áp cho tên thuộc tính", hoan_tat.ban_co_dinh("ten", "是", {}, {}) is None)
kiem("6c: so khớp nguyên chuỗi, không khớp một phần ('是否进口' không thành 'Có')", hoan_tat.ban_co_dinh("gia_tri", "是否进口", {}, {}) is None)
# quy tắc mẫu (22/10/2026)
from song_ngu_1.quy_tac_gia_tri import ban_quy_tac  # noqa: E402
_qt = {("克重", "180g(含)-250g(不含)"): "Từ 180g đến dưới 250g", ("主面料成分的含量", "91%(含)-95%(含)"): "Từ 91% đến 95%",
       ("克重", "500g（不含）以下"): "Dưới 500g", ("主面料成分的含量", "96%及以上"): "Từ 96% trở lên",
       ("主面料成分的含量", "30%及以下"): "Từ 30% trở xuống", ("年份", "1年(含)-3年(不含)"): "Từ 1 năm đến dưới 3 năm",
       ("克重", "2500g（含）~3000g（不含）"): "Từ 2500g đến dưới 3000g", ("拍照像素", "300万"): "3 MP",
       ("拍照像素", "1200万"): "12 MP", ("摄像像素", "50万"): "0.5 MP"}
kiem("quy tắc mẫu: 10 ca khoảng 含 và 像素 万 ra đúng chữ", all((ban_quy_tac(n, z) or ("",))[0] == v for (n, z), v in _qt.items()))
kiem("quy tắc mẫu: 万 ở thuộc tính không phải 像素 (日生产量 10万个) thì để mô hình", ban_quy_tac("日生产量", "10万个") is None
     and ban_quy_tac("价格", "300万") is None)
kiem("quy tắc mẫu: '以上' không kèm 含, khoảng chỉ ghi 含 một đầu, 含糖 thì để mô hình",
     ban_quy_tac("年龄", "14岁以上") is None and ban_quy_tac("尺寸", "0cm~3cm(含)") is None and ban_quy_tac("口味", "含糖") is None)
kiem("6c: quy tắc mẫu xếp sau P7 đã duyệt (P7 có thì dùng P7)",
     hoan_tat.ban_co_dinh("gia_tri", "96%及以上", {}, {"96%及以上": "Từ 96% trở lên (P7)"}, ten_zh="含量")[1] == "tu_dien_p7"
     and hoan_tat.ban_co_dinh("gia_tri", "300万", {}, {}, ten_zh="拍照像素") == ("3 MP", "quy_tac_diem_anh"))

# --- mô hình A gọi qua API (dich.py): thử bằng API giả, không gọi mạng, không dùng khoá thật ---
import importlib.util  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

spec = importlib.util.spec_from_file_location("dich_kaggle", Path(__file__).resolve().parents[2] / "src" / "song_ngu_1" / "kaggle" / "dich.py")
dich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dich)
cfg_a = dich.MO_HINH["deepseek_v41_flash"]
cfg_p = dich.MO_HINH["deepseek_flash_api"]
cfg_al = dich.MO_HINH["deepseek_alibaba"]
kiem("cấu hình: mô hình A là Gemma qua api_tuy_chon (từ 16/10/2026), không phải Qwen; cấu hình DeepSeek Alibaba vẫn còn",
     C.MO_HINH["A"] == "api_tuy_chon" and "api" in dich.MO_HINH["api_tuy_chon"] and "api" in cfg_al)
kiem("cấu hình Alibaba: mã deepseek-v4.1-flash, khoá DASHSCOPE_API_KEY, có giá, thử nhiều cách tắt suy luận",
     cfg_al["api"] == "deepseek-v4.1-flash" and cfg_al["key_env"] == "DASHSCOPE_API_KEY" and cfg_al["gia_usd_1m"]["ra"] == 1.20
     and isinstance(cfg_al["extra"], list) and {"enable_thinking": False} in cfg_al["extra"])
# --- Qwen3-30B-A3B (21/10/2026): nạp bằng vLLM với nhiều cách lượng tử, không có đường dự phòng transformers ---
cfg_q30 = dich.MO_HINH["qwen3_30b_a3b"]
kiem("cấu hình Qwen3-30B-A3B: bản nén AWQ, kiểu câu lệnh qwen, thử lượng tử tự nhận rồi moe_wna16, không có hf",
     "AWQ" in cfg_q30["vllm"] and "Instruct-2507" in cfg_q30["vllm"] and cfg_q30["kieu"] == "qwen"
     and cfg_q30["quant_thu"] == [None, "moe_wna16"] and not cfg_q30["hf"])
cfg_q14 = dich.MO_HINH["qwen3_14b"]
kiem("cấu hình Qwen3-14B: bản nén AWQ chính thức của Qwen, kiểu câu lệnh qwen, có dự phòng transformers bằng bản đầy đủ",
     cfg_q14["vllm"] == "Qwen/Qwen3-14B-AWQ" and cfg_q14["hf"] == "Qwen/Qwen3-14B" and cfg_q14["kieu"] == "qwen")
_VLLM_that = dich.VLLM
_da_thu = []


class _VllmGia:
    chi_nhan = "moe_wna16"

    def __init__(self, repo, eager=False, quant=None):
        _da_thu.append((quant, eager))
        if quant != self.chi_nhan:
            raise RuntimeError(f"kernel khong ho tro tren T4 ({quant})")
        self.phien_ban = {"quantization": quant, "enforce_eager": eager}


dich.VLLM = _VllmGia
try:
    be_q, ten_q, repo_q, loi_q = dich.tai_backend(cfg_q30, "auto")
    kiem("Qwen3-30B-A3B: lượng tử tự nhận lỗi thì thử moe_wna16 và nạp được, lỗi trước được ghi lại",
         ten_q == "vllm" and be_q.phien_ban["quantization"] == "moe_wna16" and _da_thu[:2] == [(None, False), (None, True)]
         and "kernel" in (loi_q or ""))
    _VllmGia.chi_nhan = "khong_co"
    _da_thu.clear()
    try:
        dich.tai_backend(cfg_q30, "auto")
        dung_han = False
    except SystemExit as e:
        dung_han = "không có đường dự phòng" in str(e) and len(_da_thu) == 4
    kiem("Qwen3-30B-A3B: mọi cách vLLM đều lỗi thì dừng hẳn, không thử transformers 4-bit (30B không vừa một T4)", dung_han)
finally:
    dich.VLLM = _VLLM_that
kiem("cấu hình: API chính thức tắt suy luận mặc định, có giá để theo ngân sách",
     cfg_p["extra"] == {"thinking": {"type": "disabled"}} and cfg_p["gia_usd_1m"]["ra"] > 0 and cfg_p["base_url"] == "https://api.deepseek.com")
os.environ.pop("NVIDIA_API_KEY", None)
try:
    dich.API(cfg_a)
    thieu_khoa = False
except SystemExit:
    thieu_khoa = True
kiem("API: thiếu khoá thì dừng, không chạy", thieu_khoa)
os.environ["NVIDIA_API_KEY"] = "khoa-gia-de-thu"
os.environ["DEEPSEEK_API_KEY"] = "khoa-gia-de-thu-2"
os.environ["DASHSCOPE_API_KEY"] = "khoa-gia-de-thu-3"


class ApiGia(dich.API):
    loi401 = True
    nhan_enable = False      # máy chủ giả mặc định KHÔNG hiểu enable_thinking (chỉ hiểu thinking.type = disabled)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dem = {}

    def _post(self, payload):
        nd = payload["messages"][-1]["content"]
        n = self.dem[nd] = self.dem.get(nd, 0) + 1
        if "401" in nd and self.loi401:
            return 401, "khoa sai", None
        if "429" in nd and n < 2:
            return 429, "qua han muc", "0"
        if "503" in nd and n < 3:
            return 503, "ban", None
        if "stream_options" in payload:          # máy chủ giả không nhận stream_options
            return 400, '{"detail":"unknown field stream_options"}', None
        assert payload.get("stream"), "phải gọi theo luồng"
        # giả luồng SSE y như máy chủ thật, để thử luôn hàm gộp mẩu
        self.cac_payload = getattr(self, "cac_payload", []) + [payload]
        tat_nghi = payload.get("thinking") == {"type": "disabled"} or payload.get("enable_thinking") is False and self.nhan_enable
        loi_nd = ("Your model 'deepseek-v4.1-flash' could not answer: The model endpoint is limiting the requests: retry later"
                  if "lim-" in nd and n < 3 else None)       # cổng trung gian trả lỗi trong nội dung với HTTP 200
        mau = ([] if tat_nghi else [{"choices": [{"delta": {"reasoning_content": "nghĩ"}}]}]) + [
               {"choices": [{"delta": {"content": loi_nd or "VI:"}}]},
               {"choices": [{"delta": {"content": "" if loi_nd else nd[-6:]}, "finish_reason": "stop"}]},
               {"model": "gia", "usage": {"prompt_tokens": 5, "completion_tokens": 3}, "choices": []}]
        dong = [b"data: " + json.dumps(x).encode() for x in mau] + [b": chu thich bo qua", b"", b"data: [DONE]"]
        return 200, self._doc_luong(iter(dong)), None


_ngu = time.sleep
time.sleep = lambda s: None      # không chờ thật khi thử đường thử lại
try:
    ms = [[{"role": "user", "content": t}] for t in ["a-000001", "b-429xxx", "c-503xxx", "d-401xxx", "e-000005"]]
    kq = ApiGia(cfg_a, 3).sinh(ms, [100] * 5, 1.0)
    kiem("API: kết quả đúng thứ tự dù chạy song song", [x for x, _ in kq][:3] == ["VI:000001", "VI:429xxx", "VI:503xxx"] and kq[4][0] == "VI:000005")
    kiem("API: gặp 429 và 503 thì thử lại rồi qua", kq[1][1] == "stop" and kq[2][1] == "stop")
    kiem("API: khoá sai (401) thì bỏ ngay, đánh dấu loi_api", kq[3] == ("", "loi_api"))
    bl = ApiGia(cfg_a, 1)
    bl.dem_usage = False          # máy chủ giả trả 400 cho stream_options ở lần gọi đầu; tắt đi để đếm lần gọi cho chắc
    kl = bl._mot([{"role": "user", "content": "lim-123456"}], 100)
    kiem("API: cổng trả LỖI trong nội dung (HTTP 200) thì không ghi làm bản dịch, thử lại rồi lấy bản dịch thật",
         kl == ("VI:123456", "stop") and bl.tk["loi_trong_noi_dung"] == 2 and bl.tk["yeu_cau"] == 1)
    bl2 = ApiGia(dict(cfg_a, so_lan_thu=2), 1)
    bl2.dem_usage = False
    kl2 = bl2._mot([{"role": "user", "content": "lim-654321"}], 100)
    kiem("API: lỗi trong nội dung kéo dài thì trả loi_api, KHÔNG trả chữ thông báo lỗi làm bản dịch", kl2 == ("", "loi_api"))
    be = ApiGia(cfg_a, 2)
    kiem("API: gộp mẩu theo luồng, bỏ phần suy luận", be._mot([{"role": "user", "content": "x-abcdef"}], 100) == ("VI:abcdef", "stop"))
    kiem("API: máy chủ không nhận stream_options thì tự bỏ rồi gọi lại", be.dem_usage is False and be.tk["token_suy_luan"] > 0)

    class ApiChonCach(ApiGia):      # máy chủ giả: không nhận "chat_template_kwargs"
        def _post(self, payload):
            if "chat_template_kwargs" in payload:
                return 400, '{"detail":"unknown field chat_template_kwargs"}', None
            return super()._post(payload)
    be = ApiChonCach(cfg_a, 2)
    kq_thu = be.thu_ket_noi([{"chat_template_kwargs": {"thinking": False}}, {"thinking": {"type": "disabled"}}, {}])
    kiem("API gọi thử: bỏ cách bị từ chối (400), chọn cách trả lời được và dừng thử khi đã nhanh",
         be.extra == {"thinking": {"type": "disabled"}} and len(be.tk["goi_thu"]) == 2 and "loi" in be.tk["goi_thu"][0])
    os.environ["DASHSCOPE_API_KEY"] = "sk-sp-khoa-gia-token-plan"
    try:
        dich.API(cfg_al, 1)
        chan_sp = False
    except SystemExit as e:
        chan_sp = "Token Plan" in str(e)
    kiem("khoá Token Plan (sk-sp-) của Alibaba: từ chối chạy vì điều khoản cấm script tự động", chan_sp)
    os.environ["DASHSCOPE_API_KEY"] = "khoa-gia-de-thu-3"
    # --- nhà cung cấp tuỳ chọn: bắt buộc địa chỉ và tên mô hình, chỉ gửi khoá qua https ---
    os.environ["API_KEY_A"] = "khoa-gia-de-thu-4"
    cfg_tc = dich.MO_HINH["api_tuy_chon"]
    thieu = []
    for c in (cfg_tc, dict(cfg_tc, base_url="https://x.example/v1"), dict(cfg_tc, api="m", base_url="http://x.example/v1")):
        try:
            dich.API(c, 1)
            thieu.append(False)
        except SystemExit:
            thieu.append(True)
    kiem("api_tuy_chon: thiếu địa chỉ hoặc tên mô hình thì dừng; địa chỉ http (không mã hoá) thì từ chối gửi khoá", thieu == [True, True, True])
    kiem("api_tuy_chon: có cách gọi thử cuối cùng không thêm trường lạ ({})", cfg_tc["extra"][-1] == {})
    bt = ApiGia(dict(cfg_tc, api="m", base_url="https://x.example/v1"), 1)
    bt.dem_usage = False
    bt.toi_da_goi = 3
    kq3 = bt.sinh([[{"role": "user", "content": f"t-{i:06d}"}] for i in range(6)], [10] * 6, 1.0)
    kiem("trần số lần gửi: chỉ gửi đúng 3 yêu cầu, 3 yêu cầu còn lại bị chặn (không gửi đi)",
         bt.so_goi == 3 and len(bt.cac_payload) == 3 and bt.tk["het_toi_da_goi"] == 3 and [r for _, r in kq3].count("loi_api") == 3)
    kiem("api_tuy_chon: địa chỉ https đầy đủ thì khởi tạo được",
         dich.API(dict(cfg_tc, api="m", base_url="https://x.example/v1"), 1).url == "https://x.example/v1/chat/completions")
    # --- bản miễn phí của nhà cung cấp (Google AI Studio): giữ nhịp, hết hạn mức ngày thì dừng, không nhận vai trò system ---
    class ApiGoogle(ApiGia):
        def _post(self, payload):
            self.cac_goi = getattr(self, "cac_goi", 0) + 1
            nd = payload["messages"][-1]["content"]
            if self.cac_goi > self.sau_bao_nhieu_goi_het_ngay:
                return 429, '{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[{"quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}', None
            if any(m["role"] == "system" for m in payload["messages"]):
                return 400, '{"error":{"message":"Developer instruction is not enabled for models/gemma"}}', None
            self.noi_dung_da_thay = payload["messages"][0]["content"]
            return super()._post(payload)
        sau_bao_nhieu_goi_het_ngay = 4

    bg = ApiGoogle(cfg_tc | {"api": "m", "base_url": "https://x.example/v1"}, 1)
    bg.dem_usage = False
    kg = bg._mot([{"role": "system", "content": "LOI-DAN"}, {"role": "user", "content": "g-111111"}], 100)
    kiem("máy chủ từ chối vai trò system: gộp lời dặn vào tin nhắn người dùng rồi gọi lại, trả bản dịch",
         kg == ("VI:111111", "stop") and bg.gop_he_thong and bg.noi_dung_da_thay.startswith("LOI-DAN") and "g-111111" in bg.noi_dung_da_thay)
    kg2 = [bg._mot([{"role": "user", "content": f"g-{i:06d}"}], 100) for i in range(4)]
    kiem("hết hạn mức NGÀY (429 per day): dừng ngay, các yêu cầu sau không gửi đi nữa (không đập vào máy chủ)",
         bg.dung_quota and [r for _, r in kg2].count("loi_api") >= 2 and bg.cac_goi <= 5 and bg.tk["het_quota_ngay"] >= 2)
    nhip = []
    _ngu2 = time.sleep
    time.sleep = lambda s: nhip.append(round(s, 2))
    try:
        bn = ApiGia(cfg_tc | {"api": "m", "base_url": "https://x.example/v1"}, 1)
        bn.dem_usage = False
        bn.rpm = 30
        for i in range(4):
            bn._cho_nhip()
    finally:
        time.sleep = _ngu2
    # sleep bị thay nên đồng hồ không chạy: lần thứ hai chờ 2 giây, thứ ba 4 giây, thứ tư 6 giây (mốc gửi cách nhau 60/30 = 2 giây)
    kiem("--rpm 30: các lần gửi cách nhau 2 giây (lần đầu không chờ)",
         len(nhip) == 3 and all(abs(x - 2.0 * (i + 1)) < 0.05 for i, x in enumerate(nhip)))
    # --- nhiều khoá: chọn khoá nhanh, khoá lỗi hoặc hết hạn mức thì đổi sang khoá khác ---
    os.environ.update(KH_A="khoa-a-gia", KH_B="khoa-b-gia", KH_C="khoa-a-gia")      # KH_C trùng KH_A; KH_D không đặt
    os.environ.pop("KH_D", None)
    cfg_nk = cfg_tc | {"api": "m", "base_url": "https://x.example/v1", "key_env": "KH_A,KH_B,KH_C,KH_D"}

    class ApiNhieuKhoa(ApiGia):
        hanh_vi = {}

        def _post(self, payload):
            ten = self.tl.k["ten"] if getattr(self.tl, "k", None) else "?"
            self.goi_theo_khoa = getattr(self, "goi_theo_khoa", {})
            self.goi_theo_khoa[ten] = self.goi_theo_khoa.get(ten, 0) + 1
            if self.hanh_vi.get("*") and getattr(self, "so_500_dau", 0) > 0:      # lỗi 500 nhanh trên MỌI khoá (nhà cung cấp quá tải)
                self.so_500_dau -= 1
                return 500, "loi may chu chung", None
            h = self.hanh_vi.get(ten)
            if h == "500":
                return 500, "loi may chu", None
            if h == "ngay":
                return 429, '{"error":{"status":"RESOURCE_EXHAUSTED","quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}', None
            if h == "sai":
                return 400, '{"error":{"message":"API key not valid. Please pass a valid API key."}}', None
            return super()._post(payload)

    def moi(**hv):
        b = ApiNhieuKhoa(cfg_nk, 2)
        b.dem_usage = False
        b.hanh_vi = hv
        return b
    tin = lambda i: [{"role": "user", "content": f"m-{i:06d}"}]
    n1 = moi()
    kiem("nhiều khoá: bỏ khoá trùng giá trị và biến không đặt, còn 2 khoá", [k["ten"] for k in n1.khoa] == ["KH_A", "KH_B"])
    n1.khoa[0]["ewma"], n1.khoa[1]["ewma"] = 4.0, 180.0
    c1 = n1._chon_khoa()
    n1._danh_dau(c1, 1, "bo")
    n1.khoa[0]["dang_chay"] = 100          # khoá nhanh đang quá tải (101 x 4 = 404 > 180): việc chuyển sang khoá chậm đang rảnh
    c2 = n1._chon_khoa()
    kiem("nhiều khoá: ưu tiên khoá nhanh, khoá nhanh quá tải thì chia bớt sang khoá chậm", c1["ten"] == "KH_A" and c2["ten"] == "KH_B")
    n2 = moi(KH_A="500")
    r2 = [n2._mot(tin(i), 100) for i in range(6)]
    kiem("nhiều khoá: một khoá lỗi 500 thì yêu cầu sang khoá kia, vẫn có bản dịch",
         all(r == (f"VI:{i:06d}", "stop") for i, r in enumerate(r2)) and n2.goi_theo_khoa["KH_B"] >= 6 and n2.tk["theo_khoa"]["KH_A"]["loi"] >= 1)
    kiem("nhiều khoá: khoá lỗi bị cho nghỉ, các yêu cầu sau không gửi vào khoá đó nữa", n2.goi_theo_khoa["KH_A"] == 1 and n2.khoa[0]["nghi_den"] > time.time())
    n3 = moi(KH_A="ngay")
    r3 = n3._mot(tin(1), 100)
    kiem("nhiều khoá: một khoá hết hạn mức NGÀY thì loại khoá đó, vẫn dịch bằng khoá kia, không dừng cả chương trình",
         r3 == ("VI:000001", "stop") and n3.khoa[0]["chet"] and not n3.dung_quota)
    n3.hanh_vi = {"KH_A": "ngay", "KH_B": "ngay"}
    r3b = n3._mot(tin(2), 100)
    kiem("nhiều khoá: mọi khoá hết hạn mức ngày thì dừng hẳn", r3b == ("", "loi_api") and n3.dung_quota)
    gnk = dich.GIAY_NGHI_KHOA
    dich.GIAY_NGHI_KHOA = 0.001      # thử nhanh: khoá lỗi chỉ nghỉ một phần nghìn giây
    try:
        n6 = moi(**{"*": True})
        n6.so_500_dau = 7
        r6 = n6._mot(tin(6), 100)
        n7 = moi(**{"*": True})
        n7.so_500_dau = 1000
        r7 = n7._mot(tin(7), 100)
    finally:
        dich.GIAY_NGHI_KHOA = gnk
    kiem("nhiều khoá: lỗi 500 lan trên mọi khoá (7 lần liên tiếp) thì vẫn thử đủ vòng rồi qua, KHÔNG bỏ cuộc sau 5 lần thử như một khoá",
         r6 == ("VI:000006", "stop") and n6.tk["thu_lai"] >= 7)
    kiem("nhiều khoá: lỗi 500 kéo dài mãi thì dừng ở số lần thử tối đa (số khoá x 5), trả loi_api", r7 == ("", "loi_api") and sum(n7.goi_theo_khoa.values()) == 10)
    n4 = moi(KH_A="sai")
    n4.thu_ket_noi([{"thinking": {"type": "disabled"}}])
    kiem("nhiều khoá, gọi thử: khoá đầu sai thì loại, dùng khoá sau, khoá sai không bị gửi yêu cầu thật nữa",
         n4.khoa[0]["chet"] and not n4.khoa[1]["chet"] and n4._mot(tin(3), 100) == ("VI:000003", "stop") and n4.goi_theo_khoa["KH_A"] == 1)
    n5 = moi(KH_B="sai")
    n5.thu_ket_noi([{"thinking": {"type": "disabled"}}])
    kiem("nhiều khoá, gọi thử: khoá sau sai thì bị loại ngay từ lúc thử, khoá đầu vẫn chạy",
         n5.khoa[1]["chet"] and not n5.khoa[0]["chet"] and n5._mot(tin(4), 100) == ("VI:000004", "stop") and n5.goi_theo_khoa.get("KH_B", 0) == 1)
    try:
        moi(KH_A="sai", KH_B="sai").thu_ket_noi([{"thinking": {"type": "disabled"}}])
        tat_ca_sai = False
    except SystemExit:
        tat_ca_sai = True
    kiem("nhiều khoá, gọi thử: mọi khoá đều sai thì dừng, không chạy", tat_ca_sai)
    kiem("nhiều khoá: thẻ chỉ ghi TÊN biến và thống kê từng khoá, không chứa giá trị khoá",
         "khoa-a-gia" not in json.dumps(n2.tk, ensure_ascii=False) and "khoa-b-gia" not in json.dumps(n2.tk, ensure_ascii=False)
         and set(n2.tk["theo_khoa"]) <= {"KH_A", "KH_B"})
    nhip2 = []
    _ngu3 = time.sleep
    time.sleep = lambda s: nhip2.append(round(s, 2))
    try:
        nr = moi()
        nr.rpm = 30
        nr._cho_nhip(nr.khoa[0])
        nr._cho_nhip(nr.khoa[1])
        nr._cho_nhip(nr.khoa[0])
    finally:
        time.sleep = _ngu3
    kiem("nhiều khoá: --rpm tính riêng từng khoá (khoá khác không phải chờ nhau)", len(nhip2) == 1 and abs(nhip2[0] - 2.0) < 0.05)
    # --- Gemma 4 (Google AI Studio): suy nghĩ lẫn trong nội dung dạng <thought>...</thought>, không tắt bằng trường lạ ---
    kiem("don_dep: bỏ khối <thought>...</thought>, giữ bản dịch ra sau", dich.don_dep("<thought>* 否 means no\n* Output only</thought>Không") == "Không")
    kiem("don_dep: khối <think>...</think> của Qwen vẫn bị bỏ", dich.don_dep("<think>x</think>Tốt") == "Tốt")
    kiem("don_dep: suy nghĩ mở mà chưa đóng (bị cắt) thì không có bản dịch", dich.don_dep("<thought>* 否 means no and then") == "")
    sse = [b"data: " + json.dumps({"choices": [{"delta": {"content": "<thought>nghi nhieu</thought>Tốt"}, "finish_reason": "stop"}]}).encode(), b"data: [DONE]"]
    kiem("đọc luồng: đếm ký tự suy nghĩ nằm trong nội dung", dich.API(cfg_tc | {"api": "m", "base_url": "https://x.example/v1"}, 1)._doc_luong(iter(sse))["ky_tu_suy_luan"] == len("nghi nhieu"))

    class ApiGemma(ApiGia):
        def _post(self, payload):
            if any(k in payload for k in ("thinking", "enable_thinking", "chat_template_kwargs")):
                return 400, '{"error":{"code":400,"message":"Invalid JSON payload received. Unknown name \\"thinking\\": Cannot find field."}}', None
            if payload.get("reasoning_effort") == "none" and not getattr(self, "khong_tat_duoc", False):
                noi = "Xin chào"
            else:
                noi = "<thought>Source language: Chinese. Translating the greeting</thought>Xin chào"
            lines = [b"data: " + json.dumps({"choices": [{"delta": {"content": noi}, "finish_reason": "stop"}]}).encode(), b"data: [DONE]"]
            return 200, self._doc_luong(iter(lines)), None

    cfg_gm = cfg_tc | {"api": "gemma-4-31b-it", "base_url": "https://x.example/v1"}
    bgm = ApiGemma(cfg_gm, 1)
    bgm.dem_usage = False
    bgm.thu_ket_noi(cfg_tc["extra"])
    kiem("Gemma: các trường lạ bị 400 thì bỏ qua, chọn reasoning_effort=none vì không còn suy nghĩ", bgm.extra == {"reasoning_effort": "none"})
    bgk = ApiGemma(cfg_gm, 1)
    bgk.dem_usage = False
    bgk.khong_tat_duoc = True
    try:
        bgk.thu_ket_noi(cfg_tc["extra"])
        dung_gm = False
    except SystemExit as e:
        dung_gm = "suy luận" in str(e)
    kiem("Gemma: không cách nào tắt được suy nghĩ thì dừng (mặc định, tránh chạy chậm vô ích)", dung_gm)
    bgc = ApiGemma(dict(cfg_gm, cho_suy_luan=True), 1)
    bgc.dem_usage = False
    bgc.khong_tat_duoc = True
    bgc.thu_ket_noi(cfg_tc["extra"])
    kiem("Gemma: có --cho_phep_suy_luan thì vẫn chạy được dù không tắt được suy nghĩ", True)
    with tempfile.TemporaryDirectory() as td3:
        td3 = Path(td3)
        pd.DataFrame({"ma_don_vi": [f"G{i}" for i in range(3)], "loai": "gia_tri", "name_zh": "颜色", "category": "bags",
                      "van_ban_gui": ["红色", "蓝色", "绿色"]}).to_parquet(td3 / "dv.parquet")

        class BeRong(ApiGia):
            def sinh(self, msgs, max_new, lap):
                return [("", "length") if "红色" in m[-1]["content"] else ("Xanh", "stop") for m in msgs]
        goc2 = dich.tai_backend
        dich.tai_backend = lambda cfg, backend, s=8, e=None: (BeRong(cfg, 1), "api", cfg["api"], None)
        sys.argv = ["dich.py", "--vao", str(td3 / "dv.parquet"), "--ra", str(td3 / "ra"), "--mo_hinh", "deepseek_alibaba", "--ngan_sach_usd", "1"]
        try:
            dich.main()
        finally:
            dich.tai_backend = goc2
        got3 = pd.concat([pd.read_parquet(f) for f in (td3 / "ra" / "deepseek_alibaba").glob("lo_*.parquet")])
        kiem("API: câu trả lời rỗng (bị cắt trước khi ra bản dịch) không được ghi, lần sau dịch lại; 2 đơn vị kia được ghi",
             sorted(got3.ma_don_vi) == ["G1", "G2"])
    class ApiCat(ApiGia):
        def _post(self, payload):
            self.lan = getattr(self, "lan", 0) + 1
            if self.lan == 1:     # luồng đứt giữa chừng: có chữ nhưng không có finish_reason
                lines = [b"data: " + json.dumps({"choices": [{"delta": {"content": "G-JM291màu đen, G-"}}]}).encode()]
            else:
                lines = [b"data: " + json.dumps({"choices": [{"delta": {"content": "G-JM291 màu đen, G-JM291 màu be"}, "finish_reason": "stop"}]}).encode(),
                         b"data: [DONE]"]
            return 200, self._doc_luong(iter(lines)), None
    bc = ApiCat(cfg_tc | {"api": "m", "base_url": "https://x.example/v1"}, 1)
    bc.dem_usage = False
    kc = bc._mot([{"role": "user", "content": "c-1"}], 100)
    kiem("luồng đứt giữa chừng (không có finish_reason): không nhận bản dịch cụt, thử lại rồi lấy bản đủ",
         kc == ("G-JM291 màu đen, G-JM291 màu be", "stop") and bc.tk["luong_bi_cat"] == 1 and bc.tk["yeu_cau"] == 1)
    # --- gộp nhiều đơn vị mỗi yêu cầu ---
    import re as _re
    pt = dich.phan_tich_gop
    kiem("gộp: bóc đúng 3 mục", pt("[1] Màu đỏ\n[2] Màu xanh\n[3] Tốt", 3) == {1: "Màu đỏ", 2: "Màu xanh", 3: "Tốt"})
    kiem("gộp: dòng thừa ở đầu và dòng trống được bỏ qua", pt("Đây là bản dịch:\n\n[1] A\n\n[2] B\n", 2) == {1: "A", 2: "B"})
    kiem("gộp: thiếu một mục thì mục đó không có trong kết quả", pt("[1] A\n[3] C", 3) == {1: "A", 3: "C"})
    kiem("gộp: mục trùng số bị bỏ cả hai", pt("[1] A\n[2] B\n[2] BB\n[3] C", 3) == {1: "A", 3: "C"})
    kiem("gộp: bản dịch trải nhiều dòng thì không tin mục đó", pt("[1] A\ndòng lạc\n[2] B", 2) == {2: "B"})
    kiem("gộp: số ngoài phạm vi bị bỏ", pt("[1] A\n[9] Z", 2) == {1: "A"})
    kiem("gộp: mục rỗng bị bỏ", pt("[1]\n[2] B", 2) == {2: "B"})
    rows_t = [("gia_tri", "颜色", "bags", "红色"), ("gia_tri", "颜色", "bags", "蓝色"), ("ten", "", "bags", "颜色"),
              ("gia_tri", "材质", "bags", "长" * 700), ("gia_tri", "材质", "bags", "有\n换行"), ("gia_tri", "材质", "bags", "皮革")]
    nh, ri = dich.chia_gop(rows_t, 5)
    kiem("chia gộp: tách theo loại, đơn vị dài hoặc có xuống dòng đi riêng", nh == [[0, 1], [2], [5]] and ri == [3, 4])
    nh2, _ = dich.chia_gop([("gia_tri", "a", "bags", "字" * 500)] * 6, 20)
    kiem("chia gộp: tổng ký tự mỗi yêu cầu không vượt giới hạn (6 đơn vị 500 ký tự: nhóm 4 rồi nhóm 2)",
         all(500 * len(g) <= dich.GOP_MAX_KY_TU_MOT_YEU_CAU for g in nh2) and [len(g) for g in nh2] == [4, 2])
    kiem("chia gộp: tối đa n đơn vị mỗi nhóm", all(len(g) <= 3 for g in dich.chia_gop([("gia_tri", "a", "bags", "字")] * 10, 3)[0]))

    class BeGop:
        def __init__(self, bo_muc=None, hong_api=False, cat=False):
            self.bo_muc, self.hong_api, self.cat, self.so_goi_gop, self.so_goi_rieng = bo_muc, hong_api, cat, 0, 0

        def sinh(self, msgs, max_new, lap):
            out = []
            for m in msgs:
                u = m[-1]["content"]
                if u.startswith(dich.HD_GOP[:25]):
                    self.so_goi_gop += 1
                    if self.hong_api:
                        out.append(("", "loi_api"))
                        continue
                    it = _re.findall(r"Item (\d+)\ncontext: [^\n]*\ntext: ([^\n]*)", u)
                    dong = [f"[{k}] G:{t}" for k, t in it if int(k) != self.bo_muc]
                    out.append(("\n".join(dong), "length" if self.cat else "stop"))
                else:
                    self.so_goi_rieng += 1
                    out.append(("R:" + u.splitlines()[-1], "stop"))
            return out

    def chuan_bi_lo(rows):
        return ([dich.cau_lenh("chat", l, n, c, t) for l, n, c, t in rows], [dich.so_token_toi_da(t) for _, _, _, t in rows])

    rows_g = [("gia_tri", "颜色", "bags", t) for t in ["红色", "蓝色", "绿色", "黄色"]] + [("ten", "", "bags", "颜色")]
    md, td = chuan_bi_lo(rows_g)
    be1 = BeGop()
    kq1, tk1 = dich.dich_gop(be1, rows_g, md, td, 1.0, 3)
    kiem("dich_gop: bản dịch đúng thứ tự từng đơn vị, 2 yêu cầu gộp cho nhóm gia_tri (3+1) và 1 cho ten",
         [x for x, _ in kq1] == ["G:红色", "G:蓝色", "G:绿色", "G:黄色", "G:颜色"] and be1.so_goi_gop == 3 and be1.so_goi_rieng == 0
         and tk1["yeu_cau_gop"] == 3 and tk1["don_vi_roi_ve_rieng"] == 0)
    be2 = BeGop(bo_muc=2)
    kq2, tk2 = dich.dich_gop(be2, rows_g, md, td, 1.0, 3)
    kiem("dich_gop: mô hình bỏ sót mục 2 thì đơn vị đó được dịch riêng, các đơn vị khác giữ bản gộp",
         [x for x, _ in kq2][:3] == ["G:红色", "R:蓝色", "G:绿色"] and be2.so_goi_rieng == 1 and tk2["don_vi_roi_ve_rieng"] == 1)
    be3 = BeGop(hong_api=True)
    kq3, tk3 = dich.dich_gop(be3, rows_g, md, td, 1.0, 3)
    kiem("dich_gop: yêu cầu gộp lỗi API thì các đơn vị của nó là loi_api (không dịch riêng thêm, không tốn thêm)",
         all(r == "loi_api" for _, r in kq3) and be3.so_goi_rieng == 0 and tk3["yeu_cau_gop_loi_api"] == 3)
    be4 = BeGop(cat=True)
    kq4, tk4 = dich.dich_gop(be4, rows_g[:3], md[:3], td[:3], 1.0, 3)
    kiem("dich_gop: câu trả lời bị cắt (length) thì bỏ mục cuối và dịch riêng mục đó",
         [x for x, _ in kq4] == ["G:红色", "G:蓝色", "R:绿色"] and tk4["don_vi_roi_ve_rieng"] == 1)
    rows_r = rows_t[3:5]
    mr, tr_ = chuan_bi_lo(rows_r)
    be5 = BeGop()
    kq5g, tk5 = dich.dich_gop(be5, rows_r, mr, tr_, 1.0, 5)
    kiem("dich_gop: đơn vị quá dài hoặc có xuống dòng luôn đi riêng, không vào yêu cầu gộp",
         be5.so_goi_gop == 0 and be5.so_goi_rieng == 2 and tk5["don_vi_di_rieng"] == 2)
    # --- Alibaba Cloud: thử các cách tắt suy luận; địa chỉ gọi ghi đè được ---
    ba = ApiGia(cfg_al, 2)
    ba.thu_ket_noi(cfg_al["extra"])
    kiem("Alibaba: cách enable_thinking=false không có tác dụng thì thử tiếp, chọn thinking.type=disabled (không còn suy luận)",
         ba.extra == {"thinking": {"type": "disabled"}} and len(ba.tk["goi_thu"]) == 2 and ba.tk["goi_thu"][0]["ky_tu_suy_luan"] > 0)
    ba = ApiGia(cfg_al, 2)
    ba.nhan_enable = True
    ba.thu_ket_noi(cfg_al["extra"])
    kiem("Alibaba: enable_thinking=false có tác dụng thì chọn luôn cách đầu, dừng thử", ba.extra == {"enable_thinking": False} and len(ba.tk["goi_thu"]) == 1)

    class ApiKhongTat(ApiGia):
        def _post(self, payload):
            payload = {k: v for k, v in payload.items() if k not in ("thinking", "enable_thinking")}
            return super()._post(payload)
    try:
        ApiKhongTat(cfg_al, 2).thu_ket_noi(cfg_al["extra"])
        dung_lai = False
    except SystemExit as e:
        dung_lai = "suy luận" in str(e)
    kiem("mô hình trả phí mà không tắt được suy luận: dừng, không chạy tiếp", dung_lai)
    with tempfile.TemporaryDirectory() as td2:
        td2 = Path(td2)
        pd.DataFrame({"ma_don_vi": ["G0"], "loai": "gia_tri", "name_zh": "颜色", "category": "bags", "van_ban_gui": ["红色"]}).to_parquet(td2 / "dv.parquet")
        thay = {}
        goc_tb = dich.tai_backend

        def gia_tb(cfg, backend, s=8, e=None):
            thay["base_url"] = cfg["base_url"]
            return ApiGia(cfg, 1), "api", cfg["api"], None
        dich.tai_backend = gia_tb
        sys.argv = ["dich.py", "--vao", str(td2 / "dv.parquet"), "--ra", str(td2 / "ra"), "--mo_hinh", "deepseek_alibaba",
                    "--api_base_url", "https://abc.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "--ngan_sach_usd", "1"]
        try:
            dich.main()
        finally:
            dich.tai_backend = goc_tb
        kiem("--api_base_url ghi đè địa chỉ gọi của mô hình", thay.get("base_url") == "https://abc.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")

    # --- API trả phí: ngân sách, chi phí, tham số gửi đi ---
    bp = ApiGia(cfg_p, 1)
    bp.dem_usage = False          # máy chủ giả không nhận stream_options
    ms5 = [[{"role": "user", "content": f"z-{i:06d}"}] for i in range(5)]
    kq5 = bp.sinh(ms5, [10] * 5, 1.0)
    pl = bp.cac_payload[0]
    kiem("API trả phí: gửi đúng mô hình, tắt suy luận, nhiệt độ 0, token ra = ước + 256",
         pl["model"] == "deepseek-flash" and pl["thinking"] == {"type": "disabled"} and pl["temperature"] == 0.0 and pl["max_tokens"] == 10 + 256)
    kiem("API trả phí: ước chi phí mỗi lần gọi = (5 token vào × 0,30 + 3 token ra × 1,20) / 1 triệu",
         abs(bp.chi_phi - 5 * (5 * 0.30 + 3 * 1.20) / 1e6) < 1e-12 and abs(bp.tk["chi_phi_uoc_usd"] - round(bp.chi_phi, 5)) < 1e-9)
    bp = ApiGia(cfg_p, 1)
    bp.dem_usage = False
    bp.ngan_sach = 2.5 * (5 * 0.30 + 3 * 1.20) / 1e6      # đủ cho 3 lần gọi (chạm sau lần thứ 3)
    kq5 = bp.sinh(ms5, [10] * 5, 1.0)
    kiem("API trả phí: chạm ngân sách thì ngừng gửi yêu cầu mới (3 gọi, 2 bị chặn), phần chặn trả loi_api",
         bp.tk["yeu_cau"] == 3 and bp.tk["het_ngan_sach"] == 2 and [r for _, r in kq5].count("loi_api") == 2 and len(bp.cac_payload) == 3)
    kiem("API trả phí: ước chi phí dư khi máy chủ không trả thống kê token",
         ApiGia(cfg_p, 1)._tinh_phi([{"role": "user", "content": "x" * 400}], {}, "y" * 100) > 0)
    # --- đường HTTP thật (urllib + luồng SSE) với máy chủ giả trên máy này, mô phỏng định dạng của DeepSeek ---
    import http.server
    import threading as _th
    nhan = {}

    class May(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            nhan["thu"] = self.headers.get("Authorization")
            nhan["dich"] = self.path
            nhan["body"] = json.loads(self.rfile.read(n).decode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for g in [{"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                      {"choices": [{"delta": {"content": "Màu "}}]},
                      {"choices": [{"delta": {"content": "đỏ"}, "finish_reason": "stop"}]},
                      {"model": "deepseek-flash", "choices": [],
                       "usage": {"prompt_tokens": 100, "completion_tokens": 4, "prompt_cache_hit_tokens": 90, "prompt_cache_miss_tokens": 10}}]:
                self.wfile.write(b"data: " + json.dumps(g).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), May)
    _th.Thread(target=srv.serve_forever, daemon=True).start()
    cfg_h = dict(cfg_p, base_url=f"http://127.0.0.1:{srv.server_address[1]}")
    bh = dich.API(cfg_h, 1)
    kq_h = bh._mot([{"role": "system", "content": "s"}, {"role": "user", "content": "红色"}], 10)
    srv.shutdown()
    kiem("HTTP thật: đường dẫn /chat/completions, khoá gửi qua Authorization Bearer, thân yêu cầu đúng",
         nhan.get("dich") == "/chat/completions" and nhan.get("thu") == "Bearer khoa-gia-de-thu-2"
         and nhan["body"]["model"] == "deepseek-flash" and nhan["body"]["stream"] is True and nhan["body"]["thinking"] == {"type": "disabled"})
    kiem("HTTP thật: gộp luồng SSE ra 'Màu đỏ', đếm đúng token trúng bộ nhớ đệm khi tính phí",
         kq_h == ("Màu đỏ", "stop") and abs(bh.chi_phi - (10 * 0.30 + 90 * 0.006 + 4 * 1.20) / 1e6) < 1e-12)
    be.han_chot = time.time() - 1
    n0 = sum(be.dem.values())
    kiem("API: quá hạn chót thì không gửi yêu cầu mới, trả loi_api",
         be._mot([{"role": "user", "content": "y-123456"}], 100) == ("", "loi_api") and sum(be.dem.values()) == n0 and be.tk["het_gio"] == 1)

    # chạy cả dich.py: đơn vị lỗi API không được ghi, lần sau chạy tiếp chỉ dịch phần thiếu
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        dv = pd.DataFrame({"ma_don_vi": [f"G{i}" for i in range(6)], "loai": "gia_tri", "name_zh": "颜色", "category": "bags",
                           "van_ban_gui": ["红色", "蓝色", "绿色401", "黄色", "黑色", "白色"]})
        dv.to_parquet(td / "dv.parquet")
        gia = ApiGia(cfg_a, 2)
        dich.tai_backend = lambda cfg, backend, s=8, e=None: (gia, "api", "x", None)
        sys.argv = ["dich.py", "--vao", str(td / "dv.parquet"), "--ra", str(td / "ra"), "--mo_hinh", "deepseek_v41_flash", "--lo", "3"]
        dich.main()
        got = pd.concat([pd.read_parquet(f) for f in (td / "ra" / "deepseek_v41_flash").glob("lo_*.parquet")])
        kiem("dich.py API: đơn vị lỗi không được ghi, 5 đơn vị còn lại được ghi", len(got) == 5 and "G2" not in set(got.ma_don_vi))
        gia.loi401 = False
        dich.main()
        got = pd.concat([pd.read_parquet(f) for f in (td / "ra" / "deepseek_v41_flash").glob("lo_*.parquet")])
        kiem("dich.py API: chạy lại chỉ dịch phần thiếu, không trùng", len(got) == 6 and got.ma_don_vi.is_unique)
        sys.argv = ["dich.py", "--vao", str(td / "dv.parquet"), "--ra", str(td / "ra_phi"), "--mo_hinh", "deepseek_flash_api"]
        try:
            dich.main()
            doi_ngan_sach = False
        except SystemExit as e:
            doi_ngan_sach = "ngan_sach_usd" in str(e)
        kiem("dich.py: mô hình trả phí mà không đặt --ngan_sach_usd thì từ chối chạy", doi_ngan_sach)
        the_a = json.loads((td / "ra" / "the_deepseek_v41_flash.json").read_text(encoding="utf-8"))
        kiem("thẻ ghi số yêu cầu API và không chứa khoá", the_a["api"]["yeu_cau"] > 0 and "khoa-gia-de-thu" not in json.dumps(the_a))
finally:
    time.sleep = _ngu
    os.environ.pop("NVIDIA_API_KEY", None)

print(f"\n{DAT}/{TONG} ca đạt")
sys.exit(0 if DAT == TONG else 1)
