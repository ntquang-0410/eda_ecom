# -*- coding: utf-8 -*-
"""Thử pipeline bộ song ngữ thứ hai (Tiki Việt -> Trung) bằng câu tổng hợp: định tuyến, luật kiểm bản gương, câu lệnh dich.py
chiều vi_zh, bước chuẩn bị trên dữ liệu giả. Không đọc dữ liệu thật, không cần GPU. Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_2\\test_song_ngu_2.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "song_ngu_1" / "kaggle"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_2 import chuan_bi  # noqa: E402
from song_ngu_2 import cau_hinh as C  # noqa: E402
from song_ngu_2.dinh_tuyen import dinh_tuyen, la_do_luong  # noqa: E402
from song_ngu_2.luat_kiem import go_nhan, kiem_mot, lech_so, ma_bat_buoc, mau_anh_chua_dich, muc_khop  # noqa: E402
import dich  # noqa: E402

DAT, TONG = 0, 0


def kiem(ten, dung):
    global DAT, TONG
    TONG += 1
    DAT += bool(dung)
    print(("OK   " if dung else "SAI  ") + ten)


def tr(vi, zh):
    return kiem_mot(vi, vi, zh)


# --- định tuyến ---
kiem("thương hiệu chép nguyên (chữ Latin)", dinh_tuyen("Thương hiệu", "Vinamilk")[0] == "ten_hang")
kiem("thương hiệu có dấu Việt vẫn chép nguyên (tên hãng không dịch)", dinh_tuyen("Thương hiệu", "Trung Nguyên Legend")[0] == "ten_hang")
kiem("thương hiệu OEM chép nguyên", dinh_tuyen("Thương hiệu", "OEM")[0] == "ten_hang")
kiem("giá trị chỉ có số: chép nguyên", dinh_tuyen("Khối lượng (kg)", "0.126")[0] == "chep_nguyen")
kiem("số + đơn vị đo: chép nguyên", dinh_tuyen("Kích thước", "12.5 x 7 x 0.5 cm")[0] == "do_luong")
kiem("danh sách đo: chép nguyên", dinh_tuyen("Dung tích", "10ml; 15ml; 30ml")[0] == "do_luong")
kiem("số + chữ Việt không phải đơn vị: gửi mô hình", dinh_tuyen("Số lượng", "1 cái; 2 cái")[0] == "mo_hinh")
kiem("la_do_luong: có chữ Việt có dấu thì không", not la_do_luong("5 ngày"))
kiem("mã model chép nguyên", dinh_tuyen("Model", "TQ-03")[0] == "ma_model")
kiem("Model là chữ thường (Mới nhất): gửi mô hình", dinh_tuyen("Model", "Mới nhất")[0] == "mo_hinh")
kiem("Model: tên có dấu hai chấm cuối vẫn nhận", dinh_tuyen("Model:", "KS-TU720")[0] == "ma_model")
kiem("cỡ chữ S; M; L; XL: chép nguyên", dinh_tuyen("Size", "S; M; L; XL; 2XL")[0] == "co_chu")
kiem("cỡ chữ không nhầm với từ", dinh_tuyen("Size", "Free size")[0] == "mo_hinh")
kiem("màu bằng từ Anh: gửi mô hình", dinh_tuyen("Màu sắc", "Black; Silver")[0] == "mo_hinh")
kiem("tên đơn vị chịu trách nhiệm: cất riêng", dinh_tuyen("Tên đơn vị/tổ chức chịu trách nhiệm về hàng hóa", "Công ty ABC")[:2] == ("van_loai", "to_chuc"))
kiem("địa chỉ tổ chức: cất riêng", dinh_tuyen("Địa chỉ tổ chức chịu trách nhiệm về hàng hóa", "Hà Nội")[0] == "van_loai")
kiem("số điện thoại: cất riêng", dinh_tuyen("Lưu ý", "Gọi 0985419715")[:2] == ("van_loai", "lien_he"))
kiem("số điện thoại có dấu chấm: cất riêng", dinh_tuyen("Chọn số", "0704.054.866; 0705.335.866")[0] == "van_loai")
kiem("địa chỉ web: cất riêng", dinh_tuyen("Thương hiệu", "Goce.vn")[0] == "van_loai")
kiem("zalo: cất riêng", dinh_tuyen("Lưu ý", "Liên hệ zalo để được tư vấn")[0] == "van_loai")
kiem("mô tả thường không bị cất riêng", dinh_tuyen("Chất liệu", "Cotton cao cấp")[0] == "mo_hinh")
kiem("mã vạch 13 chữ số không phải điện thoại", dinh_tuyen("Mã vạch", "8936038391234")[0] == "chep_nguyen")

# --- làm sạch giá trị như bước EDA ---
from song_ngu_2.chuan_bi import lam_gia_tri, lam_ten  # noqa: E402
kiem("làm sạch: bỏ emoji ✅ ở đầu", lam_gia_tri("✅ Chống nước") == "Chống nước")
kiem("làm sạch: bỏ hashtag, giữ phần còn lại", lam_gia_tri("Đèn LED #ĐènBắtMuỗi #MáyBắtMuỗi") == "Đèn LED")
kiem("làm sạch: giá trị toàn hashtag thành rỗng", lam_gia_tri("#ĐènBắtMuỗi #MáyBắtMuỗi") == "")
kiem("làm sạch: gạch đầu dòng '- ' ở đầu bị bỏ", lam_gia_tri("- Dùng cho xe máy") == "Dùng cho xe máy")
kiem("làm sạch: mã model có dấu # giữa chữ (A#1) không bị coi là hashtag", lam_gia_tri("Mẫu A#1") == "Mẫu A#1")
kiem("làm sạch: dấu ':' cuối tên bị bỏ", lam_ten("Model:") == "Model")

# --- luật kiểm bản gương: bản dịch đúng thì qua ---
kiem("bản dịch đúng, chữ Hán không bị coi là lỗi", tr("Màu đỏ", "红色")[0] == [])
kiem("danh sách 3 mục dùng ', '", tr("Đen, Trắng, Xám", "黑色, 白色, 灰色")[0] == [])
kiem("số thập phân Việt 1,5kg thành 1.5kg", tr("Trọng lượng 1,5kg", "重量1.5kg")[0] == [])
kiem("hàng nghìn Việt 1.000ml thành 1000ml", tr("Dung tích 1.000ml", "容量1000ml")[0] == [])
kiem("10.000 mAh thành 1万毫安时", tr("Pin 10.000 mAh", "电池1万毫安时")[0] == [])
kiem("ngày 24/05/2025 thành 2025年5月24日 (số 05 thành 5)", tr("Ngày sản xuất 24/05/2025", "生产日期2025年5月24日")[0] == [])
kiem("hai cái thành 2个 hoặc 两个 không bị nhắc số thừa", "so_thua" not in tr("Hai cái", "2个")[1])
kiem("chữ Việt in hoa không dấu không bị coi là mã", tr("MAU XANH DUONG", "蓝色")[0] == [])
kiem("giữ mã model ABC-123", tr("Model: ABC-123", "型号：ABC-123")[0] == [])
kiem("mã chữ-số có phần chữ không phải đơn vị: bắt buộc giữ", ma_bat_buoc("Mã ABC-123 màu đỏ") == ["ABC-123"])
kiem("20cm là số + đơn vị, không bắt buộc giữ nguyên chữ", ma_bat_buoc("Dài 20cm, nặng 1,5kg, 12V") == [])
kiem("20cm dịch thành 20厘米 thì qua", tr("Dài 20cm", "长20厘米")[0] == [])
kiem("mất mã ABC-123 thì trượt", "mat_ma" in tr("Model: ABC-123", "型号：未知")[0])
kiem("giữ USB thì qua", tr("Cổng USB", "USB接口")[0] == [])
kiem("viết tắt tiếng Việt NSX dịch ra 自生产之日起 thì qua", tr("3 năm kể từ NSX", "自生产之日起3年")[0] == [])
kiem("3h dịch ra 3小时 thì qua (h là đơn vị giờ)", tr("Khoảng 3h", "约 3 小时")[0] == [])
kiem("S11-Kaki: chỉ đòi S11, Kaki dịch ra 卡其色 là đúng", ma_bat_buoc("S11-Kaki") == ["S11"] and tr("S11-Kaki", "S11-卡其色")[0] == [])
kiem("S11-Kaki mất S11 thì trượt", "mat_ma" in tr("S11-Kaki", "卡其色")[0])
kiem("dấu ** có ở nguồn thì bản dịch giữ nguyên không bị coi là lời thừa", "loi_thua" not in tr("Niacinamide*, Panthenol**", "烟酰胺*, 泛醇**")[0])
kiem("dấu ** mô hình tự thêm thì trượt loi_thua", "loi_thua" in tr("Màu đỏ", "**红色**")[0])
kiem("giá trị dài mở đầu bằng tiêu đề dịch ra 成分： không trượt, chỉ nhắc nhan_dau",
     tr("Thành phần Bột mì 67,16%, dầu thực vật 8,87%, ớt 6,45%, muối, bột tỏi", "成分：小麦粉 67.16%, 植物油 8.87%, 辣椒 6.45%, 食盐, 蒜粉")[0] == []
     and "nhan_dau" in tr("Thành phần Bột mì 67,16%, dầu thực vật 8,87%, ớt 6,45%, muối, bột tỏi", "成分：小麦粉 67.16%, 植物油 8.87%, 辣椒 6.45%, 食盐, 蒜粉")[1])
kiem("go_nhan: nguồn mở đầu bằng chính tên thuộc tính thì không gỡ", go_nhan("成分：小麦粉 67%", "Thành phần Bột mì 67%", "成分", "Thành phần") is None)
# --- sửa sau lần chạy thật 23/10/2026 (A trượt 371/13.518, nhiều ca oan) ---
kiem("tên riêng Việt giữ nguyên chữ (Nhà Chíp, Đen Đá) trong bản Trung có chữ Hán: chỉ nhắc viet_giu",
     kiem_mot("Đặt bởi shop Nhà Chíp", "Đặt bởi shop Nhà Chíp", "由 Nhà Chíp 店铺提供")[0] == [] and "viet_giu" in kiem_mot("x", "Hòa một gói Đen Đá với nước", "将一包 Đen Đá 加入水中")[1])
kiem("chữ Việt viết thường còn sót (đỏ) vẫn trượt viet_sot", "viet_sot" in tr("Màu đỏ", "红色 đỏ")[0])
kiem("bản dịch không có chữ Hán mà còn chữ Việt (để nguyên) vẫn trượt viet_sot", "viet_sot" in tr("U Mũ", "U Mũ")[0])
kiem("tên thuộc tính (loai = ten) còn chữ Việt viết hoa vẫn trượt", "viet_sot" in kiem_mot("Mẫu SP", "Mẫu SP", "Mẫu 产品", loai="ten")[0])
kiem("quá 4 từ Việt có dấu thì không còn là tên riêng: trượt", "viet_sot" in tr("Nấu Món Ăn Rất Ngọt Và Đậm Đà Cho Cả Nhà", "Nấu Món Ăn Rất Ngọt Và Đậm Đà Cho 家人")[0])
kiem("nguồn dài mất 1 số (4 muỗng ... 1 ly) chỉ nhắc so_thieu, không trượt", (lambda r: "lech_so" not in r[0] and "so_thieu" in r[1])(
     tr("Hòa tan 4 muỗng (khoảng 22g) bột Milo với 100ml nước nóng vào 1 ly rồi khuấy đều, thêm đá và thưởng thức ngay khi còn lạnh nhé các bạn", "将 4 勺（约 22g）Milo 粉倒入杯中，加入 100ml 热水搅拌均匀，加入冰块即可享用")))
kiem("nguồn ngắn mất số vẫn trượt lech_so", "lech_so" in tr("2 vít 4 cạnh", "螺丝")[0])
kiem("nguồn dài mất 5 số trong 8 số vẫn trượt", "lech_so" in tr("Mã A1 B2 C3 D4 E5 F6 G7 H8 và các loại sản phẩm khác dùng cho nhiều mục đích sử dụng khác nhau trong gia đình và văn phòng", "各种产品")[0])
kiem("hai cụm ngắn 'Vải nỉ mỏng, nhẹ' dịch gộp thành một cụm: không trượt số mục", tr("Vải nỉ mỏng, nhẹ", "轻薄绒布")[0] == [])
kiem("hai mục ngắn 'Hồng, Tím' mất một mục vẫn trượt", "lech_so_muc" in tr("Hồng, Tím", "粉色")[0])
kiem("D54xR28xC5 dịch ra 长54x宽28x高5: D, R, C là chữ viết tắt kích thước, không phải mã", ma_bat_buoc("D54xR28xC5") == [] and tr("Kích thước: D66 x R44 x C57 (cm)", "尺寸: 直径66 x 宽44 x 高57 (cm)")[0] == [])
kiem("千斤顶 (cái kích) chứa chữ 斤 nhưng đúng", "jin_sai" not in tr("Kích nâng ô tô", "汽车千斤顶")[0])
kiem("nguồn tự lặp tên hãng ×4 thì bản dịch lặp theo không trượt lap_ky_tu",
     "lap_ky_tu" not in tr("Lót Chuột FIRO - Pad Chuột FIRO - Tấm Lót Chuột FIRO - Miếng Lót Chuột FIRO - FIRO", "FIRO 鼠标垫 - FIRO 鼠标垫 - FIRO 鼠标垫 - FIRO 鼠标垫 - FIRO")[0])
kiem("vòng lặp thật (澳洲 ×40) vẫn trượt lap_ky_tu", "lap_ky_tu" in tr("100% Hạt Macca nhập khẩu úc.", "100% 澳洲进口" + "澳洲" * 40)[0])
kiem("2mx2 (hai mét nhân hai) không phải mã: dịch ra 2米x2米 thì qua", tr("1m6x2m, 1m8x2m, 2mx2,2m", "1米6x2米, 1米8x2米, 2米x2,2米")[0] == [])
kiem("一包Nescafé: từ Việt/Pháp có dấu nằm sát chữ Hán vẫn nhận ra là tên riêng (viet_giu)", "viet_giu" in kiem_mot("x", "Hòa một gói Nescafé 3IN1 với nước", "将一包Nescafé 3IN1与热水混合")[1] and tr("Hòa một gói Nescafé 3IN1 với nước", "将一包Nescafé 3IN1与热水混合")[0] == [])
kiem("mã có chữ Đ giữ nguyên (ĐQ EMR09, CVG35Đ, BTĐ): chỉ nhắc ma_viet", tr("ĐQ EMR09", "ĐQ EMR09")[0] == [] and tr("CVG35N, CVG35Đ", "CVG35N, CVG35Đ")[0] == [] and tr("BTĐ", "BTĐ")[0] == [])
kiem("chữ Việt in hoa 4 chữ trở lên giữ nguyên (TRÙN QUẾ) vẫn trượt", "viet_sot" in tr("1 TRÙN QUẾ 2KG SFARM", "1 TRÙN QUẾ 2KG SFARM")[0])
kiem("phồn thể 銀色 trượt phon_the", "phon_the" in tr("Bạc", "銀色")[0])
kiem("go_nhan: nguồn không có tiêu đề, bản dịch tự thêm 成分： thì gỡ", go_nhan("成分：茄子 (60%), 虾酱", "Cà pháo (60%), nước mắm tôm", "成分", "Thành phần") == "茄子 (60%), 虾酱")
kiem("danh sách đủ 4 mục thì qua", tr("Xanh, Trắng, Be, Hồng", "蓝色, 白色, 米色, 粉色")[0] == [])
kiem("danh sách thiếu mục thì trượt", "lech_so_muc" in tr("Xanh, Trắng, Be, Hồng", "蓝色, 白色, 粉色")[0])
kiem("mô hình đổi , thành 、 vẫn đủ số mục thì qua", muc_khop("Xanh, Trắng, Be", "蓝色、白色、米色"))
kiem("một mục: không đòi dấu ngăn", muc_khop("Cotton cao cấp", "优质棉"))
kiem("câu văn dài có dấu phẩy không phải danh sách: không kiểm số mục", muc_khop("Bảo quản nơi khô ráo, thoáng mát, tránh ánh nắng trực tiếp của mặt trời", "存放于干燥通风处避免阳光直射"))
kiem("số thập phân 1,5 không tách mục (bản Trung giữ dấu phẩy thập phân)", muc_khop("Dài 1,5m, rộng 2m", "长1,5m, 宽2m"))

# --- sửa sau lần thử Gemma 23/10/2026 (35/200 trượt, nhiều ca oan) ---
kiem("danh sách mã 'QC1505, QC1507' giữ nguyên: đủ 2 mục (dấu phẩy sau chữ số không phải số thập phân)", tr("QC1505, QC1507", "QC1505, QC1507")[0] == [])
kiem("danh sách cỡ 'L42, XL40, XXL38, XXXL36' giữ nguyên thì qua", tr("L42, XL40, XXL38, XXXL36", "L42, XL40, XXL38, XXXL36")[0] == [])
kiem("mục dính dấu phẩy không cách 'Hồng,0-6m Lime' tách thành 'Hồng, 0-6m' vẫn qua",
     "lech_so_muc" not in tr("Hồng,0-6m Lime, Xanh,6-18m Lime", "粉色, 0-6m 青柠色, 蓝色, 6-18m 青柠色")[0])
kiem("danh sách bị mất 1 mục vẫn trượt sau khi nới", "lech_so_muc" in tr("Xanh, Trắng, Be, Hồng", "蓝色, 白色, 粉色")[0])
kiem("'1PCS-Blue' dịch thành '1PCS-蓝色': Blue là từ, không phải mã", tr("1PCS-Blue, 1PCS-Dark Blue", "1PCS-蓝色, 1PCS-深蓝色")[0] == [])
kiem("'4in1' dịch thành '4合1' là đúng", tr("Hub 4in1", "4合1集线器")[0] == [])
kiem("'MATCHA' trong dòng viết hoa nhiều (Cacao, DÂU, MATCHA) dịch ra 抹茶 là đúng", tr("Cacao, DÂU, KHOAI MÔN, MATCHA, TRỨNG", "可可, 草莓, 香芋, 抹茶, 鸡蛋")[0] == [])
kiem("từ in hoa (USB, SX, NSX, BH, TB) không còn bị đòi giữ: phần lớn là chữ viết tắt tiếng Việt", ma_bat_buoc("Cổng USB") == [] and ma_bat_buoc("3 năm kể từ NSX") == [] and ma_bat_buoc("Cỡ TB 32 x 32 cm") == [])
kiem("mã lẫn hoa thường iPhone15 vẫn bắt buộc", ma_bat_buoc("Ốp iPhone15") == ["iPhone15"])
kiem("giá trị dài > 80 ký tự không kiểm mã (văn bản dài hay bỏ mã phụ)", ma_bat_buoc("Water, Glycerin, Acrylates/C10-30 Alkyl Acrylate Crosspolymer, PEG-60 Hydrogenated Castor Oil, Disodium EDTA") == [])
kiem("'2 ngăn' dịch thành 双格 là đúng", "lech_so" not in tr("Bộ kẻ mày 2 ngăn Mira", "Mira 双格眉笔套装")[0])
kiem("'1 ô' dịch thành 单格 là đúng", "lech_so" not in tr("Kích Thước 1 ô: 2.5 (cm)", "单格尺寸: 2.5 (cm)")[0])
kiem("go_nhan: 'Ý' -> '品牌产地: 意大利' gỡ nhãn khi trùng bản dịch tên thuộc tính", go_nhan("品牌产地: 意大利", "Ý", "品牌产地") == "意大利")
kiem("go_nhan: nhãn chứa bản dịch tên ('产地 (Made in): 意大利' với tên '原产地 (Made in)') cũng gỡ", go_nhan("产地 (Made in): 意大利", "Ý", "原产地 (Made in)") == "意大利")
kiem("go_nhan: nguồn có dấu hai chấm thì không gỡ", go_nhan("型号: ABC", "Model: ABC", "型号") is None)
kiem("go_nhan: nhãn không liên quan tên thuộc tính thì không gỡ (không đoán)", go_nhan("保修: 12个月", "Bảo hành 12 tháng", "颜色") is None)
kiem("go_nhan: bản dịch không có nhãn thì None", go_nhan("意大利", "Ý", "品牌产地") is None)
kiem("go_nhan: chưa có bản dịch tên thì không gỡ", go_nhan("品牌产地: 意大利", "Ý", None) is None)
kiem("dich.py vi_zh: câu lệnh cấm lặp tên thuộc tính", "Never output the attribute name" in dich.HE_THONG_VI_ZH and "never repeat the attribute name" in dich.HD_GOP_VI_ZH)

# --- luật kiểm: bản dịch sai thì trượt ---
kiem("sót tiếng Việt có dấu thì trượt", "viet_sot" in tr("Màu đỏ", "红色 đỏ")[0])
kiem("để nguyên bản Việt thì trượt viet_sot", "viet_sot" in tr("Màu đỏ", "Màu đỏ")[0])
kiem("bản dịch rỗng thì trượt", tr("Màu đỏ", "")[0] == ["rong"])
kiem("lời thừa 'Bản dịch:' tiếng Trung thì trượt", "loi_thua" in tr("Màu đỏ", "译文：红色")[0])
kiem("markdown **...** thì trượt", "loi_thua" in tr("Màu đỏ", "**红色**")[0])
kiem("lọt ngữ cảnh '产品类别' thì trượt", "lo_ngu_canh" in tr("Màu đỏ", "产品类别：食品 红色")[0])
kiem("gắn nhãn 'XX：' khi nguồn không có dấu hai chấm thì trượt", "lo_ngu_canh" in tr("Màu đỏ", "颜色：红色")[0])
kiem("nguồn có dấu hai chấm thì nhãn là đúng", tr("Model: ABC-123", "型号：ABC-123")[0] == [])
kiem("lặp chữ Hán (không dấu cách) thì trượt", "lap_ky_tu" in tr("Chất liệu", "材质" * 10)[0])
kiem("lặp chữ Hán mà nguồn cũng lặp thì không bắt", "lap_ky_tu" not in tr("哈哈哈哈哈", "哈哈哈哈哈")[0])
kiem("cân bị dịch thành 斤 thì trượt jin_sai", "jin_sai" in tr("Trọng lượng 5 cân", "重量5斤")[0])
kiem("cân dịch đúng 公斤 thì qua", tr("Trọng lượng 5 cân", "重量5公斤")[0] == [])
kiem("kg bị đổi thành 斤 thì trượt", "jin_sai" in tr("Trọng lượng 5kg", "重量10斤")[0])
kiem("mất số (đổi 5 thành 6) thì trượt lech_so", "lech_so" in tr("Dung tích 500ml", "容量600ml")[0])
kiem("bản Trung phồn thể thì trượt", "phon_the" in tr("Màu đỏ", "紅色")[0])
kiem("chữ giản thể không bị nhầm phồn thể (窗 và 惠 dùng chung)", tr("Cửa sổ", "窗户")[0] == [] and tr("Ưu đãi", "优惠")[0] == [])
kiem("lech_so: 1.000 và 1000 như nhau", not lech_so("1.000ml", "1000ml"))
kiem("lech_so: 1,5 và 1.5 như nhau", not lech_so("1,5kg", "1.5kg"))
kiem("lech_so: thiếu số 5 thì lệch", lech_so("5 cái", "个"))
kiem("không dịch (ASCII giữ nguyên) thì chỉ nhắc khong_dich", "khong_dich" in tr("Black; Silver", "Black; Silver")[1] and tr("Black; Silver", "Black; Silver")[0] == [])

# --- dich.py chiều vi_zh ---
dich.HUONG = "vi_zh"
m = dich.cau_lenh("qwen", "gia_tri", "Màu sắc", "food", "Đen, Trắng ngà")
kiem("dich.py vi_zh: câu lệnh hệ thống dịch sang Hán giản thể", "Simplified Chinese" in m[0]["content"] and "Tiki.vn" in m[0]["content"])
kiem("dich.py vi_zh: ngữ cảnh danh mục bằng tiếng Trung và tên thuộc tính Việt", "食品" in m[1]["content"] and "Màu sắc" in m[1]["content"])
g = dich.cau_lenh_gop([("ten", "", "home", "Thương hiệu"), ("ten", "", "home", "Chất liệu")])
kiem("dich.py vi_zh: câu lệnh gộp ra Hán giản thể, danh mục tiếng Trung", "Simplified Chinese" in g[1]["content"] and "家居用品" in g[1]["content"])
kiem("dich.py vi_zh: dò sót ngôn ngữ nguồn là chữ Việt có dấu", dich.sot_ngon_ngu_nguon().search("红色 đỏ") and not dich.sot_ngon_ngu_nguon().search("红色 ABC-1"))
kiem("dich.py vi_zh: hỏi lại dùng câu tiếng Anh vi_zh", "Vietnamese" in dich.hoi_lai("qwen", m, "红色 đỏ")[-1]["content"])
dich.HUONG = "zh_vi"
kiem("dich.py zh_vi (mặc định) không đổi: dịch sang tiếng Việt", "Vietnamese" in dich.cau_lenh("qwen", "ten", "", "food", "颜色")[0]["content"]
     and dich.sot_ngon_ngu_nguon().search("còn 红色") is not None)

# --- bước chuẩn bị trên dữ liệu giả ---
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    nguon = tmp / "tiki_gia.parquet"
    P = pd.DataFrame({"product_id": ["1", "2"], "category": ["food", "home"], "split": ["train", "test"], "row_hash": ["a", "b"], "bt_attributes_text": ["", ""], "attributes_vi": [
        [{"name": "Thương hiệu", "values": ["Vinamilk"]}, {"name": "Màu sắc", "values": ["Đen", "Trắng ngà"]},
         {"name": "Ghi chú", "values": ["✅", "#TagNay"]},
         {"name": "Khối lượng (kg)", "values": ["0.5"]}, {"name": "Tên đơn vị/tổ chức chịu trách nhiệm về hàng hóa", "values": ["Công ty X"]}],
        [{"name": "Màu sắc", "values": ["Đen", "Trắng ngà"]}, {"name": "Chất liệu", "values": ["Cotton"]},
         {"name": "Kích thước", "values": ["10 x 5 cm"]}]]})
    P.to_parquet(nguon)
    sha_cu, C.SHA256_FILE_NGUON = C.SHA256_FILE_NGUON, chuan_bi.sha256(nguon)
    try:
        B, DV, the = chuan_bi.chay(nguon, tmp / "ra", in_ra=lambda *a: None)
    finally:
        C.SHA256_FILE_NGUON = sha_cu
    kiem("chuẩn bị: 7 cặp, 1 cất riêng, 6 dùng", len(B) == 7 and (B.trang_thai_vao == "dung").sum() == 6)
    kiem("chuẩn bị: thuộc tính toàn emoji/hashtag bị bỏ như bước EDA ('Ghi chú' = ✅, #TagNay)", "Ghi chú" not in set(B.name_vi))
    kiem("chuẩn bị: giữ split và row_hash của nguồn, không tự chia", set(B.split_nguon) == {"train", "test"} and set(B.row_hash) == {"a", "b"})
    kiem("chuẩn bị: nhiều giá trị nối bằng ', '", (B.gia_tri_vi == "Đen, Trắng ngà").sum() == 2)
    kiem("chuẩn bị: gộp trùng — 'Màu sắc = Đen, Trắng ngà' của 2 sản phẩm thành 1 đơn vị",
         (DV[(DV.loai == "gia_tri") & (DV.van_ban_nguon == "Đen, Trắng ngà")].so_cap == 2).all()
         and len(DV[(DV.loai == "gia_tri") & (DV.van_ban_nguon == "Đen, Trắng ngà")]) == 1)
    kiem("chuẩn bị: đơn vị tên = 4 tên khác nhau của cặp được dùng (không có tên đơn vị chịu trách nhiệm)",
         sorted(DV[DV.loai == "ten"].van_ban_nguon) == ["Chất liệu", "Kích thước", "Khối lượng (kg)", "Màu sắc", "Thương hiệu"][:0] + sorted(
             ["Chất liệu", "Kích thước", "Khối lượng (kg)", "Màu sắc", "Thương hiệu"]))
    kiem("chuẩn bị: thương hiệu, số, đo lường chép nguyên (không gửi mô hình)",
         set(B[B.dinh_tuyen.isin(["ten_hang", "chep_nguyen", "do_luong"])].gia_tri_zh_quy_tac) == {"Vinamilk", "0.5", "10 x 5 cm"})
    kiem("chuẩn bị: file vào Kaggle đủ cột nguồn", {"name_nguon", "van_ban_nguon", "van_ban_gui", "loai", "ma_don_vi"} <= set(DV.columns))
    kiem("chuẩn bị: có dich.py và cham_kiwi.py kèm theo", (tmp / "ra" / "vao_kaggle" / "dich.py").exists()
         and (tmp / "ra" / "vao_kaggle" / "cham_kiwi.py").exists())
    # SHA256 lạ thì dừng
    try:
        chuan_bi.chay(nguon, tmp / "ra2", in_ra=lambda *a: None)
        dung = False
    except SystemExit:
        dung = True
    kiem("chuẩn bị: file nguồn khác SHA256 đã khảo sát thì dừng", dung)


# --- lọc thông tin riêng tư còn sót ---
from song_ngu_2.rieng_tu import rieng_tu  # noqa: E402
kiem("riêng tư: địa chỉ 'Đ/c: 63 ...' cất riêng", rieng_tu("Hướng dẫn sử dụng", "Nhập khẩu 100% UAE. Đ/c: 63 Nguyễn Thị Căn, Tân Thới Hiệp, Quận 12") == "dia_chi")
kiem("riêng tư: 'Phường 6, Quận 3' cất riêng", rieng_tu("Địa chỉ sử dụng", "Khách hàng đến cửa hàng Moto 165 Pasteur, Phường 6, Quận 3, TP Hồ Chí Minh") == "dia_chi")
kiem("riêng tư: shop có tên trong ngoặc kép cất riêng", rieng_tu("Lưu ý", 'Sản phẩm được bảo hành 3-6 tháng bởi shop " Nhà Chíp " ***Quý khách lưu ý') == "ten_shop")
kiem("riêng tư: 'Cung cấp bởi HT SYS OFFICIAL STORE' cất riêng", rieng_tu("Lưu ý", "Sản phẩm Được Cung Cấp Bởi HT SYS OFFICIAL STORE") == "ten_shop")
kiem("riêng tư: thương hiệu 'Anh Lam Store' cất riêng", rieng_tu("Thương hiệu", "Anh Lam Store") == "ten_shop")
kiem("riêng tư: 'Store' không phải thương hiệu thì giữ ('Book Store' trong phân loại)", rieng_tu("Phân loại", "2 chai_Book Store, 2 chai_Flower Shop") is None)
kiem("riêng tư: 'shop' nói chung giữ ('nhắn tin cho shop', 'Shop cập nhật thường xuyên')", rieng_tu("Lưu ý", "Vui lòng nhắn tin cho shop để được hỗ trợ") is None and rieng_tu("Hạn sử dụng", "Shop cập nhật thường xuyên") is None)
kiem("riêng tư: 'Số 1', 'Quận thiếc hàn', 'Dây Số 3' không phải địa chỉ", rieng_tu("PHÂN LOẠI", "Dây Số 1, Dây Số 2, Dây Số 3") is None and rieng_tu("khác", "50W, 80W, Quận thiếc hàn") is None)
kiem("riêng tư: tên doanh nghiệp (Công ty TNHH ...) giữ", rieng_tu("Kích thước", "Nhập khẩu & phân phối: Công ty TNHH Richell Việt Nam") is None)
kiem("riêng tư: tên shop viết liền 'shopquyetnam' cất riêng", rieng_tu("Quy cách đóng gói", "Lưu ý : Để giảm cước vận chuyển cho quý khách hàng thân yêu shopquyetnam xin mạn phép") == "ten_shop")
kiem("riêng tư: 'PET SHOP UYTINPRO' cất riêng", rieng_tu("Chất liệu", "Sản phẩm có sẵn tại PET SHOP UYTINPRO * ĐẶC ĐIỂM") == "ten_shop")
kiem("riêng tư: đại lý 'nhà bán hàng Honda Head Phát Tiến' cất riêng", rieng_tu("Điều kiện sử dụng", "trả góp của nhà bán hàng Honda Head Phát Tiến trên Tiki") == "ten_shop")
kiem("riêng tư: 'Shopee', 'SHOP CÓ PHÂN LOẠI', 'nhà bán hàng tại cửa hàng' giữ",
     rieng_tu("Chất liệu", "Cotton theo bài chuẩn Shopee") is None and rieng_tu("Lưu ý", "SHOP CÓ PHÂN LOẠI KÍNH MÀU VÀNG") is None
     and rieng_tu("Lưu ý", "thanh toán phí cho nhà bán hàng tại cửa hàng") is None)
from song_ngu_2.hoan_tat import bo_rac  # noqa: E402
kiem("ghép cặp: bỏ chữ rác _x000D_ (Excel) ở cả hai cột, chuỗi sạch giữ nguyên",
     bo_rac("Chiều cao: 136mm_x000D_ Đường kính: 106mm") == "Chiều cao: 136mm Đường kính: 106mm"
     and bo_rac("高度: 136mm_x000D_ 直径: 106mm") == "高度: 136mm 直径: 106mm" and bo_rac("Đen,  Trắng") == "Đen,  Trắng" and bo_rac(None) is None)

# --- bước hoàn tất (hoan_tat.chay) trên dữ liệu giả ---
from song_ngu_2 import hoan_tat  # noqa: E402
from song_ngu_2.quy_tac_gia_tri import ban_quy_tac  # noqa: E402
from song_ngu_2.van_ban import chuan_hoa_zh  # noqa: E402
kiem("quy tắc thời hạn: '12 Tháng' -> 12个月", ban_quy_tac("12 Tháng") == ("12个月", "quy_tac_thoi_han"))
kiem("quy tắc thời hạn: '3 năm kể từ ngày sản xuất' -> 自生产之日起3年", ban_quy_tac("3 năm kể từ ngày sản xuất")[0] == "自生产之日起3年")
kiem("quy tắc thời hạn: '2 năm kể từ NSX' và '24 tháng từ ngày sản xuất'", ban_quy_tac("2 năm kể từ NSX")[0] == "自生产之日起2年" and ban_quy_tac("24 tháng từ ngày sản xuất")[0] == "自生产之日起24个月")
kiem("quy tắc thời hạn: '06 tháng' bỏ số 0 đầu", ban_quy_tac("06 tháng")[0] == "6个月")
kiem("quy tắc thời hạn: câu dài hơn mẫu thì không nhận", ban_quy_tac("Độ bền trên 10 năm. - Không cong vênh") is None and ban_quy_tac("3 năm kể từ ngày sản xuất - Xem trên bao bì") is None)
kiem("quy tắc thời hạn: '2 giờ kể từ NSX' không nhận (giờ không đi với ngày sản xuất)", ban_quy_tac("2 giờ kể từ NSX") is None)
kiem("quy tắc khối lượng: '500gram' -> 500克, '250 gam' -> 250克, '1,5 kilogram' -> 1,5千克", ban_quy_tac("500gram") == ("500克", "quy_tac_khoi_luong")
     and ban_quy_tac("250 gam")[0] == "250克" and ban_quy_tac("1,5 kilogram")[0] == "1,5千克")
kiem("mau_anh_chua_dich: nguồn và bản dịch đều còn Black/Blue thì True; đã dịch thì False; nguồn không có màu Anh thì False",
     mau_anh_chua_dich("Blue, Green", "Blue, Green") and not mau_anh_chua_dich("Blue, Green", "蓝色, 绿色") and not mau_anh_chua_dich("Xanh", "Blue"))
kiem("quy tắc khối lượng: câu dài hơn mẫu không nhận", ban_quy_tac("500gram/gói") is None and ban_quy_tac("Khoảng 500 gram") is None)
kiem("chuan_hoa_zh: dấu toàn/nửa góc và khoảng trắng như nhau", chuan_hoa_zh("白色，灰色，黑色") == chuan_hoa_zh("白色, 灰色, 黑色") and chuan_hoa_zh("100% 有机") == chuan_hoa_zh("100%有机"))
kiem("chuan_hoa_zh: 、 và × như , và x", chuan_hoa_zh("红、蓝") == chuan_hoa_zh("红,蓝") and chuan_hoa_zh("3×4") == chuan_hoa_zh("3x4"))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    nguon = tmp / "tiki_gia2.parquet"
    P = pd.DataFrame({"product_id": ["1", "2"], "category": ["food", "home"], "split": ["train", "test"], "row_hash": ["a", "b"],
                      "bt_attributes_text": ["", ""], "attributes_vi": [
        [{"name": "Thương hiệu", "values": ["Vinamilk"]}, {"name": "Màu sắc", "values": ["Đen", "Trắng"]},
         {"name": "Hạn sử dụng", "values": ["12 Tháng"]}, {"name": "Sản phẩm có được bảo hành không?", "values": ["Có"]},
         {"name": "Xuất xứ", "values": ["Ý"]}, {"name": "Tên đơn vị/tổ chức chịu trách nhiệm về hàng hóa", "values": ["Công ty X"]}],
        [{"name": "Màu sắc", "values": ["Đen", "Trắng"]}, {"name": "Chất liệu", "values": ["Nhựa"]}, {"name": "Hình dạng", "values": ["Tròn"]},
         {"name": "Kích cỡ", "values": ["Xám"]}, {"name": "Phân loại", "values": ["Black"]}]]})
    P.to_parquet(nguon)
    sha_cu, C.SHA256_FILE_NGUON = C.SHA256_FILE_NGUON, chuan_bi.sha256(nguon)
    try:
        ra = tmp / "ra"
        B, DV, the = chuan_bi.chay(nguon, ra, in_ra=lambda *a: None)
    finally:
        C.SHA256_FILE_NGUON = sha_cu
    # A, B, điểm A, điểm B cho từng đơn vị (theo chữ nguồn)
    dich_gia = {
        "Thương hiệu": ("品牌", "品牌", .8, .8), "Màu sắc": ("颜色", "颜色", .8, .8), "Hạn sử dụng": ("保质期", "使用期限", .8, .8),
        "Sản phẩm có được bảo hành không?": ("产品是否有保修？", "该产品是否有保修？", .8, .8), "Xuất xứ": ("产地", "产地", .8, .8),
        "Chất liệu": ("材质", "材质", .8, .8), "Hình dạng": ("形状", "形状", .8, .8), "Kích cỡ": ("尺寸", "尺寸", .8, .8),
        "Đen, Trắng": ("黑色, 白色", "黑色，白色", .8, .75),             # giống nhau sau chuẩn hoá: giong_het
        "Nhựa": ("Nhựa", "塑料", .8, .8),                                # A sót chữ Việt: dùng B
        "Tròn": ("Tròn", "Tròn", .8, .8),                                # cả hai trượt: cất riêng
        "Phân loại": ("分类", "分类", .8, .8), "Black": ("Black", "黑色", .8, .8),   # A để nguyên tên màu Anh, B đã dịch: dùng B
        "Xám": ("灰色", "银灰", .10, .10),                               # khác nhau, cả hai dưới sàn cất riêng: cất riêng
        "Ý": ("产地: 意大利", "意大利", .8, .8),                           # A tự thêm nhãn: gỡ nhãn, hai bản giống nhau
        "Có": ("是", "有", .8, .8), "12 Tháng": ("12 个月", "12个月", .8, .8)}
    kg = tmp / "kaggle"
    for sub, i in (("api_tuy_chon", 0), ("qwen3_14b", 1)):
        d = kg / "ra_tiki" / sub
        d.mkdir(parents=True)
        pd.DataFrame({"ma_don_vi": DV.ma_don_vi, "ban_dich": [dich_gia[v][i] for v in DV.van_ban_nguon], "ly_do_dung": "stop",
                      "dich_lai": False, "backend": "x"}).to_parquet(d / "lo_00000.parquet")
        dk = kg / "ra_tiki" / f"kiwi_{sub}"
        dk.mkdir(parents=True)
        pd.DataFrame({"ma_don_vi": DV.ma_don_vi, "diem": [dich_gia[v][2 + i] for v in DV.van_ban_nguon]}).to_parquet(dk / "lo_00000.parquet")
    D, CR, U, the2 = hoan_tat.chay(kg, ra, in_ra=lambda *a: None, tu_dien={"ten": {"Chất liệu": "材料(từ điển)"}, "gia_tri": {},
                                                                              "co_khong": {("Sản phẩm có được bảo hành không?", "Có"): "是"}})
    Ut = U.set_index(["loai", "van_ban_nguon"])
    kiem("hoàn tất: hai bản giống nhau sau chuẩn hoá (，/, ) -> giong_het, độ tin cậy cao", Ut.loc[("gia_tri", "Đen, Trắng")].cach_chon == "giong_het"
         and Ut.loc[("gia_tri", "Đen, Trắng")].do_tin_cay == "cao")
    kiem("hoàn tất: A sót chữ Việt (Nhựa) -> dùng B (chi_B, tin cậy thấp)", Ut.loc[("gia_tri", "Nhựa")].ban == "塑料" and Ut.loc[("gia_tri", "Nhựa")].cach_chon == "chi_B"
         and Ut.loc[("gia_tri", "Nhựa")].do_tin_cay == "thap")
    kiem("hoàn tất: cả hai trượt (Tròn) -> cất riêng ca_hai_truot", pd.isna(Ut.loc[("gia_tri", "Tròn")].ban) and Ut.loc[("gia_tri", "Tròn")].ly_do_cat == "ca_hai_truot")
    kiem("hoàn tất: khác nhau và cả hai dưới sàn cất riêng (Xám) -> cất riêng", pd.isna(Ut.loc[("gia_tri", "Xám")].ban))
    kiem("hoàn tất: A để nguyên tên màu Anh (Black), B đã dịch -> dùng B (mau_chua_dich)", Ut.loc[("gia_tri", "Black")].ban == "黑色"
         and "mau_chua_dich" in Ut.loc[("gia_tri", "Black")].truot_A)
    kiem("hoàn tất: gỡ nhãn '产地: ' khỏi bản A (Ý) -> 意大利, ghi cờ go_nhan", Ut.loc[("gia_tri", "Ý")].A == "意大利" and bool(Ut.loc[("gia_tri", "Ý")].go_nhan_A)
         and Ut.loc[("gia_tri", "Ý")].ban == "意大利")
    kiem("hoàn tất: quy tắc thời hạn đè (12 Tháng -> 12个月, cao)", Ut.loc[("gia_tri", "12 Tháng")].ban == "12个月" and Ut.loc[("gia_tri", "12 Tháng")].cach_chon == "quy_tac_thoi_han")
    kiem("hoàn tất: bảng có/không theo thuộc tính đè (Có -> 是)", Ut.loc[("gia_tri", "Có")].ban == "是" and Ut.loc[("gia_tri", "Có")].cach_chon == "quy_tac_co_khong")
    kiem("hoàn tất: khoá tra bảng không phân biệt hoa thường (CHẤT LIỆU và chất liệu cùng mục)",
         hoan_tat.ban_co_dinh("ten", "CHẤT  LIỆU", "", {"chất liệu": "材料"}, {}, {}) == ("材料", "tu_dien_p6")
         and hoan_tat.ban_co_dinh("gia_tri", "CÓ", "Hỗ Trợ Sạc", {}, {}, {("hỗ trợ sạc", "có"): "是"}) == ("是", "quy_tac_co_khong"))
    kiem("hoàn tất: hai khoá khác hoa thường mà bản dịch khác nhau thì giữ mục đầu và đếm xung đột", hoan_tat.chuan_bang({"Có": "是", "có": "有", "Không": "否"}) == ({"có": "是", "không": "否"}, 1))
    kiem("hoàn tất: từ điển tên đè (Chất liệu -> 材料(từ điển), cao)", Ut.loc[("ten", "Chất liệu")].ban == "材料(từ điển)" and Ut.loc[("ten", "Chất liệu")].do_tin_cay == "cao")
    kiem("hoàn tất: tên không có trong từ điển lấy bản A (Hạn sử dụng -> 保质期)", Ut.loc[("ten", "Hạn sử dụng")].ban == "保质期")
    d1 = D.set_index(["product_id", "name_vi"])
    kiem("hoàn tất: thương hiệu chép nguyên, tin cậy cao", d1.loc[("1", "Thương hiệu")].gia_tri_zh == "Vinamilk" and d1.loc[("1", "Thương hiệu")].do_tin_cay == "cao"
         and d1.loc[("1", "Thương hiệu")].name_zh == "品牌")
    kiem("hoàn tất: tin cậy của cặp là mức thấp hơn của tên và giá trị (Nhựa thap)", d1.loc[("2", "Chất liệu")].do_tin_cay == "thap")
    kiem("hoàn tất: cặp có giá trị bị cất riêng không có trong bản xuất (Tròn, Xám)", ("2", "Hình dạng") not in d1.index and ("2", "Kích cỡ") not in d1.index)
    kiem("hoàn tất: tên đơn vị chịu trách nhiệm nằm ở file cất riêng, lý do van_loai:to_chuc",
         (CR.ly_do_cat == "van_loai:to_chuc").sum() == 1)
    kiem("hoàn tất: bản công bố đúng 8 cột cố định, không có cột nội bộ", list(D[C.COT_CONG_BO].columns) == ["pair_id", "product_id", "category", "name_vi", "gia_tri_vi",
                                                                                                          "name_zh", "gia_tri_zh", "do_tin_cay"])
    kiem("hoàn tất: mọi cặp xuất ra không còn ô rỗng", not D[C.COT_CONG_BO].isna().any().any())
    kiem("hoàn tất: tổng cặp giữ + cất riêng = tổng cặp nguồn", len(D) + len(CR) == len(B))
    kiem("hoàn tất: bản đầy đủ giữ split và row_hash của nguồn", set(D.split_nguon) == {"train", "test"} and set(D.row_hash) == {"a", "b"})
    kiem("hoàn tất: thẻ nguồn gốc có ngưỡng cất riêng và số gỡ nhãn", the2["nguong"]["kappa_cat"] == C.KAPPA_CAT and the2["go_nhan"]["A"] == 1)

print(f"\n{DAT}/{TONG} ca đạt")
sys.exit(0 if DAT == TONG else 1)
