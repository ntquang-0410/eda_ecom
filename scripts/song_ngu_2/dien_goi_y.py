# -*- coding: utf-8 -*-
"""Điền cột goi_y_claude và ly_do_claude cho ba file duyệt của bộ Tiki (P6, P7, P8), sau khi Claude đọc từng dòng (23/10/2026).
Chỉ ghi hai cột đó, không đụng cột anh_duyet và anh_ghi_chu; DỪNG nếu anh đã điền anh_duyet ở bất kỳ dòng nào (không ghi đè phần anh duyệt); chạy lại được khi anh chưa duyệt (xoá gợi ý cũ).
Mỗi gợi ý phải khớp ít nhất một dòng, không thì dừng (file đã đổi). Chạy từ gốc repo:

    .venv\\Scripts\\python.exe scripts\\song_ngu_2\\dien_goi_y.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from song_ngu_2 import cau_hinh as C  # noqa: E402

# ---- P6: tên thuộc tính: (ten_vi chính xác, ban gợi ý, lý do) ----
P6 = [
    ("Xuất xứ (Made in)", "原产地", "bỏ chú thích tiếng Anh '(Made in)' thừa; sàn Trung dùng 原产地 (nước xuất xứ); tên này có nhiều cặp nhất (12.672); anh có thể chọn 产地"),
    ("Hạn sử dụng", "保质期", "giá trị là '3 năm kể từ ngày sản xuất', '12 tháng' = hạn dùng của thực phẩm, mỹ phẩm; thuật ngữ chuẩn là 保质期, còn 使用期限 nghe như thời hạn dùng thiết bị"),
    ("Hướng dẫn bảo quản", "保存方法", "tự nhiên hơn 保存指南 (Qwen chọn); cùng nghĩa"),
    ("Phương thức giao hàng Seller Delivery", "配送方式 卖家配送", "Seller Delivery là 卖家配送 (người bán tự giao); Gemma viết 卖家自送 lệch nghĩa"),
    ("Dòng máy", "机型", "giá trị là iPhone, Samsung Galaxy...; Taobao dùng 机型 (anh có thể chọn 适用机型 nếu muốn nhấn mạnh phụ kiện)"),
    ("Phanh", "刹车", "xe máy, xe điện: giá trị CBS, ABS, Phanh đĩa, Phanh cơ; 刹车 là cách gọi thông dụng, 制动器 là thuật ngữ kỹ thuật"),
    ("Thắng xe", "刹车", "'thắng' là phanh (cách nói miền Nam); cùng một tên với 'Phanh'"),
    ("Số km/mỗi lần sạc", "单次充电续航里程", "tự nhiên hơn '每充电一次的公里数'"),
    ("Chất liệu nồi & chảo", "锅具材质", "ngắn gọn, 锅具 gồm cả nồi và chảo; bỏ dấu & tiếng Anh"),
    ("Sử dụng cho bếp", "适用灶具", "'bếp' ở đây là bếp từ, bếp gas (giá trị: 'Mọi loại bếp', 'Bếp từ, bếp điện, bếp gas'); Gemma dịch 用于厨房 (nhà bếp) sai nghĩa"),
    ("Bluetooth", "蓝牙", "tên chuẩn bằng tiếng Trung; Gemma giữ nguyên chữ Latin (Qwen: 蓝牙)"),
    ("Phụ kiện đi kèm", "附带配件", "'đi kèm' cần có 附带; Gemma bỏ mất ('配件')"),
    ("Thời gian pin", "电池续航时间", "tự nhiên hơn '电池时间'"),
    ("Đời xe", "年款", "giá trị là các năm 2018, 2019...; 年款 là năm đời xe; Gemma '年份', Qwen '车型' sai"),
    ("Hàm lượng dinh dưỡng", "营养成分", "thuật ngữ chuẩn trên bao bì thực phẩm"),
    ("Dòng Xe", "车型", "giá trị Mazda 2, Toyota Alphard...; 车型 phổ biến hơn 车系 (dòng xe theo series)"),
    ("Khả năng giữ nhiệt", "保温性能", "thuật ngữ chuẩn (Qwen); 保温能力 ít dùng"),
    ("Dạng bào chế", "剂型", "thuật ngữ dược chuẩn"),
    # Đã bỏ gợi ý 尺码 cho Size/SIZE/size/Kích cỡ (23/10/2026, kiểm lại bằng dữ liệu): trong 210 cặp 'Size' chỉ 41 là chữ cỡ S/M/L, 126 là kích thước
    # (cm, mm, m); 'Kích cỡ' 87 chữ cỡ so với 143 kích thước. 尺寸 của Gemma đúng với phần lớn nên giữ.
    ("Đối tượng", "适用人群", "giá trị Người trưởng thành, Nữ, Trẻ em (nhóm làm đẹp); 对象 quá chung chung"),
]
# ---- P8: có/không theo thuộc tính: (ten_vi chính xác, gia_tri_vi chính xác, ban gợi ý, lý do) ----
LD_HOI = "tên đã dịch thành câu hỏi/hỗ trợ (是否/支持/含) nên trả lời 是/否; '无/有' (Gemma) không khớp"
P8 = [
    ("Sản phẩm có được bảo hành không?", "Không", "否", "tên '产品是否有保修？' là câu hỏi có/không nên trả lời 是/否; Gemma dịch Không là 无, lệch với 是 của 'Có' (9.425 cặp, cặp nhiều nhất bộ)"),
    ("Có thuế VAT", "Có", "是", "tên '含增值税' trả lời 是/否 (cặp 'Không' đã là 否)"),
    ("Hỗ trợ lắp đặt", "Có", "是", LD_HOI),
    ("Hỗ trợ lắp đặt", "Không", "否", LD_HOI),
    ("Hỗ trợ sạc nhanh", "Có", "是", LD_HOI),
    ("Hỗ trợ sạc nhanh", "có", "是", LD_HOI),
    ("Hỗ trợ sạc nhanh", "CÓ", "是", LD_HOI),
    ("Hỗ trợ sạc nhanh", "không", "否", LD_HOI),
    ("Hỗ trợ sạc nhanh", "không có", "否", LD_HOI),
    ("Hỗ trợ đàm thoại", "Có", "是", LD_HOI),
    ("Chống nước", "Không", "无", "cùng thuộc tính 'Có' đã là 有 nên 'Không' là 无 (Gemma viết 否 lẫn 有)"),
]
# ---- P7: giá trị: (tiền tố ten_vi, tiền tố gia_tri_vi, ban gợi ý, lý do) ----
P7 = [
    ("Hình thức bảo hành", "Điện tử", "电子保修", "937 cặp; '电子' đứng một mình không rõ nghĩa, đây là hình thức bảo hành điện tử"),
    ("Hình thức bảo hành", "Hóa đơn", "发票保修", "908 cặp; đồng bộ với 电子保修, 保修卡, 保修标签 (cùng thuộc tính 'Hình thức bảo hành'); '发票' đứng một mình chưa rõ là bảo hành bằng hóa đơn"),
    ("Hình thức bảo hành", "Tem bảo hành", "保修标签", "'保修贴' lạ; Kiwi chỉ 0,41"),
    ("Thành phần", "Tham khảo mô tả", "详见描述", "cách nói thông dụng của sàn Trung ('参考描述' nghe máy móc)"),
    ("Phương thức giao hàng Seller", "Nhà bán giao hàng cho khách hàng", "卖家配送给买家", "dùng 买家 (người mua) thay 顾客 cho nhất quán với tên thuộc tính 卖家配送; Gemma '卖家配送' (Kiwi 0,59) quá cụt"),
    ("Phương thức giao hàng Seller", "Khách hàng đến lấy hàng tại nhà bá", "买家到卖家处自提", "dùng 买家 cho nhất quán; bỏ '客户/顾客' và '货物' thừa"),
    ("Chất liệu", "ABS nguyên sinh", "ABS原生料", "'nguyên sinh' là nhựa nguyên liệu mới (virgin) = 原生料 (Qwen); '原生ABS' nghe thiếu"),
    ("Hạn sử dụng", "xem bb", "见包装", "'bb' viết tắt của 'bao bì'; Gemma '见 BB' và Qwen '请查看产品说明' đều sai"),
    ("Phân loại", "Black", "黑色", "Gemma để nguyên chữ Anh; nên dịch tên màu"),
    ("Model", "MỚI NHẤT", "最新款", "thuộc tính Model nên là 最新款 (Qwen); 最新 thiếu '款'"),
    ("Model", "Mới nhất", "最新款", "như trên"),
    ("Model", "mới nhất", "最新款", "như trên"),
    ("Chất liệu", "Giấy carton", "纸板", "'giấy carton' là bìa các-tông nói chung: 纸板 an toàn; '卡纸' (Gemma) là giấy bìa mỏng làm thiệp, '瓦楞纸' (Qwen) chỉ loại có sóng"),
    ("Phanh", "Đĩa", "碟刹", "'Đĩa' ở thuộc tính Phanh là phanh đĩa; Gemma '制动盘' (chi tiết cơ khí), Qwen '碟片' sai"),
    ("Phanh", "Phanh Cơ", "机械刹车", "đồng bộ với 刹车 ở tên thuộc tính; Gemma '机械制动'"),
    ("Dạng bào chế", "Nước", "液体", "dạng bào chế dạng nước là chế phẩm lỏng; Gemma '水质' (chất lượng nước) sai nghĩa"),
    ("Chất liệu", "Tổng hợp", "合成材料", "vật liệu tổng hợp; Gemma '合成' thiếu 材料, Qwen '综合' sai nghĩa"),
    ("Đặc sản vùng miền", "Không phân biệt", "不限", "cách nói thông dụng ('无特定地区' nghe máy móc)"),
    ("Sử dụng cho bếp", "Mọi loại bếp", "所有灶具", "đồng bộ với tên '适用灶具'; Gemma '所有类型的炉灶' dài"),
    ("Loại", "Da Vân Carbon, Nhung Lông Cừu", "碳纤维纹皮革, 羊羔绒", "'nhung lông cừu' là nhung lông cừu non (羊羔绒); '羊绒' là len cashmere, sai"),
    ("Trọng lượng sản phẩm", "500gram", "500克", "Gemma để nguyên chữ 'gram'; đã thêm quy tắc chung cho 'N gram'"),
    ("Trọng lượng sản phẩm", "500 gram", "500克", "như trên"),
]


def doc(file):
    return pd.read_csv(file, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def ap_dung(file, quy_tac, khoa):
    t = doc(file)
    if (t.anh_duyet.str.strip() != "").any():
        raise SystemExit(f"{Path(file).name}: anh đã điền anh_duyet: dừng, không ghi đè.")
    t["goi_y_claude"] = ""          # anh chưa duyệt gì: được xoá gợi ý cũ của Claude và điền lại bản mới
    t["ly_do_claude"] = ""
    dem = 0
    for q in quy_tac:
        *k, goi_y, ly_do = q
        m = pd.Series(True, index=t.index)
        for cot, gt in zip(khoa, k):
            m &= (t[cot] == gt) if khoa is not P7_KHOA else t[cot].str.startswith(gt)
        if not m.any():
            raise SystemExit(f"{Path(file).name}: không khớp dòng nào: {k}")
        t.loc[m, "goi_y_claude"] = goi_y
        t.loc[m, "ly_do_claude"] = ly_do
        dem += int(m.sum())
    t.to_csv(file, index=False, encoding="utf-8-sig")
    print(f"{Path(file).name}: {len(quy_tac)} gợi ý, {dem} dòng được điền")


P7_KHOA = ["ten_vi", "gia_tri_vi"]
if __name__ == "__main__":
    ap_dung(C.FILE_TU_DIEN_TEN, P6, ["ten_vi"])
    ap_dung(C.FILE_CO_KHONG, P8, ["ten_vi", "gia_tri_vi"])
    ap_dung(C.FILE_TU_DIEN_GIA_TRI, P7, P7_KHOA)
