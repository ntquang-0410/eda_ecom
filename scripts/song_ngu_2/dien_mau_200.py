# -*- coding: utf-8 -*-
"""Điền đánh giá của Claude (23/10/2026, Nhật Anh xem lại) vào file mẫu 200 cặp của bộ song ngữ 2. Chạy bằng Python 3.9 có openpyxl:

    <python3.9> scripts/song_ngu_2/dien_mau_200.py <file xlsx>

Mặc định mọi dòng là Đúng (tên và giá trị); các dòng em đọc thấy lỗi nằm trong TEN và GIA_TRI dưới đây (stt -> (đánh giá, bản sửa đề xuất, ghi chú)).
DỪNG nếu file đã có nội dung ở bốn cột nền vàng (không ghi đè phần anh đã điền).
"""
import sys

from openpyxl import load_workbook

sys.stdout.reconfigure(encoding="utf-8")
# tên thuộc tính: stt -> (đánh giá, bản sửa, ghi chú)
TEN = {
    17: ("Sai nhẹ", "输出接口", "'Đầu ra' của sạc là cổng ra; '输出' quá chung"),
    19: ("Sai", "传动系统", "'Bộ chuyển động' của xe đạp là bộ truyền động/đề; '运动组件' (linh kiện thể thao) sai nghĩa"),
    42: ("Sai nhẹ", "形态", "'Dạng' ở đây là dạng sản phẩm (hạt, xay sẵn), không phải hình dạng 形状"),
}
# giá trị: stt -> (đánh giá, bản sửa, ghi chú)
GIA_TRI = {
    8: ("Sai nhẹ", "高度: 136mm 直径: 106mm", "nguồn có ký tự rác _x000D_ (xuống dòng của Excel) và nó còn nguyên ở cả hai cột; đã làm sạch từ bản 20261023_1656"),
    32: ("Sai nhẹ", "长54x宽28x高5", "C5 (Cao) chưa dịch; D54xR28xC5 dịch dở"),
    34: ("Sai nhẹ", "01 薰衣草香氛蜡烛, 柚子香, 香茅柠檬香, 白千层香 5 纹", "'Tràm' là 白千层/茶树, không phải 尤加利 (bạch đàn)"),
    36: ("Sai nhẹ", "缠绕膜包裹", "'Quấn màn co' là quấn màng co/màng quấn; '收缩膜卷' là cuộn màng co"),
    40: ("Sai nhẹ", "双USB接口", "'端' lạ; thuật ngữ sàn là 接口"),
    41: ("Sai nhẹ", "", "nguồn 'kh' mơ hồ (có thể là kg), bản dịch giữ nguyên 'kh'"),
    42: ("Sai nhẹ", "咖啡豆, 已研磨", "'Hạt' của cà phê là hạt nguyên (咖啡豆), '颗粒' là dạng hạt nhỏ"),
    52: ("Sai nhẹ", "100% 阿拉比卡", "Arabica để nguyên chữ Latin; sàn Trung dùng 阿拉比卡"),
    56: ("Sai nhẹ", "台湾 / 中国", "Gemma để nguyên 'Taiwan / China' (bản Qwen dịch 台湾 / 中国)"),
    59: ("Sai nhẹ", "蛋白质 > 40%", "PROTEIN để nguyên tiếng Anh"),
    65: ("Sai nhẹ", "白红, 银金, 黑色", "đảo thứ tự màu: 'Trắng Đỏ' thành 红白, 'bạc vàng' thành 金银"),
    67: ("Sai nhẹ", "随机 Logo", "'logo ngầu nhiên' là 'ngẫu nhiên' (随机); Gemma dịch 酷炫 (ngầu)"),
    77: ("Sai nhẹ", "500克 自封袋包装", "'gr' để nguyên"),
    79: ("Sai nhẹ", "阿拉比卡、罗布斯塔、艾克赛萨和卡蒂莫咖啡", "tên giống cà phê để nguyên chữ Latin"),
    80: ("Sai nhẹ", "蓝牙 6.0", "lỗi chính tả nguồn 'Bluetoth' được giữ nguyên, nên dịch 蓝牙"),
    94: ("Sai nhẹ", "", "dịch dở: 'Cannon Jeans Fluo Or' để nguyên còn các màu sau dịch (tên màu mũ bảo hiểm, cần người hiểu mẫu)"),
    98: ("Sai", "80% Bảo Lộc 罗布斯塔咖啡和 20% Cầu Đất 阿拉比卡咖啡", "địa danh dịch sai: Cầu Đất thành '梂土' (vô nghĩa); nên giữ tên riêng"),
    101: ("Sai nhẹ", "碳纤维地垫 C1, 碳纤维地垫 C2, 碳纤维地垫 C3, 碳纤维地垫 C4, 碳纤维地垫 C5", "'Carbon' để nguyên; 'Thảm' là 地垫 (thảm lót sàn)"),
    103: ("Sai nhẹ", "黑色，海军蓝，灰色", "'Blue Navy' là 海军蓝/藏青; '蓝色海军' sai"),
    106: ("Sai nhẹ", "", "'chẩm chéo' là món chấm đặc sản, bản dịch 湿拌蘸酱 gần đúng, nên giữ tên riêng"),
    109: ("Sai nhẹ", "", "'cây sim' là 桃金娘, bản dịch viết 越橘 (sai loài); câu đầu cũng gượng"),
    111: ("Sai nhẹ", "24支螺丝刀头, LUCKY", "'đầu tua vít' là mũi vít; '24头螺丝刀' nghe như bộ tua vít 24 đầu"),
    114: ("Sai", "千年健精油, 蜂蜡, 辅料适量", "'Thiên niên kiện' là 千年健; Gemma dịch 续断 (cây khác), Qwen dịch đúng"),
    119: ("Sai", "蜡牛皮 - 170, 颗粒纹米色 - 144, 颗粒纹藏青 - 136, 颗粒纹红 - 165", "'KEM HẠT' thành '谷物奶油' vô nghĩa; 'hạt' là vân hạt của da; cần người am hiểu xe"),
    120: ("Sai nhẹ", "黑色，军绿色，灰绿色，橙色，天蓝色", "'Green Army' là army green (军绿色), không phải '绿色军用'"),
    124: ("Sai nhẹ", "2片装多功能挂物墙贴", "'Treo Đồ' (treo đồ vật) bị bỏ"),
    125: ("Sai nhẹ", "", "'Lauryl' thành '劳里尔' (phiên âm sai, nên là 月桂基); vài thành phần dịch gượng"),
    128: ("Sai", "", "'cà gai leo' dịch thành 刺蒺藜 (cây khác); nên giữ tên Việt kèm tên khoa học (Solanum procumbens)"),
    131: ("Sai nhẹ", "", "'Lauryl' thành '劳里尔', một thành phần dịch thành chuỗi vô nghĩa"),
    137: ("Sai nhẹ", "灰钛, 黑色, 白色, 藏青色", "'Màu xanh đen' bị dịch trùng thành 黑色"),
    138: ("Sai nhẹ", "米粉, 麦芽糖, 蜂蜜, 杏仁, 南瓜籽, 核桃, 澳洲坚果, 腰果, 向日葵籽, 燕麦, 葡萄, 奇异果, 酸梅, 芒果, 爆米花, 蓝莓, 草莓", "'bí xanh' (hạt) thành 青瓜籽 (hạt dưa leo) sai; dòng 37 cùng cụm này dịch đúng 南瓜子, không nhất quán"),
    142: ("Sai nhẹ", "2 包 250克, 4 包 250克, 1 包 250克", "'GR' để nguyên"),
    144: ("Sai nhẹ", "链条-428H-金色", "đảo thứ tự các phần của mã 'SÊN-428H-VÀNG'"),
    145: ("Sai", "", "'Hồng' (hồng) thành 红色 (đỏ); 'Crunch White' thành 鳄鱼白 (nhầm với Croco); tên mẫu mũ nhiều chỗ dịch sai"),
    # soát lại lần 2 (23/10/2026, theo yêu cầu của anh): thêm 8 chỗ lần đầu bỏ qua, và ghi chú riêng tư cho dòng 90
    46: ("Sai nhẹ", "… 不可为 Con Thỏ、Con Ó 等一次性碱性电池充电", "'alkaline' để nguyên tiếng Anh (pin kiềm là 碱性电池)"),
    81: ("Sai nhẹ", "1 x 双头磁吸充电线", "'Cáp sạc từ đôi': chữ 'từ' (nam châm, 磁吸) bị bỏ"),
    90: ("Đúng", "", "dịch đúng, nhưng giá trị có tên cửa hàng 'PET SHOP UYTINPRO' nên từ bản 20261023_1656 cặp này bị cất riêng (lọc riêng tư)"),
    97: ("Sai nhẹ", "防腐剂 焦亚硫酸钾 (E224)", "'Kali Metabisulfit' (E224) là 焦亚硫酸钾; Gemma dịch 亚硫酸氢钾 (kali bisulfit, chất khác)"),
    110: ("Sai nhẹ", "仅机器, 机器+过滤棉, 机器+过滤棉+益生菌10G", "'bông lọc' là bông lọc (过滤棉), không phải bộ lọc 过滤器"),
    118: ("Sai nhẹ", "仅机器, 机器+气管+气石", "máy sủi oxy: 'dây' là dây dẫn khí (气管), '线' là dây điện"),
    132: ("Sai nhẹ", "… 藿香, 维生素原 B5", "'pro vitamin B5' (panthenol) bị dịch thành 维生素 B5, mất chữ 'pro' (维生素原 B5 / 泛醇)"),
    140: ("Sai nhẹ", "单机 - B93, 机器含4个替换头 - B93", "'4 đầu' là 4 đầu thay của máy, '配件' (phụ kiện) quá chung"),
    147: ("Sai nhẹ", "DK81386 装甲恐龙, DK81385 翼龙, DK81387 剑鳍恐龙", "'vây kiếm' là vây hình kiếm (剑鳍), Gemma dịch 剑齿 (răng kiếm); 'thằn lằn bay' là 翼龙, không phải 飞蜥"),
}
GC = "Claude chấm 23/10/2026, Nhật Anh xem lại"

wb = load_workbook(sys.argv[1])
ws = wb["Mau"]
da_co = [r for r in range(2, ws.max_row + 1) if any(ws.cell(row=r, column=c).value not in (None, "") for c in (8, 9, 10, 11))]
assert not da_co, f"đã có nội dung ở dòng {da_co[:5]}: không ghi đè"
dem = {"Đúng": 0, "Sai nhẹ": 0, "Sai": 0}
for r in range(2, ws.max_row + 1):
    stt = ws.cell(row=r, column=1).value
    if stt is None:
        continue
    dt, st, gt = TEN.get(stt, ("Đúng", "", ""))
    dg, sg, gg = GIA_TRI.get(stt, ("Đúng", "", ""))
    ws.cell(row=r, column=8, value=dt)
    ws.cell(row=r, column=9, value=dg)
    sua = " | ".join(x for x in [f"tên: {st}" if st else "", f"giá trị: {sg}" if sg else ""] if x)
    ws.cell(row=r, column=10, value=sua or None)
    ghi = " ; ".join(x for x in [f"tên: {gt}" if gt else "", gg] if x)
    ws.cell(row=r, column=11, value=(ghi + " | " if ghi else "") + GC)
    dem[dg] += 1
wb.save(sys.argv[1])
print("đã điền", sum(dem.values()), "dòng; giá trị:", dem, "| tên Sai/Sai nhẹ:", {k: sum(1 for v in TEN.values() if v[0] == k) for k in ("Sai nhẹ", "Sai")})
