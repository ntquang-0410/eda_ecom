import sys
import zipfile
import shutil
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.chart.data import CategoryChartData
from pptx.oxml.ns import qn

OUT = sys.argv[1]

# ---------- bảng màu (một nơi duy nhất) ----------
NAVY = "0E2A47"      # nền tối, chữ tiêu đề
INK = "17212B"       # chữ chính
BODY = "3B4656"      # chữ phụ đậm
MUTED = "6B778A"     # chú thích
ORANGE = "E8590C"    # 1688
BLUE = "1C7ED6"      # Tiki
GREEN = "2B8A5E"     # giữ lại
RED = "B42318"       # bị loại
GOLD = "F2B01E"      # nhấn trên nền tối
WHITE = "FFFFFF"
TINT_OR = "FDEBDD"
TINT_BL = "E2EEFB"
TINT_RED = "FDECEA"
TINT_GR = "E3F4EC"
GREY_BG = "F2F4F7"
CARD_DARK = "17395C"

HEAD = "Cambria"
BASE = "Calibri"
CJK = "Microsoft YaHei"


def rgb(h):
    return RGBColor.from_string(h)


def fmt(n):
    return f"{n:,}".replace(",", ".")


prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(5.625)
LAYOUT = prs.slide_layouts[5]  # Title Only


def no_shadow(shape):
    spPr = shape._element.spPr
    for el in spPr.findall(qn("a:effectLst")):
        spPr.remove(el)
    etree.SubElement(spPr, qn("a:effectLst"))


def style_run(r, size=None, bold=None, color=None, font=None, cjk=False, italic=None):
    f = r.font
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color:
        f.color.rgb = rgb(color)
    name = CJK if cjk else font
    if name:
        f.name = name
    if cjk:
        rPr = r._r.get_or_add_rPr()
        for tag in ("a:ea", "a:cs"):
            for el in rPr.findall(qn(tag)):
                rPr.remove(el)
            e = etree.SubElement(rPr, qn(tag))
            e.set("typeface", CJK)


def fill_tf(tf, paras, anchor="t", align="l", space_before=0):
    """paras: list of paragraphs; each paragraph = list of (text, opts) runs."""
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    for i, runs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if i > 0 and space_before:
            p.space_before = Pt(space_before)
        for text, o in runs:
            r = p.add_run()
            r.text = text
            style_run(r, **o)


def tb(slide, x, y, w, h, paras, anchor="t", align="l", name=None, space_before=0):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    fill_tf(s.text_frame, paras, anchor, align, space_before)
    return s


def box(slide, x, y, w, h, fill, radius=0.0, shape=None, name=None):
    kind = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / min(w, h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    no_shadow(s)
    if name:
        s.name = name
    return s


def set_title(slide, text, dark=False):
    t = slide.shapes.title
    if dark:
        t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.95), Inches(8.8), Inches(1.5)
    else:
        t.left, t.top, t.width, t.height = Inches(0.5), Inches(0.3), Inches(9), Inches(0.7)
    tf = t.text_frame
    tf.clear()
    fill_tf(tf, [[(text, dict(size=38 if dark else 30, bold=True, color=WHITE if dark else NAVY, font=HEAD))]],
            anchor="t" if dark else "m")
    t.name = "Tiêu đề"


def new_slide(dark=False, number=None):
    s = prs.slides.add_slide(LAYOUT)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(NAVY if dark else WHITE)
    if number:
        tb(s, 9.1, 5.22, 0.4, 0.25, [[(str(number), dict(size=10, color=MUTED))]], align="r", name="Số trang")
    return s


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def hide_value_axis(chart):
    ax = chart.value_axis
    ax.has_major_gridlines = False
    ax.visible = False
    ax._element.find(qn("c:delete")).set("val", "1")  # python-pptx để trống val, PowerPoint vẫn vẽ trục


def vn_labels(series, values, pos, size, color, bold=False):
    """Nhãn số tự viết theo dấu chấm hàng nghìn (nhãn tự động theo locale máy, hay ra dấu phẩy)."""
    for j, v in enumerate(values):
        if not v:
            continue
        dl = series.points[j].data_label
        dl.position = pos
        tf = dl.text_frame
        tf.text = fmt(v)
        f = tf.paragraphs[0].runs[0].font
        f.size = Pt(size)
        f.bold = bold
        f.color.rgb = rgb(color)


# ================= 1. Mở đầu =================
s = new_slide(dark=True)
set_title(s, "Bộ dữ liệu song ngữ Trung Việt cho thương mại điện tử", dark=True)
tb(s, 0.6, 0.5, 6, 0.3, [[("BÁO CÁO TIẾN ĐỘ", dict(size=12, bold=True, color=GOLD))]], name="Nhãn")
# giãn chữ cho nhãn
for r in s.shapes[-1].text_frame.paragraphs[0].runs:
    r._r.get_or_add_rPr().set("spc", "300")
tb(s, 0.6, 2.55, 8.8, 0.4, [[("Từ dữ liệu crawl thô của 1688 và Tiki đến bộ chính thức", dict(size=18, color="CBD5E1"))]], name="Phụ đề")
stats = [
    ("26.038", "sản phẩm trong bộ chính thức"),
    ("2", "nguồn: 1688 và Tiki"),
    ("9", "nhóm hàng"),
    ("2", "mô hình dịch: Gemma và Qwen"),
]
for i, (n, l) in enumerate(stats):
    x = 0.6 + i * 2.25
    box(s, x, 3.4, 2.05, 1.45, CARD_DARK, radius=0.08, name=f"Ô số {i+1}")
    tb(s, x + 0.18, 3.5, 1.7, 0.65, [[(n, dict(size=36, bold=True, color=GOLD, font=HEAD))]], anchor="m", name=f"Số {i+1}")
    tb(s, x + 0.18, 4.15, 1.72, 0.6, [[(l, dict(size=13, color="E2E8F0"))]], name=f"Nhãn số {i+1}")
tb(s, 0.6, 5.1, 3, 0.25, [[("Tháng 10/2026", dict(size=11, color="94A3B8"))]], name="Ngày")
notes(s, "Thưa thầy, hôm nay em báo cáo về bộ dữ liệu song ngữ Trung Việt cho thương mại điện tử mà nhóm vừa hoàn thành. "
         "Kết quả cuối là 26.038 sản phẩm lấy từ hai sàn 1688 và Tiki, chia thành 9 nhóm hàng, được dịch bằng hai mô hình là Gemma 4 và Qwen3 14B. "
         "Em sẽ đi từ dữ liệu thô đến bộ chính thức qua một sơ đồ duy nhất.")

# ================= 2. Dữ liệu thô =================
s = new_slide(number=2)
set_title(s, "Dữ liệu thô từ hai sàn")
cards = [
    dict(x=0.5, tint=TINT_OR, col=ORANGE, name="1688", sub="Gốc tiếng Trung, cần dịch sang Việt",
         num="16.348", unit="sản phẩm, 9 nhóm hàng",
         rows=[("Trung (gốc)", "牛奶洗面奶深层清洁补水保湿男女洁面乳", True),
               ("Việt (dịch)", "Sữa rửa mặt sữa tươi làm sạch sâu cung cấp nước dưỡng ẩm cho nam nữ", False),
               ("Mô tả", "化妆品功效: 补水保湿; 成分: 甘油", True),
               ("", "Công dụng mỹ phẩm: Cấp nước, dưỡng ẩm; Thành phần: Glycerin", False)]),
    dict(x=5.1, tint=TINT_BL, col=BLUE, name="Tiki", sub="Gốc tiếng Việt, cần dịch sang Trung",
         num="13.200", unit="sản phẩm, 6 nhóm hàng",
         rows=[("Việt (gốc)", "Bánh mì hoa cúc Mini Harrys 210gr PHÁP", False),
               ("Trung (dịch)", "迷你雏菊面包 Harrys 210克 法式", True),
               ("Mô tả", "Thương hiệu: Harrys; Xuất xứ thương hiệu: Pháp", False),
               ("", "品牌: Harrys; 品牌产地: 法国", True)]),
]
for c in cards:
    x = c["x"]
    box(s, x, 1.2, 4.4, 3.65, c["tint"], radius=0.08, name=f"Thẻ {c['name']}")
    tb(s, x + 0.25, 1.3, 1.4, 0.5, [[(c["name"], dict(size=26, bold=True, color=c["col"], font=HEAD))]], anchor="m", name=f"Tên {c['name']}")
    tb(s, x + 1.5, 1.3, 2.75, 0.5, [[(c["sub"], dict(size=12, color=BODY))]], anchor="m", name=f"Mô tả {c['name']}")
    tb(s, x + 0.25, 1.85, 3.95, 0.6,
       [[(c["num"], dict(size=32, bold=True, color=NAVY, font=HEAD)), ("  " + c["unit"], dict(size=14, color=BODY))]],
       anchor="m", name=f"Số lượng {c['name']}")
    box(s, x + 0.2, 2.6, 4.0, 2.1, WHITE, radius=0.06, name=f"Khung ví dụ {c['name']}")
    paras = []
    for lab, txt, zh in c["rows"]:
        if lab:
            paras.append([(lab, dict(size=10, bold=True, color=MUTED))])
        paras.append([(txt, dict(size=12, color=INK, cjk=zh))])
    t = tb(s, x + 0.35, 2.7, 3.7, 1.9, paras, name=f"Ví dụ {c['name']}")
    # khoảng cách trước nhãn "Mô tả"
    t.text_frame.paragraphs[2].space_before = Pt(5)
    t.text_frame.paragraphs[4].space_before = Pt(9)
tb(s, 0.5, 4.98, 8.5, 0.3, [[("Mỗi sản phẩm gồm tiêu đề (Title) và mô tả (Description, là danh sách thuộc tính)", dict(size=13, color=BODY))]], name="Chú thích")
notes(s, "Dữ liệu thô có hai nguồn đi ngược chiều nhau. 1688 có 16.348 sản phẩm, bản gốc là tiếng Trung và cần dịch sang tiếng Việt. "
         "Tiki có 13.200 sản phẩm, bản gốc là tiếng Việt và cần dịch sang tiếng Trung. Mỗi sản phẩm có tiêu đề và phần mô tả. "
         "Mô tả ở đây là danh sách thuộc tính như thương hiệu, xuất xứ, chất liệu. Hai ví dụ trên slide cho thầy thấy bản gốc và bản dịch của một sản phẩm.")

# ================= 3. Phễu =================
s = new_slide(number=3)
set_title(s, "Phễu xử lý từ thô đến chính thức")
tb(s, 0.5, 0.95, 9, 0.3, [[("Hai phần chạy song song: Description (Nhật Anh) và Title (Huy)", dict(size=14, color=MUTED))]], name="Phụ đề")
W, GAP = 1.62, 0.225
colx = lambda i: 0.5 + i * (W + GAP)
stages = [
    dict(n="29.548", u="sản phẩm thô", name="Thu thập thô", cap=["1688: 16.348", "Tiki: 13.200"], h=1.4),
    dict(n="494.575", u="cặp thuộc tính", name="Làm sạch", cap=["Tách mô tả thành cặp, sửa bằng từ điển, bỏ thông tin riêng tư"], h=1.25),
    dict(n="2", u="mô hình dịch", name="Dịch từng cặp", cap=["Gemma 4 làm chính, Qwen3-14B dự phòng, CometKiwi chấm điểm"], h=1.1),
    dict(n="29.529", u="sản phẩm có mô tả dịch", name="Dịch theo cả cụm", cap=["Hiệu đính cả danh sách thuộc tính theo ngữ cảnh sản phẩm"], h=0.95),
    dict(n="26.038", u="sản phẩm chính thức", name="Gộp và lọc", cap=["Loại 3.510 sản phẩm theo 6 quy tắc"], h=0.8, last=True),
]
cy = 2.7
for i, st in enumerate(stages):
    x = colx(i)
    tb(s, x, 1.3, W, 0.45, [[(st["n"], dict(size=26, bold=True, color=GREEN if st.get("last") else NAVY, font=HEAD))]], anchor="b", name=f"Số bước {i+1}")
    tb(s, x, 1.77, W, 0.25, [[(st["u"], dict(size=11, color=MUTED))]], name=f"Đơn vị bước {i+1}")
    b = box(s, x, cy - st["h"] / 2, W, st["h"], GREEN if st.get("last") else NAVY, radius=0.06, name=f"Khối bước {i+1}")
    fill_tf(b.text_frame, [[(st["name"], dict(size=15, bold=True, color=WHITE))]], anchor="m", align="c")
    b.text_frame.margin_left = b.text_frame.margin_right = Inches(0.08)
    if i < len(stages) - 1:
        box(s, x + W + 0.03, cy - 0.12, 0.16, 0.24, MUTED, shape=MSO_SHAPE.CHEVRON, name=f"Mũi tên {i+1}")
    tb(s, x, 3.5, W, 0.95, [[(c, dict(size=12, color=BODY))] for c in st["cap"]], name=f"Chú thích bước {i+1}")
bx = colx(1)
bw = colx(3) + W - colx(1)
box(s, bx, 4.6, bw, 0.55, GREY_BG, radius=0.06, name="Dải Title")
tb(s, bx + 0.15, 4.6, bw - 0.3, 0.55,
   [[("Title (Huy): ", dict(size=12, bold=True, color=NAVY)),
     ("dịch bằng 2 mô hình, chọn bản tốt hơn theo điểm COMETKiwi, sửa chữ Hán còn sót", dict(size=12, color=BODY))]],
   anchor="m", name="Chữ dải Title")
notes(s, "Đây là sơ đồ chính của buổi báo cáo. Em và bạn Huy làm song song hai phần: em làm phần mô tả, bạn Huy làm phần tiêu đề. "
         "Bước một là dữ liệu thô, 29.548 sản phẩm. "
         "Bước hai là làm sạch: em tách mô tả thành gần 495 nghìn cặp thuộc tính, sửa lỗi bằng quy tắc và từ điển đã duyệt tay, và bỏ thông tin riêng tư như tên công ty, địa chỉ nhà sản xuất. "
         "Bước ba là dịch từng cặp bằng hai mô hình, Gemma 4 làm chính, Qwen3 14B chỉ thay khi Gemma sai quy tắc kiểm tra, còn điểm CometKiwi dùng để chấm độ tin cậy của từng cặp, và cặp nào cả hai bản đều dưới ngưỡng thì bị loại. "
         "Bước bốn là dịch lại theo cả cụm thuộc tính của một sản phẩm để tên và giá trị nhất quán, lúc này còn 29.529 sản phẩm có mô tả. "
         "Bước cuối là gộp hai sàn và lọc, ra 26.038 sản phẩm. "
         "Phần tiêu đề bạn Huy cũng dịch bằng hai mô hình, chọn bản tốt hơn theo điểm COMETKiwi và sửa chữ Hán còn sót.")

# ================= 4. Bước gộp =================
s = new_slide(number=4)
set_title(s, "Bước gộp cuối: loại 3.510 sản phẩm")
for i, (n, l, col, y) in enumerate([
    ("29.548", "sản phẩm vào bước gộp", NAVY, 1.3),
    ("3.510", "bị loại, chiếm 11,9%", RED, 2.6),
    ("26.038", "sản phẩm giữ lại", GREEN, 3.9),
]):
    tb(s, 0.5, y, 3.1, 0.65, [[(n, dict(size=40, bold=True, color=col, font=HEAD))]], anchor="m", name=f"Số gộp {i+1}")
    tb(s, 0.5, y + 0.65, 3.1, 0.3, [[(l, dict(size=14, color=BODY))]], name=f"Nhãn gộp {i+1}")

cd = CategoryChartData()
cd.categories = ["Vượt trần 3.500 mỗi nhóm", "Mô tả trùng, giữ tối đa 3", "Lạc nhóm hàng", "Cặp thuộc tính dịch yếu", "Title lỗi", "Không có mô tả"]
cd.add_series("Số sản phẩm bị loại", (1822, 825, 594, 214, 36, 19))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(3.9), Inches(1.2), Inches(5.6), Inches(3.4), cd)
gf.name = "Biểu đồ lý do loại"
ch = gf.chart
ch.has_legend = False
ch.has_title = False
ch.font.size = Pt(12)
ch.font.name = BASE
ch.font.color.rgb = rgb(BODY)
ch.category_axis.reverse_order = True
ch.category_axis.format.line.fill.background()
ch.category_axis.has_major_gridlines = False
hide_value_axis(ch)
ch.value_axis.minimum_scale = 0
ch.value_axis.maximum_scale = 2200
pl = ch.plots[0]
pl.gap_width = 45
pl.vary_by_categories = False
pl.series[0].format.fill.solid()
pl.series[0].format.fill.fore_color.rgb = rgb(RED)
vn_labels(pl.series[0], (1822, 825, 594, 214, 36, 19), XL_LABEL_POSITION.OUTSIDE_END, 12, INK, bold=True)
tb(s, 3.9, 4.7, 5.6, 0.5, [[("Mỗi sản phẩm chỉ ghi một lý do: quy tắc đầu tiên nó vi phạm. Danh sách loại được lưu riêng, không xóa dòng nào ở file nguồn.", dict(size=11, color=MUTED))]], name="Ghi chú")
notes(s, "Bước gộp nhận vào 29.548 sản phẩm và giữ lại 26.038, loại 3.510, tức là khoảng 12 phần trăm. "
         "Lý do loại nhiều nhất là vượt trần: em đặt mỗi nhóm hàng tối đa 3.500 sản phẩm để các nhóm không lệch nhau quá xa, nên nhóm nào vượt thì bỏ bớt. "
         "Tiếp theo là mô tả trùng, cùng nhóm có hơn ba sản phẩm mô tả giống hệt thì chỉ giữ ba. "
         "Sau đó là sản phẩm lạc nhóm, ví dụ phụ kiện bể cá nằm trong nhóm thực phẩm, rồi các sản phẩm có nhiều cặp dịch yếu, title lỗi và sản phẩm không có mô tả. "
         "Mỗi sản phẩm bị loại đều có ghi một lý do, và em không xóa dòng nào trong file nguồn.")

# ================= 5. Cân bằng nhóm =================
s = new_slide(number=5)
set_title(s, "Cân bằng 9 nhóm hàng")
cats = ["mother_baby", "home", "beauty", "electronics", "food", "bags", "auto", "shoes", "fashion"]
a = [1958, 1779, 1292, 1084, 1406, 2874, 960, 2309, 1334]
t = [1542, 1721, 2095, 2094, 1710, 0, 1880, 0, 0]
assert sum(a) == 14996 and sum(t) == 11042 and sum(a) + sum(t) == 26038
cd = CategoryChartData()
cd.categories = [f"{c} ({fmt(a[i] + t[i])})" for i, c in enumerate(cats)]
cd.add_series("1688", a)
cd.add_series("Tiki", t)
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(0.5), Inches(1.1), Inches(6.3), Inches(4.1), cd)
gf.name = "Biểu đồ nhóm hàng theo nguồn"
ch = gf.chart
ch.has_title = False
ch.font.size = Pt(12)
ch.font.name = BASE
ch.font.color.rgb = rgb(BODY)
ch.category_axis.reverse_order = True
ch.category_axis.format.line.fill.background()
ch.category_axis.has_major_gridlines = False
hide_value_axis(ch)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12)
pl = ch.plots[0]
pl.gap_width = 40
pl.overlap = 100
for ser, col, vals in zip(pl.series, (ORANGE, BLUE), (a, t)):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(col)
    vn_labels(ser, vals, XL_LABEL_POSITION.CENTER, 11, WHITE)
tb(s, 7.1, 1.25, 2.4, 0.5, [[("Lớn nhất so với nhỏ nhất", dict(size=12, bold=True, color=MUTED))]], name="Nhãn so sánh")
tb(s, 7.1, 1.8, 2.4, 0.6, [[("3,5 lần", dict(size=32, bold=True, color=RED, font=HEAD))]], anchor="m", name="Trước")
tb(s, 7.1, 2.4, 2.4, 0.3, [[("trước khi cân bằng", dict(size=13, color=BODY))]], name="Nhãn trước")
tb(s, 7.1, 2.95, 2.4, 0.6, [[("2,6 lần", dict(size=32, bold=True, color=GREEN, font=HEAD))]], anchor="m", name="Sau")
tb(s, 7.1, 3.55, 2.4, 0.3, [[("sau khi cân bằng", dict(size=13, color=BODY))]], name="Nhãn sau")
tb(s, 7.1, 4.2, 2.4, 0.6, [[("bags, shoes, fashion chỉ có sản phẩm từ 1688", dict(size=12, color=MUTED))]], name="Ghi chú nhóm")
notes(s, "Đây là kết quả sau khi cân bằng. Trước khi gộp, nhóm lớn nhất là mẹ và bé với 4.696 sản phẩm, nhóm nhỏ nhất là thời trang với 1.351 sản phẩm, chênh 3,5 lần. "
         "Sau khi áp trần 3.500 mỗi nhóm, nhóm lớn nhất là 3.500 và nhóm nhỏ nhất là 1.334, chênh còn 2,6 lần. "
         "Màu cam là sản phẩm từ 1688, màu xanh là từ Tiki. Ba nhóm túi, giày và thời trang chỉ có từ 1688 vì Tiki không có các nhóm này, nên nhóm thời trang còn ít, em không có nguồn khác để bù.")

# ================= 6. Hạn chế và việc tiếp theo =================
s = new_slide(number=6)
set_title(s, "Hạn chế và việc tiếp theo")
cols = [
    dict(x=0.5, head="Cần lưu ý", tint=TINT_RED, col=RED, items=[
        "Bản dịch do máy, mới kiểm theo mẫu, chưa có người rà toàn bộ",
        "Title trùng nhau: 1688 có 757 sản phẩm thuộc 337 title lặp, Tiki có 153 sản phẩm thuộc 73 title",
        "Túi, giày, thời trang chỉ có từ 1688; thời trang còn 1.334 sản phẩm"]),
    dict(x=5.1, head="Việc tiếp theo (dự kiến)", tint=TINT_GR, col=GREEN, items=[
        "Gom nhóm các title trùng rồi mới chia train, dev, test",
        "Chọn tập test và nhờ người dịch rà soát",
        "Cả nhóm chốt mô hình để huấn luyện"]),
]
for c in cols:
    x = c["x"]
    box(s, x, 1.2, 4.4, 3.7, c["tint"], radius=0.08, name=f"Thẻ {c['head']}")
    tb(s, x + 0.3, 1.35, 3.8, 0.5, [[(c["head"], dict(size=20, bold=True, color=c["col"], font=HEAD))]], anchor="m", name=f"Tiêu đề {c['head']}")
    for i, it in enumerate(c["items"]):
        y = 2.0 + i * 0.95
        o = box(s, x + 0.3, y + 0.03, 0.36, 0.36, c["col"], shape=MSO_SHAPE.OVAL, name=f"Số mục {i+1}")
        fill_tf(o.text_frame, [[(str(i + 1), dict(size=14, bold=True, color=WHITE))]], anchor="m", align="c")
        tb(s, x + 0.85, y, 3.3, 0.85, [[(it, dict(size=14, color=INK))]], name=f"Nội dung mục {i+1}")
notes(s, "Em xin nêu rõ các hạn chế. Thứ nhất, toàn bộ bản dịch là do máy, em mới kiểm theo mẫu chứ chưa có người rà hết, nên chưa dùng bộ này làm tập test được. "
         "Thứ hai, có nhiều sản phẩm trùng title: ở 1688 có 757 sản phẩm thuộc 337 title lặp, ở Tiki là 153 sản phẩm thuộc 73 title. Vì vậy khi chia train, dev, test phải gom các sản phẩm trùng title vào cùng một tập để tránh rò rỉ. "
         "Thứ ba, ba nhóm túi, giày và thời trang chỉ có từ 1688. "
         "Việc tiếp theo, dự kiến là gom nhóm title trùng rồi chia tập, chọn tập test và nhờ người dịch rà, sau đó cả nhóm chốt mô hình huấn luyện. Em xin hết, cảm ơn thầy.")

# ================= 7. Dự phòng =================
s = new_slide(number=7)
set_title(s, "Dự phòng: 6 quy tắc loại sản phẩm")
data = [
    ("R0  Không có mô tả", "Sản phẩm không có mô tả để dịch", "19"),
    ("R1  Lạc nhóm hàng", "Thuộc 56 danh mục không hợp nhóm, ví dụ phụ kiện bể cá nằm trong nhóm food", "594"),
    ("R2  Cặp dịch yếu", "Từ 25% số cặp thuộc tính trở lên có độ tin cậy thấp", "214"),
    ("R3  Mô tả trùng", "Cùng nguồn và nhóm có hơn 3 mô tả giống hệt, chỉ giữ tối đa 3", "825"),
    ("R4  Title lỗi", "Title còn chữ Hán, chưa dịch hoặc mô hình từ chối dịch", "36"),
    ("R5  Vượt trần nhóm", "Nhóm quá 3.500 sản phẩm: bỏ bớt, ưu tiên danh mục kém hợp rồi sản phẩm có nhiều cặp yếu hơn", "1.822"),
]
assert sum(int(d[2].replace(".", "")) for d in data) == 3510
rows_h = [0.4, 0.42, 0.55, 0.42, 0.42, 0.42, 0.55, 0.42]
gt = s.shapes.add_table(len(rows_h), 3, Inches(0.5), Inches(1.2), Inches(9), Inches(sum(rows_h)))
gt.name = "Bảng quy tắc"
tbl = gt.table
# bỏ kiểu bảng mặc định (có kẻ ô)
tblPr = tbl._tbl.tblPr
for k in ("firstRow", "bandRow"):
    tblPr.set(k, "0")
sid = tblPr.find(qn("a:tableStyleId"))
if sid is None:
    sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
for j, w in enumerate((2.3, 5.7, 1.0)):
    tbl.columns[j].width = Inches(w)
for i, h in enumerate(rows_h):
    tbl.rows[i].height = Inches(h)


def cell(i, j, text, bold=False, color=INK, fill=None, align="l"):
    c = tbl.cell(i, j)
    c.margin_left = c.margin_right = Inches(0.1)
    c.margin_top = c.margin_bottom = Inches(0.03)
    c.vertical_anchor = MSO_ANCHOR.MIDDLE
    c.text_frame.word_wrap = True
    p = c.text_frame.paragraphs[0]
    p.alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[align]
    r = p.add_run()
    r.text = text
    style_run(r, size=13, bold=bold, color=color)
    if fill:
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(fill)
    else:
        c.fill.background()


for j, (h, al) in enumerate((("Quy tắc", "l"), ("Loại khi", "l"), ("Số loại", "r"))):
    cell(0, j, h, bold=True, color=WHITE, fill=NAVY, align=al)
for i, (a1, b1, c1) in enumerate(data, start=1):
    f = GREY_BG if i % 2 else WHITE
    cell(i, 0, a1, bold=True, color=NAVY, fill=f)
    cell(i, 1, b1, fill=f)
    cell(i, 2, c1, bold=True, color=RED, fill=f, align="r")
cell(7, 0, "Tổng", bold=True)
cell(7, 1, "")
cell(7, 2, "3.510", bold=True, color=RED, align="r")
tb(s, 0.5, 4.95, 9, 0.3, [[("Quy tắc áp theo thứ tự R0 đến R5; mỗi sản phẩm bị ghi theo quy tắc đầu tiên nó vi phạm.", dict(size=12, color=MUTED))]], name="Ghi chú")
notes(s, "Trang này chỉ mở khi thầy hỏi loại theo tiêu chí nào. Có sáu quy tắc, áp theo thứ tự từ R0 đến R5, mỗi sản phẩm được ghi theo quy tắc đầu tiên nó vi phạm. "
         "Quy tắc R5 là trần 3.500 sản phẩm mỗi nhóm, khi vượt thì em ưu tiên bỏ những sản phẩm thuộc danh mục kém hợp với nhóm, rồi đến sản phẩm có tỷ lệ cặp dịch yếu cao hơn. Tổng cộng loại 3.510 sản phẩm.")

prs.core_properties.title = "Bộ dữ liệu song ngữ Trung Việt cho thương mại điện tử"
prs.core_properties.subject = "Báo cáo tiến độ cho thầy hướng dẫn"
prs.core_properties.author = "Nhật Anh"
tmp = OUT + ".tmp"
prs.save(tmp)

# ---------- vá theme: màu và phông của bộ slide ----------
CLR = dict(dk1=INK, lt1=WHITE, dk2=NAVY, lt2=GREY_BG, accent1=ORANGE, accent2=BLUE, accent3=GREEN,
           accent4=RED, accent5=MUTED, accent6=GOLD, hlink=BLUE, folHlink=MUTED)
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
with zipfile.ZipFile(tmp) as zin, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data_b = zin.read(item.filename)
        if item.filename.startswith("ppt/theme/theme") and item.filename.endswith(".xml"):
            root = etree.fromstring(data_b)
            cs = root.find(f".//{{{A}}}clrScheme")
            cs.set("name", "Song ngu TMDT")
            for k, v in CLR.items():
                el = cs.find(f"{{{A}}}{k}")
                for ch_ in list(el):
                    el.remove(ch_)
                etree.SubElement(el, f"{{{A}}}srgbClr").set("val", v)
            fs = root.find(f".//{{{A}}}fontScheme")
            fs.set("name", "Song ngu TMDT")
            fs.find(f"{{{A}}}majorFont/{{{A}}}latin").set("typeface", HEAD)
            fs.find(f"{{{A}}}minorFont/{{{A}}}latin").set("typeface", BASE)
            data_b = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        zout.writestr(item, data_b)
import os
os.remove(tmp)
print("done", OUT)
