# Dựng file Excel mẫu kiểm tra 200 cặp của bộ song ngữ 2 (Tiki, Việt sang Trung). Chạy bằng Python 3.9 có openpyxl (.venv của dự án không có openpyxl);
# dữ liệu lấy từ mau_200.json do lay_mau_200.py sinh ra. Cách chạy: <python3.9> dung_excel_mau.py <file xlsx ra> <thư mục có mau_200.json>
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

RA = Path(sys.argv[1])
TAM = Path(sys.argv[2])
mau = json.loads((TAM / "mau_200.json").read_text(encoding="utf-8"))
meta = json.loads((TAM / "mau_200_meta.json").read_text(encoding="utf-8"))
N = len(mau)
CUOI = N + 1                      # dòng dữ liệu cuối của sheet Mau

F = "Arial"
f_thuong = Font(name=F, size=10)
f_dam = Font(name=F, size=10, bold=True)
f_tieu_de = Font(name=F, size=14, bold=True)
f_xanh = Font(name=F, size=10, color="0000FF")          # số nhập tay (đếm từ dữ liệu)
f_xam = Font(name=F, size=10, color="595959")
nen_vang = PatternFill("solid", fgColor="FFF2CC")       # ô anh điền
nen_dau = PatternFill("solid", fgColor="D9E1F2")        # tiêu đề cột dữ liệu
nen_tham_khao = PatternFill("solid", fgColor="EDEDED")  # cột tham khảo
vien = Border(*(Side(style="thin", color="BFBFBF"),) * 4)
boc = Alignment(wrap_text=True, vertical="top")
LUA_CHON = '"Đúng,Sai nhẹ,Sai"'

wb = Workbook()

# ---------------- Sheet 1: Hướng dẫn ----------------
hd = wb.active
hd.title = "Huong_dan"
dong = [
    ("Mẫu kiểm tra bộ song ngữ thứ hai (description của Tiki, Việt sang Trung)", f_tieu_de),
    (f"Bản dữ liệu: bo_song_ngu_2_day_du_{meta['nhan']}.parquet ({meta['tong_cap']:,} cặp). Mẫu {N} cặp, lấy ngẫu nhiên có phân tầng, hạt giống {meta['seed']}. Tiếng Việt là bản gốc của Tiki, tiếng Trung là bản dịch máy.", f_thuong),
    ("", f_thuong),
    ("Anh làm gì", f_dam),
    ("1. Mở sheet Mau. Mỗi dòng là một cặp thuộc tính của một sản phẩm. Chỉ điền 4 cột nền vàng (H đến K).", f_thuong),
    ("2. Cột H (Đánh giá tên) và I (Đánh giá giá trị): chọn trong danh sách thả xuống: Đúng, Sai nhẹ, Sai. Chấm tên và giá trị riêng.", f_thuong),
    ("3. Cột J: nếu Sai hoặc Sai nhẹ, ghi bản dịch anh cho là đúng (không bắt buộc). Cột K: ghi chú tự do.", f_thuong),
    ("4. Đọc cột D đến G (Việt gốc và Trung dịch) rồi chấm trước; các cột tham khảo (L đến S, nền xám) chỉ xem sau, để không bị bản dịch khác ảnh hưởng.", f_thuong),
    ("5. Sheet Tong_hop tự tính tỉ lệ đúng theo từng nhóm và ước cho cả bộ. Không cần sửa sheet đó.", f_thuong),
    ("", f_thuong),
    ("Cách chấm", f_dam),
    ("Đúng: bản Trung đúng nghĩa, đủ ý, là tiếng Trung giản thể tự nhiên, dùng được ngay trong tin đăng thương mại điện tử (cách diễn đạt khác anh nghĩ vẫn là Đúng).", f_thuong),
    ("Sai nhẹ: đúng nghĩa chính nhưng có lỗi nhỏ: diễn đạt gượng, thừa hoặc thiếu chữ không quan trọng, dấu câu, đơn vị, từ chưa chuẩn thuật ngữ sàn Trung.", f_thuong),
    ("Sai: sai nghĩa, thiếu hoặc thêm ý, bịa thông tin, sai số hoặc mã, lệch số mục trong danh sách, còn chữ Việt chưa dịch (trừ tên riêng, tên hãng), chữ phồn thể, hoặc không đọc hiểu được.", f_thuong),
    ("Danh sách nhiều mục (cách nhau bởi dấu phẩy): một mục sai là Sai nhẹ nếu ý chính vẫn đúng, nhiều mục sai hoặc thiếu mục là Sai.", f_thuong),
    ("Ở Việt \"cân\" là kg: bản Trung phải là 公斤 hoặc 千克; viết 斤 là Sai (sai gấp đôi). Tên thương hiệu và mã giữ nguyên chữ Latin là Đúng.", f_thuong),
    ("", f_thuong),
    ("Ví dụ cách điền (dòng minh hoạ, không nằm trong mẫu)", f_dam),
]
for i, (t, f) in enumerate(dong, start=1):
    hd.cell(row=i, column=1, value=t).font = f
vd_dau = ["Tên thuộc tính (Việt)", "Tên thuộc tính (Trung)", "Giá trị (Việt)", "Giá trị (Trung)", "Đánh giá tên", "Đánh giá giá trị", "Bản sửa đề xuất", "Ghi chú"]
vd = ["Trọng lượng", "重量", "Nặng 5 cân", "重5斤", "Đúng", "Sai", "重5公斤", "Cân của Việt Nam là kg, 斤 sai gấp đôi"]
r0 = len(dong) + 1
for j, (a, b) in enumerate(zip(vd_dau, vd), start=1):
    c = hd.cell(row=r0, column=j, value=a)
    c.font, c.fill, c.border = f_dam, nen_dau, vien
    c = hd.cell(row=r0 + 1, column=j, value=b)
    c.font, c.border = f_thuong, vien
    if j >= 5:
        c.fill = nen_vang
hd.cell(row=r0 + 3, column=1, value="Các nhóm trong mẫu (số mẫu nhiều hơn ở nhóm máy dịch có độ tin cậy thấp, vì lỗi hay nằm ở đó)").font = f_dam
for k, g in enumerate(meta["nhom"]):
    hd.cell(row=r0 + 4 + k, column=1, value=g["nhom"]).font = f_dam
    hd.cell(row=r0 + 4 + k, column=2, value=f"{g['mo_ta']} ({g['so_mau']} mẫu, {g['so_cap_trong_bo']:,} cặp trong bộ)").font = f_thuong
hd.column_dimensions["A"].width = 24
for col in "BCDEFGH":
    hd.column_dimensions[col].width = 22

# ---------------- Sheet 2: Mẫu ----------------
ws = wb.create_sheet("Mau")
COT = [("STT", "stt", 6), ("Nhóm", "nhom", 7), ("Danh mục", "category", 10),
       ("Tên thuộc tính (Việt, gốc)", "name_vi", 26), ("Tên thuộc tính (Trung, dịch)", "name_zh", 20),
       ("Giá trị (Việt, gốc)", "gia_tri_vi", 42), ("Giá trị (Trung, dịch)", "gia_tri_zh", 42),
       ("Đánh giá tên", None, 12), ("Đánh giá giá trị", None, 12), ("Bản sửa đề xuất", None, 30), ("Ghi chú", None, 24),
       ("Tham khảo: cách chọn", "cach_chon", 18), ("Tham khảo: độ tin cậy", "do_tin_cay", 11),
       ("Tham khảo: bản A (Gemma 26B)", "gia_tri_zh_A", 34), ("Tham khảo: bản B (Qwen3-14B)", "gia_tri_zh_B", 34),
       ("Tham khảo: Kiwi A", "diem_kiwi_A", 9), ("Tham khảo: Kiwi B", "diem_kiwi_B", 9),
       ("Tham khảo: luật trượt", "luat_truot", 18), ("pair_id", "pair_id", 9)]
for j, (ten, _, rong) in enumerate(COT, start=1):
    c = ws.cell(row=1, column=j, value=ten)
    c.font, c.border, c.alignment = f_dam, vien, Alignment(wrap_text=True, vertical="center")
    c.fill = nen_vang if 8 <= j <= 11 else (nen_tham_khao if j >= 12 else nen_dau)
    ws.column_dimensions[c.column_letter].width = rong
for i, r in enumerate(mau, start=2):
    for j, (_, khoa, _) in enumerate(COT, start=1):
        v = r.get(khoa) if khoa else None
        if isinstance(v, float) and v != v:      # NaN
            v = None
        c = ws.cell(row=i, column=j, value=v)
        c.font, c.border, c.alignment = (f_xam if j >= 12 else f_thuong), vien, boc
        if 8 <= j <= 11:
            c.fill = nen_vang
        if j in (16, 17) and v is not None:
            c.number_format = "0.00"
ws.freeze_panes = "A2"
ws.auto_filter.ref = f"A1:S{CUOI}"
dv = DataValidation(type="list", formula1=LUA_CHON, allow_blank=True, showErrorMessage=True,
                    errorTitle="Giá trị không hợp lệ", error="Chọn Đúng, Sai nhẹ hoặc Sai")
ws.add_data_validation(dv)
dv.add(f"H2:I{CUOI}")
ws.cell(row=1, column=8).comment = Comment("Điền ô vàng H đến K. H, I chọn trong danh sách: Đúng, Sai nhẹ, Sai.", "Claude")

# ---------------- Sheet 3: Tổng hợp ----------------
th = wb.create_sheet("Tong_hop")
th.cell(row=1, column=1, value="Tổng hợp kết quả chấm (tự tính từ sheet Mau)").font = f_tieu_de
th.cell(row=2, column=1, value="Tỉ lệ = số Đúng / số dòng đã chấm của nhóm. Ước cả bộ = trung bình tỉ lệ các nhóm, trọng số là số cặp của nhóm trong bộ (chỉ tính nhóm đã chấm).").font = f_xam
DAU = ["Nhóm", "Mô tả", "Số cặp trong bộ", "Số mẫu", "Đã chấm", "Đúng", "Sai nhẹ", "Sai", "Tỉ lệ Đúng", "Tỉ lệ Đúng hoặc Sai nhẹ", "Trọng số dùng"]
RNG_NHOM = f"Mau!$B$2:$B${CUOI}"


def bang(dong_dau, tieu_de, cot_cham):
    """Bảng theo nhóm cho một cột chấm (I: giá trị, H: tên). Trả dòng kế tiếp."""
    th.cell(row=dong_dau, column=1, value=tieu_de).font = f_dam
    h = dong_dau + 1
    for j, t in enumerate(DAU, start=1):
        c = th.cell(row=h, column=j, value=t)
        c.font, c.fill, c.border, c.alignment = f_dam, nen_dau, vien, Alignment(wrap_text=True, vertical="center")
    rc = f"Mau!${cot_cham}$2:${cot_cham}${CUOI}"
    d0 = h + 1
    for k, g in enumerate(meta["nhom"]):
        d = d0 + k
        th.cell(row=d, column=1, value=g["nhom"])
        th.cell(row=d, column=2, value=g["mo_ta"])
        c = th.cell(row=d, column=3, value=g["so_cap_trong_bo"])
        c.font, c.number_format = f_xanh, "#,##0"
        c.comment = Comment(f"Đếm từ bo_song_ngu_2_day_du_{meta['nhan']}.parquet: số cặp có cach_chon của giá trị thuộc {g['cach_chon']}.", "Claude")
        th.cell(row=d, column=4, value=f"=COUNTIF({RNG_NHOM},A{d})")
        th.cell(row=d, column=5, value=f'=COUNTIFS({RNG_NHOM},A{d},{rc},"<>")')
        th.cell(row=d, column=6, value=f'=COUNTIFS({RNG_NHOM},A{d},{rc},"Đúng")')
        th.cell(row=d, column=7, value=f'=COUNTIFS({RNG_NHOM},A{d},{rc},"Sai nhẹ")')
        th.cell(row=d, column=8, value=f'=COUNTIFS({RNG_NHOM},A{d},{rc},"Sai")')
        th.cell(row=d, column=9, value=f"=IF(E{d}=0,0,F{d}/E{d})")
        th.cell(row=d, column=10, value=f"=IF(E{d}=0,0,(F{d}+G{d})/E{d})")
        th.cell(row=d, column=11, value=f"=IF(E{d}>0,C{d},0)")
    d1 = d0 + len(meta["nhom"]) - 1
    t = d1 + 1
    th.cell(row=t, column=1, value="Cả bộ")
    th.cell(row=t, column=2, value="Ước cho toàn bộ cặp (có trọng số theo số cặp)")
    th.cell(row=t, column=3, value=f"=SUM(C{d0}:C{d1})")
    for j, col in zip(range(4, 9), "DEFGH"):
        th.cell(row=t, column=j, value=f"=SUM({col}{d0}:{col}{d1})")
    th.cell(row=t, column=9, value=f'=IF(SUM(K{d0}:K{d1})=0,"",SUMPRODUCT(K{d0}:K{d1},I{d0}:I{d1})/SUM(K{d0}:K{d1}))')
    th.cell(row=t, column=10, value=f'=IF(SUM(K{d0}:K{d1})=0,"",SUMPRODUCT(K{d0}:K{d1},J{d0}:J{d1})/SUM(K{d0}:K{d1}))')
    th.cell(row=t, column=11, value=f"=SUM(K{d0}:K{d1})")
    for d in range(d0, t + 1):
        for j in range(1, 12):
            c = th.cell(row=d, column=j)
            if c.font != f_xanh:
                c.font = f_dam if d == t else f_thuong
            c.border = vien
            if j in (9, 10):
                c.number_format = "0.0%"
            elif j in (4, 5, 6, 7, 8, 11) or (j == 3 and d == t):
                c.number_format = "#,##0"
    return t + 2


nxt = bang(4, "Giá trị (cột I của sheet Mau)", "I")
nxt = bang(nxt, "Tên thuộc tính (cột H của sheet Mau)", "H")
# theo độ tin cậy: chỉ để tham khảo, mẫu không phân tầng theo cột này nên không ước cả bộ
th.cell(row=nxt, column=1, value="Giá trị theo độ tin cậy của cặp (tham khảo, không có trọng số)").font = f_dam
h = nxt + 1
for j, t in enumerate(["Độ tin cậy", "Số mẫu", "Đã chấm", "Đúng", "Sai nhẹ", "Sai", "Tỉ lệ Đúng"], start=1):
    c = th.cell(row=h, column=j, value=t)
    c.font, c.fill, c.border = f_dam, nen_dau, vien
RNG_TIN = f"Mau!$M$2:$M${CUOI}"
RC = f"Mau!$I$2:$I${CUOI}"
for k, muc in enumerate(["cao", "vua", "thap"]):
    d = h + 1 + k
    th.cell(row=d, column=1, value=muc)
    th.cell(row=d, column=2, value=f"=COUNTIF({RNG_TIN},A{d})")
    th.cell(row=d, column=3, value=f'=COUNTIFS({RNG_TIN},A{d},{RC},"<>")')
    th.cell(row=d, column=4, value=f'=COUNTIFS({RNG_TIN},A{d},{RC},"Đúng")')
    th.cell(row=d, column=5, value=f'=COUNTIFS({RNG_TIN},A{d},{RC},"Sai nhẹ")')
    th.cell(row=d, column=6, value=f'=COUNTIFS({RNG_TIN},A{d},{RC},"Sai")')
    th.cell(row=d, column=7, value=f'=IF(C{d}=0,"",D{d}/C{d})')
    for j in range(1, 8):
        c = th.cell(row=d, column=j)
        c.font, c.border = f_thuong, vien
        c.number_format = "0.0%" if j == 7 else "#,##0"
for col, w in zip("ABCDEFGHIJK", [10, 58, 14, 9, 9, 9, 9, 9, 11, 14, 12]):
    th.column_dimensions[col].width = w

for s in wb.worksheets:
    for row in s.iter_rows():
        for c in row:
            if c.value is not None and (c.font is None or c.font.name != F):
                c.font = Font(name=F, size=c.font.size if c.font else 10, bold=c.font.bold if c.font else False,
                              color=c.font.color if c.font else None)
from openpyxl.workbook.properties import CalcProperties
wb.calculation = CalcProperties(fullCalcOnLoad=True)      # Excel tính lại mọi công thức khi mở
RA.parent.mkdir(parents=True, exist_ok=True)
wb.save(RA)
sys.stdout.reconfigure(encoding="utf-8")
print("ghi", RA, RA.stat().st_size, "byte;", N, "dong mau")


