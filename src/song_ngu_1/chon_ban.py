# Bước 6b (laptop, CPU): chọn một bản Việt cho mỗi đơn vị từ bản A và bản B, dựa vào luật kiểm (bước 5),
# chrF giữa hai bản và điểm CometKiwi của từng bản (bước 6a, chấm trên Kaggle).
#
# Bản sửa 15/10/2026 sau lần chạy thử 4 (91 đơn vị có cả hai bản, đọc tay, chưa có nhãn đúng sai):
# - Hai bản giống hệt sau chuẩn hoá và cùng qua luật thì giữ luôn. Hai mô hình độc lập cùng ra một bản là bằng chứng mạnh,
#   mà CometKiwi chấm chuỗi thuộc tính ngắn khá thấp (5 bản đúng như "Túi đeo chéo" bị loại vì điểm 0,44).
# - Điểm sàn kappa nới xuống kappa_ngan với chuỗi Trung ngắn (từ DO_DAI_ZH_NGAN ký tự trở xuống): bản đúng 0,45 đến 0,48 bị loại
#   ở sàn 0,5, còn các bản sai rõ nằm ở 0,29 đến 0,33.
# - Hai bản khác nhau và cùng đạt sàn: lấy bản của mô hình ưu tiên (mặc định A, DeepSeek), trừ khi điểm CometKiwi của bản kia cao hơn
#   từ delta trở lên. Lý do: CometKiwi thiên về câu dài (刷子颜色 "Màu sắc của chiếc bàn chải" 0,71 so với "Màu bàn chải" 0,40),
#   còn mô hình ưu tiên đúng nghĩa chuyên ngành hơn ở mẫu đã đọc. Đây là TẠM, cần kiểm lại trên khoảng 2.000 đơn vị.
import math

from .van_ban import chrf, chuan_hoa

DO_DAI_ZH_NGAN = 8     # chuỗi Trung từ ngần này ký tự trở xuống thì dùng điểm sàn kappa_ngan


def _so(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def chon(a, b, truot_a, truot_b, nhac_a, nhac_b, kiwi_a, kiwi_b, tau, kappa, delta, do_dai_ngan=6, chrf_ab=None,
         dai_zh=None, kappa_ngan=None, uu_tien="A", chi_doi_khi_truot=False, norm=None, dai_ngan_nguon=None, kappa_cat=None,
         kappa_cat_ngan=None):
    """Trả về dict: ban (văn bản chọn hoặc None), tu ('A'/'B'/None), cach_chon, do_tin_cay ('cao'/'vua'/'thap'/None),
    chrf_ab, ly_do_cat (khi không chọn được). Năm trường hợp theo trang thiết kế.
    dai_zh: độ dài chuỗi Trung đã gửi; kappa_ngan: điểm sàn cho chuỗi Trung ngắn (None thì dùng kappa);
    uu_tien: mô hình được ưu tiên khi hai bản cùng đạt ('A' hoặc 'B'); delta: chênh điểm CometKiwi đủ để đổi sang bản kia.
    chi_doi_khi_truot (anh duyệt 22/10/2026): True thì khi hai bản khác nhau và cùng qua luật, luôn lấy bản ưu tiên, điểm CometKiwi chỉ
    quyết định độ tin cậy (và cất riêng khi cả hai dưới sàn); bản kia chỉ được dùng khi bản ưu tiên trượt luật. Lý do: ở mẫu 100 cặp,
    10/12 lỗi rõ là Gemma đúng mà bị thay bằng bản Qwen sai nhưng trơn tru vì Kiwi chấm cao hơn (汕头 "Thâm Quyền" 0,74, "Sán Đầu" 0,36).
    False: luật cũ (bản kia vượt từ delta, hoặc chỉ bản kia đạt sàn, hoặc ít cờ nhắc hơn thì đổi sang bản kia).
    norm (bộ 2, 23/10/2026): hàm chuẩn hoá để so hai bản, mặc định chuan_hoa của bản Việt; dai_ngan_nguon: ngưỡng độ dài nguồn để dùng
    kappa_ngan, mặc định DO_DAI_ZH_NGAN (8 ký tự Trung; bộ 2 nguồn là tiếng Việt nên truyền ngưỡng khác).
    kappa_cat, kappa_cat_ngan (bộ 2, 23/10/2026): sàn riêng cho việc CẤT RIÊNG (cả hai bản dưới sàn này thì bỏ); sàn kappa, kappa_ngan chỉ còn
    quyết định độ tin cậy. Mặc định None: dùng chung sàn như cũ."""
    qua_a = a is not None and not truot_a
    qua_b = b is not None and not truot_b
    ka = kiwi_a if _so(kiwi_a) else None
    kb = kiwi_b if _so(kiwi_b) else None
    norm = norm or chuan_hoa
    na, nb = norm(a) if a else "", norm(b) if b else ""
    c = chrf_ab if _so(chrf_ab) else (round(chrf(na, nb), 1) if na and nb else None)
    kq = {"ban": None, "tu": None, "cach_chon": None, "do_tin_cay": None, "chrf_ab": c, "ly_do_cat": None}
    san = kappa_ngan if (kappa_ngan is not None and dai_zh is not None and dai_zh <= (dai_ngan_nguon or DO_DAI_ZH_NGAN)) else kappa
    ngan_nguon = kappa_ngan is not None and dai_zh is not None and dai_zh <= (dai_ngan_nguon or DO_DAI_ZH_NGAN)
    san_cat = (kappa_cat_ngan if ngan_nguon else kappa_cat) if kappa_cat is not None else san
    san_cat = san if san_cat is None else san_cat
    kiwi = {"A": ka, "B": kb}
    nhac = {"A": nhac_a, "B": nhac_b}
    khac = "B" if uu_tien == "A" else "A"

    def lay(tu, cach, tin):
        kq.update(ban=a if tu == "A" else b, tu=tu, cach_chon=f"{cach}_{tu}" if cach != "giong_het" else cach, do_tin_cay=tin)
        return kq

    def dat(tu):
        return kiwi[tu] is not None and kiwi[tu] >= san

    def dat_cat(tu):
        return kiwi[tu] is not None and kiwi[tu] >= san_cat

    def chon_hai():
        """Chọn bản khi hai bản khác nhau và cùng qua luật. Trả về (tu, hai_dat) hoặc None khi không bản nào đạt sàn.
        hai_dat: cả hai đạt sàn và bản chọn không bị bản kia vượt từ delta trở lên."""
        if not dat_cat("A") and not dat_cat("B"):
            return None
        if chi_doi_khi_truot:          # luôn lấy bản ưu tiên; hai_dat = bản ưu tiên có đạt sàn không (quyết định độ tin cậy)
            return uu_tien, dat(uu_tien)
        if dat("A") != dat("B"):
            return ("A" if dat("A") else "B"), False
        if kiwi[khac] - kiwi[uu_tien] >= delta:      # bản kia vượt rõ: bỏ ưu tiên
            return khac, False
        if len(nhac[khac]) < len(nhac[uu_tien]):    # ít cờ nhắc hơn thì tin hơn
            return khac, True
        return uu_tien, True

    # 5. cả hai trượt (hoặc thiếu bản)
    if not qua_a and not qua_b:
        kq["ly_do_cat"] = "ca_hai_truot"
        return kq
    # 4. chỉ một bản qua luật
    if qua_a != qua_b:
        tu = "A" if qua_a else "B"
        if dat_cat(tu):
            return lay(tu, "chi", "thap")
        kq["ly_do_cat"] = "mot_ban_qua_luat_diem_thap"
        return kq
    # cả hai qua luật
    ngan = min(len(na), len(nb)) <= do_dai_ngan
    cao = max(x for x in (ka, kb) if x is not None) if (ka is not None or kb is not None) else None
    # 1. giống hệt: giữ, không đòi điểm sàn; chỉ độ tin cậy phụ thuộc điểm
    if na == nb:
        tu = khac if len(nhac[khac]) < len(nhac[uu_tien]) else uu_tien
        return lay(tu, "giong_het", "cao" if (cao is not None and cao >= san) else "vua")
    # 2. gần giống (chỉ với chuỗi dài)
    if not ngan and c is not None and c >= tau:
        r = chon_hai()
        if r is None:
            kq["ly_do_cat"] = "gan_giong_diem_thap"
            return kq
        return lay(r[0], "gan_giong", "vua" if (r[1] or not chi_doi_khi_truot) else "thap")
    # 3. khác nhau nhiều
    if ka is None or kb is None:
        kq["ly_do_cat"] = "khac_nhieu_thieu_diem"
        return kq
    r = chon_hai()
    if r is None:
        kq["ly_do_cat"] = "khac_nhieu_diem_khong_du"
        return kq
    if r[1]:
        return lay(r[0], "khac_chu_cung_nghia", "vua")
    # luật mới: bản ưu tiên dưới sàn nhưng bản kia đạt sàn, vẫn giữ bản ưu tiên, tin cậy thấp; luật cũ: bản kia vượt theo điểm
    return lay(r[0], "uu_tien_duoi_san" if chi_doi_khi_truot else "diem_vuot", "thap")
