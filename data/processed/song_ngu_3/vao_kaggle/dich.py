# -*- coding: utf-8 -*-
"""Bước 4 (Kaggle): dịch các đơn vị chữ Trung sang tiếng Việt bằng MỘT mô hình.

Notebook song-ngu-1-kaggle.ipynb gọi script này hai lần CÙNG LÚC:
- mô hình A = DeepSeek V4.1 Flash gọi qua API của NVIDIA (build.nvidia.com, khoá NVIDIA_API_KEY lấy từ Kaggle Secrets), không cần GPU;
- mô hình B = HY-MT1.5-7B chạy trên cả hai GPU T4 (vLLM chia mô hình bản đầy đủ fp16 ra hai GPU, tensor parallel). Lý do (lần chạy
  thử M0, 09/10/2026): bản GPTQ của HY-MT không chạy được trên vLLM 0.31 (desc_act=True) và đường lùi transformers 4-bit chỉ được
  0,3 đơn vị/giây (khoảng 41 giờ cả bộ).
(Qwen3-8B vẫn còn trong MO_HINH để thay lại được, nhưng anh Nhật Anh đã chốt 10/10/2026: Qwen3-8B là baseline huấn luyện về sau
nên không dùng để dựng bộ dữ liệu.)
- Ghi kết quả TỪNG LÔ ra <ra>/<mo_hinh>/lo_*.parquet ngay khi dịch xong lô đó.
- Chạy tiếp được: bỏ qua đơn vị đã có trong <ra> và trong các thư mục --cu (đầu ra phiên bản trước, gắn vào làm dữ liệu).
- Hết --gio_toi_da thì dừng êm (không lỗi) để Kaggle lưu đầu ra; chạy phiên bản sau để dịch tiếp.
- Bản dịch còn sót chữ Hán thì hỏi lại mô hình một lần (lượt hội thoại thứ hai); giữ bản ít chữ Hán hơn, ghi cột dich_lai.
- Backend: thử vLLM (nhanh); lỗi thì dùng transformers + bitsandbytes 4-bit (rất chậm), trừ khi --backend ép một loại.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# Bản đầy đủ (không lượng tử) cho vLLM: bản AWQ của Qwen3-8B bỏ sót chữ Hán ở 33/200 câu thử; bản GPTQ của HY-MT không nạp được.
MO_HINH = {
    # API chính thức của DeepSeek (TRẢ PHÍ, đọc api-docs.deepseek.com ngày 13/10/2026). Mô hình deepseek-flash là DeepSeek-V4.1-Flash.
    # Suy luận bật sẵn nên phải tắt bằng {"thinking": {"type": "disabled"}}. gia_usd_1m là giá GIỜ CAO ĐIỂM (gấp đôi giờ thấp điểm)
    # để ước chi phí theo hướng dư; dich.py đòi --ngan_sach_usd cho mô hình có gia_usd_1m và tự dừng khi ước chi phí chạm ngân sách.
    "deepseek_flash_api": {"api": "deepseek-flash", "base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY",
                           "kieu": "chat", "lap": 1.0, "extra": {"thinking": {"type": "disabled"}}, "them_token_ra": 256,
                           "so_lan_thu": 5, "song_song": 32, "gia_usd_1m": {"vao_miss": 0.30, "vao_hit": 0.006, "ra": 1.20}},
    # DeepSeek V4.1 Flash trên ALIBABA CLOUD Model Studio (TRẢ PHÍ, đọc help.alibabacloud.com ngày 15/10/2026). Mã mô hình deepseek-v4.1-flash,
    # khoá lấy từ DASHSCOPE_API_KEY. Địa chỉ gọi tuỳ tài khoản (có thể dạng https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1):
    # lấy đúng địa chỉ trong bảng điều khiển và truyền --api_base_url (mặc định dưới đây là địa chỉ quốc tế cũ, CHƯA kiểm).
    # Suy luận bật sẵn, tắt bằng enable_thinking=false (tài liệu không nêu ví dụ tắt, nên thử từng cách và chọn cách không có chữ suy luận).
    # KHÔNG dùng khoá Token Plan (bắt đầu bằng sk-sp-): điều khoản cấm chạy tự động và gọi hàng loạt, có thể bị khoá. Giá quốc tế (Singapore),
    # mỗi triệu token: vào 0,30 giờ cao điểm hoặc 0,15 giờ thấp điểm, ra 1,20 hoặc 0,60; cao điểm là 8h đến 22h giờ UTC+8 (7h đến 21h giờ Việt Nam).
    "deepseek_alibaba": {"api": "deepseek-v4.1-flash", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                         "key_env": "DASHSCOPE_API_KEY", "kieu": "chat", "lap": 1.0,
                         "extra": [{"enable_thinking": False}, {"thinking": {"type": "disabled"}}],
                         "them_token_ra": 256, "so_lan_thu": 5, "song_song": 16,
                         "gia_usd_1m": {"vao_miss": 0.30, "vao_hit": 0.30, "ra": 1.20}},
    # Nhà cung cấp BẤT KỲ có giao diện kiểu OpenAI (nhà bán lại, cổng trung gian): bắt buộc truyền --api_base_url và --api_model;
    # khoá đọc từ biến môi trường API_KEY_A (đổi bằng --api_key_env). Giá chỉ để ước chi phí và chặn ngân sách (đổi bằng
    # --gia_vao_usd_1m và --gia_ra_usd_1m nếu nhà cung cấp tính khác). Khoá chỉ được gửi tới đúng địa chỉ --api_base_url.
    "api_tuy_chon": {"api": "", "base_url": "", "key_env": "API_KEY_A", "kieu": "chat", "lap": 1.0,
                     # reasoning_effort là trường chuẩn của giao diện OpenAI mà Google AI Studio nhận (các trường thinking, enable_thinking,
                     # chat_template_kwargs bị Google từ chối 400 "Unknown name"); {} cuối: nhà cung cấp từ chối trường lạ
                     "extra": [{"reasoning_effort": "none"}, {"reasoning_effort": "minimal"}, {"reasoning_effort": "low"},
                               {"thinking": {"type": "disabled"}}, {"enable_thinking": False},
                               {"chat_template_kwargs": {"thinking": False}}, {}],
                     "them_token_ra": 256, "so_lan_thu": 5, "song_song": 8,
                     "gia_usd_1m": {"vao_miss": 0.30, "vao_hit": 0.30, "ra": 1.20}},
    # API miễn phí của NVIDIA (dùng thử lần M0 3 và 4: quá chậm, khoảng 3 phút mỗi lần gọi). Giữ lại để đối chiếu.
    "deepseek_v41_flash": {"api": "deepseek-ai/deepseek-v4.1-flash", "base_url": "https://integrate.api.nvidia.com/v1",
                           "key_env": "NVIDIA_API_KEY", "kieu": "chat", "lap": 1.0},
    "qwen3_8b": {"vllm": "Qwen/Qwen3-8B", "hf": "Qwen/Qwen3-8B", "kieu": "qwen", "lap": 1.0},
    # eager: vLLM 0.31 chụp CUDA graph cho HunYuan thì lỗi "operation not permitted when stream is capturing" (M0 lần 2),
    # nên chạy thẳng không chụp graph. Bản vLLM sau sửa được thì bỏ cờ này.
    "hy_mt15_7b": {"vllm": "tencent/HY-MT1.5-7B", "hf": "tencent/HY-MT1.5-7B", "kieu": "hymt", "lap": 1.05, "eager": True},
    # Qwen3-30B-A3B-Instruct-2507 (Apache 2.0; MoE 30,5B tổng, ~3,3B hoạt động mỗi token; bản Instruct-2507 không có chế độ suy nghĩ).
    # Anh chọn 21/10/2026 làm ứng viên chính cho mô hình B thay HY-MT (giấy phép HY-MT hạn chế dùng đầu ra). T4 không chứa nổi bản đầy đủ
    # (~61 GB) và không chạy FP8 (cần GPU Ada trở lên), nên dùng bản nén AWQ 4-bit của cộng đồng: stelterlab (nhiều lượt tải nhất trên
    # Hugging Face ngày 21/10/2026, ~387 nghìn), CHƯA kiểm chất lượng. Thử lượng tử tự nhận trước, rồi moe_wna16 (Triton, cho GPU
    # không có Marlin như T4). Không có đường dự phòng transformers 4-bit: 30B không vừa một GPU T4 16 GB.
    "qwen3_30b_a3b": {"vllm": "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ", "hf": None, "kieu": "qwen", "lap": 1.0,
                      "quant_thu": [None, "moe_wna16"]},
    # Qwen3-14B (Apache 2.0, dày đặc 14,8B), ứng viên thứ hai cho B (anh đồng ý thử 21/10/2026, sau khi bản 30B-A3B nén cộng đồng bịa nghĩa
    # ở mẫu 200 đơn vị). Bản nén AWQ 4-bit CHÍNH THỨC của Qwen (~2,7 triệu lượt tải); bản đầy đủ ~29,6 GB không vừa hai T4 khi còn chừa
    # bộ nhớ đệm. Có chế độ suy nghĩ: tắt bằng enable_thinking=False (VLLM.sinh đã truyền). Lưu ý M0: bản AWQ của Qwen3-8B sót chữ Hán
    # 33/200 câu (có bước hỏi lại). Dự phòng transformers 4-bit vẫn được (14B nf4 ~9 GB vừa một T4), dùng bản đầy đủ.
    "qwen3_14b": {"vllm": "Qwen/Qwen3-14B-AWQ", "hf": "Qwen/Qwen3-14B", "kieu": "qwen", "lap": 1.0},
}
TOKEN_RA_API = 8192           # trần token ra khi gọi API (đã cộng dư cho trường hợp mô hình có phần suy luận)
# Thông báo lỗi mà cổng trung gian trả về trong NỘI DUNG câu trả lời (gặp ở Evolus lần thử 15/10/2026: "Your model 'deepseek-v4.1-flash'
# could not answer: The model endpoint is limiting the requests: retry..."). Bản dịch thật không chứa các cụm này.
LOI_TRONG_NOI_DUNG = re.compile(r"could not answer|is limiting the requests|model endpoint is|rate.?limit(ed|ing)? (exceeded|reached)|"
                                r"too many requests|quota exceeded|insufficient (credits|balance|quota)", re.I)
GIAY_CHO_API = 300            # chờ tối đa mỗi lần đọc từ API
KHOA_HONG = re.compile(r"api key not valid|API_KEY_INVALID|PERMISSION_DENIED|UNAUTHENTICATED|invalid api key|incorrect api key", re.I)
GIAY_NGHI_KHOA = 15.0         # nhiều khoá: khoá lỗi tạm nghỉ ngần này giây, gấp đôi mỗi lần lỗi liên tiếp
GIAY_NGHI_TOI_DA = 300.0
GIAY_KHOA_CHAM = 90          # nhiều khoá: một lần gọi lâu hơn mức này thì khoá đó nghỉ một phút để khoá khác gánh việc
PHIEN_BAN_CAU_LENH = "v2-2026-10-10"
DO_DAI_NGU_CANH = 12288       # token tối đa của một yêu cầu (câu lệnh + bản dịch)
TOKEN_RA_TOI_DA = 4096
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
DANH_MUC = {"bags": "túi xách", "mother_baby": "mẹ và bé", "home": "đồ gia dụng", "shoes": "giày dép", "food": "thực phẩm",
            "fashion": "thời trang", "beauty": "mỹ phẩm, làm đẹp", "electronics": "đồ điện tử", "auto": "phụ kiện ô tô"}
HE_THONG_QWEN = (
    "You translate Chinese product attributes from the e-commerce site 1688.com into Vietnamese.\n"
    "Rules:\n"
    "- Output only the Vietnamese translation: no explanation, no notes, no quotes, no label before it.\n"
    "- Translate EVERY Chinese word. The output must not contain any Chinese character.\n"
    "- Chinese place names use their Vietnamese names (浙江 -> Chiết Giang, 义乌 -> Nghĩa Ô). "
    "Never replace them with other places.\n"
    "- Keep numbers, units, sizes, model codes, Latin-letter brand names and bracket tokens such as [SĐT] or [URL] exactly as written.\n"
    "- If the text is a list separated by \", \", translate every item, keep the same number of items in the same order, "
    "and separate them with \", \".\n"
    "Examples:\n"
    "V字领 -> Cổ chữ V\n"
    "T恤衫 -> Áo phông\n"
    "藏青色, 酒红色 -> Xanh tím than, Đỏ rượu vang\n"
    "1000ml（含）以下 -> Từ 1000ml trở xuống")
LAI_QWEN = ("Your translation above still contains Chinese characters. Translate it again fully into Vietnamese, "
            "with no Chinese characters. Output only the translation.")
LAI_HYMT = "上面的译文里还有中文。请把全部内容翻译成越南语，不要保留任何中文，只输出译文："

# ---------------- hướng Việt -> Trung (bộ song ngữ 2, dữ liệu Tiki; anh ra lệnh 22/10/2026) ----------------
# --huong vi_zh đổi câu lệnh, ngữ cảnh và phép dò "còn sót ngôn ngữ nguồn" (chữ Việt có dấu thay cho chữ Hán). Mặc định zh_vi: không đổi gì.
HUONG = "zh_vi"
HE_THONG_VI_ZH = (
    "You translate Vietnamese product attributes from the e-commerce site Tiki.vn into Simplified Chinese, "
    "in the style of Chinese e-commerce listings (Taobao, 1688).\n"
    "Rules:\n"
    "- Output only the Chinese translation: no explanation, no notes, no quotes, no label before it, no pinyin.\n"
    "- The product category and the attribute name are context only. Translate only the given text. Never output the attribute name "
    "or a label such as \"颜色: \" before the translation (wrong: \"颜色: 红色\"; right: \"红色\").\n"
    "- Use Simplified Chinese characters only.\n"
    "- Translate EVERY Vietnamese word. The output must not contain any Vietnamese word or any Vietnamese letter with diacritics.\n"
    "- Vietnamese place names use their standard Chinese names (Hà Nội -> 河内, Việt Nam -> 越南, Nhật Bản -> 日本).\n"
    "- Keep numbers, units, sizes, model codes, Latin-letter brand names and bracket tokens exactly as written. "
    "Never translate or transliterate a brand name written in Latin letters.\n"
    "- If the text is a list separated by \", \", translate every item, keep the same number of items in the same order, "
    "and separate them with \", \".\n"
    "Examples:\n"
    "Màu đỏ -> 红色\n"
    "Đen, Trắng ngà -> 黑色, 米白色\n"
    "Hạn sử dụng 24 tháng kể từ ngày sản xuất -> 自生产之日起24个月")
LAI_VI_ZH = ("Your translation above still contains Vietnamese words. Translate it again fully into Simplified Chinese, "
             "with no Vietnamese words or Vietnamese letters (keep only numbers, units, model codes and Latin brand names). "
             "Output only the translation.")
HD_GOP_VI_ZH = ("You will receive several numbered items. Translate each item's text into Simplified Chinese following the rules above. "
                "Reply with exactly one line per item, in the same order, formatted as `[number] translation`, for example `[1] 红色`. "
                "Each reply line holds only the translation of that item's text: never repeat the attribute name or add a label before it. "
                "Never merge, skip, split or renumber items, never add explanations, and keep every translation on ONE line.")
DANH_MUC_ZH = {"food": "食品", "home": "家居用品", "beauty": "美妆护肤", "mother_baby": "母婴用品", "electronics": "数码电子",
               "auto": "汽车用品"}
# chữ cái Latin có dấu của tiếng Việt: bản dịch ra tiếng Trung mà còn những chữ này là còn sót tiếng Việt
VIET_DAU = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


def sot_ngon_ngu_nguon():
    """Biểu thức dò phần còn sót của ngôn ngữ nguồn trong bản dịch: chữ Hán (Trung -> Việt) hoặc chữ Việt có dấu (Việt -> Trung)."""
    return VIET_DAU if HUONG == "vi_zh" else CJK


def cau_lenh(kieu, loai, name_zh, cat, text):
    if kieu in ("qwen", "chat") and HUONG == "vi_zh":
        dm = DANH_MUC_ZH.get(cat, cat)
        if loai == "ten":
            u = f"Product category: {dm}.\nTranslate this attribute NAME into Simplified Chinese:\n{text}"
        else:
            u = (f"Product category: {dm}. Attribute name (Vietnamese): {name_zh}.\n"
                 f"Translate this attribute VALUE into Simplified Chinese:\n{text}")
        return [{"role": "system", "content": HE_THONG_VI_ZH}, {"role": "user", "content": u}]
    if kieu in ("qwen", "chat"):
        dm = DANH_MUC.get(cat, cat)
        if loai == "ten":
            u = f"Product category: {dm}.\nTranslate this attribute NAME into Vietnamese:\n{text}"
        else:
            u = (f"Product category: {dm}. Attribute name (Chinese): {name_zh}.\n"
                 f"Translate this attribute VALUE into Vietnamese:\n{text}")
        return [{"role": "system", "content": HE_THONG_QWEN}, {"role": "user", "content": u}]
    # HY-MT1.5: mẫu dịch Trung <-> ngôn ngữ khác ghi trên thẻ mô hình, KHÔNG kèm ngữ cảnh. Ở lần thử M0 mẫu "dịch có ngữ cảnh"
    # khiến mô hình dịch luôn phần ngữ cảnh ("Thể loại sản phẩm: ... Tên thuộc tính: ...") và bỏ mất phần cần dịch.
    return [{"role": "user", "content": f"将以下文本翻译为越南语，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"}]


def hoi_lai(kieu, msgs, ban_dau):
    lai = LAI_VI_ZH if HUONG == "vi_zh" else (LAI_HYMT if kieu == "hymt" else LAI_QWEN)
    return msgs + [{"role": "assistant", "content": ban_dau}, {"role": "user", "content": lai}]


def don_dep(s):
    # phần suy nghĩ lẫn trong nội dung: <think>...</think> (Qwen) và <thought>...</thought> (Gemma 4 trên Google AI Studio)
    s = re.sub(r"<(think|thought)>.*?</\1>", "", s or "", flags=re.S)
    if re.search(r"<(think|thought)>", s):
        return ""          # phần suy nghĩ mở mà chưa đóng: câu trả lời bị cắt trước khi ra bản dịch, không có bản dịch
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'“”":
        s = s[1:-1].strip()
    return s


def so_token_toi_da(text):
    return int(min(TOKEN_RA_TOI_DA, 3 * len(text) + 64))


# ---------------- gộp nhiều đơn vị mỗi lần gọi (giảm số yêu cầu, hợp với hạn mức ngày của bản miễn phí) ----------------
GOP_MAX_KY_TU_MOT_DON_VI = 600     # đơn vị dài hơn thì đi riêng
GOP_MAX_KY_TU_MOT_YEU_CAU = 2400   # tổng ký tự Trung tối đa trong một yêu cầu gộp
HD_GOP = ("You will receive several numbered items. Translate each item's text into Vietnamese following the rules above. "
          "Reply with exactly one line per item, in the same order, formatted as `[number] translation`, for example `[1] Màu đỏ`. "
          "Never merge, skip, split or renumber items, never add explanations, and keep every translation on ONE line.")
MARK_GOP = re.compile(r"^\s*\[(\d+)\]\s?(.*)$")


def cau_lenh_gop(rows):
    """rows: danh sách (loai, name_zh, cat, text) cùng loại. Trả về hội thoại gửi mô hình."""
    khoi = []
    vi_zh = HUONG == "vi_zh"
    for k, (loai, name_zh, cat, text) in enumerate(rows, 1):
        dm = (DANH_MUC_ZH if vi_zh else DANH_MUC).get(cat, cat)
        ctx = (f"product category = {dm}; this text is an attribute NAME" if loai == "ten"
               else f"product category = {dm}; attribute name ({'Vietnamese' if vi_zh else 'Chinese'}) = {name_zh}")
        khoi.append(f"Item {k}\ncontext: {ctx}\ntext: {text}")
    return [{"role": "system", "content": HE_THONG_VI_ZH if vi_zh else HE_THONG_QWEN},
            {"role": "user", "content": (HD_GOP_VI_ZH if vi_zh else HD_GOP) + "\n\n" + "\n\n".join(khoi)}]


def phan_tich_gop(text, n):
    """Bóc câu trả lời gộp thành {số thứ tự: bản dịch}. Chỉ giữ mục chắc chắn: số trong 1..n, không trùng, một dòng, không rỗng.
    Mục trùng, ngoài phạm vi, hoặc trải ra nhiều dòng bị bỏ (đơn vị đó sẽ được dịch riêng)."""
    muc, nghi, cur = {}, set(), None
    for dong in (text or "").splitlines():
        m = MARK_GOP.match(dong)
        if m:
            k = int(m.group(1))
            if not 1 <= k <= n or k in muc:
                nghi.add(k)
                cur = None
                continue
            muc[k] = m.group(2).strip()
            cur = k
        elif dong.strip() and cur is not None:
            nghi.add(cur)           # bản dịch tràn sang dòng sau: không tin
    return {k: v for k, v in muc.items() if k not in nghi and v}


def chia_gop(rows, n_gop, max_ky_tu=GOP_MAX_KY_TU_MOT_YEU_CAU):
    """Chia chỉ số 0..len(rows)-1 thành các nhóm gộp (cùng loại, tối đa n_gop đơn vị, tổng ký tự trong giới hạn) và danh sách đi riêng
    (đơn vị quá dài hoặc có xuống dòng)."""
    nhom, rieng, cur, cur_loai, tong = [], [], [], None, 0
    for i, (loai, _, _, t) in enumerate(rows):
        if len(t) > GOP_MAX_KY_TU_MOT_DON_VI or "\n" in t:
            rieng.append(i)
            continue
        if cur and (loai != cur_loai or len(cur) >= n_gop or tong + len(t) > max_ky_tu):
            nhom.append(cur)
            cur, tong = [], 0
        if not cur:
            cur_loai = loai
        cur.append(i)
        tong += len(t)
    if cur:
        nhom.append(cur)
    return nhom, rieng


def dich_gop(be, rows, msgs_don, toi_da_don, lap, n_gop):
    """Dịch một lô bằng cách gộp. Trả về (kết quả từng đơn vị [(bản dịch, lý do dừng)], thống kê).
    Đơn vị không bóc được từ câu trả lời gộp, hoặc đi riêng vì quá dài, được dịch riêng bằng câu lệnh từng đơn vị.
    Yêu cầu gộp bị lỗi API thì các đơn vị của nó trả ('', 'loi_api') (không ghi, lần sau dịch lại)."""
    nhom, rieng = chia_gop(rows, n_gop)
    kq = [None] * len(rows)
    tk = {"yeu_cau_gop": len(nhom), "don_vi_gop": sum(len(g) for g in nhom), "yeu_cau_gop_loi_api": 0, "don_vi_di_rieng": len(rieng),
          "don_vi_roi_ve_rieng": 0}
    if nhom:
        out = be.sinh([cau_lenh_gop([rows[i] for i in g]) for g in nhom], [sum(toi_da_don[i] for i in g) for g in nhom], lap)
        for g, (x, r) in zip(nhom, out):
            if r == "loi_api":
                for i in g:
                    kq[i] = ("", "loi_api")
                tk["yeu_cau_gop_loi_api"] += 1
                continue
            muc = phan_tich_gop(don_dep(x), len(g))
            if r == "length" and muc:
                muc.pop(max(muc), None)          # mục cuối có thể bị cắt dở
            for k, i in enumerate(g, 1):
                if k in muc:
                    kq[i] = (muc[k], "stop")
    ds_rieng = rieng + [i for g in nhom for i in g if kq[i] is None]
    tk["don_vi_roi_ve_rieng"] = len(ds_rieng) - len(rieng)
    if ds_rieng:
        for i, (x, r) in zip(ds_rieng, be.sinh([msgs_don[i] for i in ds_rieng], [toi_da_don[i] for i in ds_rieng], lap)):
            kq[i] = (don_dep(x), r)
    return kq, tk


def so_gpu_thay():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except ImportError:
        return 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------- backend ----------------
class VLLM:
    def __init__(self, repo, eager=False, quant=None):
        from vllm import LLM
        self.tp = so_gpu_thay()
        # T4 không có NVLink: tắt all-reduce tự viết của vLLM, dùng NCCL. quant: None để vLLM tự nhận kiểu lượng tử của bản nén,
        # hoặc ép một kiểu (vd "moe_wna16": kernel Triton cho MoE nén 4-bit, dùng được trên GPU không có Marlin như T4)
        them = {"quantization": quant} if quant else {}
        self.llm = LLM(model=repo, dtype="half", gpu_memory_utilization=0.90, max_model_len=DO_DAI_NGU_CANH,
                       tensor_parallel_size=self.tp, disable_custom_all_reduce=True, enforce_eager=bool(eager),
                       enable_prefix_caching=True, trust_remote_code=True, seed=0, **them)
        self.tok = self.llm.get_tokenizer()
        import vllm
        self.phien_ban = {"vllm": vllm.__version__, "tensor_parallel": self.tp, "enforce_eager": bool(eager),
                          "quantization": quant or "tu_nhan"}

    def sinh(self, msgs, max_new, lap):
        from vllm import SamplingParams
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        # câu lệnh quá dài thì bỏ qua (trả rỗng, ly_do_dung = qua_dai); còn lại cắt số token ra cho vừa độ dài ngữ cảnh
        try:
            dai = [len(self.tok.encode(p, add_special_tokens=False)) for p in prompts]
        except Exception:   # không đếm được thì ước rộng tay: mỗi ký tự một token
            dai = [len(p) for p in prompts]
        chay = [i for i, d in enumerate(dai) if d + 64 <= DO_DAI_NGU_CANH]
        ps = [SamplingParams(temperature=0.0, max_tokens=min(max_new[i], DO_DAI_NGU_CANH - dai[i] - 8), repetition_penalty=lap, seed=0)
              for i in chay]
        outs = []
        if chay:
            try:   # llm.chat tự áp mẫu hội thoại của mô hình (không thêm BOS hai lần)
                outs = self.llm.chat([msgs[i] for i in chay], ps, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
            except TypeError:   # bản vLLM cũ chưa nhận chat_template_kwargs hoặc danh sách hội thoại
                outs = self.llm.generate([prompts[i] for i in chay], ps, use_tqdm=False)
        kq = [("", "qua_dai")] * len(msgs)
        for i, o in zip(chay, outs):
            kq[i] = (o.outputs[0].text, o.outputs[0].finish_reason or "")
        return kq


class HF:
    def __init__(self, repo):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=q, device_map={"": 0},
                                                          torch_dtype=torch.float16, trust_remote_code=True).eval()
        self.phien_ban = {"transformers": transformers.__version__, "torch": torch.__version__}

    def sinh(self, msgs, max_new, lap, bs=24):
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max(max_new[i:i + bs]), do_sample=False,
                                        repetition_penalty=lap, pad_token_id=self.tok.pad_token_id)
            for row in g[:, enc["input_ids"].shape[1]:]:
                out.append((self.tok.decode(row, skip_special_tokens=True), ""))
        return out


class API:
    """Mô hình gọi qua API kiểu OpenAI (chat/completions), nhiều yêu cầu cùng lúc. Khoá lấy từ biến môi trường, không in, không ghi.
    Gặp 429 hoặc lỗi máy chủ thì chờ rồi thử lại (tối đa 8 lần); lỗi khác (khoá sai, yêu cầu sai) thì bỏ ngay.
    Đơn vị lỗi trả ('', 'loi_api'): KHÔNG ghi vào lô, để lần chạy sau dịch lại."""

    def __init__(self, cfg, song_song=8, extra=None, giay_cho=GIAY_CHO_API):
        # key_env có thể là NHIỀU tên biến cách nhau dấu phẩy (mỗi biến một khoá, nên mỗi khoá một dự án): mỗi yêu cầu chọn khoá nhanh nhất còn khoẻ,
        # khoá lỗi hoặc hết hạn mức thì nghỉ và yêu cầu chuyển sang khoá khác
        ten_env = [x.strip() for x in str(cfg["key_env"]).split(",") if x.strip()] or [cfg["key_env"]]
        self.khoa, thieu_env = [], []
        for ten in ten_env:
            v = os.environ.get(ten, "").strip()
            if not v:
                thieu_env.append(ten)
                continue
            if ten == "DASHSCOPE_API_KEY" and v.startswith("sk-sp-"):
                # khoá Token Plan của Alibaba Cloud: điều khoản cấm chạy tự động, gọi hàng loạt, tác vụ định kỳ (có thể bị khoá khoá và gói)
                raise SystemExit("Khoá bắt đầu bằng sk-sp- là khoá Token Plan của Alibaba Cloud: điều khoản cấm script tự động và gọi hàng loạt, "
                                 "có thể bị khoá khoá. Dùng khoá API trả theo lượng dùng (pay-as-you-go) của Model Studio.")
            if any(k["key"] == v for k in self.khoa):
                continue          # cùng một khoá đặt ở hai biến: chỉ giữ một
            self.khoa.append({"ten": ten, "key": v, "ewma": None, "dang_chay": 0, "nghi_den": 0.0, "loi_lien": 0, "chet": "",
                              "moc": 0.0, "yeu_cau": 0, "loi": 0, "giay": 0.0, "loi_cuoi": ""})
        if not self.khoa:
            raise SystemExit(f"Thiếu biến môi trường {ten_env[0]}: chương trình đọc khoá API từ biến này (tên đổi được bằng --api_key_env). "
                             f"Trên Kaggle: đặt bằng Kaggle Secrets. Trên laptop: trong CHÍNH cửa sổ terminal đang chạy lệnh, gõ "
                             f"$k = Read-Host 'Khoa' -AsSecureString; $env:{ten_env[0]} = [System.Net.NetworkCredential]::new('', $k).Password "
                             f"rồi dán khoá. Không ghi khoá vào file.")
        if thieu_env:
            log(f"Không thấy biến môi trường {', '.join(thieu_env)}: bỏ qua, chạy với {len(self.khoa)} khoá.")
        self.key = self.khoa[0]["key"]
        self.tl = threading.local()       # khoá đang dùng của luồng này (để _post lấy đúng khoá mà không đổi chữ ký)
        if not cfg.get("base_url") or not cfg.get("api"):
            raise SystemExit("Thiếu địa chỉ gọi API (--api_base_url) hoặc tên mô hình (--api_model).")
        if cfg["base_url"].startswith("http://") and not re.match(r"http://(127\.0\.0\.1|localhost)([:/]|$)", cfg["base_url"]):
            raise SystemExit("Địa chỉ gọi dùng http (không mã hoá): không gửi khoá qua đường này. Dùng địa chỉ https.")
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.model = cfg["api"]
        self.n = max(1, int(song_song))
        ce = cfg.get("extra", {})      # dict, hoặc danh sách các cách để gọi thử (thu_ket_noi chọn cách không còn suy luận)
        self.extra = dict(extra) if extra is not None else (dict(ce) if isinstance(ce, dict) else {})
        self.tra_phi = bool(cfg.get("gia_usd_1m"))
        self.them_ra = int(cfg.get("them_token_ra", 2048))   # token ra cho phép thêm ngoài độ dài ước (mô hình có suy luận cần nhiều)
        self.so_lan = int(cfg.get("so_lan_thu", 8))
        self.gia = cfg.get("gia_usd_1m")                     # có giá thì tính chi phí và theo ngân sách
        self.ngan_sach = None                                # USD; chạm thì không gửi yêu cầu mới
        self.chi_phi = 0.0
        self.toi_da_goi = 0                                  # số lần gửi yêu cầu tối đa (kể cả thử lại); 0 là không giới hạn
        self.so_goi = 0
        self.rpm = 0                                         # số yêu cầu tối đa mỗi phút (giữ dưới hạn mức bản miễn phí); 0 là không giới hạn
        self.lock_nhip = threading.Lock()
        self.moc_gui_tiep = 0.0
        self.cho_suy_luan = bool(cfg.get("cho_suy_luan"))    # True: không dừng khi không tắt được suy luận (bản miễn phí, chậm nhưng không tốn tiền)
        self.dung_quota = False                              # hết hạn mức NGÀY: không gửi nữa, chờ sang ngày
        self.gop_he_thong = False                            # máy chủ không nhận vai trò system: gộp lời dặn vào tin nhắn người dùng
        self.giay_cho = giay_cho
        self.dem_usage = True     # xin thống kê token kèm luồng; máy chủ không nhận thì tự bỏ
        self.han_chot = None      # mốc giờ (epoch): quá mốc thì không gửi yêu cầu mới, kể cả đang giữa lô
        self.lock = threading.Lock()
        self.tk = {"yeu_cau": 0, "thu_lai": 0, "loi_api": 0, "loi_trong_noi_dung": 0, "het_gio": 0, "het_ngan_sach": 0, "het_toi_da_goi": 0,
                   "het_quota_ngay": 0, "token_vao": 0, "token_ra": 0,
                   "token_suy_luan": 0, "giay_cac_lan_goi": 0.0, "chi_phi_uoc_usd": 0.0, "model_tra_ve": [], "theo_khoa": {}}
        self.phien_ban = {"api": cfg["api"], "base_url": cfg["base_url"], "song_song": self.n, "nhan_theo_luong": True,
                          "giay_cho": giay_cho, "extra_body": self.extra}

    def _tinh_phi(self, msgs, u, noi_dung):
        """Chi phí ước (USD) của một lần gọi theo giá giờ cao điểm. Thiếu thống kê token thì ước dư từ số ký tự."""
        g = self.gia
        vao, ra = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        if not vao and not ra:
            vao = sum(len(m.get("content", "")) for m in msgs) // 2 + 20
            ra = len(noi_dung) // 1 + 5
        hit = int(u.get("prompt_cache_hit_tokens") or 0)
        miss = int(u.get("prompt_cache_miss_tokens") or max(vao - hit, 0))
        return (miss * g["vao_miss"] + hit * g["vao_hit"] + ra * g["ra"]) / 1e6

    def _doc_luong(self, r):
        """Gộp các mẩu SSE ('data: {...}') thành một câu trả lời như khi không dùng luồng.
        Nhận theo luồng để kết nối luôn có dữ liệu chạy về, không bị quá giờ chờ như lần M0 thứ hai."""
        noi, suy_luan, ly_do, usage, model = [], 0, "", {}, ""
        for dong in r:
            dong = dong.decode("utf-8", "replace").strip()
            if not dong.startswith("data:"):
                continue
            d = dong[5:].strip()
            if d == "[DONE]":
                break
            try:
                g = json.loads(d)
            except json.JSONDecodeError:
                continue
            model = g.get("model") or model
            usage = g.get("usage") or usage
            for c in g.get("choices") or []:
                delta = c.get("delta") or {}
                noi.append(delta.get("content") or "")
                suy_luan += len(delta.get("reasoning_content") or "")     # phần suy luận: chỉ đếm, không lấy vào bản dịch
                ly_do = c.get("finish_reason") or ly_do
        day_du = "".join(noi)
        # suy nghĩ nằm lẫn trong nội dung (Gemma 4: <thought>...</thought>): đếm để biết suy luận có tắt được không
        suy_luan += sum(len(m) for m in re.findall(r"<(?:think|thought)>(.*?)</(?:think|thought)>", day_du, flags=re.S))
        if re.search(r"<(?:think|thought)>(?!.*</(?:think|thought)>)", day_du, flags=re.S):
            suy_luan += len(day_du)
        return {"choices": [{"message": {"content": day_du}, "finish_reason": ly_do}], "usage": usage,
                "model": model, "ky_tu_suy_luan": suy_luan}

    def _post(self, payload):
        kd = getattr(self.tl, "k", None)
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Authorization": "Bearer " + (kd["key"] if kd else self.key), "Content-Type": "application/json",
                                              "Accept": "text/event-stream" if payload.get("stream") else "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.giay_cho) as r:
                return r.status, (self._doc_luong(r) if payload.get("stream") else json.loads(r.read().decode("utf-8"))), None
        except urllib.error.HTTPError as e:
            try:
                nd = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                nd = ""
            return e.code, nd, e.headers.get("Retry-After")
        except Exception as e:   # mạng đứt, quá giờ
            return 0, f"{type(e).__name__}: {str(e)[:200]}", None

    @staticmethod
    def _gop_system(msgs):
        """Gộp tin nhắn system vào tin nhắn người dùng đầu tiên (cho mô hình không nhận vai trò system, ví dụ Gemma trên Gemini API)."""
        he = "\n\n".join(m["content"] for m in msgs if m.get("role") == "system")
        con = [dict(m) for m in msgs if m.get("role") != "system"]
        if he and con:
            con[0]["content"] = he + "\n\n" + con[0]["content"]
        return con

    def _cho_nhip(self, k=None):
        """Giãn đều các lần gửi để không vượt --rpm (mỗi yêu cầu cách nhau ít nhất 60/rpm giây). Nhiều khoá: nhịp tính RIÊNG từng khoá
        (hạn mức tính theo dự án của khoá)."""
        if not self.rpm:
            return
        with self.lock_nhip:
            t = time.time()
            moc = max(t, k["moc"] if k else self.moc_gui_tiep)
            if k:
                k["moc"] = moc + 60.0 / self.rpm
            else:
                self.moc_gui_tiep = moc + 60.0 / self.rpm
        if moc - t > 0:
            time.sleep(moc - t)

    def _chon_khoa(self, ep=None):
        """Chọn khoá cho một lần gửi: bỏ khoá đã hết hạn mức ngày hoặc đang nghỉ sau lỗi; trong số còn lại lấy khoá có
        (số yêu cầu đang chạy + 1) x thời gian trả lời trung bình nhỏ nhất (khoá nhanh và đang rảnh được ưu tiên). Khoá chưa đo coi như nhanh
        bằng khoá nhanh nhất để được thử; khoá chậm được 'quên dần' để thỉnh thoảng thử lại. Mọi khoá còn lại đều đang nghỉ thì chờ khoá
        dậy sớm nhất. Trả None khi mọi khoá đều đã hết hạn mức ngày hoặc bị loại."""
        if ep is not None:
            with self.lock:
                ep["dang_chay"] += 1
            return ep
        while True:
            with self.lock:
                con = [k for k in self.khoa if not k["chet"]]
                if not con:
                    return None
                t = time.time()
                san = [k for k in con if k["nghi_den"] <= t]
                if san:
                    ew = [k["ewma"] for k in san if k["ewma"] is not None]
                    tham = min(ew) if ew else 1.0
                    k = min(san, key=lambda x: (x["dang_chay"] + 1) * (x["ewma"] if x["ewma"] is not None else tham))
                    for o in san:
                        if o is not k and o["ewma"] is not None and o["ewma"] > tham:
                            o["ewma"] = max(tham, o["ewma"] * 0.98)
                    k["dang_chay"] += 1
                    return k
                cho = min(k["nghi_den"] for k in con) - t
            time.sleep(min(90.0, max(cho, 0.5)))

    def _danh_dau(self, k, dt, loai, cho=0.0, ly_do=""):
        """Ghi kết quả một lần gửi của khoá k. loai: 'ok' (có thể kèm cảnh báo chậm), 'loi' (lỗi tạm: nghỉ rồi thử lại),
        'nghi' (nghỉ đúng cho giây), 'chet' (hết hạn mức ngày hoặc khoá bị loại: không dùng nữa trong lượt chạy này).
        Nghỉ chỉ áp dụng khi có từ hai khoá trở lên (một khoá đã có chờ và thử lại riêng)."""
        nhieu = len(self.khoa) > 1
        with self.lock:
            k["dang_chay"] = max(0, k["dang_chay"] - 1)
            if loai == "ok":
                k["yeu_cau"] += 1
                k["giay"] += dt
                k["loi_lien"] = 0
                k["ewma"] = dt if k["ewma"] is None else 0.7 * k["ewma"] + 0.3 * dt
                if nhieu and dt > GIAY_KHOA_CHAM:
                    k["nghi_den"] = time.time() + 60.0         # trả lời được nhưng quá chậm: cho nghỉ một phút để khoá khác gánh
            elif loai == "chet":
                k["loi"] += 1
                k["chet"] = ly_do or "bị loại"
            elif loai == "bo":
                pass          # lần gửi bị bỏ vì lý do không thuộc về khoá (yêu cầu sai, đổi cách gọi): không phạt khoá
            else:
                k["loi"] += 1
                k["loi_lien"] += 1
                if nhieu:
                    k["nghi_den"] = time.time() + (cho if loai == "nghi" else min(GIAY_NGHI_TOI_DA, GIAY_NGHI_KHOA * 2 ** (k["loi_lien"] - 1)))
            self.tk["theo_khoa"][k["ten"]] = {"yeu_cau_thanh_cong": k["yeu_cau"], "loi": k["loi"],
                                              "giay_trung_binh": round(k["giay"] / k["yeu_cau"], 1) if k["yeu_cau"] else None,
                                              "trang_thai": k["chet"] or "con_dung_duoc"}

    def _mot(self, msgs, max_new, so_lan=None, khoa_ep=None):
        so_lan = so_lan or self.so_lan
        payload = {"model": self.model, "messages": self._gop_system(msgs) if self.gop_he_thong else msgs, "temperature": 0.0,
                   "stream": True, "max_tokens": int(min(TOKEN_RA_API, max_new + self.them_ra))}
        if self.dem_usage:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra)
        loi = ""
        nk = max(1, len(self.khoa))
        # nhiều khoá: số lần thử nhân với số khoá (mỗi vòng đi hết các khoá mới tính là một lần thử), vì lỗi 500 của nhà cung cấp
        # thường nhanh (dưới 1 giây) và trải trên mọi khoá: nếu chuyển khoá liên tục không chờ thì hết lượt thử trong vài giây
        tong_lan = so_lan * nk if (nk > 1 and khoa_ep is None) else so_lan
        for lan in range(tong_lan):
            if self.dung_quota:      # đã hết hạn mức ngày: không gửi nữa
                with self.lock:
                    self.tk["het_quota_ngay"] += 1
                return "", "loi_api"
            if self.han_chot and time.time() > self.han_chot:
                with self.lock:
                    self.tk["het_gio"] += 1
                return "", "loi_api"
            if self.ngan_sach is not None and self.chi_phi >= self.ngan_sach:      # chạm ngân sách: không tốn thêm đồng nào
                with self.lock:
                    self.tk["het_ngan_sach"] += 1
                return "", "loi_api"
            with self.lock:       # trần số lần gửi (để giữ tín dụng khi nhà cung cấp tính theo tín dụng, không theo token)
                if self.toi_da_goi and self.so_goi >= self.toi_da_goi:
                    self.tk["het_toi_da_goi"] += 1
                    return "", "loi_api"
                self.so_goi += 1
            if khoa_ep is not None and khoa_ep["chet"]:      # gọi thử đúng một khoá mà khoá đó vừa bị loại: không gọi lại
                break
            k = self._chon_khoa(khoa_ep)
            if k is None:            # mọi khoá đều đã hết hạn mức ngày hoặc bị loại
                self.dung_quota = True
                with self.lock:
                    self.tk["het_quota_ngay"] += 1
                return "", "loi_api"
            self.tl.k = k
            try:
                self._cho_nhip(k)
                t = time.time()
                ma, nd, cho = self._post(payload)
            finally:
                self.tl.k = None
            dt = time.time() - t
            if ma == 200 and isinstance(nd, dict) and nd.get("choices"):
                c = nd["choices"][0]
                u = nd.get("usage") or {}
                noi_dung = (c.get("message") or {}).get("content") or ""
                if payload.get("stream") and not c.get("finish_reason"):
                    # luồng kết thúc mà không có finish_reason: kết nối bị cắt giữa chừng (gặp ở Google AI Studio bản miễn phí:
                    # 'G-JM291màu đen, G-' bị lưu như bản dịch). Không nhận, thử lại
                    with self.lock:
                        self.tk["luong_bi_cat"] = self.tk.get("luong_bi_cat", 0) + 1
                    ma, nd, cho = 0, "luồng kết thúc không có finish_reason (bị cắt giữa chừng)", None
                elif LOI_TRONG_NOI_DUNG.search(noi_dung[:300]):
                    # cổng trung gian (Evolus) trả LỖI của nhà cung cấp phía sau như thể là câu trả lời (HTTP 200, finish_reason stop):
                    # tuyệt đối không ghi làm bản dịch; coi như bị giới hạn tốc độ, chờ rồi thử lại
                    with self.lock:
                        self.tk["loi_trong_noi_dung"] += 1
                    ma, nd, cho = 429, noi_dung[:200], None
                else:
                    with self.lock:
                        self.tk["giay_cac_lan_goi"] += time.time() - t
                        self.tk["yeu_cau"] += 1
                        self.tk["token_vao"] += int(u.get("prompt_tokens") or 0)
                        self.tk["token_ra"] += int(u.get("completion_tokens") or 0)
                        self.tk["token_suy_luan"] += int(nd.get("ky_tu_suy_luan") or 0)
                        if self.gia:
                            self.chi_phi += self._tinh_phi(msgs, u, noi_dung)
                            self.tk["chi_phi_uoc_usd"] = round(self.chi_phi, 5)
                        if nd.get("model") and nd["model"] not in self.tk["model_tra_ve"]:
                            self.tk["model_tra_ve"].append(nd["model"])
                    self._danh_dau(k, dt, "ok")
                    return noi_dung, c.get("finish_reason") or ""
            loi = f"HTTP {ma}: {str(nd)[:200]}"
            k["loi_cuoi"] = loi
            nhieu = len(self.khoa) > 1
            if ma == 400 and "stream_options" in str(nd) and "stream_options" in payload:
                self._danh_dau(k, dt, "bo")
                payload.pop("stream_options")      # máy chủ không nhận: bỏ phần xin thống kê token rồi gọi lại
                self.dem_usage = False
                continue
            if ma == 400 and not self.gop_he_thong and re.search(
                    r"developer instruction|system instruction|system role|role.{0,20}system|system message|unsupported.{0,30}role",
                    str(nd), re.I) and any(m.get("role") == "system" for m in msgs):
                self._danh_dau(k, dt, "bo")
                self.gop_he_thong = True       # máy chủ không nhận vai trò system: gộp lời dặn vào tin nhắn người dùng rồi gọi lại
                payload["messages"] = self._gop_system(msgs)
                continue
            if ma == 429 and re.search(r"per ?day|daily|requests per day", str(nd), re.I):
                # hết hạn mức NGÀY (ví dụ Google AI Studio bản miễn phí): khoá này không dùng nữa; còn khoá khác thì sang khoá khác,
                # hết mọi khoá thì dừng hẳn (thử lại ngay cũng vô ích)
                self._danh_dau(k, dt, "chet", ly_do="hết hạn mức ngày")
                if all(o["chet"] for o in self.khoa):
                    self.dung_quota = True
                    with self.lock:
                        self.tk["het_quota_ngay"] += 1
                        self.tk["loi_cuoi"] = loi
                    return "", "loi_api"
                continue
            if nhieu and (ma == 401 or KHOA_HONG.search(str(nd))):
                self._danh_dau(k, dt, "chet", ly_do=f"khoá bị từ chối ({loi[:60]})")      # khoá sai hoặc đã thu hồi: loại, đổi khoá
                if all(o["chet"] for o in self.khoa):
                    break
                continue
            if not (ma in (0, 408, 409, 429) or 500 <= ma < 600):
                if nhieu and ma == 403:
                    self._danh_dau(k, dt, "nghi", cho=600.0)      # có thể khoá bị chặn tạm: nghỉ 10 phút, đổi khoá
                    continue
                self._danh_dau(k, dt, "bo")
                break
            try:
                cho = float(cho) if cho else 0.0
            except ValueError:
                cho = 0.0
            self._danh_dau(k, dt, "nghi" if cho > 0 else "loi", cho=min(cho, 300.0))
            # chỉ bỏ qua việc chờ khi có khoá khác đang HOÀN TOÀN khoẻ (chưa lỗi liên tiếp, không nghỉ); lỗi lan trên mọi khoá thì phải lùi dần
            sang_khoa_khac = any(not o["chet"] and o is not k and o["nghi_den"] <= time.time() and o["loi_lien"] == 0 for o in self.khoa)
            cho = min(90.0, max(cho, 2.0 ** (lan // nk))) + random.random()
            if self.han_chot and time.time() + cho > self.han_chot:
                break
            with self.lock:
                self.tk["thu_lai"] += 1
            if not sang_khoa_khac:       # còn khoá khác đang khoẻ thì thử ngay bằng khoá đó, không chờ
                time.sleep(cho)
        with self.lock:
            self.tk["loi_api"] += 1
            self.tk["loi_cuoi"] = loi
        return "", "loi_api"

    def thu_ket_noi(self, ung_vien=None):
        """Gọi thử một câu với từng cách thêm vào yêu cầu (ung_vien, ví dụ các cách tắt suy luận), chọn cách trả lời nhanh nhất.
        Lần M0 thứ ba: để mặc định, một câu "你好" mất khoảng 4 đến 5 phút."""
        cau_thu = [{"role": "user", "content": "Dịch sang tiếng Việt: 你好"}]
        kq = []
        for kk in self.khoa:        # thử lần lượt từng khoá cho tới khi một khoá gọi được (một khoá: như trước)
            kq = []
            for ex in (ung_vien if ung_vien else [self.extra]):
                self.extra = ex
                sl0, t = self.tk["token_suy_luan"], time.time()
                x, r = self._mot(cau_thu, 64, so_lan=2, khoa_ep=kk if len(self.khoa) > 1 else None)
                d = {"extra": ex, "giay": round(time.time() - t, 1), "tra_loi": x[:40], "ky_tu_suy_luan": self.tk["token_suy_luan"] - sl0}
                if r == "loi_api":
                    d["loi"] = self.tk.get("loi_cuoi", "")
                log("Gọi thử:", json.dumps(d, ensure_ascii=False))
                kq.append(d)
                if "loi" not in d and d["tra_loi"].strip() and d["giay"] < 30 and d["ky_tu_suy_luan"] == 0:
                    break      # đã đủ nhanh và không còn suy luận, khỏi thử các cách sau (cách mặc định có thể mất vài phút)
            if any("loi" not in d and d["tra_loi"].strip() for d in kq):
                break
            if len(self.khoa) > 1:
                self._loai_neu_khoa_hong(kk, kq[-1].get("loi", ""))
        self.tk["goi_thu"] = kq
        dat = [d for d in kq if "loi" not in d and d["tra_loi"].strip()]
        if not dat:
            raise SystemExit(f"Không gọi được API ({kq[-1].get('loi')}). Kiểm khoá (Kaggle Secrets), Internet, "
                             f"địa chỉ gọi (--api_base_url), tên mô hình và hạn mức của tài khoản.")
        if self.tra_phi and not self.cho_suy_luan and all(d["ky_tu_suy_luan"] > 0 for d in dat):
            # mô hình trả phí mà suy luận không tắt được: mỗi câu tốn thêm token suy luận (tính như token ra) và chậm hơn nhiều
            raise SystemExit("Không tắt được chế độ suy luận (cả các cách thử đều còn chữ suy luận). Dừng để khỏi tốn tiền; "
                             "xem 'goi_thu' ở log, thử --api_extra khác.")
        dat = [d for d in dat if d["ky_tu_suy_luan"] == 0] or dat
        tot = min(dat, key=lambda d: d["giay"])
        self.extra = tot["extra"]
        self.phien_ban["extra_body"] = self.extra
        if len(self.khoa) > 1:
            # thử nốt các khoá còn lại cùng lúc bằng cách đã chọn: khoá sai bị loại ngay, khỏi làm hỏng từng yêu cầu thật
            con_lai = [k for k in self.khoa if k is not kk and not k["chet"]]      # kk: khoá đã gọi thử được ở vòng trên

            def thu(k):
                t = time.time()
                x, r = self._mot(cau_thu, 64, so_lan=2, khoa_ep=k)
                return k, x, r, round(time.time() - t, 1)
            with concurrent.futures.ThreadPoolExecutor(max(1, len(con_lai))) as ex:
                for k, x, r, g in ex.map(thu, con_lai):
                    log(f"Gọi thử khoá {k['ten']}: {g} giây | " + ("lỗi " + k["loi_cuoi"][:120] if r == "loi_api" else f"'{x[:30]}'"))
                    if r == "loi_api":
                        self._loai_neu_khoa_hong(k, k["loi_cuoi"])
            for k in self.khoa:        # lần gọi thử đầu thường chậm bất thường (khởi động nguội): không dùng làm số đo tốc độ
                k.update(ewma=None, nghi_den=0.0, loi_lien=0, yeu_cau=0, giay=0.0, loi=0)
            self.tk["theo_khoa"] = {}
            log(f"Khoá dùng được: {sum(1 for k in self.khoa if not k['chet'])}/{len(self.khoa)}"
                + "".join(f" | {k['ten']} bị loại: {k['chet']}" for k in self.khoa if k["chet"]))
            if all(k["chet"] for k in self.khoa):
                raise SystemExit("Mọi khoá đều bị loại khi gọi thử (xem log). Kiểm khoá, hạn mức và địa chỉ gọi.")
        return f"chọn {json.dumps(self.extra, ensure_ascii=False)}: '{tot['tra_loi']}' sau {tot['giay']} giây"

    def _loai_neu_khoa_hong(self, k, loi):
        """Gọi thử khoá k thất bại: lỗi kiểu khoá sai hoặc yêu cầu bị từ chối (HTTP 400, 401, 403) thì loại khoá; lỗi tạm (500, quá giờ, 429) thì giữ."""
        if re.search(r"HTTP (400|401|403)\b", loi or "") or KHOA_HONG.search(loi or ""):
            k["chet"] = f"gọi thử bị từ chối ({(loi or '')[:70]})"

    def sinh(self, msgs, max_new, lap):
        with concurrent.futures.ThreadPoolExecutor(self.n) as ex:
            return list(ex.map(self._mot, msgs, max_new))


class Gia:
    """Backend giả, CHỈ để thử trên máy không có GPU: trả lại chính câu gửi đi, kiểm việc ghi lô và chạy tiếp."""
    phien_ban = {"gia": "1"}

    def sinh(self, msgs, max_new, lap):
        return [("[gia] " + m[-1]["content"].splitlines()[-1], "stop") for m in msgs]


def tai_backend(cfg, backend, song_song=8, extra=None):
    loi = None
    if backend == "gia":
        return Gia(), "gia", "gia", None
    if "api" in cfg:
        # extra là một dict (dùng luôn) hoặc danh sách dict (gọi thử từng cái, chọn cái nhanh nhất và không còn suy luận)
        if extra is None and isinstance(cfg.get("extra"), list):
            extra = cfg["extra"]
        be = API(cfg, song_song, None if isinstance(extra, list) else extra)
        host = re.sub(r"^https?://([^/]+).*$", r"\1", cfg["base_url"])
        log(f"Thử kết nối API: mô hình {cfg['api']} | khoá chỉ gửi tới {host} | giá ước {cfg.get('gia_usd_1m')}")
        log("Kết quả gọi thử:", be.thu_ket_noi(extra if isinstance(extra, list) else None))
        return be, "api", cfg["api"], None
    if backend in ("auto", "vllm"):
        cach = [(q, e) for q in cfg.get("quant_thu", [None]) for e in ([True] if cfg.get("eager") else [False, True])]
        cac_loi = []
        for k, (quant, eager) in enumerate(cach):
            try:
                log("Nạp vLLM:", cfg["vllm"], f"| lượng tử {quant}" if quant else "",
                    "| chạy thẳng không chụp CUDA graph" if eager else "")
                return VLLM(cfg["vllm"], eager, quant), "vllm", cfg["vllm"], loi
            except Exception as e:  # T4 có thể không được vLLM bản mới hỗ trợ: ghi lỗi rồi thử cách khác
                loi = f"{f'{quant}: ' if quant else ''}{'eager: ' if eager else ''}{type(e).__name__}: {str(e)[:400]}"
                cac_loi.append(loi)
                log("vLLM không chạy được:", loi)
                if backend == "vllm" and k == len(cach) - 1:
                    raise
        if not cfg.get("hf"):
            # mô hình quá lớn cho đường dự phòng transformers 4-bit trên một GPU T4: dừng hẳn, ghi đủ các lỗi để xem
            raise SystemExit(f"Không nạp được {cfg['vllm']} bằng vLLM và mô hình này không có đường dự phòng transformers. "
                             f"Các lỗi: {' || '.join(cac_loi)}")
    log("Nạp transformers + bitsandbytes 4-bit:", cfg["hf"])
    return HF(cfg["hf"]), "hf", cfg["hf"], loi


def sha_mo_hinh(repo):
    try:
        from huggingface_hub import model_info
        return model_info(repo).sha
    except Exception as e:
        return f"khong_lay_duoc: {type(e).__name__}"


# ---------------- chính ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vao", required=True)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--mo_hinh", required=True, choices=list(MO_HINH))
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf", "gia"])
    ap.add_argument("--song_song", type=int, default=None, help="số yêu cầu API cùng lúc (chỉ cho mô hình gọi API; mặc định theo cấu hình mô hình, 8 nếu không có)")
    ap.add_argument("--api_base_url", default="", help="địa chỉ gọi API (ghi đè base_url của mô hình), ví dụ địa chỉ trong bảng điều khiển Alibaba Cloud")
    ap.add_argument("--cho_phep_suy_luan", action="store_true",
                    help="vẫn chạy khi không tắt được chế độ suy luận (phần suy nghĩ được cắt khỏi bản dịch). Chỉ nên dùng với API miễn phí: "
                         "chậm hơn nhiều và tốn token")
    ap.add_argument("--them_token_ra", type=int, default=-1,
                    help="token ra cho phép thêm ngoài độ dài ước (ghi đè cấu hình mô hình; mô hình còn suy nghĩ cần khoảng 3000)")
    ap.add_argument("--gop", type=int, default=0,
                    help="gộp tối đa N đơn vị cùng loại vào MỘT yêu cầu (chỉ cho mô hình kiểu chat; 0 hoặc 1 là không gộp). "
                         "Đơn vị bóc không được hoặc quá dài sẽ được dịch riêng")
    ap.add_argument("--rpm", type=int, default=0,
                    help="số yêu cầu API tối đa mỗi phút (giữ dưới hạn mức, ví dụ bản miễn phí của Google AI Studio); 0 là không giới hạn")
    ap.add_argument("--toi_da_yeu_cau", type=int, default=0,
                    help="số lần gửi yêu cầu API tối đa trong lần chạy này (kể cả thử lại, kể cả gọi thử đầu); 0 là không giới hạn. "
                         "Dùng khi nhà cung cấp tính theo tín dụng chứ không theo token")
    ap.add_argument("--api_model", default="", help="tên mô hình gửi cho API (ghi đè), bắt buộc với --mo_hinh api_tuy_chon")
    ap.add_argument("--api_key_env", default="", help="tên biến môi trường chứa khoá (ghi đè)")
    ap.add_argument("--gia_vao_usd_1m", type=float, default=0.0, help="giá vào mỗi triệu token (USD) để ước chi phí")
    ap.add_argument("--gia_ra_usd_1m", type=float, default=0.0, help="giá ra mỗi triệu token (USD) để ước chi phí")
    ap.add_argument("--ngan_sach_usd", type=float, default=0.0,
                    help="ngân sách tối đa (USD) cho mô hình gọi API trả phí: chạm thì dừng. BẮT BUỘC với mô hình trả phí")
    ap.add_argument("--api_extra", default="", help='JSON thêm vào yêu cầu API: một dict, hoặc danh sách dict để gọi thử '
                                                    'từng cái rồi chọn cái nhanh nhất, ví dụ \'[{"thinking":{"type":"disabled"}}, {}]\'')
    ap.add_argument("--gio_toi_da", type=float, default=8.0)
    ap.add_argument("--thu", type=int, default=0, help="chỉ dịch N đơn vị (thử khói); 0 là tất cả")
    ap.add_argument("--lo", type=int, default=2000)
    ap.add_argument("--cu", action="append", default=[], help="thư mục đầu ra của phiên bản trước")
    ap.add_argument("--thu_ket_noi", action="store_true", help="chỉ gọi thử API một câu rồi thoát (0 = được, 1 = không)")
    ap.add_argument("--huong", default="zh_vi", choices=["zh_vi", "vi_zh"],
                    help="zh_vi: dịch Trung sang Việt (bộ song ngữ 1, mặc định); vi_zh: dịch Việt sang Trung (bộ song ngữ 2, dữ liệu Tiki). "
                         "Hướng vi_zh chỉ cho mô hình kiểu chat/qwen")
    a = ap.parse_args()
    global HUONG
    HUONG = a.huong
    t0 = time.time()
    cfg = MO_HINH[a.mo_hinh]
    if HUONG == "vi_zh" and cfg["kieu"] not in ("qwen", "chat"):
        raise SystemExit(f"--huong vi_zh chưa có câu lệnh cho mô hình kiểu {cfg['kieu']}: chỉ dùng mô hình kiểu chat hoặc qwen.")
    if "api" in cfg:
        if a.api_base_url.strip():
            cfg = dict(cfg, base_url=a.api_base_url.strip())
        if a.api_model.strip():
            cfg = dict(cfg, api=a.api_model.strip())
        if a.api_key_env.strip():
            cfg = dict(cfg, key_env=a.api_key_env.strip())
        if a.cho_phep_suy_luan:
            cfg = dict(cfg, cho_suy_luan=True)
        if a.them_token_ra >= 0:
            cfg = dict(cfg, them_token_ra=a.them_token_ra)
        if cfg.get("gia_usd_1m") and (a.gia_vao_usd_1m or a.gia_ra_usd_1m):
            g = dict(cfg["gia_usd_1m"])
            if a.gia_vao_usd_1m:
                g["vao_miss"] = g["vao_hit"] = a.gia_vao_usd_1m
            if a.gia_ra_usd_1m:
                g["ra"] = a.gia_ra_usd_1m
            cfg = dict(cfg, gia_usd_1m=g)
    if a.thu_ket_noi:
        if "api" not in cfg:
            log(a.mo_hinh, "không gọi qua API, không phải thử")
            return 0
        try:
            tai_backend(cfg, a.backend, a.song_song or 8, json.loads(a.api_extra) if a.api_extra.strip() else None)
            return 0
        except SystemExit as e:
            log("Thử kết nối hỏng:", e)
            return 1
    ra = Path(a.ra) / a.mo_hinh
    ra.mkdir(parents=True, exist_ok=True)

    DV = pd.read_parquet(a.vao)
    # tên cột chứa tên thuộc tính ở ngôn ngữ nguồn: name_zh (bộ 1) hoặc name_nguon (bộ 2, hướng vi_zh)
    cot_ten_nguon = "name_nguon" if "name_nguon" in DV.columns else "name_zh"
    if a.thu:
        h = DV.ma_don_vi.map(lambda x: hashlib.md5(x.encode()).hexdigest())
        n_ten = max(1, a.thu // 4)
        DV = pd.concat([DV[DV.loai == "ten"].assign(h=h).sort_values("h").head(n_ten),
                        DV[DV.loai == "gia_tri"].assign(h=h).sort_values("h").head(a.thu - n_ten)]).drop(columns="h")
    # gom phần đã dịch: trong thư mục ra và trong đầu ra phiên bản trước (chép sang để đầu ra phiên bản này đầy đủ)
    for k, cu in enumerate(a.cu):
        for f in sorted(Path(cu).glob(f"{a.mo_hinh}/lo_*.parquet")):
            dich = ra / f"lo_cu{k}_{f.name}"
            if not dich.exists():
                shutil.copy2(f, dich)
    da = set()
    for f in ra.glob("lo_*.parquet"):
        da |= set(pd.read_parquet(f, columns=["ma_don_vi"]).ma_don_vi)
    con = DV[~DV.ma_don_vi.isin(da)].copy()
    con["dd"] = con.van_ban_gui.str.len()
    # theo loại rồi theo độ dài: cùng loại nằm liền nhau để nhóm gộp đầy (xen kẽ tên và giá trị làm ~4.100 yêu cầu thay vì ~2.750)
    con = con.sort_values(["loai", "dd"], kind="stable").reset_index(drop=True)
    log(f"{a.mo_hinh}: tổng {len(DV):,} đơn vị, đã có {len(DV) - len(con):,}, còn {len(con):,}")

    the = {"mo_hinh": a.mo_hinh, "phien_ban_cau_lenh": PHIEN_BAN_CAU_LENH if HUONG == "zh_vi" else "vi_zh-v1-2026-10-22",
           "huong": HUONG, "bat_dau": time.strftime("%Y-%m-%d %H:%M:%S"),
           "tong": int(len(DV)), "da_co_truoc": int(len(DV) - len(con)), "thu": a.thu, "tham_so": {
               "temperature": 0.0, "repetition_penalty": cfg["lap"],
               "max_tokens": f"min({TOKEN_RA_API}, 3*len+64+2048)" if "api" in cfg else f"min({TOKEN_RA_TOI_DA}, 3*len+64)",
               "max_model_len": DO_DAI_NGU_CANH, "hoi_lai_khi_sot_han": 1}}
    f_the = Path(a.ra) / f"the_{a.mo_hinh}.json"
    if len(con) == 0:
        the.update(ket_thuc=time.strftime("%Y-%m-%d %H:%M:%S"), ghi_chu="đã đủ, không phải dịch thêm")
        f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    if cfg.get("gia_usd_1m") and a.backend != "gia" and a.ngan_sach_usd <= 0:
        raise SystemExit(f"{a.mo_hinh} là mô hình trả phí: bắt buộc đặt --ngan_sach_usd (ví dụ 0.5 khi chạy thử, 8 khi chạy thật).")
    song_song = a.song_song if a.song_song is not None else int(cfg.get("song_song", 8))
    be, ten_be, repo, loi_vllm = tai_backend(cfg, a.backend, song_song, json.loads(a.api_extra) if a.api_extra.strip() else None)
    if hasattr(be, "tk") and cfg.get("gia_usd_1m"):
        be.ngan_sach = a.ngan_sach_usd
        the["ngan_sach_usd"] = a.ngan_sach_usd
    if hasattr(be, "tk") and a.toi_da_yeu_cau:
        be.toi_da_goi = a.toi_da_yeu_cau
        the["toi_da_yeu_cau"] = a.toi_da_yeu_cau
    if hasattr(be, "tk") and a.rpm:
        be.rpm = a.rpm
        the["rpm"] = a.rpm
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "khong_co"
    except ImportError:
        gpu = "khong_co"
    if ten_be == "api":
        be.han_chot = t0 + a.gio_toi_da * 3600
    sha = {"gia": "gia", "api": "api (không có mã phiên bản; ghi ngày gọi ở bat_dau)"}.get(ten_be) or sha_mo_hinh(repo)
    the.update(backend=ten_be, repo=repo, sha_repo=sha, loi_vllm=loi_vllm, phien_ban=be.phien_ban, gpu=gpu)
    log("Nạp xong sau", round(time.time() - t0), "giây; backend", ten_be)

    so = [int(m.group(1)) for f in ra.glob("lo_*.parquet") for m in [re.fullmatch(r"lo_(\d+)", f.stem)] if m]
    so_lo_cu = max(so) + 1 if so else 0      # số thứ tự lô mới nối tiếp lô cũ, không trùng tên
    xong, dung_vi_het_gio, so_hoi_lai, so_sua_duoc = 0, False, 0, 0
    t_dich = time.time()
    for i in range(0, len(con), a.lo):
        if time.time() - t0 > a.gio_toi_da * 3600:
            dung_vi_het_gio = True
            log("Hết giờ cho phép, dừng êm để lưu đầu ra")
            break
        lo = con.iloc[i:i + a.lo]
        ten_nguon = lo[cot_ten_nguon]
        msgs = [cau_lenh(cfg["kieu"], l, n, c, t) for l, n, c, t in zip(lo.loai, ten_nguon, lo.category, lo.van_ban_gui)]
        toi_da = [so_token_toi_da(t) for t in lo.van_ban_gui]
        if a.gop > 1 and cfg["kieu"] in ("qwen", "chat"):
            rows = list(zip(lo.loai, ten_nguon, lo.category, lo.van_ban_gui))
            kq, tk_gop = dich_gop(be, rows, msgs, toi_da, cfg["lap"], a.gop)
            for k_, v_ in tk_gop.items():
                the.setdefault("gop", {})
                the["gop"][k_] = the["gop"].get(k_, 0) + v_
        else:
            kq = [(don_dep(x), r) for x, r in be.sinh(msgs, toi_da, cfg["lap"])]
        if hasattr(be, "tk"):
            # API: câu trả lời rỗng (bị cắt hoặc hết token trước khi ra bản dịch) coi như lỗi: không ghi, lần sau dịch lại
            rong = [j for j, (x, r) in enumerate(kq) if not x.strip() and r not in ("loi_api", "qua_dai")]
            for j in rong:
                kq[j] = ("", "loi_api")
            be.tk["tra_loi_rong"] = be.tk.get("tra_loi_rong", 0) + len(rong)
        # còn sót chữ Hán: hỏi lại một lần, giữ bản ít chữ Hán hơn
        SOT = sot_ngon_ngu_nguon()     # chữ Hán (zh_vi) hoặc chữ Việt có dấu (vi_zh)
        sot = [j for j, (x, r) in enumerate(kq) if r != "qua_dai" and SOT.search(x)]
        dich_lai = [False] * len(kq)
        if sot:
            kq2 = be.sinh([hoi_lai(cfg["kieu"], msgs[j], kq[j][0]) for j in sot], [toi_da[j] for j in sot], cfg["lap"])
            for j, (x2, r2) in zip(sot, kq2):
                x2 = don_dep(x2)
                if x2 and r2 != "qua_dai" and len(SOT.findall(x2)) < len(SOT.findall(kq[j][0])):
                    kq[j] = (x2, r2)
                    dich_lai[j] = True
        ok = [r != "loi_api" for _, r in kq]      # đơn vị gọi API lỗi: không ghi, lần chạy sau dịch lại
        if not any(ok):
            if getattr(be, "tk", {}).get("het_quota_ngay"):
                the["dung_vi_het_quota_ngay"] = True
                log("Hết hạn mức NGÀY của nhà cung cấp, dừng. Chạy lại cùng lệnh sau 00:00 giờ Thái Bình Dương "
                    "(khoảng 14:00 giờ Việt Nam khi còn giờ mùa hè Mỹ, 15:00 sau đó); phần đã dịch được giữ, chỉ dịch phần thiếu.")
            elif getattr(be, "tk", {}).get("het_toi_da_goi"):
                the["dung_vi_het_toi_da_goi"] = True
                log(f"Đã gửi đủ {a.toi_da_yeu_cau} yêu cầu cho phép, dừng. Muốn làm tiếp: nâng --toi_da_yeu_cau và chạy lại.")
            elif getattr(be, "tk", {}).get("het_ngan_sach"):
                the["dung_vi_het_ngan_sach"] = True
                log(f"Chạm ngân sách {a.ngan_sach_usd} USD (ước chi {be.chi_phi:.4f}), dừng. Muốn làm tiếp: nâng --ngan_sach_usd và chạy lại.")
            elif getattr(be, "tk", {}).get("het_gio"):
                dung_vi_het_gio = True
                log("Hết giờ cho phép giữa lô, dừng êm để lưu đầu ra")
            else:
                the["dung_vi_loi_api"] = getattr(be, "tk", {}).get("loi_cuoi", "")
                log("Cả lô đều lỗi API, dừng để khỏi mất giờ:", the["dung_vi_loi_api"])
            break
        df = pd.DataFrame({"ma_don_vi": lo.ma_don_vi.values, "ban_dich": [x for x, _ in kq],
                           "ly_do_dung": [r for _, r in kq], "dich_lai": dich_lai, "backend": ten_be})[ok]
        tam = ra / f".tam_{so_lo_cu + i // a.lo:05d}.parquet"
        df.to_parquet(tam, index=False)
        tam.rename(ra / f"lo_{so_lo_cu + i // a.lo:05d}.parquet")      # đổi tên sau khi ghi xong: không để lại file dở
        xong += int(sum(ok))
        so_hoi_lai += len(sot)
        so_sua_duoc += sum(dich_lai)
        toc_do = xong / max(time.time() - t_dich, 1)
        log(f"{a.mo_hinh}: xong {xong:,}/{len(con):,} ({toc_do:.2f} đơn vị/giây; còn khoảng {(len(con) - xong) / max(toc_do, 1e-9) / 3600:.1f} giờ)"
            f" | sót chữ Hán hỏi lại {len(sot)}, sửa được {sum(dich_lai)}" + (f" | lỗi API {len(ok) - sum(ok)}" if not all(ok) else "")
            + (f" | trung bình {be.tk['giay_cac_lan_goi'] / max(1, be.tk['yeu_cau']):.1f} giây mỗi lần gọi" if hasattr(be, "tk") else "")
            + (f" | chi phí ước {be.chi_phi:.4f} USD / ngân sách {a.ngan_sach_usd}" if getattr(be, "gia", None) else ""))
        the.update(da_dich_phien_nay=xong, toc_do_don_vi_giay=round(toc_do, 3), hoi_lai_vi_sot_han=so_hoi_lai,
                   hoi_lai_sua_duoc=so_sua_duoc)
        if hasattr(be, "tk"):
            the["api"] = be.tk
        f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    the.update(ket_thuc=time.strftime("%Y-%m-%d %H:%M:%S"), dung_vi_het_gio=dung_vi_het_gio, da_dich_phien_nay=xong,
               con_lai=int(len(con) - xong), giay=round(time.time() - t0))
    f_the.write_text(json.dumps(the, ensure_ascii=False, indent=1), encoding="utf-8")
    log("Kết thúc", a.mo_hinh, "| còn lại", len(con) - xong)


if __name__ == "__main__":
    sys.exit(main())
