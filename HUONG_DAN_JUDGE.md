# HƯỚNG DẪN BÀN GIAO: CHẠY JUDGE (DeepSeek chọn bản dịch Gemma hay Qwen)

**Dành cho:** Huy và AI của Huy (người nhận bàn giao để chạy judge).
**Người bàn giao:** Nhật Anh. Tài liệu này soạn cùng Claude, ngày 31/10/2026, từ toàn bộ các quyết định đã bàn và chốt.
**Trạng thái:** mã đã viết và kiểm thử xong; **lần chạy lớn CHƯA bắt đầu** (mới có các lần thí điểm nhỏ). Nhiệm vụ của bên nhận: lấy mã và dữ liệu, cấu hình API (Alibaba, khóa riêng của Huy), kiểm tra, chạy thí điểm, chạy lớn trong ngân sách **15 triệu token**, kiểm kết quả, ghép hai bộ dữ liệu cuối.

---

## 0. Đọc cái này trước (tóm tắt 1 phút)

1. Ta có bộ dữ liệu dịch thương mại điện tử Việt–Trung `zh-vie_ecom` (bộ official v2 mà Huy đã gộp Tiki vào 1688). Mỗi sản phẩm có tiêu đề và mô tả, và **mỗi chỗ có hai bản dịch ứng viên**: một của **Gemma 4**, một của **Qwen3-14B**.
2. Ta dùng **DeepSeek (`deepseek-v4.1-flash`) làm judge**: với mỗi chỗ, judge **xem nguồn + hai bản dịch (không biết bản nào của mô hình nào) và chọn bản tốt hơn**, dựa trên một bộ quy tắc đã được siết chặt (mục 5).
3. Judge được áp cho **cả hai bộ dữ liệu**: **bộ CỤM** và **bộ PAIR**, mỗi bộ có **Title + Mô tả**. Kết quả là **hai bộ dữ liệu cuối** để fine-tune nhiều mô hình rồi so sánh xem bộ nào tốt hơn.
4. Judge chấm **từng mảnh** (cả tiêu đề; từng thuộc tính `tên: giá trị` ở bộ cụm; riêng từng tên / từng giá trị ở bộ pair), **chỉ một lần mỗi mảnh** (không hỏi hai lần đổi chỗ), **gom 15 mảnh mỗi lời gọi** để tiết kiệm token.
5. Mảnh nào judge quyết rõ thì **thay** bằng bản được chọn; mảnh judge khai **tương đương** hoặc **chưa chấm** thì **giữ nguyên bản hiện tại**.
6. Ngân sách: **15 triệu token**. Kế hoạch chạy chính ước **khoảng 9,7 triệu** (mục 7). Mọi giai đoạn có trần token và **chạy tiếp được** nếu bị ngắt.
7. **Lấy mã và tài liệu:** nhánh `nhatanh_updating` của `https://github.com/ntquang-0410/eda_ecom` (mục 15.1). Dữ liệu: phần có trên HuggingFace tải công khai (mục 15.2), phần còn lại Nhật Anh gửi riêng (mục 15.3). Huy dùng **chính loại API Alibaba Cloud Model Studio như Nhật Anh** (mục 9).
8. Việc đầu tiên của AI bên Huy: đọc mục 1–5, làm theo **quy trình ở mục 10** (có lệnh cụ thể và tiêu chí đạt cho từng bước). Có **prompt mẫu** cho AI ở mục 14.
9. **Không đẩy lên HuggingFace khi chưa có lệnh của Nhật Anh.** Không dán khóa API vào chat/log/file.

---

## 1. Mục tiêu và ý tưởng

**Bối cảnh.** Bộ dữ liệu có hai nguồn: **1688** (nguồn tiếng Trung, dịch sang tiếng Việt) và **Tiki** (nguồn tiếng Việt, dịch sang tiếng Trung). Mỗi nội dung được dịch bởi hai mô hình, Gemma 4 và Qwen3-14B. Trước đây việc chọn giữa hai bản được làm bằng **CometKiwi** (điểm tự động) và theo thiết lập của pipeline dịch. Hai hạn chế đã đo được:
- CometKiwi **thiên về bản dài hơn** (verbosity bias) và chỉ trùng ý judge khoảng 53–61%.
- Điểm CometKiwi **không có cho bản cụm** (chỉ có cho tiêu đề và đơn vị bản pair).

**Ý tưởng chính.** Dùng một mô hình ngôn ngữ mạnh (DeepSeek) làm **judge**: đọc nguồn + hai bản dịch rồi chọn bản tốt hơn theo tiêu chí rõ ràng. Đây là hướng "LLM-as-judge" kẹp CometKiwi (Kiwi chỉ còn là tín hiệu phụ để đối chiếu, **không** dùng để phá hòa).

**Điều Nhật Anh yêu cầu rõ (phải giữ):**
- Judge **chọn 1 trong 2** (Qwen hoặc Gemma), **bắt buộc chọn**, dựa trên các tiêu chí.
- **Che tên mô hình**: judge không được thấy mô hình nào dịch bản nào, cũng không thấy điểm CometKiwi, id hay thông tin gợi ý.
- **Không thiên vị mô hình nào** (xem cách chống thiên vị vị trí ở mục 5).
- Áp cho **Title + Mô tả**, ở **cả bản cụm và bản pair**.
- Kết quả: **hai bộ dữ liệu**, rồi Nhật Anh fine-tune cả hai trên vài mô hình để chọn bộ nào tốt hơn.
- Kết quả chỉ lưu **trên máy**; chỉ đẩy HF khi Nhật Anh ra lệnh.

---

## 2. Thuật ngữ

| Từ | Nghĩa |
|---|---|
| **Bản cụm** | Bộ mà mô hình dịch **khi đã thấy cả tiêu đề và danh sách thuộc tính** của sản phẩm và rà soát bản nháp. Lưu theo từng cặp `tên: giá trị`. Mô tả = các cặp nối bằng `"; "`. |
| **Bản pair** | Bộ mà **mỗi tên hoặc giá trị thuộc tính được dịch riêng** (không ngữ cảnh), gọi là một **đơn vị**. Mô tả = các dòng `tên: giá trị` nối bằng xuống dòng. |
| **Mảnh** | Đơn vị mà judge chấm: một tiêu đề; một thuộc tính (bản cụm); một tên hoặc một giá trị (bản pair). |
| **Ứng viên A / B** | Bản dịch của Gemma (A) và của Qwen (B) cho cùng một mảnh. |
| **Bản hiện tại** | Bản đang nằm sẵn trong bộ official (một trong hai ứng viên, hoặc từ điển/quy tắc). |
| **Cuộc đấu / mảnh tranh chấp** | Mảnh mà hai ứng viên **khác nhau thật sự** (sau khi bỏ khác biệt khoảng trắng, dấu câu) và **cả hai không rỗng**. Chỉ mảnh tranh chấp mới đưa cho judge. |
| **Tương đương (`equivalent`)** | Judge khai hai bản ngang nhau (ví dụ chỉ khác từ đồng nghĩa). |
| **Bền vững** | Mảnh mà khi hỏi cả hai thứ tự (Gemma trước / Qwen trước) judge đều chọn cùng một mô hình. Dùng để **đo** judge, không dùng ở lần chạy chính. |
| **Một lượt** | Mỗi mảnh hỏi **một lần**, thứ tự hai bản **ngẫu nhiên theo băm cố định** của `item_id`. |
| **Gom (`--gom K`)** | K mảnh trong **một** lời gọi API để quy tắc chỉ gửi một lần. |
| **Cụm lai** | Cách chạy bản cụm đã chốt: cặp dùng bởi đúng 1 sản phẩm có ngữ cảnh; cặp dùng bởi ≥ 2 sản phẩm không có ngữ cảnh. |
| **Ngữ cảnh** | Tiêu đề nguồn + các thuộc tính nguồn của sản phẩm (chỉ phía nguồn, không có bản dịch nào). |
| **Top N (pair)** | Chỉ chấm N đơn vị pair được nhiều sản phẩm dùng nhất để vừa ngân sách. |

---

## 3. Dữ liệu

**Nền (danh sách sản phẩm và cấu trúc) = bộ official v2 do Huy gộp**, đã qua bước làm sạch:
- Bộ official v2 gốc (HF `ntquang0410/zh-vie_ecom`, `data/LLM/bilingual_official`, commit `4a23574`): **25.956 sản phẩm** (1688 14.928 + Tiki 11.028).
- Bước làm sạch của Nhật Anh/Claude **chỉ loại bỏ** 381 sản phẩm (1688 91, Tiki 290; lý do: rò lời nhắc trong tiêu đề, lệch số đoạn mô tả, chữ Hán sót, vượt trần số mục của một khuôn). Nhật Anh đã duyệt. **Không sửa chữ nào** của các sản phẩm giữ lại. → `ft_buoc4_giu.parquet` (bản cụm) và `ft_buoc4_giu_pair.parquet` (bản pair): **25.575 sản phẩm** (1688 14.837 + Tiki 10.738).
- **Judge KHÔNG gộp, thêm hay bớt sản phẩm nào.** Việc gộp Tiki vào 1688 theo tiêu chí của Huy đã có sẵn trong bộ official; bộ cuối giữ nguyên các hàng đó và **chỉ thay chữ ở cột bản dịch**. (Đã kiểm: ghép với 0 quyết định của judge cho lại ĐÚNG bộ official từng ký tự.)
- Sản phẩm ngoài bộ official (khoảng 3.973 sản phẩm có bản dịch hai mô hình nhưng Huy không đưa vào bộ) **không** được chấm.

**Hai bản dịch ứng viên lấy từ các file dịch của hai mô hình** (không dịch lại gì):

| Nội dung | File nguồn (xem mục 15) |
|---|---|
| Tiêu đề | `title_candidates/{1688,tiki}_title_best_cometkiwi.parquet` (có cả hai bản và điểm Kiwi; điểm Kiwi **không bao giờ** đưa vào lời nhắc; **có trên HF**, mục 15.2) |
| Bản cụm 1688 | `song_ngu_4/cap_cum_*.parquet` (Nhật Anh gửi riêng, mục 15.3) |
| Bản cụm Tiki | `song_ngu_3/so_sanh_cum_*.parquet` (lọc theo các cặp đã phát hành trong `cap_cum_*.parquet`; Nhật Anh gửi riêng) |
| Bản pair 1688 / Tiki | `song_ngu_1` và `song_ngu_2`: `don_vi_da_chon_*.parquet` + `bo_song_ngu_*_day_du_*.parquet` (Nhật Anh gửi riêng) |

**Các bảng mục** (dựng bằng `scripts/judge/so_cap_du_lieu.py`, ghi vào `data/processed/so_cap/`; số dòng và mã băm để đối chiếu ở Phụ lục C):

| Bảng | Số mục | Dùng cho |
|---|---|---|
| `muc_title.parquet` | 25.374 | tiêu đề (chạy chính) |
| `muc_cum_lai.parquet` | 4.446 (3.838 có ngữ cảnh + 608 không) | bản cụm (chạy chính) |
| `muc_pair.parquet` | 34.534 | bản pair (chạy chính, lấy top N) |
| `muc_cum.parquet` | 4.446 | cặp duy nhất không ngữ cảnh (đo) |
| `muc_cum_ctx.parquet` | 17.170 | mọi (sản phẩm, cặp) có ngữ cảnh (đo) |
| `muc_title_ctx.parquet`, `muc_pair_ctx.parquet` | 25.374 / 34.534 | tầng 2 (không dùng ở lần chạy chính) |

**Hai điều quan trọng về dữ liệu (đã từng gây lỗi, đã sửa):**
1. **Ứng viên rỗng không phải cuộc đấu.** Ở bản cụm 1688, 6.193 cặp `dich_moi` **chỉ có bản của Gemma** (Qwen không dịch, `v_cuoi_B` rỗng). Chúng bị loại khỏi bảng mục (nếu không sẽ "thắng Gemma" tầm thường, chiếm 23% mục cụm có ngữ cảnh). Phần mô tả đó giữ bản của Gemma như bộ official.
2. **Đơn vị bản pair là THEO NHÓM HÀNG.** Cùng một chữ nguồn (ví dụ `其他`) có nhiều `ma_don_vi`, mỗi nhóm hàng một đơn vị. `san_pham_n` (số sản phẩm dùng) và việc ghép bộ cuối đều tính theo (chữ, nhóm hàng). Con số độ phủ top N ở mục 7 đã tính đúng theo cách này.

---

## 4. Các quyết định đã chốt (Nhật Anh chốt với Claude)

| # | Quyết định | Lý do ngắn |
|---|---|---|
| D1 | Judge cố định `deepseek-v4.1-flash`, **tắt suy luận (thinking off)**, nhiệt độ 0 | Rẻ, nhanh; đã đo ổn định (chạy lặp giống 97%) |
| D2 | Áp cho **Title + Mô tả**, **cả bản cụm và bản pair**; ra **hai bộ dữ liệu cuối** | Để fine-tune rồi so sánh hai bộ |
| D3 | **Mù**: lời nhắc chỉ có hướng dịch, loại văn bản, nhóm hàng, (ngữ cảnh nguồn), nguồn, Translation 1/2. Không tên mô hình, không điểm Kiwi, không id | Tránh judge thiên vị theo mô hình |
| D4 | Tiêu chí: **nghĩa, đầy đủ, số/đơn vị/mã/tên, thuật ngữ, trôi chảy**, được siết thành bộ quy tắc v2 (mục 5) | Quy tắc v1 chưa đủ chặt nên judge lệch theo vị trí |
| D5 | Judge chấm **từng mảnh**, không chấm cả mô tả | Dữ liệu dịch lưu theo từng thuộc tính; lấy tốt nhất của từng bên cho từng chỗ |
| D6 | **Chấm một lần** (một lượt), **buộc chọn 1 trong 2**, thứ tự hai bản **ngẫu nhiên theo băm cố định** của `item_id` | Nhật Anh yêu cầu; thiên vị vị trí đã đo là có thật (~63–70% chọn vị trí 1) nên ngẫu nhiên hóa để không lợi mô hình nào |
| D7 | Chính sách **trung lập** cho 4 điểm (thương hiệu Latin giữ hay viết sang chữ đích; từ ngoại ngữ đã có ở nguồn; từ đồng nghĩa ngang vị thế; định dạng đơn vị/số) **kèm một ngoại lệ**: đổi thương hiệu/tên model thành **tên khác** (phiên âm bịa hoặc sai) là lỗi WRONG | Mọi quy tắc "ưu tiên" mà trùng thói quen một mô hình sẽ nghiêng kết quả về mô hình đó |
| D8 | Mảnh judge khai **tương đương** → **giữ bản hiện tại** (vẫn là một trong hai bản) và ghi nhãn | Hai bản ngang nhau thì không có lý do đổi; tránh đổi nhầm khi judge khai tương đương sai (~16,6% mục bền vững) |
| D9 | **Tiêu đề: judge cho TẤT CẢ** (25.374) | Kết quả đối xứng, đầy đủ |
| D10 | **Bản cụm "lai"**: cặp dùng bởi 1 sản phẩm → **có ngữ cảnh**; cặp dùng bởi ≥ 2 sản phẩm → không ngữ cảnh | Ngữ cảnh quan trọng ở cặp riêng của sản phẩm; cặp dùng chung là cặp chung chung |
| D11 | **Pair: top N = 5.000** đơn vị dùng nhiều nhất để bắt đầu; khi còn ngân sách tăng `-TopPair` (phần đã chấm không hỏi lại) | Chấm hết 34.534 đơn vị tốn ~6,9 triệu token, vượt ngân sách |
| D12 | **Gom 15 mảnh mỗi lời gọi** (mảnh có ngữ cảnh gom ít hơn: `max(3, K//2)`) | 85% token vào của title/pair là chữ quy tắc lặp lại |
| D13 | **Cách 1 cho cột nhãn mô tả**: giữ nhãn cũ nhưng **đổi tên cột `desc_nguon_dich` thành `desc_nguon_dich_goc`**, và **thêm cột `desc_so_manh_da_doi`** (số mảnh mô tả judge đã đổi) | Nhãn cũ có thể sai sau khi đổi mảnh (mô tả thành hỗn hợp); không ghi sai, chi tiết từng mảnh ở file `danh_dau` |
| D14 | **Không đẩy HF** khi chưa có lệnh của Nhật Anh; kết quả chỉ lưu trên máy | Nhật Anh yêu cầu |
| D15 | **Không có người chấm kiểm lại** (Nhật Anh không biết tiếng Trung, chưa tìm được người) | Phải nêu giới hạn này trong báo cáo (mục 6) |
| D16 | **CometKiwi chỉ là tín hiệu phụ** để báo cáo mức đồng thuận, **không** dùng để phá hòa | Kiwi thiên về bản dài, nặng nhất đúng ở các ca khó |
| D17 | Bộ cuối giữ **nguyên các hàng của bộ official** (đã loại 381 sản phẩm); judge chỉ thay chữ ở cột bản dịch | Không tự gộp/bớt; theo tiêu chí gộp của Huy |
| D18 | Mô hình tắt suy luận; **dừng ngay khi lỗi nặng (401/402/403)**; mỗi giai đoạn có trần token | An toàn ngân sách |

---

## 5. Quy tắc chọn bản dịch (tiêu chí đã siết chặt)

Toàn văn lời nhắc (tiếng Anh, đây là thứ judge thực sự nhận) nằm ở **Phụ lục A**; mã ở `scripts/judge/so_cap_rubric.py` (`HE_THONG_V2`, `QUY_TAC_CHINH_SACH`) và `so_cap_gom.py`. **Không sửa lời nhắc giữa các lần chạy** (kết quả sẽ không so sánh được).

### 5.1 Bốn bước judge phải làm
1. **Liệt kê lỗi cụ thể của TỪNG bản** (bắt buộc chỉ đúng chữ mới được nói "thiếu" hay "thêm"):
   - **WRONG**: nghĩa sai hoặc ngược (sai loại sản phẩm, màu, chất liệu, số lượng, chức năng), sai số / quy đổi đơn vị / mã, hoặc **đổi tên thương hiệu/model/sản phẩm thành tên KHÁC**.
   - **MISSING**: thiếu thông tin của nguồn (tính năng, thuộc tính, số, thương hiệu, mã).
   - **ADDED**: thêm thông tin không có ở nguồn **và không được ngữ cảnh sản phẩm hỗ trợ**. Từ ngữ pháp, lượng từ, cách diễn đạt thuộc tính chuẩn không phải là thêm.
   - **BROKEN**: lời nhắc rò rỉ, chuỗi giữ chỗ/rác (ví dụ `ZXQ…`), từ nguồn bị dịch sang sai ngôn ngữ (ví dụ chữ Anh khi cần tiếng Trung và từ nguồn không phải tiếng Anh), hoặc không đọc được. (Lời nhắc v2 không nhắc riêng "bản rỗng": ở lần chạy chính các ứng viên rỗng đã bị loại khỏi bảng mục từ trước, xem mục 3.)
   - **TERM**: thuật ngữ sai rõ hoặc không chuẩn cho nhóm hàng.
2. **Bản ít lỗi hơn thắng.** Mức nặng: WRONG và BROKEN > MISSING và ADDED > TERM.
3. **Những khác biệt sau KHÔNG BAO GIỜ là lỗi và KHÔNG là lý do chọn** (đây là **chính sách trung lập đã chốt**, D7):
   - viết thương hiệu/tên Latin như nguồn, **hoặc** viết **cùng** thương hiệu sang chữ đích (phiên âm, tên bản địa chính thức);
   - giữ hay dịch từ ngoại ngữ thông thường **đã là tiếng nước ngoài ngay ở nguồn** (không phải tên hay mã);
   - từ đồng nghĩa ngang vị thế, kể cả dạng rút gọn và đầy đủ (ví dụ `电泵` và `电动泵`), nhãn thuộc tính chuẩn ngang nhau;
   - định dạng đơn vị và số (`5L`/`5升`, `kg`/`kilogram`, dấu chấm/phẩy thập phân, khoảng trắng, dấu câu, gạch chéo, ngoặc);
   - thứ tự từ giữ nguyên nghĩa, mức trang trọng;
   - độ trôi chảy/tự nhiên, **trừ khi** một bản sai ngữ pháp hoặc khó đọc;
   - độ dài, và việc bản nào đứng trước/sau.
4. **Nếu sau các bước vẫn tương đương** → judge ghi `equivalent = true` và **chọn Translation 1**. Vì **thứ tự hai bản đã được ngẫu nhiên hóa theo băm**, đây là **tung đồng xu công bằng**, không lợi mô hình nào, và được **ghi cờ** để xử lý ở bước ghép.

Judge **bắt buộc** trả lời 1 hoặc 2, **không có "tie"**.

### 5.2 Định dạng trả lời
- Một lời gọi một mảnh (rubric v2, dùng khi đo): `{"defects_1":[...],"defects_2":[...],"equivalent":bool,"winner":1|2,"criterion":"...","reason":"..."}`.
- **Chế độ gom (lần chạy chính):** `{"results":[{"i":1,"d1":[...],"d2":[...],"eq":bool,"w":1|2,"c":"W|M|A|B|T|N"},...]}` (một mục mỗi mảnh theo đúng thứ tự). Trường theo thứ tự **lỗi → tương đương → người thắng** để quyết định theo sau phân tích.

### 5.3 Ngữ cảnh (bản cụm lai)
Với cặp dùng bởi 1 sản phẩm, lời nhắc có khối `PRODUCT CONTEXT` (tiêu đề nguồn + danh sách thuộc tính nguồn của sản phẩm đó, tối đa 3.000 ký tự, giữ cặp đang xét và các cặp kề). Judge **chỉ dùng** ngữ cảnh để hiểu nguồn (ví dụ hiểu `cốt đặc biệt` là nước mắm); thông tin do ngữ cảnh hỗ trợ **không** tính là ADDED. Ngữ cảnh **không** có bản dịch nào.

### 5.4 Cách dùng kết quả của judge khi ghép bộ cuối
| Judge trả lời | Việc làm | Nhãn trong file `danh_dau` |
|---|---|---|
| Quyết rõ (`equivalent=false`) | Dùng bản judge chọn, **sau kiểm an toàn** (bản mới không được làm xuất hiện chữ Hán sót ở bản Việt của 1688, hay thêm chữ Việt có dấu vào bản Trung của Tiki; chỉ thay khi bản hiện tại đúng bằng A hoặc B, nếu không là bản từ điển/quy tắc: giữ) | `judge_t1` |
| **Tương đương** (`equivalent=true`) | **Giữ bản hiện tại** (mặc định D8; có thể đổi bằng `--khi-tuong-duong theo_judge` để lấy bản judge chọn, không tốn token) | `judge_tuong_duong` |
| Chưa chấm (đơn vị pair ngoài top N) | Giữ bản hiện tại | `chua_cham` |

### 5.5 Cột của bộ cuối (Cách 1, D13)
Bộ cuối có **17 cột** (đã kiểm bằng lần chạy thử toàn quy trình):
`product_id, source, category, title_zh, title_vi, description_zh, description_vi, title_nguon_dich, desc_nguon_dich_goc, huong_goc, khuon_id, nhom_mo_ta_id, can_hang_chac, mem_latin_la_ban_trung, mem_viet_sot_title_zh, mem_v2_review, desc_so_manh_da_doi`.
- 9 cột đầu là cột của bộ official (riêng `desc_nguon_dich` đã đổi tên thành `desc_nguon_dich_goc`, xem dưới).
- `huong_goc, khuon_id, nhom_mo_ta_id, can_hang_chac, mem_latin_la_ban_trung, mem_viet_sot_title_zh, mem_v2_review` là **cột phụ do bước làm sạch thêm** (hướng dịch, mã khuôn mô tả, mã nhóm mô tả giống nhau dùng để nhóm khi chia tập tránh rò rỉ, và các cờ mềm không loại sản phẩm). Chúng được **tính trên bản gốc TRƯỚC judge và KHÔNG tính lại** sau khi judge đổi chữ; chỉ dùng để nhóm/lọc, không ảnh hưởng judge.
- `desc_so_manh_da_doi` là cột mới (cuối cùng).
- `desc_nguon_dich_goc` = nhãn gốc (`gemma4`, `qwen3_14b_awq`, `gemma4+qwen3_14b_awq`, `tu_dien_quy_tac`) của bộ official, **không cập nhật** sau khi judge đổi mảnh.
- `desc_so_manh_da_doi` = số mảnh mô tả judge đã đổi (0 = mô tả y hệt bản gốc).
- `title_nguon_dich` giữ tên cũ và **đã cập nhật đúng** khi tiêu đề được đổi.
- File đi kèm `bo_<cum|pair>_danh_dau.parquet`: mỗi sản phẩm một dòng (`title_nhan`, `title_da_doi`, `mo_ta_so_manh_tranh_chap`, `mo_ta_so_manh_judge_t1`, `..._judge_tuong_duong`, `..._chua_cham`, `..._da_doi`, …).

---

## 6. Số đo thực tế (để biết kỳ vọng) và giới hạn

Đo trên mẫu chia tầng (lệch về Tiki so với tỉ lệ thật), mô hình `deepseek-v4.1-flash`, tắt suy luận, 8 luồng.

| Chỉ số | Giá trị |
|---|---|
| Thiên vị vị trí (chọn vị trí 1) | ~63–70%; Gemma thắng 85–91% khi đứng trước nhưng chỉ ~50–59% khi đứng sau |
| Judge khai tương đương | ~37–40% số mục (tiêu đề ~11–23%, cụm ~42–50%, pair ~50–54%) |
| Mục bền vững (hai thứ tự đồng ý) | ~55% (tiêu đề ~73%, cụm ~51%, pair ~41%) |
| Trong mục bền vững: Gemma / Qwen | ~87% / 13% (Tiki: Qwen thắng nhiều hơn, ~15–24%) |
| Lệch hai thứ tự mà judge **không** khai tương đương | ~12,7% (một mục một lời gọi), ~14,4% (gom 15) |
| Gom 15 so với một mục một lời gọi, trên mục bền vững | đúng 81,8%, khai tương đương 16,6% (giữ hiện trạng nên không sai), **đảo ngược không khai chỉ 1,6%** |
| Token mỗi mục (gom 15) | tiêu đề ~239, cụm có ngữ cảnh ~642, pair ~200 (không gom: ~959 / 1.333 / 889) |
| Tốc độ | ~8–10 lời gọi/giây với 8 luồng, 0 lỗi 429 |
| Độ ổn định judge | chạy lặp cùng lời nhắc giống 97% (gần như tất định) |

**Giới hạn phải ghi trong báo cáo:**
- **Judge chưa được người kiểm** (D15). Chỉ có kiểm tra gián tiếp: nhất quán, tái lập, thiên vị vị trí/độ dài đã đo, đối chiếu CometKiwi.
- ~13–14% mảnh judge quyết mà **không nhận ra là không chắc** (hai thứ tự sẽ lệch); trong một lượt không phân biệt được.
- Judge **có thể sai ở kiến thức** (ví dụ `Hồng` là màu hồng, `水绿`, tên bánh `bánh xếp`); quy tắc không sửa được.
- **Bộ pair chỉ phủ ~84% mảnh** nếu top N = 5.000; bộ cụm phủ 100% mảnh tranh chấp. Phải ghi độ phủ khi so hai bộ.
- Hai bộ có **giao thức judge khác nhau** (cụm lai có ngữ cảnh một phần, pair không): ghi rõ trong báo cáo.
- Kết quả phụ thuộc **đúng mô hình judge** này; nếu dùng mô hình khác, phải thí điểm lại (mục 10, B6).

---

## 7. Ngân sách 15 triệu token

(Nhật Anh đã dùng ~2,89 triệu token cho các lần thí điểm trên gói của Nhật Anh; **không tính vào ngân sách của Huy**.)

**Token mỗi mục đã đo (gom 15):** tiêu đề 239; cụm có ngữ cảnh 642; cụm không ngữ cảnh ~200; pair 200. Token vào chiếm ~92% (tính cả token vào và ra).

| Giai đoạn | Số mục | Token ước tính | Trần mặc định của `chay_lon.ps1` |
|---|---|---|---|
| Pair, top 5.000 đơn vị (phủ 83,9% mảnh) | 5.000 | ~1,0 triệu | 1,4 triệu |
| Cụm lai | 4.446 (3.838 có ngữ cảnh) | ~2,6 triệu | 3,3 triệu |
| Tiêu đề, tất cả | 25.374 | ~6,1 triệu | 7,5 triệu |
| **Tổng** | | **~9,7 triệu** | |

Còn dư ~5 triệu. Nếu còn ngân sách có thể tăng top N pair (độ phủ theo **mảnh** thật, đã đo):

| Top N | Phủ mảnh | Token thêm so với N = 5.000 |
|---|---|---|
| 3.347 | 81,7% | (ít hơn) |
| **5.000** | **83,9%** | 0 |
| 8.000 | 86,4% | +0,6 triệu |
| 12.000 | 88,5% | +1,4 triệu |
| tất cả 34.534 | 100% | +5,9 triệu (khó vừa nếu chạy cả tiêu đề) |

**Lưu ý:** "token" = token vào + token ra theo số liệu `usage` của API. Nhà cung cấp có thể tính hạn mức theo cách khác (Credits, token vào rẻ hơn token ra); **đối chiếu đồng hồ của gói với số mà script báo** sau thí điểm để biết hệ số. Kế hoạch có chừa dư để chịu sai lệch.

---

## 8. Cài đặt môi trường

- **Python 3.13** (đã chạy thật); thư viện: **pandas, numpy, pyarrow**. **Không cần SDK nào** (script gọi API bằng `urllib`, chuẩn OpenAI `chat/completions`).
- Các file `.py` **dời được** (đường dẫn gốc suy ra từ vị trí file). Các file `.ps1` (PowerShell, Windows) **ghi cứng** đường dẫn Python `D:\nam4_ky1\e-commerce - crawl\ecom_crawler\.venv\Scripts\python.exe`: nếu chạy trên máy khác thì **sửa dòng `$py = ...`** ở đầu mỗi `.ps1`, hoặc dùng lệnh Python trực tiếp (mục 10 đều ghi cả hai cách).
- Mọi lệnh Python chạy từ thư mục gốc repo, dạng `python -I -X utf8 scripts/judge/<tệp>.py ...` (`-X utf8` để in tiếng Việt/Trung không lỗi).
- Cấu trúc thư mục cần có (đúng tên): xem Phụ lục E.

---

## 9. Gắn API

Script chỉ cần một API **tương thích OpenAI `POST {base_url}/chat/completions`** có: tham số `model`, `messages`, `max_tokens`, `temperature`, và **JSON mode** `response_format: {"type":"json_object"}` (script đã gửi). Phản hồi cần có `choices[0].message.content` và `usage.{prompt_tokens,completion_tokens}`. Alibaba Cloud Model Studio đáp ứng đủ.

### 9.1 Nhà cung cấp: dùng `alibaba` (đã có sẵn trong mã)
Huy dùng **cùng loại API với Nhật Anh: Alibaba Cloud Model Studio, mô hình `deepseek-v4.1-flash`**. Mã đã có sẵn nhà cung cấp `alibaba` trong `scripts/judge/cau_hinh.py` (`base_url` lấy từ biến môi trường, khóa từ `DASHSCOPE_API_KEY`, tắt suy luận bằng `enable_thinking=false`). **Không cần sửa mã.** Việc cần làm:
1. **Mỗi người dùng khóa của chính mình** (Huy tạo/nhận khóa riêng; không dùng lại khóa của Nhật Anh, không gửi khóa qua chat).
2. Đặt hai biến môi trường trong **cùng cửa sổ** chạy lệnh:
   - `ALIBABA_BASE_URL` = địa chỉ gốc **khớp với loại khóa**, kết thúc bằng `/compatible-mode/v1`. Ví dụ gói **Token Plan** (Singapore): `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`; gói trả theo lượng có dạng `https://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`. **Khóa và địa chỉ phải cùng một gói**, lệch là lỗi 401 `Invalid API-key`.
   - Khóa: biến `DASHSCOPE_API_KEY` (các launcher `.ps1` hỏi khóa ẩn và tự đặt biến này, rồi xóa khi xong).
3. Mọi lệnh dùng `--nha-cung-cap alibaba` (launcher: `-NhaCungCap alibaba`).

(Nếu sau này cần nhà cung cấp khác: thêm một mục vào `CAC_NHA_CUNG_CAP` theo mẫu `alibaba`, gồm `base_url`, `mo_hinh`, `bien_khoa`, `rpm`, `phien`, `tat_suy_luan`; và sửa `ValidateSet` trong các `.ps1`.)
### 9.2 Tắt suy luận (rất quan trọng, để token ra ~40 mỗi mục thay vì hàng trăm)
Tên tham số **khác nhau theo nhà cung cấp** (đã tra tài liệu 31/10/2026):
- DeepSeek trực tiếp (`api.deepseek.com`, mô hình `deepseek-flash` = DeepSeek-V4.1-Flash-1017): `{"thinking": {"type": "disabled"}}`.
- Alibaba Cloud Model Studio (`deepseek-v4.1-flash`): `{"enable_thinking": false}`.
- OpenCode/NVIDIA: `{"thinking": {"type": "disabled"}}`.
Kiểm bằng lần chạy thí điểm: nếu token ra mỗi mục **lớn hơn nhiều** so với ~40 (gom) hoặc ~100 (một mục) thì suy luận chưa tắt.

### 9.3 Khóa API
- Khóa nằm trong **biến môi trường** (`DASHSCOPE_API_KEY`), **không** ghi vào file/chat/log. Script hỏi khóa ẩn nếu chưa có biến; log của launcher **không** chứa khóa.
- Chạy `--kiem-tra` trước: `python -I -X utf8 scripts/judge/chay_judge.py --kiem-tra --nha-cung-cap alibaba` → phải thấy danh sách mô hình và một lời gọi thử `OK`.
- **Lưu ý điều khoản:** tài liệu của Alibaba ghi gói "Token Plan" là **chỉ dành cho công cụ lập trình tương tác, không cho dịch vụ backend** (FAQ của họ không nêu hình phạt). Nhật Anh đã chạy thí điểm nhỏ trên gói này không gặp vấn đề, nhưng với hơn 60.000 lời gọi, Huy nên **đọc điều khoản của gói Huy đang dùng** và xác nhận với người cấp gói trước khi chạy lớn.

### 9.4 Nếu dùng mô hình khác `deepseek-v4.1-flash`
Các số đo ở mục 6 và quy tắc ở mục 5 được hiệu chuẩn cho mô hình này. Mô hình khác (hoặc phiên bản khác, kể cả `deepseek-v4.1-flash-1017` hay `deepseek-v4-pro` có trong danh sách của gói) có thể có thiên vị vị trí và tỉ lệ khai tương đương khác → **bắt buộc thí điểm lại** (B6) và so với kỳ vọng ở mục 11 trước khi chạy lớn. **Không** trộn kết quả của hai mô hình judge trong cùng một lần chạy lớn.

---

## 10. Quy trình từng bước (có tiêu chí đạt)

> Ký hiệu: **[PS]** = launcher PowerShell, **[PY]** = lệnh Python tương đương (dùng khi không có PowerShell).

**B0. Nhận file và kiểm.** Nhận các file ở mục 15, đặt đúng đường dẫn tương đối. Kiểm SHA256 (Phụ lục C).
*Đạt:* mọi mã băm khớp.

**B1. Chạy test (không tốn token, không cần khóa).**
```
python -I -X utf8 scripts/judge/test_so_cap.py      # kỳ vọng: 39/39
python -I -X utf8 scripts/judge/test_ghep_bo.py     # kỳ vọng: 13/13
python -I -X utf8 scripts/judge/test_judge.py       # kỳ vọng: 18/18
python -I -X utf8 scripts/ft_clean/test_ft_clean.py # kỳ vọng: 6/6
```
*Đạt:* tất cả `bài thử đạt` đủ số. (Nếu thiếu `scripts/ft_clean/luat_ft.py` thì `ghep_bo.py` không chạy: xem mục 15.)

**B2. Dựng bảng mục và đối chiếu.**
```
python -I -X utf8 scripts/judge/so_cap_du_lieu.py
```
*Đạt:* in `title khác nhau: 25374`, `CỤM: ... 4446`, `CỤM CÓ NGỮ CẢNH: 17170`, `CỤM LAI: 4446 mục | có ngữ cảnh ... 3838 | không ngữ cảnh ... 608`, `PAIR: ... 34534`, và mã băm `item_id` khớp Phụ lục C. Lệch số mục = dữ liệu đầu vào khác bản gốc → **dừng, hỏi Nhật Anh**.

**B3. Kiểm vòng (ghép với 0 quyết định phải cho lại đúng bộ official).**
```
python -I -X utf8 scripts/judge/ghep_bo.py --kiem-vong
```
*Đạt:* mọi cột `true`, thoát mã 0, `title:chua_cham 25374`, `mo_ta:chua_cham 17170` (cụm) và `195311` (pair).

**B4. Chạy thử toàn quy trình bằng judge giả (không tốn token).**
- [PS] `.\scripts\judge\chay_lon.ps1 -Gia`
- [PY] `python -I -X utf8 scripts/judge/so_cap_chon.py --loai title --tat-ca --rubric v2 --che-do mot_luot --gom 15 --gia --ra <thư mục tạm> --ten t` (rồi tương tự `--loai cum_lai`, `--loai pair --top 5000`, rồi `ghep_bo.py --thu-muc-qd <thư mục tạm> --ra <thư mục tạm 2> --title gia_t --cum gia_c --pair gia_p`).
*Đạt:* thoát 0 và có hai bộ 25.575 sản phẩm. (Judge giả chọn bản dài hơn, **chỉ để thử máy móc**, số liệu vô nghĩa.)

**B5. Gắn API và kiểm tra.** Mục 9 (đặt `ALIBABA_BASE_URL`, khóa riêng của Huy). *Đạt:* `--kiem-tra` OK, mô hình `deepseek-v4.1-flash` có trong danh sách và lời gọi thử trả `OK`.

**B6. Thí điểm hiệu chuẩn (≤ ~0,3 triệu token).** Chạy gom 15 trên **450 mục** (150 tiêu đề, 150 cụm có ngữ cảnh, 150 pair; mẫu chia tầng cố định theo hạt giống, nên luôn cùng mẫu):
- [PS] `.\scripts\judge\thu_gom.ps1 -NhaCungCap alibaba -SongSong 8` — cần có sẵn 3 file đối chứng `ket_qua_m2_v2_{title,cum_ctx,pair}_150.jsonl` (mục 15) để so với kết quả hai thứ tự.
- Nếu **không có** file đối chứng hoặc **đổi mô hình**: chỉ chạy gom và đọc các chỉ số ở mục 11 (không cần so; hoặc chạy thêm hai thứ tự v2 `--che-do hai_luot` trên 450 mục, ~1,5 triệu token, để có đối chứng riêng).
*Đạt:* 0 mục lỗi/thiếu (< 2%), token mỗi mục ≈ 240 / 640 / 200, các chỉ số ở mục 11 trong khoảng cho phép. **Nếu không đạt: dừng và báo Nhật Anh.**

**B7. Chạy lớn theo giai đoạn** (thứ tự ưu tiên: pair → cụm → tiêu đề; mỗi giai đoạn có trần, chạy tiếp được).
- [PS] `.\scripts\judge\chay_lon.ps1 -NhaCungCap alibaba -SongSong 8` (mặc định gom 15, pair top 5.000, trần 1,4 / 3,3 / 7,5 triệu, tự ghép khi xong). Chạy từng giai đoạn: `-ChiGiaiDoan pair|cum|title|ghep`. Tăng pair: `-TopPair 8000`.
- [PY]
```
python -I -X utf8 scripts/judge/so_cap_chon.py --loai pair     --tat-ca --top 5000 --rubric v2 --che-do mot_luot --gom 15 --tat-suy-luan --song-song 8 --tran-token 1400000 --nha-cung-cap alibaba --ten lon_pair
python -I -X utf8 scripts/judge/so_cap_chon.py --loai cum_lai  --tat-ca            --rubric v2 --che-do mot_luot --gom 15 --tat-suy-luan --song-song 8 --tran-token 3300000 --nha-cung-cap alibaba --ten lon_cum
python -I -X utf8 scripts/judge/so_cap_chon.py --loai title    --tat-ca            --rubric v2 --che-do mot_luot --gom 15 --tat-suy-luan --song-song 8 --tran-token 7500000 --nha-cung-cap alibaba --ten lon_title
python -I -X utf8 scripts/judge/ghep_bo.py --title lon_title --cum lon_cum --pair lon_pair --cum-che-do lai --khi-tuong-duong giu_hien_tai
```
- Nếu một giai đoạn in `chạm trần ... dừng gửi thêm`: **chưa xong**, **đừng ghép**; kiểm số token đã dùng rồi chạy lại **đúng lệnh đó** để tiếp tục (mục đã xong không hỏi lại). Lỗi `401/402/403` = dừng ngay (khóa/số dư/quyền).
- **Theo dõi ngân sách** sau mỗi giai đoạn bằng `tom_tat_lon_*.json` (`tokens_vao`, `tokens_ra`) và so với đồng hồ của nhà cung cấp.

**B8. Kiểm kết quả từng giai đoạn** (chỉ đọc file, không tốn token): `python -I -X utf8 scripts/judge/so_cap_chon.py --bao-cao lon_title` (tương tự `lon_cum`, `lon_pair`) → in và ghi `bao_cao_lon_*.json`. Đối chiếu mục 11.

**B9. Ghép hai bộ cuối** (tự động ở B7; hoặc chạy riêng `-ChiGiaiDoan ghep`). Đầu ra ở `data/processed/bo_chon/`: `bo_cum_chon_deepseek.parquet`, `bo_pair_chon_deepseek.parquet`, `bo_cum_danh_dau.parquet`, `bo_pair_danh_dau.parquet`, `ghep_bo_thong_ke.json`.
*Đạt:* mỗi bộ 25.575 sản phẩm, đúng 17 cột ở 5.5, `product_id` cùng thứ tự bộ official, các cột **nguồn** (`title_zh`/`description_zh` của 1688, `title_vi`/`description_vi` của Tiki) **giống hệt** bộ official, `desc_so_manh_da_doi` ≥ 0 và bằng cột `mo_ta_so_manh_da_doi` trong `danh_dau`.

**B10. Báo cáo cho Nhật Anh:** số mảnh theo nhãn (`judge_t1`, `judge_tuong_duong`, `chua_cham`), tỉ lệ Gemma/Qwen ở các mảnh đã đổi, token thực dùng, các số ở mục 6, và **không đẩy HF** đến khi Nhật Anh ra lệnh.

---

## 11. Kiểm tra chất lượng sau khi chạy (kỳ vọng và dấu hiệu bất thường)

Các khoảng dưới đây rút từ các lần đo với đúng mô hình `deepseek-v4.1-flash`; lệch xa là dấu hiệu có vấn đề (sai lời nhắc, sai cấu hình suy luận, mô hình khác, lỗi dữ liệu).

| Chỉ số (đọc từ `bao_cao_*.json` / `danh_dau`) | Kỳ vọng | Nếu lệch xa |
|---|---|---|
| Mục lỗi/thiếu | 0 (chấp nhận < 2%); chạy tiếp sẽ làm lại | Kiểm `ly_do` trong `ket_qua_*.jsonl` (`thieu_trong_lan_goi`, `doc_loi`, `bi_cat_het_token`) |
| Tỉ lệ khai tương đương | tiêu đề ~10–25%; cụm ~35–50%; pair ~45–60% | Quá thấp/cao nhiều: lời nhắc bị sửa hoặc mô hình khác |
| Gemma/Qwen trong mục judge chọn | Gemma chiếm ~70–90% (1688 cao hơn Tiki) | Gần 50/50 hoặc 100/0: nghi thiên vị vị trí/thứ tự hỏng |
| Token ra mỗi mục (gom) | ~40–50 | Hàng trăm: suy luận chưa tắt |
| Token vào mỗi mục (gom) | ~120–200 (title/pair), ~500–650 (cụm có ngữ cảnh) | Cao hơn nhiều: gom không hoạt động (kiểm `gom` trong bản ghi) |
| Tiêu đề đổi so với hiện tại | ~25% tiêu đề sẽ đổi, hầu hết Qwen → Gemma | Quá khác: kiểm kết nối Kiwi/ghép |

**Kiểm ngẫu nhiên bằng mắt** (nên làm, Huy hoặc AI đọc): 30 mảnh `judge_t1` đã đổi, xem lý do (`ly_do` ghi `d1=[...]; d2=[...]`) có hợp lý không; 30 mảnh `judge_tuong_duong` xem có thật sự ngang nhau không.

---

## 12. Đầu ra và cách dùng

- `bo_cum_chon_deepseek.parquet` và `bo_pair_chon_deepseek.parquet`: hai bộ để **fine-tune** (cột ở mục 5.5).
- `bo_*_danh_dau.parquet`: để **báo cáo, lọc** (ví dụ chỉ giữ sản phẩm có `mo_ta_so_manh_da_doi` ≤ 1) và truy vết.
- Biến thể "lấy bản judge chọn cho mục tương đương" (nếu Nhật Anh muốn so sánh): chạy lại **chỉ bước ghép** với `--khi-tuong-duong theo_judge --ra <thư mục khác>`; **không tốn token** vì dùng cùng kết quả judge.
- **Không** ghi đè file cũ khi ghép lại: dùng `--ra` thư mục mới.

---

## 13. Những điều KHÔNG được làm

1. **Không** đưa tên mô hình, điểm CometKiwi, id mục hay bất kỳ gợi ý nào về nguồn gốc bản dịch vào lời nhắc.
2. **Không** đổi lời nhắc/quy tắc, thứ tự băm (`thu_tu_bang_bam`), `K` gom, hay cấu hình suy luận **giữa các lần chạy** của cùng một đợt (kết quả sẽ không so sánh được). Muốn đổi: chạy đợt mới với tên mới.
3. **Không** thêm quy tắc "ưu tiên" cho một mô hình hay một phong cách (đã chốt trung lập, D7).
4. **Không** dùng CometKiwi để phá hòa (D16).
5. **Không** hỏi lại các mảnh đã xong; **không** xóa `ket_qua_lon_*.jsonl` giữa chừng.
6. **Không** đẩy lên HF; **không** dán khóa vào chat/log/file/commit.
7. **Không** tự gộp/loại sản phẩm; cấu trúc sản phẩm theo bộ official (D17).
8. **Không** sửa tay file kết quả `.jsonl`.

---

## 14. Prompt mẫu cho AI của Huy (copy-paste)

```
Bạn nhận bàn giao dự án "chạy judge DeepSeek chọn bản dịch Gemma/Qwen" cho bộ dữ liệu zh-vie_ecom.
Hãy đọc theo thứ tự: (1) HUONG_DAN_JUDGE.md toàn bộ; (2) scripts/judge/README.md (các mục "QUY TRÌNH CHÍNH", "BỘ CỤM CÓ NGỮ CẢNH", "TẦNG 2", "SỬA LỖI BẢN RỖNG", "CHẾ ĐỘ GOM", "KẾT QUẢ THÍ ĐIỂM GOM"); (3) scripts/judge/so_cap_rubric.py và so_cap_gom.py (quy tắc và định dạng); (4) so_cap_du_lieu.py, so_cap_chon.py, ghep_bo.py; (5) cau_hinh.py và chay_judge.py (phần gọi API). CLAUDE.md và WALKTHROUGH.md chỉ để tra cứu lịch sử khi cần.
Sau khi đọc, hãy: (a) tóm tắt lại cho tôi bằng lời của bạn: mục tiêu, 18 quyết định D1-D18, quy tắc chọn bản dịch 4 bước, và cách xử lý mảnh tương đương/chưa chấm; nêu chỗ bạn chưa chắc; (b) làm B0-B5 trong mục 10, báo kết quả từng bước; (c) KHÔNG bắt đầu B6 trở đi cho đến khi tôi đồng ý.
Ràng buộc: không đổi lời nhắc/quy tắc; không đưa tên mô hình/điểm Kiwi/id vào lời nhắc; không đẩy HF; không in hay lưu khóa API; ngân sách tổng 15 triệu token, dừng khi chạm trần mỗi giai đoạn.
```

---

## 15. Lấy mã và dữ liệu về (ai lấy gì, từ đâu)

### 15.1 Mã và tài liệu: lấy từ GitHub
Repo `https://github.com/ntquang-0410/eda_ecom`, nhánh **`nhatanh_updating`**:
```
git clone --branch nhatanh_updating https://github.com/ntquang-0410/eda_ecom.git
# hoặc nếu đã có repo:
git fetch origin && git checkout nhatanh_updating && git pull origin nhatanh_updating
```
Các file cần chú ý:
- Tài liệu: `HUONG_DAN_JUDGE.md` (file này), `CLAUDE.md` (ghi chú dự án), `WALKTHROUGH.md` (nhật ký toàn bộ các lượt bàn, **tùy chọn**, chỉ tra lịch sử, tìm theo "Prompt 6xx").
- Mã: toàn bộ `scripts/judge/` và `scripts/ft_clean/` (gồm `luat_ft.py`, `test_ft_clean.py`, `xay_ft_clean.py`, `xuat_ban_pair.py`, `README.md`).
- `.gitignore` đã thêm các thư mục kết quả của judge (`data/processed/so_cap/`, `bo_chon/`, `ft_clean/`, `judge/`, `song_ngu_*/`) để không vô tình commit dữ liệu.
Thư mục gốc của repo chính là "thư mục gốc" mà các đường dẫn tương đối trong tài liệu này nói tới (các `.py` tự suy ra từ vị trí file).

### 15.2 Dữ liệu CÓ trên HuggingFace (công khai, tải không cần token)
Dataset `ntquang0410/zh-vie_ecom`. URL tải: `https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/resolve/main/<đường dẫn trên HF>`.

| Đường dẫn trên HF | Lưu thành (trong repo) | SHA256 |
|---|---|---|
| `data/LLM/1688/title_translations/best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet` | `data/snapshot/title_candidates/1688_title_best_cometkiwi.parquet` (**đổi tên**) | `6b4f504f409ea2b1c6d33ce66d7d5e3b3b5825309e071c3bf8901646b9bca93c` |
| `data/LLM/tiki/title_translations/best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet` | `data/snapshot/title_candidates/tiki_title_best_cometkiwi.parquet` (**đổi tên**) | `9b5af6aa02649506f2422c04fa83f8de81110e7609f4bad1d02ed08e4bd05b6e` |

**Bắt buộc** cho judge: hai file tiêu đề ở trên (có cả bản Gemma, bản Qwen và điểm CometKiwi của từng sản phẩm; điểm Kiwi **không** đưa vào lời nhắc).

Lệnh tải (PowerShell, từ thư mục gốc repo):
```
New-Item -ItemType Directory -Force data\snapshot\title_candidates | Out-Null
$hf = "https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/resolve/main"
Invoke-WebRequest "$hf/data/LLM/1688/title_translations/best_cometkiwi/1688_zh_to_vi_best_cometkiwi.parquet" -OutFile data\snapshot\title_candidates\1688_title_best_cometkiwi.parquet
Invoke-WebRequest "$hf/data/LLM/tiki/title_translations/best_cometkiwi/tiki_vi_to_zh_best_cometkiwi.parquet" -OutFile data\snapshot\title_candidates\tiki_title_best_cometkiwi.parquet
Get-FileHash data\snapshot\title_candidates\*.parquet -Algorithm SHA256
```
(Linux/Mac: dùng `curl -L -o <đích> <url>` và `sha256sum`.)

**Chỉ cần nếu tự dựng lại `ft_buoc4_*` (mục 15.4), không cần cho judge:** bộ official v2 do Huy gộp (commit `4a23574`), thư mục `data/LLM/bilingual_official/` trên HF, lưu vào `data/snapshot/bilingual_official_v2/` **cùng tên file**:

| File (cùng tên trên HF và khi lưu) | SHA256 |
|---|---|
| `bilingual_official_context.parquet` | `8fd10827c9e21ab2cffc645365bf5bfd22ab387c1896f5ad2cfdb90ed1129975` |
| `bilingual_official_pair.parquet` | `5feec0cf2c2799845b7b3d7aaa4e8f337529264d3e2061009acb65f49ff66936` |
| `excluded_products.parquet` | `7c1587e90a840c934bb63843a00ef0159f6f3ef5e22f75f9ff4c87bd76ef9819` |
| `review_queue.parquet` | `a3657fe198b757fcf3d53607fa01ed91752504a665d695ebc11e8723faf1e117` |
| `quality_filter_audit.parquet` | `379655476a17405e8e82173c3e63ca8364d6d8b175d49a9a66673ac45ca7a426` |

Tiêu chí gộp Tiki vào 1688 của Huy nằm trong chính thư mục đó: `data/LLM/bilingual_official/criteria.json`, `README.md`, `selection_report.json`, `manifest.json`, `SHA256SUMS.txt`. **Nên đọc để hiểu bộ official** (judge không thay đổi cấu trúc này).

### 15.3 Dữ liệu CHỈ nằm trên máy Nhật Anh (không có trên HF): Nhật Anh gửi riêng (Drive/zip), **không** đưa lên HF/GitHub
Đặt đúng đường dẫn tương đối trong repo. Mã băm ở Phụ lục C.

| File | MB | Bắt buộc? |
|---|---|---|
| `data/processed/ft_clean/ft_buoc4_giu.parquet` (nền bản cụm, 25.575 sản phẩm) | 12,9 | Có (hoặc tự dựng, 15.4) |
| `data/processed/ft_clean/ft_buoc4_giu_pair.parquet` (nền bản pair) | 11,9 | Có (hoặc tự dựng, 15.4) |
| `data/processed/song_ngu_4/cap_cum_20261026_2123.parquet` (cụm 1688: hai ứng viên từng cặp) | 32,9 | Có |
| `data/processed/song_ngu_3/so_sanh_cum_20261024_1616.parquet` (cụm Tiki: hai ứng viên) | 8,6 | Có |
| `data/processed/song_ngu_3/cap_cum_20261024_1616.parquet` (cụm Tiki: các cặp đã phát hành) | 2,6 | Có |
| `data/processed/song_ngu_1/don_vi_da_chon_20261022_2126.parquet` (pair 1688: hai ứng viên từng đơn vị) | 12,5 | Có |
| `data/processed/song_ngu_1/bo_song_ngu_1_day_du_20261022_2126.parquet` (pair 1688: từng cặp của từng sản phẩm) | 18,1 | Có |
| `data/processed/song_ngu_2/don_vi_da_chon_20261023_1656.parquet` (pair Tiki) | 3,9 | Có |
| `data/processed/song_ngu_2/bo_song_ngu_2_day_du_20261023_1656.parquet` (pair Tiki) | 4,5 | Có |
| `data/processed/so_cap/ket_qua_m2_v2_{title,cum_ctx,pair}_150.jsonl` (3 file, ~0,5 MB; đối chứng cho thí điểm B6) | 0,5 | Nên có |
| `data/processed/so_cap/ket_qua_m2_v1_*_150.jsonl`, `ket_qua_b_*_150.jsonl`, `so_sanh_quy_tac.json`, `so_sanh_gom.json` | nhỏ | Tùy chọn (tham khảo) |

(Huy có thể **đã có** các file `song_ngu_*` vì cùng nhóm tạo ra; nếu có thì chỉ cần đối chiếu SHA256 ở Phụ lục C. Mã băm **khác** nghĩa là dữ liệu khác bản đã dùng: báo Nhật Anh trước khi chạy.)

### 15.4 Tùy chọn: tự dựng `ft_buoc4_*` từ bộ official trên HF thay vì nhận file
Có thể dựng lại từ các file official ở 15.2 bằng `scripts/ft_clean/xay_ft_clean.py` rồi `xuat_ban_pair.py` (xem `scripts/ft_clean/README.md`). **Chưa thử trên máy khác**; nếu số dòng khác 25.575 hoặc khác mã băm Phụ lục C thì dùng file Nhật Anh gửi.

### 15.5 Không cần gửi / không đưa lên
Các bảng `muc_*.parquet` (Huy tự dựng ở B2), `bo_chon/` (chưa có kết quả lớn), mọi **khóa API**.

---
## 16. Việc còn mở / cần xác nhận

Đã giải quyết: API của Huy là **Alibaba Cloud Model Studio** như của Nhật Anh (mục 9.1).

Còn mở:
1. **Hạn mức thật của gói của Huy** (15 triệu token tính theo token hay Credits) và **điều khoản dịch vụ** cho việc chạy hàng loạt (mục 9.3): Huy đối chiếu đồng hồ của gói sau thí điểm B6.
2. **N pair** (mặc định 5.000; tăng nếu còn ngân sách) và chính sách mục tương đương (mặc định `giu_hien_tai`): Nhật Anh đã đồng ý hướng này; muốn đổi thì đổi **trước** khi chạy.
3. **Đích lưu bộ cuối trên HF** (khi Nhật Anh ra lệnh đẩy): tên thư mục/nhánh trong `ntquang0410/zh-vie_ecom` chưa chốt. **Không đẩy** đến khi Nhật Anh ra lệnh.
4. **381 sản phẩm đã loại ở bước làm sạch:** mặc định vẫn loại (Nhật Anh đã duyệt). Muốn giữ đủ 25.956 sản phẩm của Huy thì cần chỉnh bước ghép.
5. **Không có người chấm kiểm lại:** cân nhắc dùng một mô hình judge thứ hai (ví dụ `deepseek-v4-pro`, có trong danh sách của gói) chấm lại ~300 mảnh để xem mức đồng thuận (chưa làm).

---
## Phụ lục A. Lời nhắc hệ thống chính xác (rubric v2, một mảnh)

(Trong mã: `so_cap_rubric.HE_THONG_V2`; bản có ngữ cảnh là `HE_THONG_V2_NGU_CANH`.)

```
You are an impartial reviewer of Chinese-Vietnamese e-commerce translations. You will see one SOURCE text and two candidate translations, labelled Translation 1 and Translation 2. You must decide which one is better. You may not answer "tie": always choose 1 or 2.

Work in this order and base the decision ONLY on what you find.

STEP 1. List the defects of each translation. Check every key element of the SOURCE one by one (product type, each attribute, each number and unit, brand, model code). Do not claim an omission or an addition unless you can point to the exact words. A defect is one of:
 - WRONG: a wrong or reversed meaning (wrong product type, colour, material, quantity or function), a wrong number, wrong unit conversion or wrong code, or a brand, model or product name turned into a DIFFERENT name (an invented or incorrect transliteration).
 - MISSING: information of the SOURCE that is absent (a feature, attribute, number, brand or code).
 - ADDED: information that is not in the SOURCE and is not supported by the PRODUCT CONTEXT (a new feature, material, product type or nuance). Grammar words, measure words and standard attribute wording are not additions.
 - BROKEN: leaked instructions, placeholder or garbled text, a source word rendered in a wrong language (for example an English word where the target language is required, when the SOURCE word was not already in that foreign form), or unreadable text.
 - TERM: a domain term that is clearly wrong or non-standard for this product category.

STEP 2. The translation with FEWER defects wins. If the defects differ in seriousness, WRONG and BROKEN outweigh MISSING and ADDED, which outweigh TERM.

STEP 3. These differences are NEVER defects and NEVER a reason to prefer a translation:
 - keeping a Latin-script brand, model or product name as in the SOURCE, or writing the SAME brand in the target script (transliteration, official local brand name);
 - keeping or translating ordinary foreign (for example English) words that already appear in that foreign form in the SOURCE and are not names or codes;
 - synonyms of equal standing, including a shortened and a full form of the same term, and equally standard attribute labels;
 - unit and number formatting (for example L and a local unit word, kg and kilogram, decimal point and decimal comma, spacing, punctuation, slashes, brackets);
 - word order that keeps the meaning, and being more or less formal;
 - naturalness or fluency, unless one translation is ungrammatical or hard to read;
 - length, and whether a translation is listed first or second.

STEP 4. If after steps 1 to 3 the two translations are equivalent, set "equivalent" to true and choose Translation 1. Otherwise set "equivalent" to false and choose the translation with fewer or lighter defects.

Reply with ONE JSON object and nothing else, with the fields in exactly this order:
{"defects_1": ["short phrases, at most 4"], "defects_2": ["short phrases, at most 4"], "equivalent": true or false, "winner": 1 or 2, "criterion": "meaning|completeness|numbers_codes|terminology|fluency|none", "reason": "at most 20 words"}
```

Phần **khác** của bản **gom** (`so_cap_gom.HE_THONG_GOM`): câu mở đầu đổi thành "You will see several numbered ITEMS ... Judge every ITEM independently of the others." và khối trả lời cuối là:

```
Reply with ONE JSON object and nothing else: {"results": [ ... ]} with exactly one entry per ITEM, in the same order. Each entry has the fields in exactly this order: {"i": the ITEM number, "d1": ["defects of Translation 1, at most 2, at most 6 words each"], "d2": ["defects of Translation 2, at most 2, at most 6 words each"], "eq": true or false (equivalent), "w": 1 or 2 (the winner), "c": "W|M|A|B|T|N" (the kind of the most serious defect of the losing translation: W wrong, M missing, A added, B broken, T term, N none)}.
```

Lời nhắc người dùng (mỗi mảnh) có dạng:

```
ITEM 3 | text type: one product attribute written as 'name: value' | product category: shoes
[PRODUCT CONTEXT (Chinese, for understanding only): ... — chỉ có ở mảnh có ngữ cảnh]
SOURCE (Chinese): 图案: 纯色
Translation 1 (Vietnamese): ...
Translation 2 (Vietnamese): ...
```
(thứ tự hai bản của từng mảnh theo `thu_tu_bang_bam(item_id)`; `item_id` không xuất hiện trong lời nhắc.)

## Phụ lục B. Cách chống thiên vị (tóm tắt kỹ thuật)

- **Mù:** hàm dựng lời nhắc chỉ đọc đúng các trường trong `DANH_SACH_TRUONG`; có test kiểm không rò rỉ (`test_so_cap.py`).
- **Vị trí:** thiên vị vị trí đo được ~63–70%. Ở chế độ một lượt, thứ tự ngẫu nhiên theo `md5("thu_tu:" + item_id)` (tách khỏi cách chia dev/test để không trùng; đã từng gặp lỗi trùng băm).
- **Tương đương → bản 1:** vì thứ tự ngẫu nhiên, chọn bản 1 khi tương đương là đồng xu công bằng.
- **Độ dài:** không thấy thiên vị độ dài vượt mức nền (Gemma dài hơn ở ~60% mục).
- **Không mặc định thiên vị:** không có luật "hòa thì lấy Gemma".

## Phụ lục C. Mã băm để đối chiếu

**SHA256 các file dữ liệu đầu vào:**
```
bf86f7134672331be041f83fe39ece686eae4137de0098fde028663e775da27b  data/processed/ft_clean/ft_buoc4_giu.parquet
1e7173db218a29a709203655f976d715d730db733f08bf0d34e0942eab3a8d1b  data/processed/ft_clean/ft_buoc4_giu_pair.parquet
6b4f504f409ea2b1c6d33ce66d7d5e3b3b5825309e071c3bf8901646b9bca93c  data/snapshot/title_candidates/1688_title_best_cometkiwi.parquet
9b5af6aa02649506f2422c04fa83f8de81110e7609f4bad1d02ed08e4bd05b6e  data/snapshot/title_candidates/tiki_title_best_cometkiwi.parquet
cae8eeff62fb099ba157fd2cc0bf5f51e197f04c993eca677baa9d211331ac5c  data/processed/song_ngu_4/cap_cum_20261026_2123.parquet
2e3311ee2fb1e4144b4c06dd60758fa4fd0b2198b8faa7a34c635d40de927e68  data/processed/song_ngu_3/so_sanh_cum_20261024_1616.parquet
4541039e4ec161a6c331c3e7ea30e6b9ce3c1cb7ab6ed80327b2aab69f54d64c  data/processed/song_ngu_3/cap_cum_20261024_1616.parquet
5a79bb8c8f77f65a3018165542bea736e628031f71ae1cad4547beaf43b298a4  data/processed/song_ngu_1/don_vi_da_chon_20261022_2126.parquet
8c9f0458424790ff0b4345ebe5fa6d0271bc75711950d7be70d74f619bd53494  data/processed/song_ngu_1/bo_song_ngu_1_day_du_20261022_2126.parquet
779c05183a7f726a399e59a452f024d27228fc2a0c143f43eefbcb247955a73c  data/processed/song_ngu_2/don_vi_da_chon_20261023_1656.parquet
acef46a9c947013fdd20e04e3e7b5d953673e9696a4b05803c9323642372f534  data/processed/song_ngu_2/bo_song_ngu_2_day_du_20261023_1656.parquet
```
**Bảng mục dựng lại bằng `so_cap_du_lieu.py`** (số dòng | md5 của `item_id` đã sắp xếp, ghép bằng ký tự xuống dòng `\n`):
```
muc_title        25374 | 7e13d6e619eac883ebe8b53fe9595112
muc_cum           4446 | 4025f3f28dfbe7bb748893ea6485e8c1
muc_cum_ctx      17170 | 343b05872c074f3d901e8d3f326bbdb8
muc_cum_lai       4446 | cc266bfc5394359e556599e6de3c687e
muc_pair         34534 | 3485e426dfe3409f650f1c4708adb594
muc_title_ctx    25374 | 7e13d6e619eac883ebe8b53fe9595112
muc_pair_ctx     34534 | 3485e426dfe3409f650f1c4708adb594
```
Ghi chú: `muc_title_ctx` trùng id với `muc_title`; `muc_pair_ctx` trùng id với `muc_pair`. Môi trường tham chiếu: Python 3.13.7, pandas 3.0.5, numpy 2.5.3, pyarrow 25.0.1 (mã băm theo `item_id` không phụ thuộc phiên bản).

## Phụ lục D. Các lỗi đã gặp và đã sửa (để không lặp lại)

| Lỗi | Hậu quả | Sửa |
|---|---|---|
| Ứng viên rỗng bị đưa vào bảng mục (Qwen không dịch cặp `dich_moi`) | 23% mục cụm có ngữ cảnh "thắng Gemma" tầm thường | `co_chu`, `co_noi_dung`: ứng viên rỗng không phải cuộc đấu |
| Đơn vị pair đếm/ghép theo chữ, bỏ qua nhóm hàng | `san_pham_n` đếm lặp (769.654 so 195.311 mảnh thật); độ phủ top N báo sai (95% thật ra ~84%); ghép chỉ lấy biến thể đầu | Đếm, chọn đại diện, ghép theo (chữ, nhóm hàng) |
| Băm thứ tự trùng băm chia dev/test | Mô phỏng một lượt: dev toàn Gemma trước | Băm thứ tự có muối `thu_tu:` + test |
| `return` trong `try` của launcher làm thoát mã 0 khi lỗi | Lỗi nặng bị che | Dùng `exit 2` |
| Hàm PowerShell trả cả các dòng log | Mã thoát sai | `Ghi` chỉ `Write-Host`, không đưa vào pipeline |
| Quy tắc v2 mâu thuẫn (BROKEN so với chính sách từ ngoại ngữ) | Judge lẫn | Chính sách chỉ áp cho từ đã là ngoại ngữ ở nguồn |
| Nhận định "bản hiện tại gần như toàn Gemma (~95%)" áp cho cả tiêu đề | Sai: tiêu đề do CometKiwi chọn (Gemma 56%, Qwen 44%); chỉ ở mức đơn vị pair mới nghiêng Gemma rõ | Sửa mô tả; mục 5.4 và D8 đã nói rõ |

## Phụ lục E. Cấu trúc mã và thư mục

```
e-commerce - EDA/
  HUONG_DAN_JUDGE.md  CLAUDE.md  WALKTHROUGH.md
  scripts/
    ft_clean/luat_ft.py  test_ft_clean.py
    judge/
      cau_hinh.py          # cấu hình mô hình + nhà cung cấp (thêm "huy" ở đây)
      chay_judge.py        # gọi API (goi_api), --kiem-tra, khóa từ biến môi trường
      rubric.py            # tiện ích đọc JSON dùng chung
      so_cap_rubric.py     # LỜI NHẮC v1/v2, chính sách, đọc kết quả, băm thứ tự
      so_cap_gom.py        # chế độ gom
      so_cap_du_lieu.py    # dựng bảng mục (title, cụm, cụm ngữ cảnh, cụm lai, pair, tầng 2)
      so_cap_chon.py       # chạy judge: --loai, --gom, --top, --rubric, --che-do, --bao-cao
      ghep_bo.py           # ghép hai bộ cuối + file đánh dấu + Cách 1
      so_sanh_*.py         # các phép so/ đo (quy tắc, gom, ngữ cảnh, tầng 2)
      chay_lon.ps1  thu_gom.ps1  thu_quy_tac.ps1  do_lai_quy_tac.ps1  thu_so_cap.ps1  thu_cum_ngu_canh.ps1  thu_tang2.ps1
      test_*.py
  data/
    processed/ft_clean/  ft_buoc4_giu.parquet  ft_buoc4_giu_pair.parquet
    processed/song_ngu_1..4/ ...   snapshot/title_candidates/ ...
    processed/so_cap/    (kết quả và bảng mục sinh ra ở đây)
    processed/bo_chon/   (bộ cuối sinh ra ở đây)
```
Mỗi lần chạy sinh log: `data/processed/so_cap/log_*.txt` (không chứa khóa), kết quả `ket_qua_<tên>.jsonl` (mỗi dòng một mảnh, có `ket_qua`, `luot[0].tuong_duong`, `tokens_vao/ra`, `gom`), tóm tắt `tom_tat_<tên>.json`.
