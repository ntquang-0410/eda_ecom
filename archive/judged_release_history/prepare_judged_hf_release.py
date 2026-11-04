from pathlib import Path
import pandas as pd,json,hashlib
p=Path(r'D:\download\NCKH\judge_kaggle_preflight\corrected_reviewed_release');q=json.loads((p/'quality_signals_summary.json').read_text());m=pd.read_parquet(p/'bo_pair_danh_dau.parquet');total=int(m.mo_ta_so_manh_tranh_chap.sum());un=int(m.mo_ta_so_manh_chua_cham.sum());q['pair_usage_coverage']=(total-un)/total;q['pair_disputed_occurrences_total']=total;q['pair_disputed_occurrences_judged']=total-un;q['pair_disputed_occurrences_unjudged']=un;q['coverage_basis']='Actual merged product-fragment marking counts, not sum of san_pham_n';pending=pd.read_parquet(p/'pending_judge_items.parquet');q['pending_context_product_examples_unique']=int(pending.vi_du_product_id.nunique());(p/'quality_signals_summary.json').write_text(json.dumps(q,indent=2),encoding='utf8')
text=f'''# Bilingual judged — bản đã rà soát, còn các mục chưa được chấm

## Trạng thái và phạm vi
Hai bộ có **25.575 sản phẩm**, mỗi bộ **17 cột**, cùng tập sản phẩm, cùng thứ tự và cùng title. Nguồn 1688 là Trung → Việt; Tiki là Việt → Trung. Description context được dịch theo ngữ cảnh thuộc tính; pair dịch riêng từng tên/giá trị. Không dùng cả hai bản của cùng sản phẩm ở hai phía train/test.

Đây là bản sau DeepSeek và sửa có đối chiếu nguồn, **không phải dữ liệu đã được người xác minh toàn bộ ngữ nghĩa**. Không gọi dữ liệu này là gold/reference test set đã kiểm nghĩa. Không có dòng sản phẩm nào bị loại thêm trong bước sửa này.

## Phạm vi đã được judge
- Model `deepseek-v4.1-flash`, Alibaba Token Plan Team; rubric v2, một lượt, che tên Gemma/Qwen và điểm COMETKiwi, gom yêu cầu theo mã gốc. Không dịch lại bằng một model khác.
- Title: **25.374/25.374 mục tranh chấp thành công**, không thiếu/thừa/trùng item_id. Các title không tranh chấp không cần chấm.
- Context: **4.439/4.446 mục thành công**, còn **7 mục lỗi**. Context lai dùng ngữ cảnh cho cặp dùng bởi một sản phẩm, không ngữ cảnh cho cặp được nhiều sản phẩm dùng.
- Pair: chấm **5.000/34.534 đơn vị tranh chấp** theo tần suất sử dụng; **29.534 đơn vị không được gửi judge**. Đo trên file đánh dấu sau ghép: đã chấm **{total-un:,}/{total:,} lượt mảnh sản phẩm ({(total-un)/total:.2%})**, còn **{un:,} lượt chưa chấm**. Không tính mức phủ bằng cộng san_pham_n vì khác đơn vị đếm.
- Judge khai tương đương thì giữ bản hiện tại. Mảnh chưa chấm cũng giữ bản hiện tại. Không dùng COMETKiwi để phá hòa; không mặc định chọn Gemma cho mọi trường hợp.

## 7 mục context bị API chặn — chưa giải quyết bằng judge
Nhà cung cấp trả HTTP 400 `data_inspection_failed` khi kiểm tra đầu vào cho nhóm yêu cầu. Điều đó **không chứng minh từng mục trong nhóm đều có nội dung vi phạm hoặc bản dịch sai**. Phản hồi không chỉ ra từ gây chặn. Không thay prompt, không né bộ lọc, không tự dựng người thắng.

Bảy mục giữ nguyên bản dịch trước judge; các phần đã chấm khác của sản phẩm vẫn áp dụng bình thường. File `pending_judge_items.parquet` ghi item_id, nguồn, ID sản phẩm đại diện, số lượt sử dụng, trạng thái `unjudged_provider_blocked` và fallback. Có **{q['pending_context_product_examples_unique']} ID sản phẩm đại diện riêng biệt**, tổng 7 lượt dùng. Không được gọi kết quả này là chấm đủ 100% context. File đánh dấu gốc vẫn ghi các mảnh đó chưa chấm.

## Hai lỗi ngữ nghĩa đã sửa thủ công
1. Sản phẩm 1688 `41907586179`: bản judge chọn bỏ ý `8合1`; đã bổ sung **8 trong 1** vào title Việt, giữ SNJUE8 và nội dung còn lại.
2. Sản phẩm Tiki `275246947`: `Xuất Nhật` bị đổi thành `日本产` (sản xuất ở Nhật); đã sửa thành cách diễn đạt **xuất sang Nhật**, không khẳng định xuất xứ Nhật. Những phần khác của title được giữ.

`manual_title_corrections.parquet` có nguyên văn trước/sau, lý do và model tạo bản nháp. Với hai dòng này, `title_nguon_dich` là model tạo bản nháp **trước sửa tay**, không có nghĩa model đã sinh nguyên vẹn bản cuối. File đánh dấu judge không bị viết lại để giả rằng DeepSeek thực hiện hai sửa đổi này. Hai sửa đổi đã áp dụng đồng bộ ở context và pair.

## Các nghi vấn còn tồn tại sau sửa — KHÔNG coi là lỗi đã xác nhận
Bảng `title_review_signals.parquet` cho tra theo source + product_id. Các nhóm có thể trùng nhau:
- **1.156 title 1688**, **805 title Tiki** có tập chuỗi chữ số khác title nguồn theo regex `digits + optional decimal part`. Tín hiệu này có thể đến từ số viết bằng chữ, 双 → 2, 03 → 3, ký hiệu kích thước, tên model hoặc sai số thực sự. **Không kết luận toàn bộ 1.961 dòng là dịch sai.** Regex này chưa quy đổi đầy đủ đơn vị, số viết bằng chữ và mọi cách biểu diễn số.
- **431 title Tiki** còn ký tự trong nhóm dấu tiếng Việt. Có thể là tên hãng/model, cũng có thể chưa dịch hết; chưa phân loại nghĩa toàn bộ. Bộ dò ký tự không bắt được mọi tiếng Việt không dấu.
- **16 title Tiki** không có ký tự Hán. Có thể là tên Latin/model hợp lệ hoặc thiếu dịch loại sản phẩm; chưa có kết luận từng dòng.
- Không phát hiện chữ Hán trong title Việt của 1688 theo dải U+4E00–U+9FFF ở bản này; đó là kiểm ký tự, không phải chứng minh đúng nghĩa.
- Judge vẫn có thể sai: hai ví dụ sửa tay ở trên được phát hiện trong mẫu chọn theo nguy cơ. Chưa có mẫu người chấm đại diện để ước lượng tỷ lệ lỗi toàn bộ. Không ngoại suy mẫu 12 dòng thành accuracy.

## Các giới hạn phương pháp
- Chỉ một thứ tự A/B cho mỗi mục ở lượt lớn; cách đặt thứ tự dùng băm cố định. Thiên vị vị trí và tính nhất quán không đồng nghĩa độ đúng. Trong lượt full, tỷ lệ chọn vị trí 1 trên các quyết định **không khai tương đương** khoảng 55,0% title, 55,3% context, 55,9% pair. Không so trực tiếp với số thí điểm có mẫu/mẫu số khác.
- Hai phương pháp có mức phủ và ngữ cảnh judge khác nhau; không quy mọi khác biệt chất lượng cuối thành tác động của cách dịch description.
- Cờ cũ là thông tin trên đầu vào trước judge, không tự tính lại thành chứng nhận bản sau judge đạt QA.
- Không tự sửa các nghi vấn còn lại bằng thay thế hàng loạt hoặc loại thêm sản phẩm.

## Schema 17 cột và metadata đã sửa
`product_id, source, category, title_zh, title_vi, description_zh, description_vi, title_nguon_dich, desc_nguon_dich_goc, huong_goc, khuon_id, nhom_mo_ta_id, can_hang_chac, mem_latin_la_ban_trung, mem_viet_sot_title_zh, mem_v2_review, desc_so_manh_da_doi`.

Bản pair bàn giao ban đầu thiếu 7 cột metadata và sau ghép chỉ có 10 cột. Bản này đã bổ sung đủ: hướng gốc; khuôn; nhóm description; cờ căn hàng; Latin lạ; dấu Việt ở title Trung; thành viên hàng đợi review v2. Tính trên **bản pair gốc trước judge**, tách thuộc tính bằng xuống dòng cho pair. Chỉ membership review v2 và hướng được giữ theo cùng sản phẩm; không sao chép mù khuôn/căn hàng của context sang pair. Đây là cờ heuristic; xuống dòng trong giá trị tự do vẫn có thể làm phép tách đoạn không hoàn hảo.

`khuon_id` dựa trên chuỗi nhãn nguồn; `nhom_mo_ta_id` dựa trên hai description gốc đã chuẩn hóa. Hai mã này đóng băng trước judge; nhóm ở hai phương pháp có thể khác. Chia tập cần gom các thành phần liên thông của các nhóm liên quan, kể cả title/nội dung trùng, không chỉ random theo product_id.

`desc_nguon_dich_goc` giữ nguồn dịch description **trước judge**, không phải nhãn nguồn chính xác cho từng mảnh trong description đã phối trộn. `desc_so_manh_da_doi` đếm mảnh judge đã đổi và khớp file đánh dấu. Muốn truy vết đầy đủ cần dùng file đánh dấu, checkpoint riêng và bảng sửa tay; không suy ra mọi mảnh được chấm từ số mảnh đã đổi.

## File và cách dùng
- `bo_cum_chon_deepseek.parquet`, `bo_pair_chon_deepseek.parquet`: dữ liệu chính.
- `bo_cum_danh_dau.parquet`, `bo_pair_danh_dau.parquet`: đánh dấu quyết định và mức phủ theo sản phẩm.
- `pending_judge_items.parquet`: 7 mục provider-blocked và fallback.
- `manual_title_corrections.parquet`: hai sửa đổi có đối chiếu nguồn.
- `title_review_signals.parquet`, `quality_signals_summary.json`: các nghi vấn kỹ thuật còn lại, không phải tập lỗi xác nhận.
- `validation_report.json`, `manifest.json`, `SHA256SUMS.txt`: kiểm tra và toàn vẹn file.

Ví dụ: `pd.read_parquet('hf://datasets/ntquang0410/zh-vie_ecom/data/LLM/bilingual_judged/bo_cum_chon_deepseek.parquet')`.

## Provenance và kiểm tra đã thực hiện
Đầu vào official v2: HF commit `4a2357499b1e352b300927d544aa7193e890c787`, sau bước ft_clean còn 25.575 sản phẩm. Mã judge: nhánh nhatanh_updating commit `4d6a8d05d3b7f99e21b0cf2372139ac8c6560846`. Input bàn giao và 5 file HF được kiểm SHA256; bảng item_id khớp hướng dẫn. Không đưa API key hoặc mã notebook chứa key vào gói phát hành.

Kiểm tra sau sửa: cùng ID/thứ tự giữa hai bản, ID duy nhất, không null, 17 cột, nguồn-language fields nguyên vẹn, category và description không bị sửa trong lượt khắc phục, bộ đếm thay đổi khớp đánh dấu, hai patch đúng cả hai bản, checksum toàn bộ file khớp. Những kiểm tra này không chứng nhận toàn bộ ngữ nghĩa. Bộ official hiện tại được giữ riêng, không ghi đè bởi lần phát hành này.
''';(p/'README.md').write_text(text,encoding='utf8');print('README written; pair occurrence coverage',q['pair_usage_coverage'])
files=[x for x in p.iterdir() if x.is_file() and x.name not in ['manifest.json','SHA256SUMS.txt']];manifest={'dataset_status':'reviewed_with_unjudged_items','rows_each':25575,'columns_each':17,'files':{x.name:{'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files}};(p/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');files.append(p/'manifest.json');(p/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+x.name+'\n' for x in sorted(files)),encoding='ascii')
