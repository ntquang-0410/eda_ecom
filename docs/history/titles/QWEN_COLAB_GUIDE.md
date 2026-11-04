# Hướng dẫn chạy full title bằng Qwen riêng trên Google Colab

## Đã chuẩn bị

- Notebook riêng: `title_translation_qwen35_colab.ipynb`.
- Bundle mã pipeline tương ứng: `title_translation_qwen35_colab_bundle.zip`.
- Model: `Qwen/Qwen3.5-9B`, tải trọng số công khai về runtime Colab và chạy local bằng Transformers 8-bit. Đây không phải Qwen3-14B/30B, cũng không gọi API suy luận.
- Nguồn: `ntquang0410/zh-vie_ecom`, file `data/cleaned/bilingual_zh_vi_cado.parquet`, khóa tại revision `e098c4dbd40a5ab1b4f2670f7d4b028aa4e09f88` và kiểm bằng SHA-256 trước khi dịch.
- Run folder riêng của Qwen: `MyDrive/title_translation_cado_qwen35_full_v1`. Không đọc hay ghi checkpoint Gemma.

## Cài đặt và bắt đầu

1. Tải cả notebook và bundle ở cùng thư mục project.
2. Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook** và tải `title_translation_qwen35_colab.ipynb` lên.
3. Chọn **Runtime → Change runtime type → GPU**. Chọn GPU có ít nhất 14 GiB VRAM; T4 16 GB trở lên là mục tiêu phù hợp. Nếu Colab không cấp GPU đủ bộ nhớ, notebook dừng trước khi tải model.
4. Chọn **Runtime → Run all**. Cho phép mount Google Drive. Khi được hỏi, tải lên `title_translation_qwen35_colab_bundle.zip`.
5. Notebook tự khóa phiên bản thư viện, tải đúng Parquet nguồn, chạy preflight 20 title rồi bắt đầu full 15.863 title duy nhất. Batch bắt đầu ở 4; nếu CUDA OOM thì tự giảm 2 rồi 1. Mỗi 100 title xử lý xong, SQLite checkpoint được mirror từ SSD runtime lên Drive.

Không cần Gemini key hoặc HF token. Không chỉnh tên `WORKDIR`, model, prompt, bundle hay batch size giữa các lần resume.

## Nếu Colab ngắt giữa chừng

1. Kết nối lại Colab; nếu runtime mới, chọn GPU cùng loại nếu có thể.
2. Chạy **Runtime → Run all** lại trên đúng notebook.
3. Notebook dùng cùng thư mục Drive, khôi phục SQLite gần nhất và bỏ qua các title đã hoàn tất. Có thể phải tải model weights lại. Nếu runtime bị dừng ngay trước lần mirror, tối đa khoảng 99 title cuối có thể được chạy lại.

Đổi GPU được cho phép khi phiên bản Python, thư viện và CUDA vẫn khớp với lock. Nếu các thư viện/CUDA nền tảng đã thay đổi, không ép trộn môi trường: hãy dừng và giữ nguyên checkpoint để xử lý riêng.

## Kiểm tra sau khi full hoàn tất

Khi `qwen35` có trạng thái `all_titles_complete`, chạy tiếp cell **Rà soát sau khi dịch xong**. Cell kiểm số dòng 16.348, tạo:

- `bilingual_zh_vi_cado_title_qwen35.parquet`: đủ các cột gốc và ứng viên Qwen, kèm cờ QA kỹ thuật.
- `qwen35_postrun_qa_all_unique_titles.csv`: QA của toàn bộ title duy nhất.
- `qwen35_postrun_qa_flagged_titles.csv`: các title có hard/advisory flag để xem trước.
- `qwen35_postrun_qa_summary.md` và `.json`: tổng hợp trạng thái/cờ và SHA-256.

QA được **hoãn đến sau khi dịch full** và không tự loại bản dịch. Các cờ chỉ phát hiện vấn đề kỹ thuật rõ ràng; chúng không đánh giá đầy đủ ngữ nghĩa. `title_vi_qwen35_semantic_verified` luôn là `False`. Sau đó cell cuối tải ZIP kết quả/checkpoint về máy. Không có bước upload HF tự động.

## Giới hạn cần biết

- T4/Colab không được bảo đảm cấp liên tục; thời gian full phụ thuộc GPU được cấp, độ dài title và thời hạn runtime. Hãy dự trù reconnect/resume; notebook giữ checkpoint trên Drive.
- Nếu OOM ngay cả batch 1, runtime sẽ dừng an toàn và giữ checkpoint gần nhất. Chọn GPU nhiều VRAM hơn rồi resume; không xóa run folder.
- Môi trường Colab managed không được điều khiển bằng SSH. Chạy các cell trong giao diện Colab; notebook không chạy dịch vụ nền và không publish dữ liệu ra Hugging Face.
