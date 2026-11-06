# Dữ liệu song ngữ sau judge

Thư mục này gồm bốn file Parquet, mỗi file 25.575 sản phẩm. Nguồn 1688 được dịch Trung → Việt; Tiki được dịch Việt → Trung. Hai bản cùng tập sản phẩm và thứ tự.

| File | Nội dung |
|---|---|
| `bo_cum_chon_deepseek.parquet` | Title và description bản dịch theo cụm sau luật, judge và rà soát. |
| `bo_pair_chon_deepseek.parquet` | Title và description bản dịch theo đơn vị pair sau luật, judge và rà soát. |
| `bo_cum_danh_dau.parquet` | Trạng thái và số lượng quyết định judge theo sản phẩm của bản cụm. |
| `bo_pair_danh_dau.parquet` | Trạng thái và số lượng quyết định judge theo sản phẩm của bản pair. |

Hai file dữ liệu có 9 cột: `product_id`, `source`, `category`, `title_zh`, `title_vi`, `description_zh`, `description_vi`, `judge_chon_title`, `judge_chon_description`.

`judge_chon_title` là model tạo bản title cuối được giữ (`gemma` hoặc `qwen`), không nhất thiết là lựa chọn trực tiếp của DeepSeek. Hai title đã được sửa tay; model thể hiện nguồn bản nháp.

`judge_chon_description` đếm nguồn nội dung cuối: `gemma`, `qwen`, `quy_tac`, `chua_ro`; nhóm bằng 0 được ẩn. Đây không phải điểm hay số lần judge chọn. Cụm đếm một cặp tên–giá trị là một mảnh; pair đếm tên và giá trị không rỗng riêng. Không so trực tiếp số đếm giữa hai bản. `chua_ro` biểu thị giới hạn thông tin nguồn, không tự kết luận dịch sai; trong bản cụm còn bao gồm mảnh có tên và giá trị từ hai model khác nhau.

DeepSeek đã chấm 4.439/4.446 mục cụm; bảy mục bị nhà cung cấp chặn đã được trợ lý GPT-6.1-sol rà riêng (hai chọn Gemma, bốn tương đương, một chưa chắc chắn). Bản pair chấm 5.000/34.534 đơn vị tranh chấp; các đơn vị chưa chấm giữ bản trước theo pipeline. Các cờ trong file đánh dấu phản ánh trạng thái judge, không chứng nhận ngữ nghĩa. Bộ dữ liệu chưa được người kiểm chứng toàn bộ và không nên coi là tập gold đã kiểm nghĩa.

Các file truy vết chi tiết, metadata cũ và báo cáo được lưu ngoài bản hiện tại. Có thể xem phiên bản trước khi thu gọn tại [revision 126a564f88c72ac7feb2d4cc79e7183348f5aad9](https://huggingface.co/datasets/ntquang0410/zh-vie_ecom/tree/126a564f88c72ac7feb2d4cc79e7183348f5aad9/data/LLM/bilingual_judged); lịch sử repository vẫn chứa các file đó. `manifest.json` ghi số dòng và SHA-256 của bốn file dữ liệu và README.
