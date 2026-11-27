# Audit EcomGPT-7B B — 18/27 hyp rỗng trên nhóm 858

**Câu hỏi:** 18/27 rỗng là bug harness hay degenerate thật?

**Kết luận: degenerate thật (model), không phải bug harness; và một phần là hiệu ứng chọn mẫu của 831.**

## 1. Số liệu

| Nhóm (627 dòng Tiki) | Rỗng | Tỉ lệ |
|---|---:|---:|
| Attribute | **21/224** | 9,4% |
| Description | 0/112 | 0% |
| Title | 0/291 | 0% |
| **27 dòng bổ sung** | 18/27 | 66,7% — **toàn bộ là attribute** |

9 dòng description của nhóm bổ sung (dài 3,0–3,7k ký tự) **đều có hyp**; 18 dòng attribute (ngắn,
median 20 ký tự) **đều rỗng**.

## 2. Loại trừ nguyên nhân harness

| Giả thuyết | Bằng chứng phủ định |
|---|---|
| `max_seq` / truncation | Dòng rỗng **ngắn nhất** (4–1.189 ký tự, median 18), không phải dài; 279 attribute khác cùng độ dài vẫn có hyp |
| decoding config (max_new/EOS) | EcomGPT **A** chạy cùng code, cùng template, cùng `max_length=1024` → **0 rỗng** trên đúng 27 dòng |
| prompt template | Template giống hệt nhóm attribute không rỗng |
| post-processing `split("###")` | Áp dụng như nhau cho A và B; A không rỗng ⇒ không phải nguồn khác biệt |

## 3. Vì sao B rỗng trên attribute

- B được train trên **title + description** (`1688_title`, `1688_description`, `tiki_title`,
  `tiki_description`) — **không có group attribute**.
- Attribute là phân bố ngoài miền train (các fragment kiểu "A50s", "Nôi Z1", "màu ghi xám");
  model B thường phát EOS ngay → hyp rỗng.
- Đây là hành vi model (OOD), không phải lỗi cấu hình; **B-natural** còn thoái hoá toàn cục
  (1.471/1.500 rỗng, lặp `...`/`* * *`) — mức độ nặng hơn, cùng bản chất suy biến.

## 4. Lưu ý về chọn mẫu 831

27 dòng bổ sung của 858 chính là các dòng **từng có ít nhất một cell rỗng** trong lần chạy 1.500 dòng.
Với EcomGPT B, 18 dòng đó là **chính các dòng nó fail** — vì vậy "18/27" không phải phát hiện độc lập
mà là hệ quả của cách chọn 831 cũ. Đây là lý do phải chuyển sang 858 (đã làm).

## 5. Khuyến nghị

1. Giữ nguyên cell EcomGPT B (balanced) và báo cáo **theo stratum** — attribute là điểm yếu thật.
2. Không "sửa" bằng harness; nếu muốn cải thiện, thử prompt có ngữ cảnh ("Thuộc tính sản phẩm: …")
   hoặc thêm attribute vào train — đó là thí nghiệm mới, không phải bug fix.
3. EcomGPT B-natural vẫn **FAILED**, không dùng để so sánh.
