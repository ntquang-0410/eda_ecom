# Các điểm cần Nhật Anh chốt

Lập 12/11/2026. Mỗi điểm ghi: câu hỏi, đề xuất của em, và vì sao chưa tự quyết được. Chưa điểm nào được chốt. Nhãn: **[CHỜ]** chờ Nhật Anh; **[CHƯA KIỂM]** chưa kiểm trên máy thật.

## A. Chọn mô hình cho bản cuối
1. **[CHỜ] Chạy công thức cuối (rank 64, 2 epoch, `w_title`) cho Qwen3-8B bản cụm không?** Đề xuất: có, ước 5–6 giờ RTX Pro 6000 [ƯỚC]. Lý do: cụm 8B hơn 4B +1.03 [+0.71; +1.36]. Bản pair giữ 4B vì 8B không hơn rõ.
2. **[CHỜ] Extension dùng định dạng nào: pair (dịch từng dòng "tên: giá trị") hay cụm (cả mô tả nối bằng "; ")?** Ảnh hưởng thẳng đến việc chọn 8B hay 4B. Đề xuất: quyết sau khi xem cách extension lấy chữ từ trang 1688 (từng ô riêng hay cả khối).
3. **[CHỜ] Có đưa mô hình cổ điển (OpenNMT/Moses) vào demo làm phương án dự phòng nhẹ không?** Ở pair, Moses khớp 4M token ngang 4B. OpenNMT chạy được trên CPU.

## B. Cách đưa mô hình vào extension
4. **[CHỜ] Mô hình chạy ở đâu khi demo?** (a) máy chủ GPU của nhóm hoặc Kaggle/Colab mở đường hầm; (b) chạy cục bộ trên laptop sau khi lượng tử hóa; (c) chỉ dùng OpenNMT trên CPU. Laptop có RTX 3050 (VRAM thường 4–6 GB [CHƯA KIỂM]) nên 4B bf16 (~9–10 GB) không vừa.
5. **[CHỜ] Ai viết phần extension và máy chủ?** Repo EDA hiện không có mã extension nào.
6. **[CHỜ] Cách bắt chữ và chèn bản dịch lên trang 1688** (thay chữ tại chỗ hay hiện lớp phủ; gom theo ô hay theo khối).
7. **[CHỜ] Xử lý mô tả dài hơn 2.048 token** (16–20% mô tả vượt 960 token): cắt theo thuộc tính rồi dịch từng đoạn?

## C. Chất lượng và bằng chứng
8. **[CHỜ] Đo test một lần cho mô hình được chọn.** Bản cuối trên toàn pool (07/11) chưa đo lại trên test.
9. **[CHỜ] B9: chấm so cặp bằng LLM** (Gemini chính, Claude hiệu chuẩn). Tham chiếu của mọi điểm hiện có là bản dịch máy, chưa có người hay LLM-judge kiểm bản dịch của mô hình.
10. **[CHỜ] Sửa title thật (cách 2) cho bản v2 không?** Bản cuối loại 6.386/23.017 title bị cờ; tham chiếu title có lỗi lõi ~5–14%.
11. **[CHỜ] Quang xác nhận cấu hình vi→zh** và nhóm quyết bản cụm có cần mô hình cổ điển huấn luyện riêng không.

## E. Hướng đi sau bảng kết quả năm mô hình
16. **[CHỜ] Dừng huấn luyện thêm để demo và báo cáo, hay tiếp tục tinh chỉnh?** Đề xuất của em: dừng huấn luyện siêu tham số. Lý do: ba mô hình đầu bảng ở pair chênh nhau trong khoảng nhiễu, chrF chạm trần vì tham chiếu là bản dịch máy, và điểm nghẽn thật là chất lượng title, kiểm bằng người/LLM và dựng extension. Chi tiết trong câu trả lời ngày 12/11.
17. **[CHỜ] Bộ thước đo dùng để kết luận:** giữ chrF làm thước chọn nhanh, thêm kiểm số/đơn vị/thương hiệu (đã có) và B9 (LLM chấm so cặp). COMET là tùy chọn phụ. Không dùng BLEU làm thước chính.
18. **[CHỜ] Nếu muốn cải thiện chất lượng, ưu tiên sửa dữ liệu title (cách 2, bản v2) trước khi huấn luyện thêm.** Title chrF chỉ ~50–57 so với mô tả ~83–91.

## D. Việc dọn
12. **[CHỜ] Commit mã** (`scripts/ft_train/`, `docs/`; chưa commit gì).
13. **[CHỜ] Thu hồi token Kaggle tài khoản 2** khi xong chuỗi chạy GPU.
14. **[CHỜ] Xác nhận tên tác giả ghi trên các notebook tài khoản 2** (Hoàng Huyền Anh).
15. **[CHỜ] Sửa `Phan1_Mo_dau_TLCN.docx` mục 3–4 và `BAO_CAO_FT_QWEN3_4B.md`** (sao lưu trước; từ chối sửa nếu có tệp khóa `~$`).
