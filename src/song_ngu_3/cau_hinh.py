# Cấu hình bộ song ngữ 3: dịch CẢ CỤM thuộc tính của từng sản phẩm Tiki (Việt -> Trung), mỗi sản phẩm một lần gọi, để mô hình thấy
# các thuộc tính khác của cùng sản phẩm (anh ra lệnh 23/10/2026, sau khi bộ 2 dịch từng đơn vị lên HF). Thử trước 200 sản phẩm bằng
# hai mô hình (Gemma 4 qua API, Qwen3-14B trên Kaggle), so với bản dịch đơn vị của bộ 2 trên đúng 200 sản phẩm đó; cả cụm không hơn
# rõ thì không chạy cả bộ.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THU_MUC_BO_2 = ROOT / "data" / "processed" / "song_ngu_2"
FILE_BANG_CAP = THU_MUC_BO_2 / "bang_cap.parquet"                 # cặp của bộ 2 (đã làm sạch, có vi_tri, định tuyến, lý do cất)
NHAN_BO_2 = "20261023_1656"                                        # bản dịch đơn vị đem so (bản đã lên HF)
FILE_BO_2_DAY_DU = THU_MUC_BO_2 / f"bo_song_ngu_2_day_du_{NHAN_BO_2}.parquet"
THU_MUC_RA = ROOT / "data" / "processed" / "song_ngu_3"

SO_SAN_PHAM_THU = 200
HAT_GIONG = 20261023
# Một lần gọi tối đa ngần này ký tự Việt (tên + giá trị). Đo 23/10/2026 trên 13.187 sản phẩm: trung vị 174, 99% dưới 1.236, chỉ 9 sản
# phẩm quá 3.000. Sản phẩm dài hơn thì chia thành nhiều phần liên tiếp (mỗi phần kèm tên các thuộc tính còn lại làm ngữ cảnh);
# một thuộc tính một mình đã dài hơn thì đi riêng một phần.
MAX_KY_TU_PHAN = 2500
MAX_KY_TU_TEN_KHAC = 600     # ngữ cảnh "tên các thuộc tính ở phần khác" tối đa ngần này ký tự

TEN_DANH_MUC_ZH = {"food": "食品", "home": "家居用品", "beauty": "美妆护肤", "mother_baby": "母婴用品", "electronics": "数码电子",
                   "auto": "汽车用品"}
PHIEN_BAN_CAU_LENH = "cum-vi_zh-v1-2026-10-23"
