# Hướng tiếp theo — phân tích & kế hoạch

**Ngày:** 2026-11-08  
**Bối cảnh:** sau weight survey A/B (quota sampler), BT description 4 model và refactor/dọn repo.

---

## 1. Hiện trạng: cái gì đã chắc, cái gì còn tạm

| Kết luận | Mức độ chắc | Vấn đề |
|---|---|---|
| A giữ `natural` | Khá chắc | 1 seed; eval đã dùng lại nhiều lần |
| B chọn `balanced` | **Tạm** | Gain nhỏ (+0,0144); 1 seed; credit chưa đầy đủ |
| BT 4B description tốt nhất | Chắc | 93,5% QC ok sau retry |
| 8B description | Chưa đủ | Mới 11.900/13.200, thiếu 1.300 dòng dài nhất |
| Tiki không đổi theo weight | Chưa kết luận | COMET-QE gần 0/âm, độ phân giải thấp |

Số liệu chính:

- A: `natural` 0,856 COMET-da; mọi reweight giảm 0,005–0,016.
- B: `balanced` +0,0144 (CI [+0,0081; +0,0211]); `title_heavy` +0,0160 nhưng credit thấp hơn.
- BT description sau retry: QC ok 71,7% (0.6B) / 83,0% (1.7B) / 93,5% (4B) / 91,4% (8B).

---

## 2. Sáu khoảng trống cần xử lý

1. **Chưa có seed variance.** Toàn bộ survey mới seed 42. Gain B +0,0144 với CI ±0,006 là có ý nghĩa theo câu, nhưng chưa biết dao động giữa các lần train.
2. **Selection trên eval đã dùng nhiều lần.** `eval_A`/`eval_B` lấy từ test và đã dùng để chọn config. Cần holdout mới (từ dev) để xác nhận.
3. **Bootstrap theo `eval_id`, không theo `product_id`.** Eval A có 1,31 dòng/sản phẩm, B 1,14 → CI hiện có thể hẹp hơn thực tế.
4. **Credit mới phủ 58–68%** (A 580/1.000, B 680/1.000). Cần chấm nốt 740 dòng sau khi quota API reset.
5. **Chất lượng số liệu.** `number_mismatch` còn 4.192 dòng ở 4B (32%), 4.139 ở 8B. Với e-commerce, sai số/kích thước nguy hiểm hơn sai văn phong.
6. **A/B tách rời ≠ phân phối production.** Production có đủ title + attribute + description; chưa có thực nghiệm C gồm 6 nhóm.

---

## 3. Kế hoạch đề xuất

### P0 — Chốt độ tin cậy (1 ngày, ~10h GPU)

| việc | chi tiết | chi phí | tiêu chí |
|---|---|---|---|
| Seed 123 | A: `natural`, `attr_heavy`; B: `natural`, `balanced`, `title_heavy` | ~8–9h | B giữ Δ1688 ≥ +0,008 và ΔTiki ≥ −0,005 |
| Credit nốt | 740 dòng còn lại | ~1h | Không đảo thứ hạng B |
| Làm đầy 8B | 1.300 description còn thiếu | ~1h | 8B đủ 13.200 để so sánh công bằng |
| Cluster bootstrap | resample theo `product_id` thay vì `eval_id` | vài phút | CI mới cho quyết định |

Nếu seed 123 không tái lập gain B → quay về `natural` cho cả A và B, tránh overfit.

### P1 — Chất lượng dữ liệu & recipe production (~1,5 ngày)

| việc | lý do | chi phí |
|---|---|---|
| Sửa `number_mismatch` | số liệu sai ảnh hưởng trực tiếp | 0,5 ngày code + reprocess |
| Quyết định 740 dòng empty còn lại | chủ yếu boilerplate giá/tax; nên **drop**, không retry thêm | 0 |
| Dataset C 6 nhóm (title/attr/desc × 1688/Tiki) | mô phỏng production | 4 config × ~100 phút ≈ 7h |
| Eval C từ dev | holdout sạch, không dùng để chọn trước đó | 0,5 ngày |
| Train final 4B | recipe chốt + BT 4B/8B | 2–4h |

Config C nên thử: `natural`, `balanced_B` (giữ attr natural, tái cân bằng title/desc), `title_boost`, `attr_boost`.

### P2 — Metric & hạ tầng

- **Tiki metric tốt hơn**: chấm lại hypothesis hiện có bằng `wmt22-cometkiwi-da` hoặc MetricX-QE (không cần train lại). Nếu ranking đổi, kết luận Tiki phải hạ mức chắc chắn.
- **Git history**: `.git` ~2GB do autosave commit cả data. Muốn gọn thật cần `git filter-repo` bỏ `data/` khỏi history (đổi hash toàn bộ).
- **Manifest/checksum** cho HF + run registry (config, seed, git commit, score) để tái lập.

---

## 4. Ba quyết định cần chốt

1. **Mục tiêu cuối**: tối ưu COMET hay chất lượng thương mại (số liệu/thuật ngữ)? Nếu ưu tiên số liệu, `number_mismatch` phải thành blocking hoặc có repair.
2. **BT target cho model final**: dùng BT của chính 4B (self-training) hay BT 8B (distillation)? 4B đang QC cao hơn trên subset chung, nhưng 8B có thể cho target "sạch" hơn.
3. **Mức đầu tư xác nhận**: chỉ 1 seed nữa cho top config, hay 2 seed + budget 8M token trước khi train final?

---

## 5. Batch đề xuất chạy ngay

```text
1. Cluster bootstrap lại A/B (không tốn GPU)
2. Seed 123 cho B: natural, balanced, title_heavy  (~5h)
3. Làm đầy 8B 1.300 dòng                         (~1h)
4. Chấm nốt 740 credit                           (~1h)
```

Sau batch này sẽ biết B có thực sự thắng hay không, và 8B có đáng đưa vào recipe cuối không.

---

## 6. Rủi ro chính

- Gain B nhỏ, có thể mất khi đổi seed hoặc tăng budget.
- Đổi BT model/target sẽ đổi phân phối dữ liệu; cần lock phiên bản BT trước khi làm C.
- Eval composition (B có 2/3 là title) có thể không khớp traffic thật → nên báo cáo thêm theo tỷ lệ production.
- Tiki QE hiện không đủ nhạy để kết luận; đừng chốt weight cho Tiki chỉ dựa trên metric này.
