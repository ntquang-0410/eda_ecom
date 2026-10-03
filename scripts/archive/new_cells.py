# -*- coding: utf-8 -*-
# Source of the new/changed notebook cells. Raw strings: the cell code must keep
# its backslashes (e.g. "[一-鿿]" is written as an escape in the cell,
# which Python turns into the real characters when the cell runs).

MD_5 = r'''## 5. Similarity Trung–Việt bằng BAAI/bge-m3 (từ Hugging Face)

Gọi thẳng mô hình `BAAI/bge-m3` từ Hugging Face Hub qua thư viện `transformers`
(`AutoTokenizer` + `AutoModel`), dùng **CLS pooling** + **chuẩn hoá L2** (đúng theo
`1_Pooling/config.json`) → cosine similarity = tích vô hướng.

Một con số similarity đứng riêng **không nói được** bản dịch tốt hay xấu, nên mục này làm theo thứ tự:

| Mục | Việc | Vì sao |
|---|---|---|
| 5.1 | Tính cho **toàn bộ** title, không lấy mẫu | Kết luận phải dựa trên cả tập, không phải 12% |
| 5.2 | **Mốc nền**: điểm của các cặp chắc chắn sai → đặt ngưỡng | Ngưỡng 0.5 chọn tay không có căn cứ |
| 5.3 | Xếp hạng **trong từng category** + file đọc tay | Trà/đồ ăn nhiều tên riêng phiên âm nên điểm thấp sẵn |
| 5.4 | **Chữ Hán còn sót** trong cột tiếng Việt | Dòng chưa dịch lại được điểm similarity cao |
| 5.5 | **Description** theo từng cặp thuộc tính | Lỗi nặng nhất nằm ở description, không phải title |
'''

CODE_ENCODE = r'''# --- Nạp bge-m3 thẳng từ Hugging Face Hub ---
MODEL_NAME = "BAAI/bge-m3"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
print("Thiết bị:", DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def encode(texts, max_length=512, batch_size=64):
    """Encode danh sách văn bản -> tensor (n, 1024) đã chuẩn hoá L2."""
    if isinstance(texts, str):
        texts = [texts]
    texts = ["" if t is None else str(t) for t in texts]
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        # CLS pooling (đúng cấu hình bge-m3)
        e = out.last_hidden_state[:, 0, :]
        e = F.normalize(e, p=2, dim=1)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)

def encode_np(texts, batch_size=128, **kw):
    """encode() -> numpy (n, 1024). Sắp xếp theo độ dài trước khi chia batch để giảm padding
    (nhanh hơn nhiều khi độ dài chênh lệch lớn), rồi trả lại đúng thứ tự ban đầu."""
    texts = ["" if t is None else str(t) for t in texts]
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    emb = encode([texts[i] for i in order], batch_size=batch_size, **kw).numpy().astype(np.float32)
    out = np.empty_like(emb)
    out[order] = emb
    return out
'''

MD_51 = r'''### 5.1 Similarity cho toàn bộ title
'''

CODE_51 = r'''# --- 5.1 Similarity cho TOÀN BỘ title (không lấy mẫu) ---
# Muốn chạy thử nhanh thì đặt SAMPLE = 2000; kết luận chính thức phải chạy SAMPLE = None.
SAMPLE = None

work = df[[c for c in ["product_id", "category", ZH_COL, VI_COL] if c in df.columns]].copy()
work[ZH_COL] = work[ZH_COL].fillna("").astype(str)
work[VI_COL] = work[VI_COL].fillna("").astype(str)
if "category" not in work.columns:
    work["category"] = "all"
if SAMPLE:
    work = work.sample(SAMPLE, random_state=42)

zh_emb = encode_np(work[ZH_COL].tolist(), batch_size=128)
vi_emb = encode_np(work[VI_COL].tolist(), batch_size=128)
work["sim_score"] = (zh_emb * vi_emb).sum(axis=1)   # cosine = tích vô hướng (đã chuẩn hoá L2)

# Gắn sim_score về df gốc để mục 6-7 dùng lại
df["sim_score"] = np.nan
df.loc[work.index, "sim_score"] = work["sim_score"]

print(f"Đã tính similarity cho {len(work):,} / {len(df):,} cặp title")
print(work["sim_score"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
'''

MD_52 = r'''### 5.2 Mốc nền: điểm của các cặp chắc chắn sai

Ghép title Trung của sản phẩm A với title Việt của **sản phẩm khác** B, rồi tính similarity như cặp thật:

- **sai - ngẫu nhiên**: B lấy ngẫu nhiên trong toàn bộ dữ liệu.
- **sai - cùng category**: B cùng category với A. Khó hơn vì cùng bộ từ vựng, chỉ khác sản phẩm. Đây là mốc dùng để đặt ngưỡng.

**AUC** = xác suất một cặp thật có điểm cao hơn một cặp sai (1.0 = tách hoàn toàn, 0.5 = như tung đồng xu).

Phần đuôi của 2 phân bố **chồng lên nhau** (5% cặp thật thấp nhất nằm lẫn trong 5% cặp sai cao nhất), nên không có
một ngưỡng nào tách sạch đúng/sai. Vì vậy dùng 2 mốc:

- **Chắc sai** = dưới **trung vị** của cặp sai cùng category: cặp thật mà điểm còn tệ hơn một nửa số cặp ghép sai.
  Bắt được lỗi thô: dịch nhầm sản phẩm, dịch cụt, mất gần hết tiêu đề.
- **Vùng chồng lấn** = từ trung vị đến **p95** của cặp sai: similarity không tự quyết được, chỉ dùng để xếp thứ tự đọc tay.

Similarity **không** bắt được lỗi chi tiết (sai đơn vị `2斤` → "2 con", mất một chữ "chay"): những lỗi này cần đọc tay.
'''

CODE_52 = r'''# --- 5.2 Mốc nền từ cặp ghép sai ---
NEG_Q_HARD = 0.50   # tệ hơn một nửa số cặp ghép sai -> gần như chắc sai
NEG_Q_GREY = 0.95   # từ trung vị đến p95 cặp sai: vùng chồng lấn, similarity không tự quyết được
rng = np.random.default_rng(42)
n = len(work)

def derange(pos):
    """Hoán vị pos sao cho không phần tử nào đứng yên (không ghép sản phẩm với chính nó)."""
    p = rng.permutation(pos)
    fixed = p == pos
    p[fixed] = np.roll(p, 1)[fixed]
    return p

perm_rand = derange(np.arange(n))
perm_same = np.arange(n)
cats = work["category"].to_numpy()
for c in np.unique(cats):
    pos = np.flatnonzero(cats == c)
    if len(pos) > 1:
        perm_same[pos] = derange(pos)

work["sim_rand"] = (zh_emb * vi_emb[perm_rand]).sum(axis=1)
work["sim_samecat"] = (zh_emb * vi_emb[perm_same]).sum(axis=1)

def auc(pos_scores, neg_scores):
    s = pd.Series(np.concatenate([pos_scores, neg_scores])).rank().to_numpy()
    n1, n2 = len(pos_scores), len(neg_scores)
    return float((s[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))

pcts = [.01, .05, .25, .5, .75, .95, .99]
display(pd.DataFrame({
    "cặp thật": work["sim_score"].describe(percentiles=pcts),
    "sai - cùng category": work["sim_samecat"].describe(percentiles=pcts),
    "sai - ngẫu nhiên": work["sim_rand"].describe(percentiles=pcts),
}).round(3))

TITLE_AUC = auc(work["sim_score"].to_numpy(), work["sim_samecat"].to_numpy())
print(f"AUC thật vs sai-cùng-category: {TITLE_AUC:.4f}")
print(f"AUC thật vs sai-ngẫu nhiên:    {auc(work['sim_score'].to_numpy(), work['sim_rand'].to_numpy()):.4f}")

SIM_THRESHOLD = float(work["sim_samecat"].quantile(NEG_Q_HARD))
GREY_THRESHOLD = float(work["sim_samecat"].quantile(NEG_Q_GREY))
hard = work["sim_score"] < SIM_THRESHOLD
grey = (work["sim_score"] >= SIM_THRESHOLD) & (work["sim_score"] < GREY_THRESHOLD)
clear = work["sim_score"] >= GREY_THRESHOLD
print(f"\nChắc sai   (< trung vị cặp sai = {SIM_THRESHOLD:.3f}): {int(hard.sum()):>6,} cặp ({100*hard.mean():.2f}%)")
print(f"Chồng lấn  ([{SIM_THRESHOLD:.3f}, {GREY_THRESHOLD:.3f}) = p50-p95 cặp sai): {int(grey.sum()):>6,} cặp ({100*grey.mean():.2f}%)"
      "  <- similarity không tự quyết được")
print(f"Tách rõ    (>= p95 cặp sai = {GREY_THRESHOLD:.3f}): {int(clear.sum()):>6,} cặp ({100*clear.mean():.2f}%)")
print(f"(Đối chiếu: ngưỡng 0.5 cũ bắt {int((work['sim_score'] < 0.5).sum())} cặp, xấp xỉ mốc 'chắc sai'.)")

print("\n--- Các cặp 'chắc sai' ---")
display(work[hard].sort_values("sim_score")[["category", ZH_COL, VI_COL, "sim_score"]])

fig, ax = plt.subplots(figsize=(10, 4))
bins = np.linspace(float(min(work["sim_rand"].min(), work["sim_score"].min())), 1.0, 80)
ax.hist(work["sim_rand"], bins=bins, alpha=.4, label="sai - ngẫu nhiên", color="#999999")
ax.hist(work["sim_samecat"], bins=bins, alpha=.55, label="sai - cùng category", color="#DD8452")
ax.hist(work["sim_score"], bins=bins, alpha=.6, label="cặp thật", color="#4C72B0")
ax.axvline(SIM_THRESHOLD, color="#C0392B", ls="--", lw=1.2, label=f"chắc sai < {SIM_THRESHOLD:.3f}")
ax.axvline(GREY_THRESHOLD, color="k", ls=":", lw=1.2, label=f"hết vùng chồng lấn = {GREY_THRESHOLD:.3f}")
ax.set_xlabel("cosine similarity (bge-m3) — title")
ax.set_ylabel("Số cặp")
ax.legend()
plt.tight_layout()
plt.show()
'''

MD_53 = r'''### 5.3 Xếp hạng trong từng category

Mỗi category có mặt bằng điểm khác nhau, nên dùng **mốc "chắc sai" riêng từng category** (trung vị cặp sai cùng category đó)
và lấy **3% cặp thấp nhất mỗi category** ra file để đọc tay (xuất ở mục 7, có sẵn cột ghi chú).
'''

CODE_53 = r'''# --- 5.3 Xếp hạng trong từng category ---
REVIEW_PCT = 0.03

work["rank_pct_in_cat"] = work.groupby("category")["sim_score"].rank(pct=True)
cat_thr = work.groupby("category")["sim_samecat"].quantile(NEG_Q_HARD)
work["below_cat_threshold"] = work["sim_score"] < work["category"].map(cat_thr)

per_cat = work.groupby("category").agg(
    so_cap=("sim_score", "size"),
    sim_tb=("sim_score", "mean"),
    sim_p05=("sim_score", lambda s: s.quantile(.05)),
    sai_cung_cat_tb=("sim_samecat", "mean"),
    chac_sai=("below_cat_threshold", "sum"),
)
per_cat["nguong_chac_sai"] = cat_thr
per_cat["chac_sai_%"] = 100 * per_cat["chac_sai"] / per_cat["so_cap"]
display(per_cat.sort_values("sim_tb").round(3))

# Lấy ceil(3% x số cặp) cặp thấp nhất mỗi category, tối thiểu 1 (category nhỏ không bị bỏ sót vì làm tròn)
rank_in_cat = work.groupby("category")["sim_score"].rank(method="first")
k_in_cat = work.groupby("category")["sim_score"].transform(lambda s: max(1, int(np.ceil(REVIEW_PCT * len(s)))))
review = work[rank_in_cat <= k_in_cat].sort_values(["category", "sim_score"])
print(f"\n{len(review):,} cặp ({100*REVIEW_PCT:.0f}% thấp nhất mỗi category) cần đọc tay. 3 cặp thấp nhất mỗi category:")
display(review.groupby("category").head(3)[["category", ZH_COL, VI_COL, "sim_score"]])
'''

MD_54 = r'''### 5.4 Chữ Hán còn sót trong cột tiếng Việt

Similarity **không** bắt được lỗi này, thậm chí chấm điểm cao hơn (vì 2 bên giống nhau từng ký tự).
Regex viết bằng ký tự thật `"[一-鿿]"` (chuỗi thường, không phải `r"..."`): pandas bản mới dùng regex của
Arrow, không hiểu cú pháp `\u` trong chuỗi raw.
'''

CODE_54 = r'''# --- 5.4 Chữ Hán còn sót trong cột tiếng Việt ---
CJK = "[一-鿿]"   # chuỗi thường: Python đổi \u thành ký tự thật trước khi đưa cho regex

for c in [c for c in ["title_vi", "description_vi"] if c in df.columns]:
    df[f"cjk_in_{c}"] = df[c].fillna("").astype(str).str.contains(CJK, regex=True)
    k = int(df[f"cjk_in_{c}"].sum())
    print(f"{c:<16} còn chữ Hán: {k:>6,} dòng ({100*k/len(df):.2f}%)")

cjk_ratio = df[VI_COL].fillna("").astype(str).map(lambda s: len(re.findall(CJK, s)) / max(len(s), 1))
print(f"{VI_COL} gần như chưa dịch (>50% ký tự là chữ Hán): {int((cjk_ratio > .5).sum())} dòng")

work["cjk_in_title_vi"] = df.loc[work.index, "cjk_in_title_vi"].to_numpy() if "cjk_in_title_vi" in df.columns else False
if work["cjk_in_title_vi"].any():
    print(f"\nSimilarity TB: dòng còn chữ Hán = {work.loc[work['cjk_in_title_vi'], 'sim_score'].mean():.3f} "
          f"| dòng sạch = {work.loc[~work['cjk_in_title_vi'], 'sim_score'].mean():.3f}  (điểm cao không có nghĩa là dịch tốt)")
    display(work[work["cjk_in_title_vi"]].sort_values("sim_score", ascending=False)
            [["category", ZH_COL, VI_COL, "sim_score", "rank_pct_in_cat"]])

if "cjk_in_description_vi" in df.columns and "category" in df.columns:
    print("\ndescription_vi còn chữ Hán, theo category:")
    display(df.groupby("category")["cjk_in_description_vi"].agg(so_dong="sum", ty_le="mean")
            .assign(ty_le=lambda t: (100 * t["ty_le"]).round(2)).sort_values("so_dong", ascending=False))
'''

MD_55 = r'''### 5.5 Description: so từng cặp thuộc tính

`description_zh` / `description_vi` được ghép lúc crawl theo **cùng thứ tự thuộc tính**, dạng
`Tên: giá trị; Tên: giá trị; ...`, nên tách theo `"; "` rồi ghép cặp theo vị trí. Chỉ dùng dòng có số thuộc tính
2 bên bằng nhau (dòng lệch là do dấu `；` trong giá trị tiếng Trung bị dịch thành `"; "`).

- So **tên thuộc tính** (bắt lỗi dịch tên lặp lại trên hàng nghìn sản phẩm, vd `皮质特征` → "Đặc điểm vỏ não").
- So **giá trị** (mốc nền: đổi giá trị với sản phẩm khác **cùng tên thuộc tính**).
- Chỉ embed mỗi chuỗi **duy nhất** một lần (rất nhiều giá trị lặp: `否`, `PVC`...).
- Thêm 2 cờ regex mà similarity không bắt được: **lỗi đánh số** (`Khác1`, `Là2`, `Không.1`) và **chữ Hán còn sót**.
'''

CODE_55 = r'''# --- 5.5 Description theo từng cặp thuộc tính ---
def split_desc(text):
    out = []
    for seg in str(text or "").split("; "):
        if seg.strip():
            name, sep, val = seg.partition(": ")
            out.append((name.strip(), val.strip()) if sep else ("", seg.strip()))
    return out

seg_zh = df["description_zh"].map(split_desc)
seg_vi = df["description_vi"].map(split_desc)
aligned = seg_zh.map(len) == seg_vi.map(len)
print(f"Dòng ghép cặp được: {int(aligned.sum()):,} / {len(df):,} ({100*aligned.mean():.2f}%)")

rows = []
for i in df.index[aligned]:
    for (nz, vz), (nv, vv) in zip(seg_zh[i], seg_vi[i]):
        rows.append((i, nz, vz, nv, vv))
attr = pd.DataFrame(rows, columns=["row", "name_zh", "val_zh", "name_vi", "val_vi"])
attr["product_id"] = df.loc[attr["row"], "product_id"].to_numpy() if "product_id" in df.columns else attr["row"]
attr["category"] = df.loc[attr["row"], "category"].to_numpy() if "category" in df.columns else "all"
print(f"Tổng cặp thuộc tính: {len(attr):,}")

def rowdot(A, ia, B, ib, chunk=50_000):
    """Tích vô hướng A[ia[k]] . B[ib[k]], chia khối để không tạo mảng khổng lồ trong RAM."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = (A[ia[s:s + chunk]] * B[ib[s:s + chunk]]).sum(axis=1)
    return out

def embed_unique(texts, batch_size):
    idx = pd.Index(pd.unique(pd.Series(texts)))
    return idx, encode_np(list(idx), batch_size=batch_size)

# (a) Giá trị thuộc tính
iz, Ez = embed_unique(attr["val_zh"], batch_size=64)
iv, Ev = embed_unique(attr["val_vi"], batch_size=64)
attr["sim_val"] = rowdot(Ez, iz.get_indexer(attr["val_zh"]), Ev, iv.get_indexer(attr["val_vi"]))

# Mốc nền: ghép giá trị Trung với giá trị Việt của sản phẩm khác CÙNG tên thuộc tính,
# chỉ tính những cặp mà giá trị Trung thật sự khác nhau (tránh "否" ghép với "否").
partner = np.arange(len(attr))
for _, pos in attr.groupby("name_zh").indices.items():
    if len(pos) > 1:
        partner[pos] = rng.permutation(pos)
vz = attr["val_zh"].to_numpy()
diff = vz[partner] != vz
neg_val = rowdot(Ez, iz.get_indexer(vz[diff]), Ev, iv.get_indexer(attr["val_vi"].to_numpy()[partner][diff]))
VAL_THRESHOLD = float(np.quantile(neg_val, NEG_Q_HARD))
VAL_AUC = auc(attr["sim_val"].to_numpy(), neg_val)
attr["below_threshold"] = attr["sim_val"] < VAL_THRESHOLD

# (b) Tên thuộc tính: mốc nền = tên Việt của một tên Trung khác
names = attr.groupby(["name_zh", "name_vi"]).size().rename("so_lan").reset_index()
inz, Enz = embed_unique(names["name_zh"], batch_size=128)
inv, Env = embed_unique(names["name_vi"], batch_size=128)
names["sim_name"] = rowdot(Enz, inz.get_indexer(names["name_zh"]), Env, inv.get_indexer(names["name_vi"]))
np_perm = derange(np.arange(len(names)))
diff_n = names["name_zh"].to_numpy()[np_perm] != names["name_zh"].to_numpy()
neg_name = rowdot(Enz, inz.get_indexer(names["name_zh"][diff_n]),
                  Env, inv.get_indexer(names["name_vi"].to_numpy()[np_perm][diff_n]))
NAME_THRESHOLD = float(np.quantile(neg_name, NEG_Q_HARD))

# (c) Cờ regex: lỗi đánh số của 1688 và chữ Hán còn sót
attr["numbering_artifact"] = attr["val_vi"].str.contains(r"(?:Không|Có|Là|Khác)\.?\d+\s*$", regex=True)
attr["cjk_in_vi"] = (attr["val_vi"].str.contains(CJK, regex=True) | attr["name_vi"].str.contains(CJK, regex=True))

print(f"\n--- Giá trị thuộc tính ({len(attr):,} cặp, {len(iz):,} + {len(iv):,} chuỗi duy nhất) ---")
display(pd.DataFrame({"cặp thật": attr["sim_val"].describe(percentiles=pcts),
                      "sai - cùng tên thuộc tính": pd.Series(neg_val).describe(percentiles=pcts)}).round(3))
print(f"AUC = {VAL_AUC:.4f} | mốc chắc sai (trung vị cặp sai) = {VAL_THRESHOLD:.3f} | "
      f"cặp dưới mốc: {int(attr['below_threshold'].sum()):,} ({100*attr['below_threshold'].mean():.2f}%)")
if VAL_AUC < 0.9:
    print("⚠️ AUC < 0.9: với chuỗi ngắn (1-3 từ), bge-m3 phân biệt đúng/sai kém — nhiều bản dịch đúng vẫn điểm thấp\n"
          "   (vd 广州 -> 'Quảng Châu'). Danh sách dưới mốc chỉ để đọc tay, KHÔNG dùng để xoá tự động.\n"
          "   Tín hiệu đáng tin ở mức thuộc tính là 2 cờ regex bên dưới.")
print(f"Lỗi đánh số (Khác1/Là2/Không.1): {int(attr['numbering_artifact'].sum()):,} cặp "
      f"({100*attr['numbering_artifact'].mean():.2f}%) — sim TB {attr.loc[attr['numbering_artifact'], 'sim_val'].mean():.3f}, "
      f"trong đó dưới mốc chắc sai: {int((attr['numbering_artifact'] & attr['below_threshold']).sum()):,}")
print(f"Chữ Hán còn sót: {int(attr['cjk_in_vi'].sum()):,} cặp ({100*attr['cjk_in_vi'].mean():.2f}%) — "
      f"sim TB {attr.loc[attr['cjk_in_vi'], 'sim_val'].mean():.3f}")

print("\nTỉ lệ (%) theo category: dưới mốc chắc sai / lỗi đánh số / chữ Hán còn sót:")
display((attr.groupby("category")[["below_threshold", "numbering_artifact", "cjk_in_vi"]].mean() * 100)
        .round(2).sort_values("below_threshold", ascending=False))

print("\nCặp giá trị dưới mốc, lặp lại nhiều nhất (ứng viên lỗi dịch có hệ thống — cần đọc tay xác nhận):")
low_vals = (attr[attr["below_threshold"]].groupby(["name_zh", "val_zh", "val_vi"])
            .agg(so_lan=("sim_val", "size"), sim=("sim_val", "first"))
            .sort_values("so_lan", ascending=False).reset_index())
display(low_vals.head(20))

print(f"\n--- Tên thuộc tính ({len(names):,} cặp tên khác nhau) | mốc chắc sai = {NAME_THRESHOLD:.3f} ---")
low_names = names[names["sim_name"] < NAME_THRESHOLD].sort_values("so_lan", ascending=False)
print(f"{len(low_names):,} cặp tên dưới mốc, xuất hiện tổng {int(low_names['so_lan'].sum()):,} lần. "
      "20 cặp gặp nhiều nhất (tên lặp trên nhiều sản phẩm nên sửa 1 lần được nhiều dòng):")
display(low_names.head(20).round(3))
'''

MD_7 = r'''## 7. Xuất báo cáo & kết luận
'''

CODE_7 = r'''# --- Bảng tổng hợp chỉ số chính ---
summary = pd.DataFrame([{
    "rows": len(df),
    "cols": df.shape[1],
    "null_%_max": round(null_df["null_%"].max(), 2),
    "empty_%_max": round(null_df["empty_%"].max(), 2),
    "dup_pair_%": round(100 * dup_pair / len(df), 2),
    "dup_zh_%": round(100 * dup_zh / len(df), 2),
    "dup_vi_%": round(100 * dup_vi / len(df), 2),
    "avg_len_zh": round(len_zh.mean(), 1),
    "avg_len_vi": round(len_vi.mean(), 1),
    "avg_tok_zh": round(tok_zh.mean(), 1),
    "avg_tok_vi": round(tok_vi.mean(), 1),
    "ratio_tok_median": round(ratio_tok.median(), 2),
    "ratio_outliers_%": round(100 * n_tok_out / len(df), 2),
    "suspect_%": round(100 * flags["is_suspect"].mean(), 2),
    "title_pairs_scored": len(work),
    "title_sim_mean": round(work["sim_score"].mean(), 3),
    "title_threshold_hard_p50_neg": round(SIM_THRESHOLD, 3),
    "title_threshold_grey_p95_neg": round(GREY_THRESHOLD, 3),
    "title_auc_vs_samecat_neg": round(TITLE_AUC, 4),
    "title_hard_wrong_n": int((work["sim_score"] < SIM_THRESHOLD).sum()),
    "title_grey_zone_%": round(100 * ((work["sim_score"] >= SIM_THRESHOLD) & (work["sim_score"] < GREY_THRESHOLD)).mean(), 2),
    "title_vi_with_chinese": int(df["cjk_in_title_vi"].sum()),
    "description_vi_with_chinese_%": round(100 * df["cjk_in_description_vi"].mean(), 2),
    "attr_pairs": len(attr),
    "attr_threshold_hard_p50_neg": round(VAL_THRESHOLD, 3),
    "attr_auc_vs_samename_neg": round(VAL_AUC, 4),
    "attr_below_threshold_%": round(100 * attr["below_threshold"].mean(), 2),
    "attr_numbering_artifact_%": round(100 * attr["numbering_artifact"].mean(), 2),
    "attr_vi_with_chinese_%": round(100 * attr["cjk_in_vi"].mean(), 2),
}]).T.rename(columns={0: "value"})

print("=" * 46)
print("BẢNG TỔNG HỢP EDA")
print("=" * 46)
display(summary)

# --- Lưu kết quả ra CSV ---
OUT_SUSPECT = "eda_suspect_pairs.csv"           # heuristic token-ratio (mục 6)
OUT_SUMMARY = "eda_summary.csv"
OUT_REVIEW = "eda_title_review_by_category.csv"  # 3% thấp nhất mỗi category, có cột để ghi nhận xét
OUT_CJK = "eda_chinese_left_in_vi.csv"
OUT_ATTR = "eda_attr_below_threshold.csv"
OUT_ATTR_NAMES = "eda_attr_names_low_sim.csv"

# Loại cột nested (attributes_zh/vi) trước khi ghi CSV: chúng là list<struct>,
# khi to_csv sẽ bị repr thành nhiều dòng -> làm hỏng cấu trúc file.
def drop_nested(frame):
    keep = [c for c in frame.columns if not frame[c].map(type).isin([list, dict]).any()]
    return frame[keep]

try:
    out = drop_nested(suspect).copy()
    out["reason"] = suspect["_reason"]
    out["ratio_tok"] = suspect["_ratio_tok"]
    out["sim_score"] = suspect["sim_score"].round(3)
    out.to_csv(OUT_SUSPECT, index=False, encoding="utf-8-sig")

    rv = work.loc[review.index, [c for c in ["product_id", "category", ZH_COL, VI_COL, "sim_score", "rank_pct_in_cat",
                                             "below_cat_threshold", "cjk_in_title_vi"] if c in work.columns]].copy()
    rv["sim_score"] = rv["sim_score"].round(3)
    rv["rank_pct_in_cat"] = rv["rank_pct_in_cat"].round(3)
    rv["danh_gia"] = ""   # đọc tay: dung / sai / nghi
    rv["ghi_chu"] = ""
    rv.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")

    cjk_cols = [c for c in ["product_id", "category", "title_zh", "title_vi", "description_zh", "description_vi",
                            "cjk_in_title_vi", "cjk_in_description_vi", "sim_score"] if c in df.columns]
    df.loc[df["cjk_in_title_vi"] | df["cjk_in_description_vi"], cjk_cols].to_csv(OUT_CJK, index=False, encoding="utf-8-sig")

    attr[attr["below_threshold"]].drop(columns=["row"]).sort_values(["category", "sim_val"]) \
        .round({"sim_val": 3}).to_csv(OUT_ATTR, index=False, encoding="utf-8-sig")
    low_names.round(3).to_csv(OUT_ATTR_NAMES, index=False, encoding="utf-8-sig")

    summary.to_csv(OUT_SUMMARY, encoding="utf-8-sig")
    print(f"\n✅ {len(out):,} mẫu nghi vấn heuristic -> {OUT_SUSPECT}")
    print(f"✅ {len(rv):,} cặp title cần đọc tay (3% thấp nhất mỗi category) -> {OUT_REVIEW}")
    print(f"✅ {int((df['cjk_in_title_vi'] | df['cjk_in_description_vi']).sum()):,} dòng còn chữ Hán -> {OUT_CJK}")
    print(f"✅ {int(attr['below_threshold'].sum()):,} cặp thuộc tính dưới ngưỡng -> {OUT_ATTR}")
    print(f"✅ {len(low_names):,} cặp tên thuộc tính dưới ngưỡng -> {OUT_ATTR_NAMES}")
    print(f"✅ Bảng tổng hợp -> {OUT_SUMMARY}")
except Exception as e:
    print("Lỗi xuất CSV:", e)

# --- Kết luận nhanh ---
print("\n--- NHẬN XÉT NHANH ---")
print(f"• Dữ liệu sạch về null: chỉ {int((null_df['null_%'] > 0).sum())} cột có null.")
print(f"• Trùng lặp cặp câu ở mức thấp ({100*dup_pair/len(df):.2f}%), nhưng {ZH_COL} trùng "
      f"{100*dup_zh/len(df):.2f}% -> nhiều sản phẩm khác nhau dùng chung tiêu đề.")
print(f"• Tỷ lệ độ dài zh/vi ổn định (median {ratio_tok.median():.2f} token vi / ký tự zh).")
print(f"• Title: AUC {TITLE_AUC:.3f} (thật vs sai cùng category). {int((work['sim_score'] < SIM_THRESHOLD).sum())} cặp "
      f"'chắc sai' (< {SIM_THRESHOLD:.3f}); {100*((work['sim_score'] >= SIM_THRESHOLD) & (work['sim_score'] < GREY_THRESHOLD)).mean():.1f}% "
      f"nằm trong vùng chồng lấn -> similarity chỉ bắt lỗi thô, lỗi chi tiết (đơn vị, mất chữ) phải đọc tay ({OUT_REVIEW}).")
print(f"• Description (theo thuộc tính): AUC {VAL_AUC:.3f}, {100*attr['below_threshold'].mean():.2f}% cặp giá trị dưới mốc chắc sai; "
      f"lỗi đánh số {100*attr['numbering_artifact'].mean():.2f}% và chữ Hán còn sót {100*attr['cjk_in_vi'].mean():.2f}% "
      f"— 2 lỗi này phải lọc bằng regex, similarity không bắt được.")
print(f"• {len(suspect)} cặp nghi vấn theo heuristic cần review (xem {OUT_SUSPECT}).")
'''
