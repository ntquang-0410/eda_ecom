# description_vi_to_zh: Vietnamese → Chinese product attribute pairs, translated by LLMs

Release date: 2026-10-23 · Build tag: `20261023_1656` · Folder: `data/description_vi_to_zh/`

This folder holds **87,056 Vietnamese–Chinese attribute pairs** (attribute name + attribute value) taken from
**13,187 products** listed on Tiki.vn, in 6 product categories. The Vietnamese side comes from the attribute table of
the product descriptions, as cleaned by the team's EDA step (`data/cleaned/tiki/tiki_eda.parquet`). The Chinese side
is a **machine translation**: every distinct Vietnamese attribute name and value was translated on its own by two
LLMs, the outputs were checked by rules and by a reference-free quality-estimation model (CometKiwi), and one version
was kept per item. Frequent names and values use dictionaries reviewed by the dataset author; brand names, numbers,
measurements and model codes are copied unchanged.

It is the opposite direction of `data/description_zh_to_vi/` (1688.com, Chinese → Vietnamese) and was built with the
same two models and the same pipeline design. The two folders cover different products from different sites.

The data was built for a student thesis project on Vietnamese–Chinese machine translation for cross-border
e-commerce. It is **machine-generated, silver-quality data**: suitable for training, fine-tuning and analysis, but
**not a human reference** and not suitable as a test set without human checking (see *Limitations*).

**Update 28 Oct 2026:** `description_products_vi_zh_pair.parquet` (one row per product, with the new column `desc_nguon_dich`) was added to this folder. The pair table and `provenance.json` did not change.

## Files

| File | Rows | Columns | Size (bytes) | SHA256 |
|---|---:|---:|---:|---|
| `description_pairs_vi_zh.parquet` | 87,056 | 8 | 2,607,592 | `0177e2b233d260dc024e0abe3797e91f11ddab714557368eb46e0a7ee30e78fc` |
| `description_products_vi_zh_pair.parquet` | 13,187 | 9 | 1,754,591 | `a807ee07801c17e8fbfe096a8ccb3164adab3e1966d7fd1af6337e6505a8c0bc` |
| `provenance.json` | – | – | 9,435 | `e94dfa5b022fed793543040e374b80efe67f813dd93d447144283e142cfb72e6` |
| `SHA256SUMS.txt` | – | – | 286 | checksums of the three files above |

```bash
cd data/description_vi_to_zh
sha256sum -c SHA256SUMS.txt
```

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/ntquang0410/zh-vie_ecom/data/description_vi_to_zh/description_pairs_vi_zh.parquet")
```

## Schema (`description_pairs_vi_zh.parquet`)

One row = one attribute of one product.

| Column | Type | Meaning |
|---|---|---|
| `pair_id` | int64 | Pair id, unique in this file. Assigned when the attributes were extracted from the source table; ids of pairs not released are missing, so the ids have gaps. |
| `product_id` | string | Tiki product id. Joins to `data/cleaned/tiki/tiki_eda.parquet`. |
| `category` | string | Category assigned by the team: `food`, `home`, `electronics`, `auto`, `mother_baby`, `beauty`. |
| `name_vi` | string | Attribute name, Vietnamese (e.g. `Hướng dẫn bảo quản`). |
| `gia_tri_vi` | string | Attribute value, Vietnamese (e.g. `Bảo quản nơi khô ráo thoáng mát`). Multi-value attributes are joined with `", "`. |
| `name_zh` | string | Attribute name, Simplified Chinese (e.g. `保存方法`). |
| `gia_tri_zh` | string | Attribute value, Simplified Chinese (e.g. `请置于阴凉干燥处保存`). |
| `do_tin_cay` | string | Confidence label: `cao` (high), `vua` (medium), `thap` (low). Defined below. |

The column names are Vietnamese (`gia_tri` = value, `do_tin_cay` = confidence) to match the other files of this
repository. No cell is empty and `pair_id` has no duplicates.

### `description_products_vi_zh_pair.parquet` (added 28 Oct 2026)

One row per product, built from the pair table of this folder. The columns are the same as the products table of the context-aware companion (`data/tiki/descriptions/vi_to_zh_context/`), in the same order, so the two can be stacked.

| Column | Type | Meaning |
|---|---|---|
| `product_id`, `category` | string | As above. |
| `so_cap` | int64 | Number of pairs of the product in this release. |
| `do_tin_cay` | string | Lowest `do_tin_cay` among the pairs of the product. |
| `so_cap_khong_cao` | int64 | Number of pairs of the product whose label is not `cao`. |
| `description_vi`, `description_zh` | string | The attribute list as one text, one `name: value` per line (joined with `\n`), in `pair_id` order. It is the description published in `data/tiki/full_dataset/tiki_full_attributes_independent_vi_zh.parquet`. |
| `so_cap_thap` | int64 | Number of pairs of the product whose label is `thap`. |
| `desc_nguon_dich` | string | Model(s) that translated at least one pair of the product: see below. |

`desc_nguon_dich` names the model(s) that produced at least one pair of the product. The reviewed dictionaries and fixed rules contribute to every product,
so it only says which model, if any, also worked on it. A pair counts for **Gemma** (Gemma 4 26B-A4B, "model A") when the unit was kept from Gemma's output
(including units where both models gave the same text, which the pipeline credits to Gemma), and for **Qwen** (Qwen3-14B-AWQ, "model B") when Qwen's output
alone was kept. The per-pair choice (`cach_chon`) is not published in the pair table; the label was computed from the build file.

| Value | Meaning | Products |
|---|---|---:|
| `gemma4` | at least one pair was translated or corrected by Gemma, none by Qwen | 7,605 (57.7%) |
| `gemma4+qwen3_14b_awq` | both models worked on at least one pair | 111 (0.8%) |
| `qwen3_14b_awq` | at least one pair was translated or corrected by Qwen, none by Gemma | 106 (0.8%) |
| `tu_dien_quy_tac` | no pair was produced by a model: dictionary and rules only | 5,365 (40.7%) |

### Examples

| category | name_vi | gia_tri_vi | name_zh | gia_tri_zh | do_tin_cay |
|---|---|---|---|---|---|
| food | Hướng dẫn bảo quản | Bảo quản nơi khô ráo thoáng mát | 保存方法 | 请置于阴凉干燥处保存 | cao |
| beauty | Lưu ý | Bao bì thay đổi theo từng đợt nhập hàng | 注意事项 | 包装随进货批次变化 | cao |
| beauty | Hạn sử dụng | 3 năm | 保质期 | 3年 | cao |
| mother_baby | Thương hiệu | New4all | 品牌 | New4all | cao |
| food | Hương vị | BẠC HÀ, CHANH DÂY, NHO | 口味 | 薄荷, 百香果, 葡萄 | vua |
| auto | Thắng xe | Thắng đĩa | 刹车 | 碟式制动器 | vua |
| electronics | chọn màu | Hồng, Xanh cốm, Xanh rêu, Xanh than, Đen, Đỏ | 选择颜色 | 粉色, 浅绿色, 苔藓绿, 深蓝色, 黑色, 红色 | thap |

## Statistics

| Category | Products | Pairs | `cao` | `vua` | `thap` |
|---|---:|---:|---:|---:|---:|
| food | 2,193 | 16,100 | 13,037 | 2,909 | 154 |
| home | 2,200 | 15,837 | 13,514 | 2,120 | 203 |
| electronics | 2,198 | 15,198 | 13,701 | 1,349 | 148 |
| auto | 2,199 | 14,914 | 13,476 | 1,250 | 188 |
| mother_baby | 2,198 | 12,937 | 12,147 | 725 | 65 |
| beauty | 2,199 | 12,070 | 11,143 | 828 | 99 |
| **Total** | **13,187** | **87,056** | **77,018** | **9,181** | **857** |

- **Heavy repetition.** The 87,056 pairs contain only **19,454 distinct Vietnamese (name, value) combinations**
  (826 distinct names, 18,407 distinct values), because the same attributes (`Xuất xứ thương hiệu`, `Thương hiệu`,
  `Sản phẩm có được bảo hành không?`, ...) occur in most products. Each distinct Vietnamese (name, value) always has
  exactly one Chinese translation in this file, and each Vietnamese name always has the same Chinese name.
  Deduplicate or down-weight if you do not want frequent attributes to dominate training.
- 6.6 attributes per product on average (median 6, max 31).
- Length (characters, median / 99th percentile / max): `gia_tri_vi` 8 / 277 / 4,193; `gia_tri_zh` 3 / 125 / 1,399.
  Long values are usage instructions, ingredient lists and package descriptions.

## How the data was built

### 1. Source and preparation

- **Input:** `data/cleaned/tiki/tiki_eda.parquet` (13,200 products, SHA256 `7af2a262…70d3bd`), the output of the
  team's Tiki EDA step.
- **Extraction:** the attribute list of each product (column `attributes_vi`) was split into (name, value) pairs with
  the same cleaning as the EDA step's `bt_attributes_text` (Unicode NFC, emoji and hashtags removed, edge dashes and
  a trailing `:` on names removed, pairs with an empty value dropped); attributes with several values were joined
  with `", "`. Joined back, the pairs match `bt_attributes_text` for 13,141 of 13,200 products (the rest differ by a
  few punctuation marks). This gives 89,300 pairs.
- **Pairs not released (2,244):**
  - 2,033 removed before translation: 1,952 names and addresses of the organisation responsible for the product
    (manufacturer, importer, distributor) and 81 contact details;
  - 128 removed by a privacy filter when the pairs were assembled: 126 that name a specific shop (a shop name in
    quotes, "provided by … Store", a brand value that is a "… Store"/"… Shop", shop handles such as `shopxxx`,
    "PET SHOP XXX", a dealer named in the text) and 2 that give a street address;
  - 83 whose translation could not be chosen with enough confidence (82 because of the value, 1 because of the name).
- **Routing.** Each value of the 87,267 translatable pairs goes to one of these routes; only the first needs a model:
  - *model* – sent to the two LLMs (66,152 pairs);
  - *brand* – values of `Thương hiệu` (brand) are copied as written, including Vietnamese brand names with
    diacritics (13,126 pairs);
  - *measurement* – a number with a unit, such as `250g` or `30 x 20 cm` (4,418 pairs);
  - *copy* – values without letters (numbers, ranges; 2,125 pairs);
  - *model code* – product and model codes such as `TB313` (1,307 pairs);
  - *letter size* – `S`, `M`, `XL` ... (139 pairs).
- **Translation units.** Identical texts are translated once: 13,518 units (830 attribute names, 12,688 values).

### 2. Translation with two models

| | Model A | Model B |
|---|---|---|
| Model | Gemma 4 26B-A4B instruction-tuned (`gemma-4-26b-a4b-it`) | Qwen3-14B, 4-bit AWQ (`Qwen/Qwen3-14B-AWQ`, revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`) |
| Served by | Google AI Studio (Gemini API, OpenAI-compatible endpoint), 23 Oct 2026 | vLLM 0.32.0 on Kaggle, 2 × Tesla T4, 23 Oct 2026 |
| Decoding | temperature 0, `reasoning_effort: minimal` | temperature 0, thinking mode off |
| Batching | several units of the same type per request (≤ 2,400 characters; units over 600 characters sent alone): 1,175 requests | one unit per request |
| Units translated | 13,518 | 13,518 |

Both models get the same English system prompt: translate a Vietnamese product attribute from Tiki.vn into
Simplified Chinese in the style of Chinese e-commerce listings; output only the translation, with no label before it;
translate every Vietnamese word; use the standard Chinese names of places (`Hà Nội` → `河内`); keep numbers, units,
sizes, model codes and Latin-letter brand names unchanged; for a `", "`-separated list, keep the same items in the
same order. The user message gives the product category and, for values, the Vietnamese attribute name as context.
If the output still contained Vietnamese letters, the model was asked once more to translate it fully. A label that
a model added before a value (its own translation of the attribute name, e.g. `产地：意大利`) was removed when the
source had no such label (23 units of A, 1 of B).

### 3. Rule checks (hard failures)

A translation fails if it: is empty; still contains Vietnamese words (Vietnamese letters with diacritics; short
capitalised proper names inside a Chinese value are tolerated, attribute names must be fully Chinese); contains
chatter, `<think>` tags or added markdown; translates the prompt context or prints the attribute name as a label
before a short value; loops on repeated characters; uses Traditional characters; loses or changes a number of the
source (Vietnamese number formats such as `1.000` and `1,5`, `万` and Chinese numerals are understood; long values
may lose one number or 10 %); loses a code that contains a digit; changes the number of list items; is abnormally
long for its length bucket (above the 99.9th percentile of the Chinese/Vietnamese length ratio); keeps English
colour words untranslated when the other model translated them; or writes the weight unit `斤`.

**`斤` (jīn = 0.5 kg).** In Vietnamese, "cân" means a kilogram. A Chinese output that writes "N cân" as `N斤` halves
the meaning of the number for a Chinese reader, so any `斤` that is not part of `公斤` (kg) or `千斤顶` (car jack)
fails.

### 4. Quality estimation

Both outputs were scored with **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`), a reference-free QE model. The
confidence floor is 0.40, or 0.30 when the Vietnamese source has at most 12 characters (CometKiwi scores very short
strings low even when they are correct). Besides rule failures, an item is dropped only when no rule-passing output
reaches the drop floor of 0.30 (0.22 for short sources); identical outputs of the two models are never dropped for
their score. Scores are used only to set the confidence and to drop items; **the scores themselves are not
released**.

### 5. Choosing one translation per unit

Model A is preferred. Model B is used only when A fails the rules. CometKiwi never replaces a rule-passing A by B
(this rule was adopted on the Chinese → Vietnamese set, where CometKiwi often scored fluent but wrong Qwen outputs
above correct Gemma outputs).

| Case | Kept | Value confidence | Value pairs |
|---|---|---|---:|
| A and B identical after normalisation (NFKC, spaces removed, `、` = `,`), both pass the rules | that text | `cao` if the best score ≥ floor, else `vua` | 4,516 |
| Different but similar (chrF ≥ 60, both strings longer than 3 characters), both pass | A | `vua` if A ≥ floor, else `thap` | 2,508 |
| Clearly different, both pass, A ≥ floor | A | `vua` | 6,386 |
| Clearly different, both pass, A < floor | A | `thap` | 465 |
| Only A passes the rules | A | `thap` | 113 |
| Only B passes the rules | B | `thap` | 218 |
| Both fail the rules; or no usable output reaches the drop floor | dropped | – | – |

### 6. Reviewed dictionaries and fixed rules (override model output)

These outputs always have value/name confidence `cao`. All dictionary entries were reviewed and accepted by the
dataset author.

| Source | Applies to | Units | Pairs |
|---|---|---:|---:|
| Attribute-name dictionary (107 names; matching ignores case, so it covers 175 spellings) | names | 175 | 85,734 |
| Value dictionary (289 entries keyed by (name, value), e.g. `Xuất xứ thương hiệu: Hàn Quốc` → `韩国`) | values | 384 | 31,928 |
| Yes/no table by attribute (55 entries, e.g. `Có` → `是` for "Is there a warranty?", `Có` → `有` for "Pattern") | values | 68 | 13,729 |
| Duration pattern (`N tháng/năm/ngày` → `N个月/年/天`; `… kể từ ngày sản xuất` → `自生产之日起…`) | values | 219 | 5,993 |
| Weight pattern (`N gram` → `N克`) | values | 72 | 113 |

### 7. Where the released values come from

| Origin of `gia_tri_zh` | Pairs | Share |
|---|---:|---:|
| Dictionary, yes/no table and patterns | 51,763 | 59.5 % |
| Copy rules (brand, measurement, number, model code, letter size) | 21,087 | 24.2 % |
| Model A (Gemma 4), chosen over B or only A passed | 9,472 | 10.9 % |
| Both models produced the same text | 4,516 | 5.2 % |
| Model B (Qwen3-14B) only | 218 | 0.3 % |

The pair-level `do_tin_cay` is the lower of the name confidence and the value confidence.

## Quality checks

The checks below were run during development. They are small samples graded by an AI assistant, **not**
independent human annotation, so read them as indicative only.

1. **200-pair stratified sample** (build `20261023_1557`; this build differs from it only by 92 more pairs removed by
   the privacy filter and 4 values cleaned of a `_x000D_` artefact). Seven strata: identical outputs (30), A kept with
   both passing (60), low confidence (60), value dictionary (20), yes/no table (10), duration/weight patterns (10),
   copy rules (10), with the pairs of each stratum drawn at random. The dataset author ran this evaluation on the
   200 randomly drawn pairs; each pair was graded by an AI assistant (Claude) in two passes. Scored against this
   build (199 pairs; one sampled pair is no longer released):
   - **Values, weighted to the whole file: 96.8 % correct, 99.9 % correct or minor error.** Attribute names: 99.5 %
     correct. Both name and value correct: 96.5 %.
   - The weighted figure is dominated by dictionary and rule outputs (84 % of pairs), which were all correct in the
     sample. **On the machine-translated values alone (16 % of pairs): 80.7 % correct, 99.5 % correct or minor.**
   - By stratum: identical outputs 30/30 correct; A kept with both passing 43/59 correct (59/59 correct or minor);
     low confidence 35/60 correct (55/60 correct or minor). By confidence label: `cao` 76/76, `vua` 47/63 (63/63
     correct or minor), `thap` 35/60 (55/60).
2. **Automatic checks on this file:** 0 duplicate `pair_id`; 0 empty cells; 0 Chinese characters in the Vietnamese
   columns; 0 Vietnamese letters in `name_zh`. 1,057 `gia_tri_zh` values contain Vietnamese letters: 984 are brand
   names copied on purpose (route *brand*) and 73 are model outputs that keep a Vietnamese proper name. 23,497
   `gia_tri_zh` values have no Chinese character (copied brands, numbers, measurements, codes). PII scan: 0 phone
   numbers, 0 e-mail addresses, 0 URLs; the privacy filter of section 1 finds nothing left.

## Limitations

- **Machine translation, no human reference.** Expect roughly 3 % wrong or imperfect values overall, about one in
  five among machine-translated values, and more in the `thap` group (about 40 % not fully correct in the sample).
  Filter on `do_tin_cay` if you need cleaner data.
- **Each text is translated without the rest of the product.** Short, ambiguous values are the main source of
  errors: colour names (`Hồng` pink → `红色` red in one helmet listing; `Xanh` can mean blue or green), helmet and
  car-part model names, and dimension letters (`d243 x r39 x c67 mm`, where `d` means length, became `直径243…`).
  A context-aware version translated per product is planned as a separate set.
- **Known recurring errors in model A:** Vietnamese herbal and place names (`Thiên niên kiện` → `续断` instead of
  `千年健`; `cà gai leo` → `刺蒺藜`, a different plant; `Cầu Đất` → `梂土`); some INCI ingredient names (`Lauryl`
  → `劳里尔` instead of `月桂基`); English words left as they are (`Taiwan / China`, `PROTEIN`, `Arabica`,
  `alkaline`, `gr`); colour pairs written in reverse order (`Trắng Đỏ` → `红白`).
- **Spelling conventions not normalised:** spaces between numbers or Latin words and Chinese characters follow the
  model output (`100% 大米` and `100%大米` both occur); `Đài Loan` is mostly `台湾`, sometimes `中国台湾`.
- **Brands** are copied as written in the listing, including Vietnamese brand names with diacritics. Some brand
  values are the name of the shop selling the product on Tiki (a business name); they were kept.
- **Do not use as a test set as-is.** Build test references by human checking. This matters especially if you
  evaluate **Qwen-family models**: 0.3 % of values come from Qwen3-14B alone, and 5.2 % are values where Qwen3-14B and
  Gemma 4 agreed. Scores of a Qwen model on Qwen-produced references can be inflated. The internal full version of
  this build has a per-pair origin column; ask the authors if you need it for an ablation.
- **No train/dev/test split** is provided. If you split, split by `product_id` at least; near-duplicate listings of
  the same shop may still fall on both sides.
- The CometKiwi floors (0.40 / 0.30, drop floors 0.30 / 0.22) and the chrF threshold (60) were set from score
  distributions and small manual reads, not tuned on labelled data.

## Licences and terms of the tools used

- **Source text:** product listings on Tiki.vn, collected for non-commercial academic research.
- **Model A outputs** were produced through the **Gemini API (Google AI Studio)**. The Gemma 4 weights are
  Apache 2.0, but use of the API service is governed by the *Gemini API Additional Terms of Service*, which include
  a restriction on using the Services to develop models that compete with Google's services. Whether this applies
  to training a small domain-specific translation model is for each user to assess; please review those terms
  before using this data to train models, especially commercially.
- **Model B:** Qwen3-14B is Apache 2.0.
- **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`) is CC BY-NC-SA 4.0. It was used only to compute the confidence
  labels and to filter; no scores are distributed.
- This folder follows the repository-level licence declaration, subject to the third-party terms above. Intended
  use: non-commercial research and education.

## Provenance file

`provenance.json` is the build card written by the pipeline (keys in Vietnamese), with API key names removed. Main
keys:

| Key | Meaning |
|---|---|
| `nguong` | thresholds: `tau` chrF similarity (60), `kappa` / `kappa_ngan` CometKiwi confidence floors (0.40 / 0.30), `kappa_cat` / `kappa_cat_ngan` drop floors (0.30 / 0.22), `do_dai_nguon_ngan` short-source length (12), `uu_tien` preferred model (A), `chi_doi_khi_truot` (true: switch to B only when A fails the rules) |
| `mo_hinh` | model ids used by the pipeline: `api_tuy_chon` = Gemma 4 26B-A4B via API, `qwen3_14b` = Qwen3-14B-AWQ |
| `huong` | translation direction (`vi_zh`) |
| `go_nhan` | number of units where a label added by the model was removed, per model |
| `kaggle` | run cards: decoding parameters, versions, token counts, retries, timing of each translation and scoring run |
| `co_dinh` | reviewed dictionaries (file name, SHA256, number of entries), units and pairs per fixed source |
| `chuan_bi` | preparation step: input file and its SHA256, number of pairs, routing counts, number of translation units |
| `don_vi` / `cap` | selection results per unit / per pair (`giu` kept, `cat_rieng` not released, `cach_chon` selection route, `ly_do_cat` reason for dropping) |

Selection route names in `cach_chon`: `giong_het` identical; `gan_giong_A` similar, A kept; `khac_chu_cung_nghia_A`
different wording, A kept; `uu_tien_duoi_san_A` A below floor, kept; `chi_A` / `chi_B` only that model passed;
`tu_dien_p6` / `tu_dien_p7` name / value dictionaries; `quy_tac_co_khong` yes/no table; `quy_tac_thoi_han` duration
pattern; `quy_tac_khoi_luong` weight pattern; `quy_tac_ten_hang` brand; `quy_tac_do_luong` measurement;
`quy_tac_chep_nguyen` copy; `quy_tac_ma_model` model code; `quy_tac_co_chu` letter size.

## Related files in this repository

- `data/cleaned/tiki/tiki_eda.parquet` – the cleaned Tiki product table this release was built from.
- `data/description_zh_to_vi/` – the companion set in the other direction (1688.com, Chinese → Vietnamese).

## Contact

Prepared by Nhật Anh (data processing, description pairs) for the team's thesis project. Please report errors
through the Community tab of this dataset.
