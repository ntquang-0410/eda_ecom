# description_zh_to_vi: Chinese → Vietnamese product attribute pairs, re-translated by LLMs

Release date: 2026-10-22 · Build tag: `20261022_2126` · Folder: `data/description_zh_to_vi/`

This folder holds **394,662 Chinese–Vietnamese attribute pairs** (attribute name + attribute value) taken from
**16,342 products** listed on 1688.com, in 9 product categories. The Chinese side comes from the product
descriptions (attribute tables and SKU lists). The Vietnamese side is a **new machine translation**: every Chinese
attribute name and value was translated independently by two LLMs, the outputs were checked by rules and by a
reference-free quality-estimation model (CometKiwi), and one version was kept per item. Frequent names and values
use a dictionary reviewed by the dataset author.

It replaces the Vietnamese side shipped by 1688 (Alibaba's own machine translation, see
`data/cleaned/1688/bilingual_zh_vi_desc_pairs.parquet`), which is of uneven quality. The Chinese side and the
pair structure are the same as in that file; only the Vietnamese side is new.

The data was built for a student thesis project on Vietnamese–Chinese machine translation for cross-border
e-commerce. It is **machine-generated, silver-quality data**: suitable for training, fine-tuning and analysis, but
**not a human reference** and not suitable as a test set without human checking (see *Limitations*).

**Update 28 Oct 2026:** `description_products_zh_vi_pair.parquet` (one row per product, with the new column `desc_nguon_dich`) was added to this folder. The pair table and `provenance.json` did not change.

## Files

| File | Rows | Columns | Size (bytes) | SHA256 |
|---|---:|---:|---:|---|
| `description_pairs_zh_vi.parquet` | 394,662 | 9 | 9,597,601 | `e8c5cfed878268e8f63cd8760f2dbd91f4c420f21107b76082d0378a42ffb9ba` |
| `description_products_zh_vi_pair.parquet` | 16,342 | 9 | 6,990,837 | `f6b715b5703b79fa3030abdba2a62e63815cb1c0724189cbbd69ae368a083ac1` |
| `provenance.json` | – | – | 12,064 | `02596448b88093421305ea8bba4ca7ea2591b733eac7c46eace221e10c213331` |
| `SHA256SUMS.txt` | – | – | 286 | checksums of the three files above |

```bash
cd data/description_zh_to_vi
sha256sum -c SHA256SUMS.txt
```

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/ntquang0410/zh-vie_ecom/data/description_zh_to_vi/description_pairs_zh_vi.parquet")
```

## Schema (`description_pairs_zh_vi.parquet`)

One row = one attribute of one product.

| Column | Type | Meaning |
|---|---|---|
| `pair_id` | int64 | Pair id, unique in this file. Same id space as `data/cleaned/1688/bilingual_zh_vi_desc_pairs.parquet` (ids of pairs not released are missing, so the ids have gaps). |
| `product_id` | string | 1688 product id. Joins to `data/snapshot/bilingual_zh_vi.parquet` and `data/cleaned/1688/*`. |
| `category` | string | Category assigned by the team: `shoes`, `mother_baby`, `home`, `bags`, `fashion`, `food`, `beauty`, `electronics`, `auto`. |
| `nhom_trung` | int64 | Near-duplicate product group ("duplicate group"), inherited from the description cleaning step. Products are linked when they share the same Chinese title, the same shop, the same cleaned description, or a near-identical description (MinHash Jaccard ≥ 0.8), with transitive linking (union-find). 6,566 groups. **Use it to split train/dev/test by group so that near-duplicate products do not leak across splits.** |
| `name_zh` | string | Attribute name, Chinese (e.g. `材质`). |
| `gia_tri_zh` | string | Attribute value, Chinese (e.g. `超细纤维`). Lists keep the `", "` separator. |
| `name_vi` | string | Attribute name, Vietnamese (e.g. `Chất liệu`). |
| `gia_tri_vi` | string | Attribute value, Vietnamese (e.g. `Sợi microfiber`). |
| `do_tin_cay` | string | Confidence label: `cao` (high), `vua` (medium), `thap` (low). Defined below. |

The column names are Vietnamese (`gia_tri` = value, `do_tin_cay` = confidence, `nhom_trung` = duplicate group) to
match the other files of this repository. No cell is empty and `pair_id` has no duplicates.

### `description_products_zh_vi_pair.parquet` (added 28 Oct 2026)

One row per product, built from the pair table of this folder. The columns are the same as the products table of the context-aware companion (`data/1688/descriptions/zh_to_vi_context/`), in the same order, so the two can be stacked.

| Column | Type | Meaning |
|---|---|---|
| `product_id`, `category` | string | As above. |
| `so_cap` | int64 | Number of pairs of the product in this release. |
| `do_tin_cay` | string | Lowest `do_tin_cay` among the pairs of the product. |
| `so_cap_khong_cao` | int64 | Number of pairs of the product whose label is not `cao`. |
| `description_zh`, `description_vi` | string | The attribute list as one text, one `name: value` per line (joined with `\n`), in `pair_id` order. It is the description published in `data/1688/full_dataset/1688_full_attributes_independent_zh_vi.parquet`. |
| `so_cap_thap` | int64 | Number of pairs of the product whose label is `thap`. |
| `desc_nguon_dich` | string | Model(s) that translated at least one pair of the product: see below. |

`desc_nguon_dich` names the model(s) that produced at least one pair of the product. The reviewed dictionaries and fixed rules contribute to every product,
so it only says which model, if any, also worked on it. A pair counts for **Gemma** (Gemma 4 26B-A4B, "model A") when the unit was kept from Gemma's output
(including units where both models gave the same text, which the pipeline credits to Gemma), and for **Qwen** (Qwen3-14B-AWQ, "model B") when Qwen's output
alone was kept. The per-pair choice (`cach_chon`) is not published in the pair table; the label was computed from the build file.

| Value | Meaning | Products |
|---|---|---:|
| `gemma4` | at least one pair was translated or corrected by Gemma, none by Qwen | 11,449 (70.1%) |
| `gemma4+qwen3_14b_awq` | both models worked on at least one pair | 4,892 (29.9%) |
| `qwen3_14b_awq` | at least one pair was translated or corrected by Qwen, none by Gemma | 0 (0.0%) |
| `tu_dien_quy_tac` | no pair was produced by a model: dictionary and rules only | 1 (0.0%) |

### Examples

| category | name_zh | gia_tri_zh | name_vi | gia_tri_vi | do_tin_cay |
|---|---|---|---|---|---|
| food | 储存条件 | 常温 | Điều kiện bảo quản | Nhiệt độ thường | cao |
| fashion | 上市年份/季节 | 2024年夏季 | Năm/mùa ra mắt | Mùa hè năm 2024 | cao |
| bags | 里料质地 | 涤纶 | Chất liệu lót | Polyester | cao |
| fashion | 主面料成分含量 | 70%（含）-80%（不含） | Tỉ lệ thành phần vải chính | Từ 70% đến dưới 80% | cao |
| bags | 风格 | 跨境风潮 | Phong cách | Xu hướng xuyên biên giới | vua |
| shoes | 适用场景 | 休闲 | Dịp sử dụng | Thường ngày | vua |
| shoes | 鞋垫材质 | 纺织品类 | Chất liệu lót giày | Chất liệu dệt may | thap |

## Statistics

| Category | Products | Pairs | `cao` | `vua` | `thap` |
|---|---:|---:|---:|---:|---:|
| shoes | 2,310 | 85,443 | 35,646 | 47,453 | 2,344 |
| mother_baby | 2,495 | 61,509 | 25,330 | 32,033 | 4,146 |
| home | 2,493 | 55,253 | 23,226 | 28,670 | 3,357 |
| bags | 2,878 | 51,534 | 32,273 | 18,450 | 811 |
| fashion | 1,350 | 45,909 | 20,513 | 21,970 | 3,426 |
| food | 1,420 | 28,193 | 13,874 | 13,425 | 894 |
| beauty | 1,317 | 26,103 | 14,728 | 9,434 | 1,941 |
| electronics | 1,108 | 20,948 | 11,784 | 8,535 | 629 |
| auto | 971 | 19,770 | 11,334 | 7,785 | 651 |
| **Total** | **16,342** | **394,662** | **188,708** | **187,755** | **18,199** |

- **Heavy repetition.** The 394,662 pairs contain only **57,877 distinct Chinese (name, value) combinations**
  (1,831 distinct names, 50,849 distinct values), because the same attributes (`品牌`, `材质`, `是否跨境出口专供货源: 是`,
  ...) occur in most products. Each distinct Chinese (name, value) always has exactly one Vietnamese translation in
  this file. Deduplicate or down-weight if you do not want frequent attributes to dominate training.
- Length (characters, median / 99th percentile / max): `gia_tri_zh` 3 / 147 / 5,111; `gia_tri_vi` 9 / 344 / 13,825.
  Long values are SKU lists (colours × sizes).

## How the data was built

### 1. Source and preparation

- **Input:** `data/cleaned/1688/bilingual_zh_vi_desc_pairs.parquet` (405,275 pairs, SHA256 `48f11ff3…a0d447`), the
  pair-level output of the description cleaning step. That step had already removed personal data (shop names,
  phone numbers, contact details, URLs) and marked masked spans with tokens such as `[电话]`.
- **Pairs not released (10,613):** 1,371 pairs already excluded by the cleaning step, 74 pairs that step flagged for
  manual review, and 9,168 pairs whose translation could not be chosen with enough confidence (9,100 because of the
  value, 62 because of the attribute name, 6 brand values `是` that are not brand names).
- **Routing.** Each value goes to one of these routes; only the first needs a model:
  - *model* – sent to the two LLMs (338,273 pairs);
  - *copy* – values with no Chinese character are copied as-is (sizes, numbers, Latin brand names; 49,128 pairs);
  - *brand* – values of `品牌` (brand) are romanised with pinyin or taken from a reviewed brand-word table
    (`贝来` → `Beilai`; 13,878 pairs). These are readable romanisations, **not** verified official brand spellings;
  - *real code* – licence and registration numbers are copied as-is, with the province in brackets when the code
    contains a province abbreviation (`粤妆20240194` → `粤妆20240194 (Quảng Đông)`; 2,479 pairs);
  - *date* – `2023年05月24日` → `24/05/2023` (72 pairs).
- **Translation units.** Identical texts are translated once: 43,687 units (1,980 attribute names, 41,707 values).

### 2. Translation with two models

| | Model A | Model B |
|---|---|---|
| Model | Gemma 4 26B-A4B instruction-tuned (`gemma-4-26b-a4b-it`) | Qwen3-14B, 4-bit AWQ (`Qwen/Qwen3-14B-AWQ`, revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`) |
| Served by | Google AI Studio (Gemini API, OpenAI-compatible endpoint), 16 Oct 2026 | vLLM 0.32.0 on Kaggle, 2 × Tesla T4, 21 Oct 2026 |
| Decoding | temperature 0, `reasoning_effort: minimal` | temperature 0, thinking mode off |
| Batching | up to 20 units of the same type per request (≤ 2,400 Chinese characters) | one unit per request |
| Units translated | 43,662 of 43,687 (25 short units missing; model B covers them) | 43,687 |

Both models get the same English system prompt: translate a Chinese product attribute from 1688.com into
Vietnamese; output only the translation; translate every Chinese word; use Vietnamese names for Chinese places
(`浙江` → `Chiết Giang`); keep numbers, units, sizes, model codes, Latin brand names and mask tokens unchanged;
for a `", "`-separated list, keep the same items in the same order. The user message gives the product category and,
for values, the Chinese attribute name as context. If the output still contained Chinese characters, the model was
asked once more to translate it fully.

### 3. Rule checks (hard failures)

A translation fails if it: is empty; still contains Chinese characters; contains chatter, `<think>` tags or
markdown; translates the prompt context or adds a label; loops on repeated words; loses a number of the source;
loses a Latin code (with a digit or ≥ 2 capitals); changes the number of list items; loses a mask token; has a
length ratio (Vietnamese/Chinese) outside the 0.5 %–99.5 % range of its length bucket; or mistranslates the Chinese
unit `斤`.

**`斤` (jīn = 0.5 kg).** Vietnamese readers understand "cân" as 1 kg, so keeping the number of `斤` and writing
"cân" doubles the weight. Such outputs fail, unless they can be converted safely: before the checks, "N cân" is
rewritten to "N/2 kg" when the conversion is unambiguous (`4斤洗车水蜡` → `2 kg sáp nước rửa xe`; 211 units of A and
347 units of B were converted). Any other use of "cân" as a unit fails when the source has `斤`: `半斤` → "nửa cân",
`一斤` → "một cân", `130元一斤` → "130 tệ một cân", `（斤/43元）` → "(cân/43 tệ)", `(100-120)斤` → "(100-120) cân". Only
"cân nặng" (body weight) and "theo cân" (sold by weight) are allowed. In the released file, no pair whose Chinese side
has `斤` uses "cân" as a unit.

### 4. Quality estimation

Both outputs were scored with **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`), a reference-free QE model. The floor is
0.50, or 0.35 when the Chinese text has at most 8 characters, because CometKiwi scores very short strings low even
when they are correct. Scores are used only to set the confidence and to drop items; **the scores themselves are not
released**.

### 5. Choosing one translation per unit

Model A is preferred. Model B is used only when A fails the rules. CometKiwi never replaces a rule-passing A by B:
in a manual check of 100 risky pairs under an earlier rule (switch to B when its score was clearly higher),
10 of 12 clear errors were correct Gemma outputs replaced by fluent but wrong Qwen outputs that CometKiwi scored
higher (e.g. `汕头` "Shantou" rendered as "Thâm Quyền" = Shenzhen).

| Case (both outputs exist) | Kept | Value confidence | Value pairs |
|---|---|---|---:|
| A and B identical after normalisation, both pass rules | that text | `cao` if the best score ≥ floor, else `vua` | 90,157 |
| Different but similar (chrF ≥ 60, strings longer than 6 characters), both pass | A | `vua` if A ≥ floor, else `thap` | 31,541 |
| Clearly different, both pass, A ≥ floor | A | `vua` | 86,082 |
| Clearly different, both pass, A < floor but B ≥ floor | A | `thap` | 4,937 |
| Only A passes the rules, A ≥ floor | A | `thap` | 4,192 |
| Only B passes the rules, B ≥ floor | B | `thap` | 5,451 |
| Both fail; or one passes but is below the floor; or both below the floor | dropped | – | – |

### 6. Reviewed dictionary and fixed rules (override model output)

These outputs always have value/name confidence `cao`:

| Source | Applies to | Units | Pairs |
|---|---|---:|---:|
| Attribute-name dictionary (119 names reviewed by the dataset author) | names | 119 | 167,686 |
| Supplementary name dictionary (7 names, reviewed) | names | 7 | 1,306 |
| Yes/no table (`是` Có, `否` Không, `有` Có, `无`/`没有` Không có, `支持` Hỗ trợ, `不支持` Không hỗ trợ) | values | 385 | 55,154 |
| Value dictionary (280 reviewed entries) | values | 628 | 48,881 |
| Range pattern (`（含）`/`（不含）`/`以上`/`以下` → "Từ … đến …", "Từ … đến dưới …", "Trên …", "Từ … trở lên/xuống"; e.g. `70%（含）-80%（不含）` → `Từ 70% đến dưới 80%`) | values | 80 | 2,566 |
| Megapixels (`400万` under a `像素` attribute → `4 MP`) | values | 14 | 153 |

### 7. Where the released values come from

| Origin of `gia_tri_vi` | Pairs | Share |
|---|---:|---:|
| Both models produced the same text | 90,157 | 22.8 % |
| Model A (Gemma 4), chosen over B or only A passed | 126,752 | 32.1 % |
| Model B (Qwen3-14B) only | 5,451 | 1.4 % |
| Dictionary, yes/no table and patterns | 106,754 | 27.0 % |
| Copy, brand, real-code and date rules | 65,548 | 16.6 % |

The pair-level `do_tin_cay` is the lower of the name confidence and the value confidence.

## Quality checks

The checks below were run on the dataset during development. They are small samples and are **not**
independent human annotation, so read them as indicative only.

1. **200-pair stratified sample** (build of 21 Oct 2026, before model B was replaced by Qwen3-14B and before the
   selection rule and the `斤` fix of section 3). The dataset author read all 200 pairs; 191 values were accepted as
   correct in one pass, and 9 disputed values were judged one by one in the product context together with an AI
   assistant (Claude). Attribute names: 100 % correct. Values, weighted to the whole dataset: **97.6 % correct,
   99.4 % correct or minor error**. By confidence: `cao` 100 %, `vua` 95.8 %, `thap` 91.7 %.
2. **100-pair risk-focused sample**, graded on the build of 22 Oct 2026 20:39 (the current selection rule, but before
   the `斤` fix and the range and megapixel patterns). The pairs were drawn from the riskiest ~39,300 pairs only:
   values that the earlier rule had taken from Qwen3-14B, low-confidence Gemma values, and attribute names taken from
   Qwen3-14B. It is **not** representative of the whole dataset. Graded by an AI assistant (Claude) and skimmed by
   the dataset author. Values: **88.6 % correct, 98.9 % correct or minor error** (67.8 % and 86.8 % on the same
   pairs under the earlier selection rule); names: 98.8 % correct. The one clear error left (a `斤` size written as
   "cân") is no longer present in this build.
3. **Automatic checks on this file:** 0 duplicate `pair_id`; 0 empty cells; 0 Chinese characters in `name_vi`;
   2,478 `gia_tri_vi` values contain Chinese characters, all of them real codes kept on purpose (route *real code*);
   PII scan: 0 phone numbers, 0 e-mail addresses.

## Limitations

- **Machine translation, no human reference.** Expect roughly 2–3 % wrong values overall and more in the `thap`
  group (≈ 8–13 % in the samples above). Filter on `do_tin_cay` if you need cleaner data.
- **Do not use as a test set as-is.** Build test references by human checking. This matters especially if you
  evaluate **Qwen-family models**: about 1.4 % of values come from Qwen3-14B alone, and 22.8 % are values where
  Qwen3-14B and Gemma 4 agreed. Scores of a Qwen model on Qwen-produced references can be inflated. The internal
  full version of this build has a per-pair origin column; ask the authors if you need it for an ablation.
- **Known recurring minor errors:** `独立站` → "Trạm độc lập"/"Trang web độc lập" (should be "website bán hàng
  riêng"); `双镜头` → "Song thấu kính" ("hai ống kính"); `吊牌` → "Thẻ bài" ("mác treo"); `纯色` → "Màu thuần"
  ("màu trơn"); clothing terms such as `套头`/`套头衫` (pullover) and `吊带` (camisole / two straps) are sometimes
  translated literally.
- **Brands** are pinyin romanisations, not official brand names.
- **`斤` quantities:** pairs that both models rendered with "cân" and that could not be converted safely (mostly size
  charts and tea priced per jīn) were dropped rather than guessed, so such items are under-represented.
- **English wording:** about 350 pairs (≈ 230 distinct values) are mostly English rather than Vietnamese, typically
  cosmetic product names (`MIDUOLA Velvet Matte Non-stick Lipstick`) and chemical names. Many of them are acceptable
  in Vietnamese e-commerce, but some are not translations.
- **Residual names of places and businesses:** about 18 values name a province or city (e.g. place of origin), and
  about 12 values contain a factory or company name fragment (e.g. `…双肩包工厂`). These are business names, not
  personal data, and were kept.
- **No train/dev/test split** is provided. Split by `nhom_trung`.
- The CometKiwi floors (0.50 / 0.35) and the chrF threshold (60) were set from score distributions and small manual
  reads, not tuned on labelled data.

## Licences and terms of the tools used

- **Source text:** product listings on 1688.com, collected for non-commercial academic research.
- **Model A outputs** were produced through the **Gemini API (Google AI Studio)**. The Gemma 4 weights are
  Apache 2.0, but use of the API service is governed by the *Gemini API Additional Terms of Service*, which include
  a restriction on using the Services to develop models that compete with Google's services. Whether this applies
  to training a small domain-specific translation model is for each user to assess; please review those terms
  before using this data to train models, especially commercially.
- **Model B:** Qwen3-14B is Apache 2.0.
- **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`) is CC BY-NC-SA 4.0. It was used only to compute the confidence
  labels and to filter; no scores are distributed.
- An earlier candidate for model B (Tencent HY-MT1.5-7B) was dropped because of its licence terms. **None of its
  output is in this release.**
- This folder follows the repository-level licence declaration, subject to the third-party terms above. Intended
  use: non-commercial research and education.

## Provenance file

`provenance.json` is the build card written by the pipeline (keys in Vietnamese), with API key names and Kaggle
paths removed. Main keys:

| Key | Meaning |
|---|---|
| `nguong` | thresholds: `tau` chrF similarity (60), `kappa` / `kappa_ngan` CometKiwi floors (0.50 / 0.35), `uu_tien` preferred model (A), `chi_doi_khi_truot` (true: switch to B only when A fails the rules) |
| `mo_hinh` | model ids used by the pipeline: `api_tuy_chon` = Gemma 4 26B-A4B via API, `qwen3_14b` = Qwen3-14B-AWQ |
| `kaggle` | run cards: decoding parameters, versions, token counts, retries, timing of each translation and scoring run |
| `jin_tu_doi` | number of units where "cân" was converted to kg, per model |
| `co_dinh` | reviewed dictionaries (file name, SHA256, number of entries), the yes/no table, units and pairs per fixed source |
| `chuan_bi` | preparation step: input file and its SHA256, routing counts, number of translation units |
| `don_vi` / `cap` | selection results per unit / per pair (`giu` kept, `cat_rieng` not released, `cach_chon` selection route, `ly_do_cat` reason for dropping) |

Selection route names in `cach_chon`: `giong_het` identical; `gan_giong_A` similar, A kept; `khac_chu_cung_nghia_A`
different wording, A kept; `uu_tien_duoi_san_A` A below floor, kept; `chi_A` / `chi_B` only that model passed;
`tu_dien_p6` / `tu_dien_p6b` / `tu_dien_p7` dictionaries; `quy_tac_co_khong` yes/no table; `quy_tac_khoang` range
pattern; `quy_tac_diem_anh` megapixels; `quy_tac_chep_nguyen` copy; `quy_tac_ten_hang` brand; `quy_tac_ma_that` real
code; `quy_tac_ngay` date.

## Related files in this repository

- `data/snapshot/bilingual_zh_vi.parquet` – raw product table (Chinese + 1688's Vietnamese).
- `data/cleaned/1688/bilingual_zh_vi_desc_pairs.parquet` – the input pairs of this release, with the original 1688
  Vietnamese translation.
- `data/cleaned/1688/README_desc.md` – how the description pairs were cleaned, including `nhom_trung`.

Pipeline code: GitHub repository `ntquang-0410/eda_ecom` (branch `nhatanh_updating`), folder `src/song_ngu_1/`.

## Contact

Prepared by Nhật Anh (data processing, description pairs) for the team's thesis project. Please report errors
through the Community tab of this dataset.
