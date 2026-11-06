# description_vi_to_zh_cum: Vietnamese → Chinese product attributes, translated with product context

Release date: 2026-10-24 · Build tag: `20261024_1616` · Folder: `data/tiki/descriptions/vi_to_zh_context/` (this folder was called `data/description_vi_to_zh_cum/` before the repository was regrouped by source on 25 Oct 2026)

This folder holds the attribute lists of **13,187 products** listed on Tiki.vn (6 categories), translated from Vietnamese to
Simplified Chinese **one product at a time**: the models saw the whole attribute list of a product (and its title), not each
attribute alone. The same translation is shipped in **two tables**:

| Table | Rows | One row is | Use it to |
|---|---:|---|---|
| `description_pairs_vi_zh_ctx.parquet` | 87,112 | one attribute (name + value) of one product | train or evaluate attribute-level translation |
| `description_products_vi_zh.parquet` | 13,187 | one product, with its whole attribute list as one text | train or evaluate translation of a whole description in one pass |

The two tables are built from the same output, so they match character by character: the `description_*` text of a product is
its pairs joined in page order.

It is the context-aware companion of `data/tiki/descriptions/vi_to_zh/` (same products and attributes, every attribute translated on
its own). That earlier translation was the **starting draft** of this one: the models only had to review it with the product in view.
**If you train on this folder, do not also train on the pairs of `description_vi_to_zh` for the same products**
(see *Using the data*).

The data was built for a student thesis project on Vietnamese–Chinese machine translation for cross-border e-commerce. It is
**machine-generated, silver-quality data**: suitable for training, fine-tuning and analysis, but **not a human reference** and not
suitable as a test set without human checking (see *Limitations*).

## Files

| File | Rows | Columns | Size (bytes) | SHA256 |
|---|---:|---:|---:|---|
| `description_pairs_vi_zh_ctx.parquet` | 87,112 | 9 | 2,701,180 | `b61ea8e42e8fccdb0e4dad492e11439f9aa27e022ca003855399c37eb751bd46` |
| `description_products_vi_zh.parquet` | 13,187 | 9 | 1,757,768 | `a17b276c9fbd81be72fd98fe29f416c6e74f4aa72fc1fb7bdfdf8d28049511ca` |
| `provenance.json` | – | – | 8,229 | `a4dab5a81cb018e643a341d6cff7ec4b8521548eb3d9ad6b557e03bf40fb0bce` |
| `SHA256SUMS.txt` | – | – | 285 | checksums of the three files above |

```bash
cd data/tiki/descriptions/vi_to_zh_context
sha256sum -c SHA256SUMS.txt
```

```python
import pandas as pd
base = "hf://datasets/ntquang0410/zh-vie_ecom/data/tiki/descriptions/vi_to_zh_context/"
pairs = pd.read_parquet(base + "description_pairs_vi_zh_ctx.parquet")
products = pd.read_parquet(base + "description_products_vi_zh.parquet")
```

## Schema

### `description_pairs_vi_zh_ctx.parquet`

| Column | Type | Meaning |
|---|---|---|
| `pair_id` | int64 | Pair id, unique. Same id space as `description_vi_to_zh` (ids of pairs not released are missing, so the ids have gaps). |
| `product_id` | string | Tiki product id. Joins to `data/tiki/cleaned/tiki_eda.parquet`. |
| `category` | string | `food`, `home`, `electronics`, `auto`, `mother_baby`, `beauty`. |
| `name_vi` | string | Attribute name, Vietnamese. |
| `gia_tri_vi` | string | Attribute value, Vietnamese. Several values are joined with `", "`. |
| `name_zh` | string | Attribute name, Simplified Chinese. |
| `gia_tri_zh` | string | Attribute value, Simplified Chinese. |
| `do_tin_cay` | string | Confidence label of the pair: `cao` (high), `vua` (medium), `thap` (low). Defined below. |
| `nguon_ban_dich` | string | Where the Chinese value comes from: `tu_dien_quy_tac`, `giu_ban_dich_don_vi`, `gemma_sua`, `qwen_sua`, `dich_moi`. Defined below. |

### `description_products_vi_zh.parquet`

| Column | Type | Meaning |
|---|---|---|
| `product_id`, `category` | string | As above. |
| `so_cap` | int64 | Number of pairs of the product in this release. |
| `do_tin_cay` | string | Lowest `do_tin_cay` among the pairs of the product. |
| `so_cap_khong_cao` | int64 | Number of pairs of the product whose label is not `cao`. Use it to choose your own threshold (`0` = every pair is `cao`). |
| `description_vi`, `description_zh` | string | The attribute list of the product as one text, `name: value; name: value; ...`, in page order (same layout as the cleaned description of the Tiki EDA step). |
| `so_cap_thap` | int64 | Number of pairs of the product whose label is `thap`. The product label is `thap` as soon as one pair is `thap`, so this count tells a product with one uncertain pair from one with many. Added on 25 Oct 2026 (the other columns did not change); the products table of `data/1688/descriptions/zh_to_vi_context/` has the same columns. |
| `desc_nguon_dich` | string | Model(s) that translated or corrected at least one pair of the product: `gemma4`, `qwen3_14b_awq`, `gemma4+qwen3_14b_awq` or `tu_dien_quy_tac`. Added on 28 Oct 2026; see below. |

The product table does not repeat the attribute list as separate fields. When you need the attributes one by one (a value can itself contain `;` or `: `, so
splitting the text is not safe), take the rows of the pair table with the same `product_id`, sorted by `pair_id`, which is page order:
`pairs.sort_values("pair_id").groupby("product_id")`.

The column names are Vietnamese (`gia_tri` = value, `do_tin_cay` = confidence, `nguon_ban_dich` = translation source, `so_cap` = number of
pairs) to match the other files of this repository. No cell is empty, `pair_id` and `product_id` have no duplicates, and `so_cap`
sums to the number of rows of the pair table. Attributes removed for privacy (see below) are absent from both tables.

### `desc_nguon_dich` (translation source of a product)

Added on 28 Oct 2026 as the **last column** of the products table; the other columns did not change. It names the model(s) that translated or corrected at
least one pair of the product. The reviewed dictionaries and fixed rules contribute to every product, so the column only says which model, if any,
also worked on it.

| Value | Meaning | Products |
|---|---|---:|
| `gemma4` | at least one pair was translated or corrected by Gemma, none by Qwen | 7,690 (58.3%) |
| `gemma4+qwen3_14b_awq` | both models worked on at least one pair | 67 (0.5%) |
| `qwen3_14b_awq` | at least one pair was translated or corrected by Qwen, none by Gemma | 86 (0.7%) |
| `tu_dien_quy_tac` | no pair was produced by a model: dictionary and rules only | 5,344 (40.5%) |

A pair counts for **Gemma** (Gemma 4 26B-A4B) when its `nguon_ban_dich` is `gemma_sua` or `dich_moi`, or when it is `giu_ban_dich_don_vi` and the draft in
`data/tiki/descriptions/vi_to_zh/` had chosen Gemma's output for it. It counts for **Qwen** (Qwen3-14B-AWQ) when its `nguon_ban_dich` is `qwen_sua`, or the draft had chosen Qwen's output
alone. A draft pair where both models produced the same text is credited to Gemma, which the pipeline prefers, so the share of Qwen is a lower bound.
The draft's per-pair choice is not published; the label was computed from the build files.

### Examples (pairs)

| category | name_vi | gia_tri_vi | name_zh | gia_tri_zh | do_tin_cay | nguon_ban_dich |
|---|---|---|---|---|---|---|
| beauty | Hạn sử dụng | 3 năm | 保质期 | 3年 | cao | tu_dien_quy_tac |
| beauty | Thành phần | Hồng Sâm Hàn Quốc, Nhưng Hưu , 20 loại thảo mộc | 成分 | 韩国红参, 鹿茸, 20种草本植物 | vua | gemma_sua |
| beauty | Màu | Nâu Đen, Đen | 颜色 | 棕黑色, 黑色 | vua | gemma_sua |
| electronics | Màu sắc | Màu đen, Màu đỏ, Nâu Da Bò | 颜色 | 黑色, 红色, 牛皮棕 | thap | gemma_sua |
| home | Loại | Chân tăm mộc, Chân tăm đỏ | 类型 | 原木竹签, 红竹签 | thap | dich_moi |

## Confidence label (`do_tin_cay`)

The label of a pair is the **lowest** of three levels: the level of its value, the level of its name, and the label that the pair had in
`description_vi_to_zh` (a pair that does not exist there is treated as `thap`). The label of a product is the lowest label among its pairs.

**Value level**, from how much the two models agreed (the percentages are the exactly-correct share measured in the blind sample of
*Quality checks*; small samples, see the counts):

| Value level | Case | Pairs | Correct in sample |
|---|---|---:|---|
| `cao` | fixed entry (reviewed dictionary, rule, copy of a brand, number, measurement or code) | 72,850 | not re-sampled here; 40/40 in the sample of `description_vi_to_zh` |
| `cao` | both models kept the draft | 11,821 | 95% (38/40) |
| `cao` | both models made the same edit | 229 | 100% (15/15) |
| `vua` | only Gemma edited | 1,679 | 87% (39/45) |
| `vua` | only one model could review (the other returned an unusable answer) and kept the draft | 28 | not sampled |
| `thap` | both edited, differently | 214 | 72% (18/25) |
| `thap` | only Qwen edited (the draft is kept in 259 of them, Qwen's edit is released in 25, and 7 are new Qwen translations) | 291 | 50% (10/20, draft kept) |

Whatever the group, a pair that is a **new translation** (it had no draft, 56 pairs) or whose edit was **rejected by a rule so the draft was
kept** (12 pairs) is `thap` (68 pairs in total, spread over the groups above).

**Name level:** `cao` when the name is fixed or the draft was kept, `vua` when Gemma changed it.

| Label | Pairs | Share | Products (lowest pair label) | Share |
|---|---:|---:|---:|---:|
| `cao` | 76,820 | 88.2 % | 6,979 | 52.9 % |
| `vua` | 8,967 | 10.3 % | 4,982 | 37.8 % |
| `thap` | 1,325 | 1.5 % | 1,226 | 9.3 % |

In the blind sample (145 value pairs outside the dictionary strata, not weighted), the exactly-correct share by final label was
`cao` 13/13, `vua` 73/80, `thap` 34/52. Most of the other pairs are minor errors; no clearly wrong text was found in the sample.

### `nguon_ban_dich` (origin of the Chinese value)

| Value | Meaning | Pairs | Share |
|---|---|---:|---:|
| `tu_dien_quy_tac` | fixed entry: reviewed dictionary, yes/no table, duration and weight rules, copy of brands, numbers, measurements, model codes | 72,850 | 83.6 % |
| `giu_ban_dich_don_vi` | the draft from `description_vi_to_zh` was kept | 12,108 | 13.9 % |
| `gemma_sua` | Gemma 4 changed the draft | 2,073 | 2.4 % |
| `dich_moi` | no draft existed (the pair had been dropped there) and a model translated it | 56 | 0.1 % |
| `qwen_sua` | Qwen3-14B changed the draft (used only when the Gemma edit failed the rules) | 25 | < 0.1 % |

The origin describes the value. Some attribute **names** were changed by Gemma too (219 names, plus 1 new name).

## Statistics

| Category | Products | Pairs | `cao` | `vua` | `thap` |
|---|---:|---:|---:|---:|---:|
| food | 2,193 | 16,110 | 13,016 | 2,833 | 261 |
| home | 2,200 | 15,854 | 13,460 | 2,101 | 293 |
| electronics | 2,198 | 15,201 | 13,638 | 1,334 | 229 |
| auto | 2,199 | 14,931 | 13,443 | 1,228 | 260 |
| mother_baby | 2,198 | 12,944 | 12,132 | 710 | 102 |
| beauty | 2,199 | 12,072 | 11,131 | 761 | 180 |
| **Total** | **13,187** | **87,112** | **76,820** | **8,967** | **1,325** |

- **Heavy repetition.** The 87,112 pairs contain only **19,507 distinct Vietnamese (name, value) combinations** (829 distinct names,
  18,460 distinct values): the same attributes (`Xuất xứ thương hiệu`, `Sản phẩm có được bảo hành không?`, ...) occur in most products.
  Deduplicate or down-weight if you do not want frequent attributes to dominate training.
- **The same Vietnamese text can have several Chinese translations.** Because the translation depends on the product, 125 (name, value)
  combinations have more than one Chinese value (for example `Chất liệu: Foam` → `海绵`, `记忆棉` or `泡沫`) and 35 attribute names have
  more than one Chinese name (for example `Bộ chuyển động` → `变速器`, `变速系统` or `套件`). This is intended, but it differs from
  `description_vi_to_zh`, where one text always has one translation.
- Pair table, characters (median / 99th percentile / max): `gia_tri_vi` 8 / 279 / 4,193; `gia_tri_zh` 3 / 125 / 1,399.
- Product table: 6.6 attributes per product (median 6, max 31); `description_vi` 197 / 1,264 / 8,110 characters and `description_zh`
  83 / 530 / 2,745 characters (median / 99th percentile / max). 6,979 products have only `cao` pairs.

## How the data was built

### 1. Source, preparation and the draft

- **Input:** `data/tiki/cleaned/tiki_eda.parquet` (13,200 products, SHA256 `7af2a262…70d3bd`). Attributes were extracted and cleaned exactly as for
  `description_vi_to_zh` (89,300 pairs).
- **Pairs not released:** 2,033 pairs removed before translation (name and address of the organisation responsible for the product, contact
  details), 128 removed by a privacy filter (shop names and street addresses), and 27 that had no usable translation (no draft and both models
  failed the rules). This leaves 87,112 pairs.
- **The draft:** the translation of `description_vi_to_zh` (build `20261023_1656`, parquet SHA256 `0177e2b2…e78fc`), made attribute by attribute
  by Gemma 4 and Qwen3-14B with rules, reviewed dictionaries and a CometKiwi quality estimate.
- **Fixed entries:** entries that need no review (reviewed name and value dictionaries, yes/no table, duration and weight rules, brands,
  numbers, measurements, model codes, letter sizes) were written into the input with their Chinese text and marked fixed. The models used them as
  context and were not allowed to output or change them. The unit `gr`/`gram` after a number was written `g` in fixed measurements.

### 2. Translation of the whole product

Each product (13,187; 15 long ones were cut into two or more parts of at most 2,500 characters, each part told the names of the other parts) is one
request with: the product title (context only, never translated and **not released**), the category, and the attribute list as a JSON array.
Each item carries the Vietnamese name and value and, per field, either the fixed Chinese text or the draft. The model must:

- keep a draft **exactly** when it is correct;
- change it only when the other attributes or the title show that its meaning is wrong (a colour, size, part, flavour, material or product type),
  when it leaves Vietnamese words or ordinary English words untranslated, when it writes `gr`, or when it loses or adds meaning; never to change
  only the style;
- translate a field that has no draft;
- answer with a JSON array that contains only the fields that are not fixed, plus a short reason for every change.

A reply that is not valid JSON, or that has missing, duplicated or renumbered items, is rejected and asked once more; a reply that still contains
Vietnamese words is also asked once more. 13,208 requests (parts) were prepared; the 5,277 whose fields were all fixed were not sent.

| | Model A | Model B |
|---|---|---|
| Model | Gemma 4 26B-A4B instruction-tuned (`gemma-4-26b-a4b-it`) | Qwen3-14B, 4-bit AWQ (`Qwen/Qwen3-14B-AWQ`, revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`) |
| Served by | Google AI Studio (Gemini API, OpenAI-compatible endpoint), 24 Oct 2026, 6.5 h, 7,997 requests | vLLM 0.32.0 on Kaggle, 2 × Tesla T4, 24 Oct 2026, 2.7 h |
| Decoding | temperature 0, `reasoning_effort: minimal` | temperature 0, thinking mode off |
| Unusable answers after the retry | 2 of 13,208 requests (draft kept) | 4 of 13,208 requests (draft kept) |

### 3. Rules and selection

Every changed field must pass the rules of `description_vi_to_zh` (no leftover Vietnamese word, no added chatter, same numbers, same codes, same
number of list items, no Traditional characters, ...) and one more rule: **an edit that loses more than half of the Chinese characters of the draft
is rejected**, because the models sometimes went back to the English source text. A rejected edit falls back to the draft.

The released value is the first of: fixed entry, Gemma's text if it passes, Qwen's text if it passes, the draft. In total 2,073 values and 219 names
come from Gemma, 25 values from Qwen, 12,108 values stay as in the draft and 56 values are new.
Stray `_x000D_` characters (a line break of the source spreadsheet, 4 values) were removed.

## Quality checks

The checks below are small samples graded by an AI assistant (Claude), **not** independent human annotation. Read them as indicative.

1. **Blind sample of the real run, 160 items** (24 Oct 2026; products not used in any earlier test; values up to 250 characters; the order of the
   versions was shuffled and the key opened only after grading). Counts are exactly-correct values.

   | Group | Pairs | Final text correct in sample | Finding |
   |---|---:|---|---|
   | both models kept the draft | 11,821 | 38/40 (95 %) | 2 minor errors |
   | both made the same edit | 229 | 15/15 | edit better than the draft in 14, equal in 1 |
   | only Gemma edited | 1,679 | 39/45 (87 %) | edit better than the draft in 20, worse in 4, equal in 21 |
   | both edited, differently | 214 | 18/25 (72 %) | Gemma better than the draft in 13, worse in 3, equal in 8; Qwen exactly correct in 9 of 25 |
   | only Qwen edited (draft kept) | 291 | 10/20 (50 %) | Qwen better than the draft in 8, worse in 5, equal in 7 |
   | attribute names changed by Gemma | 219 | 11/15 | better than the draft in 8, worse in 4, equal in 3 |

   Over the 85 sampled value edits of Gemma: **better than the draft 47, worse 7, equal 30**; the draft was clearly wrong in 9 of them, the Gemma
   text in none. The 7 regressions are of one kind: Gemma went back to the source (English or slang words such as `Silver`, `Zoo alphabet`, `6ly`),
   dropped an English sentence of the source, or changed the order of items. No clearly wrong final text was found in the 160 items (29 minor errors).
   Weighted by the sizes of the groups, about **93 % of the values outside the fixed entries are exactly correct** (about 98–99 % of all values; the fixed entries
   were correct in the sample of `description_vi_to_zh`). The same kind of estimate for the unit-level draft was about 81 % for machine-translated values
   in that data card, but the samples differ, so read it as "clearly better", not as an exact gain.
2. **Pilot on 200 products** (23 Oct 2026, a first version of the pipeline): 40 fields differed from the draft; the new text was better in 22, worse in
   6 and equal in 12 (after the rule on lost Chinese characters was added, which was decided after reading this pilot). It is not included in the numbers above.
3. **Difference from the draft:** compared with `description_vi_to_zh`, **3,008 of the 87,056 common values (3.5 %) and 228 names differ** (exact string
   comparison, so some differences are only spacing or punctuation), and this folder has 56 more pairs.
4. **Automatic checks on both tables:** no duplicate `pair_id` or `product_id`; no empty cell; no Chinese character in the Vietnamese columns; no
   Vietnamese letter in `name_zh`. 1,070 `gia_tri_zh` values contain Vietnamese letters, 984 of them brand names copied as written. 23,449 `gia_tri_zh`
   values contain no Chinese character (copied brands, numbers, measurements and codes). Privacy scan of every text column: 0 phone numbers, 0 e-mail
   addresses, 0 URLs, 0 pairs caught by the privacy filter.

## Using the data

- **Split by `product_id`.** Keep every row of a product (pairs and the product row) in the same split. There is no train/dev/test split in this folder.
- **Do not mix with the pairs of `description_vi_to_zh` for the same products.** About 3.5 % of the values have a different translation there.
- **Filter or weight by `do_tin_cay`.** For a cleaner subset use `cao`, or the products with `so_cap_khong_cao == 0` (6,979). To drop only the
  products with several low pairs, filter on `so_cap_thap` (`== 0`: 11,961 products; `<= 1`: 13,095).
- **Down-weight repeated attributes** (83.6 % of the pairs are fixed entries and repeat across products).
- **Use the product table to train a model that translates a whole description,** and the pair table to train a model that translates single
  attributes. The title of a product was used as context only and is not released; it is in `data/tiki/cleaned/tiki_eda.parquet`.

## Limitations

- **Machine translation, no human reference.** Roughly 7 % of the values outside the fixed entries are not exactly correct in the sample (minor errors
  for the most part). Do not use this data as a test set as it is.
- **Qwen-family models.** Qwen3-14B wrote 25 of the released values and agreed with Gemma on the 229 same edits and on the 11,821 kept drafts; the
  draft itself has a small share of Qwen text (see `description_vi_to_zh`). Scores of a Qwen model on this data can be inflated.
  `nguon_ban_dich` lets you remove `qwen_sua`.
- **Context is partial.** The models saw the attribute list and the title, not the description text or the images. Errors remain, mostly colour
  names, car and helmet model names, herbal and place names, and ingredient names; some values still mix Latin and Chinese.
- **Attributes of a product that were removed for privacy** (shop names, street addresses, names and contacts of responsible organisations) are
  missing from both tables, so a product text may have fewer attributes than the page.
- **Style is not normalised.** Spaces between numbers, Latin words and Chinese characters, and the commas of lists, follow the model output.
- **Brands** are copied as written in the listing. Some brand values are the name of the shop that sells the product (a business name); they were kept.
- **Edits without a reason.** For 361 changed values the model gave no reason; they are in the data, and in the sample of unreasoned edits Gemma was better than the draft in 5 of 15 and worse in none.

## Licences and terms of the tools used

- **Source text:** product listings on Tiki.vn, collected for non-commercial academic research.
- **Model A outputs** were produced through the **Gemini API (Google AI Studio)**. The Gemma 4 weights are Apache 2.0, but use of the API service is
  governed by the *Gemini API Additional Terms of Service*, which include a restriction on using the Services to develop models that compete with
  Google's services. Whether this applies to training a small domain-specific translation model is for each user to assess; please review those terms
  before using this data to train models, especially commercially.
- **Model B:** Qwen3-14B is Apache 2.0.
- **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`) is CC BY-NC-SA 4.0. It was used to compute the quality labels of the draft in `description_vi_to_zh`, which
  enter `do_tin_cay` here; no scores are distributed.
- This folder follows the repository-level licence declaration, subject to the third-party terms above. Intended use: non-commercial research and education.

## Provenance file

`provenance.json` is the build card written by the pipeline (keys in Vietnamese), with API key names, test calls and Kaggle paths removed. Main keys:

| Key | Meaning |
|---|---|
| `ban_dung` | build tag and the two table file names |
| `nguon` | the input file and its SHA256, and the draft it started from |
| `chuan_bi` | preparation counts: pairs, requests, fixed fields, draft fields, fields with no draft |
| `cau_lenh_he_thong_v2` | the full system prompt sent to both models |
| `chay` | run cards of Gemma 4 and Qwen3-14B: decoding parameters, versions, token counts, retries, timing |
| `ghep` | assembly counts: origin of each value, agreement between the models, label counts |

## Related files in this repository

- `data/tiki/descriptions/vi_to_zh/` – the attribute-by-attribute translation of the same pairs (the draft of this folder).
- `data/tiki/cleaned/tiki_eda.parquet` – the cleaned Tiki product table (titles, shops, prices, original attributes).

## Contact

Prepared by Nhật Anh (data processing, description pairs) for the team's thesis project. Please report errors through the Community tab of this dataset.
