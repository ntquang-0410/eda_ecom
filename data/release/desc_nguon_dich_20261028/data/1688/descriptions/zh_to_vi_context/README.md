# description_zh_to_vi_context: Chinese → Vietnamese product attributes, reviewed with product context

Release date: 2026-10-26 · Build tag: `20261026_2123` · Folder: `data/1688/descriptions/zh_to_vi_context/`

This folder holds the attribute lists of **16,342 products** listed on 1688.com (9 categories) with a Vietnamese translation that was **reviewed
one product at a time**: the models saw the whole attribute list of a product (and its title), not each attribute alone. The same translation is
shipped in **two tables**:

| Table | Rows | One row is | Use it to |
|---|---:|---|---|
| `description_pairs_zh_vi_ctx.parquet` | 401,692 | one attribute (name + value) of one product | train or evaluate attribute-level translation |
| `description_products_zh_vi.parquet` | 16,342 | one product, with its whole attribute list as one text | train or evaluate translation of a whole description in one pass |

The two tables are built from the same output, so they match character by character: the `description_*` text of a product is its pairs joined in
page order.

It is the context-aware companion of `data/1688/descriptions/zh_to_vi/` (same products and attributes, every attribute translated on its own). That
earlier translation was the **starting draft** of this one: the models only had to review it with the product in view. The 394,662 pairs of
`zh_to_vi` are all here with the same id, product and Chinese text; this folder adds 7,030 pairs that `zh_to_vi` did not release (there, a translation
could not be chosen with enough confidence). **If you train on this folder, do not also train on the pairs of `zh_to_vi` for the same products** (see *Using the data*).

The data was built for a student thesis project on Vietnamese–Chinese machine translation for cross-border e-commerce. It is
**machine-generated, silver-quality data**: suitable for training, fine-tuning and analysis, but **not a human reference** and not suitable as a
test set without human checking (see *Limitations*).

**Read this before using the product label.** No product is labelled `cao` in this folder. The label of a product is the lowest label of its pairs,
and a 1688 product has about 25 attributes, so every product has at least one pair that is not `cao` (a pair that was not `cao` in `zh_to_vi`,
a change by a model, or one of the 7,030 new translations). To select clean data, use the **pair** label `do_tin_cay`, or the counts
`so_cap_khong_cao` and `so_cap_thap` of the product table (see *Using the data*).

## Files

| File | Rows | Columns | Size (bytes) | SHA256 |
|---|---:|---:|---:|---|
| `description_pairs_zh_vi_ctx.parquet` | 401,692 | 10 | 10,100,372 | `1cc5500967ef337802cd49fa0f2a7c43e658491f4e271d8cce0c298fc40b9e9b` |
| `description_products_zh_vi.parquet` | 16,342 | 9 | 7,329,697 | `078827c76fcea1e98866abf53ccca4200635e3ac8e1dfea4730e6880da5bebc5` |
| `provenance.json` | – | – | 18,820 | `5f3d386d1df89f0ec3427b6824d6a67fb1aafa474db449f8d3c68038d0715530` |
| `SHA256SUMS.txt` | – | – | 285 | checksums of the three files above |

```bash
cd data/1688/descriptions/zh_to_vi_context
sha256sum -c SHA256SUMS.txt
```

```python
import pandas as pd
base = "hf://datasets/ntquang0410/zh-vie_ecom/data/1688/descriptions/zh_to_vi_context/"
pairs = pd.read_parquet(base + "description_pairs_zh_vi_ctx.parquet")
products = pd.read_parquet(base + "description_products_zh_vi.parquet")
```

## Schema

### `description_pairs_zh_vi_ctx.parquet`

Same columns as `zh_to_vi/description_pairs_zh_vi.parquet`, in the same order, plus `nguon_ban_dich` at the end.

| Column | Type | Meaning |
|---|---|---|
| `pair_id` | int64 | Pair id, unique. Same id space as `zh_to_vi` (ids of pairs not released are missing, so the ids have gaps). |
| `product_id` | string | 1688 product id. Joins to `data/1688/snapshot/bilingual_zh_vi.parquet` and `data/1688/cleaned/*`. |
| `category` | string | `shoes`, `mother_baby`, `home`, `bags`, `fashion`, `food`, `beauty`, `electronics`, `auto`. |
| `nhom_trung` | int64 | Near-duplicate product group ("duplicate group"), inherited from `zh_to_vi`. **Use it to split train/dev/test by group so that near-duplicate products do not leak across splits.** |
| `name_zh` | string | Attribute name, Chinese. |
| `gia_tri_zh` | string | Attribute value, Chinese. Lists keep the `", "` separator. |
| `name_vi` | string | Attribute name, Vietnamese. |
| `gia_tri_vi` | string | Attribute value, Vietnamese. |
| `do_tin_cay` | string | Confidence label of the pair: `cao` (high), `vua` (medium), `thap` (low). Defined below. |
| `nguon_ban_dich` | string | Where the Vietnamese value comes from: `tu_dien_quy_tac`, `giu_ban_dich_don_vi`, `gemma_sua`, `qwen_sua`, `dich_moi`. Defined below. |

### `description_products_zh_vi.parquet`

Same columns as the products table of `data/tiki/descriptions/vi_to_zh_context/`, with the Chinese text first because Chinese is the source here.

| Column | Type | Meaning |
|---|---|---|
| `product_id`, `category` | string | As above. |
| `so_cap` | int64 | Number of pairs of the product in this release. |
| `do_tin_cay` | string | Lowest `do_tin_cay` among the pairs of the product. |
| `so_cap_khong_cao` | int64 | Number of pairs of the product whose label is not `cao`. |
| `description_zh`, `description_vi` | string | The attribute list of the product as one text, `name: value; name: value; ...`, in page order. |
| `so_cap_thap` | int64 | Number of pairs of the product whose label is `thap`. |
| `desc_nguon_dich` | string | Model(s) that translated or corrected at least one pair of the product: `gemma4`, `qwen3_14b_awq`, `gemma4+qwen3_14b_awq` or `tu_dien_quy_tac`. Added on 28 Oct 2026; see below. |

The product table does not repeat the attribute list as separate fields. When you need the attributes one by one (a value can itself contain `;` or
`: `, so splitting the text is not safe), take the rows of the pair table with the same `product_id`, sorted by `pair_id`, which is page order:
`pairs.sort_values("pair_id").groupby("product_id")`.

The column names are Vietnamese (`gia_tri` = value, `do_tin_cay` = confidence, `nguon_ban_dich` = translation source, `so_cap` = number of pairs,
`nhom_trung` = duplicate group) to match the other files of this repository. No cell is empty, `pair_id` and `product_id` have no duplicates, and
`so_cap` sums to the number of rows of the pair table.

### `desc_nguon_dich` (translation source of a product)

Added on 28 Oct 2026 as the **last column** of the products table; the other columns did not change. It names the model(s) that translated or corrected at
least one pair of the product. The reviewed dictionaries and fixed rules contribute to every product, so the column only says which model, if any,
also worked on it.

| Value | Meaning | Products |
|---|---|---:|
| `gemma4` | at least one pair was translated or corrected by Gemma, none by Qwen | 13,109 (80.2%) |
| `gemma4+qwen3_14b_awq` | both models worked on at least one pair | 3,232 (19.8%) |
| `qwen3_14b_awq` | at least one pair was translated or corrected by Qwen, none by Gemma | 0 (0.0%) |
| `tu_dien_quy_tac` | no pair was produced by a model: dictionary and rules only | 1 (0.0%) |

A pair counts for **Gemma** (Gemma 4 26B-A4B) when its `nguon_ban_dich` is `gemma_sua` or `dich_moi`, or when it is `giu_ban_dich_don_vi` and the draft in
`data/1688/descriptions/zh_to_vi/` had chosen Gemma's output for it. It counts for **Qwen** (Qwen3-14B-AWQ) when its `nguon_ban_dich` is `qwen_sua`, or the draft had chosen Qwen's output
alone. A draft pair where both models produced the same text is credited to Gemma, which the pipeline prefers, so the share of Qwen is a lower bound.
The draft's per-pair choice is not published; the label was computed from the build files. 23 inherited pairs could not be traced to the draft and count for neither model.

### Examples (pairs)

| category | name_zh | gia_tri_zh | name_vi | gia_tri_vi | do_tin_cay | nguon_ban_dich |
|---|---|---|---|---|---|---|
| auto | 是否专利货源 | 否 | Có phải nguồn hàng có bằng sáng chế không | Không | cao | tu_dien_quy_tac |
| auto | 尺寸 | 可定制 | Kích thước | Có thể tùy chỉnh | thap | giu_ban_dich_don_vi |
| auto | 安装位置 | 前车窗挡风玻璃 | Vị trí lắp đặt | Kính chắn gió trước | vua | gemma_sua |
| bags | 图案 | 格纹 | Hoa văn | Caro | thap | dich_moi |
| auto | 保质期 | 三年 | Hạn sử dụng | 3 năm | thap | qwen_sua |

## Confidence label (`do_tin_cay`)

The label of a pair is the **lowest** of three levels: the level of its value, the level of its name, and the label that the pair had in `zh_to_vi`
(a pair that does not exist there is `thap`). The label of a product is the lowest label among its pairs.

**Value level**, from where the value comes (`nguon_ban_dich`, defined below):

| Value level | Case |
|---|---|
| `cao` | fixed entry (reviewed dictionary, yes/no rule, copy of a Latin brand, number, measurement, date or licence code); or the draft was kept |
| `vua` | Gemma changed the draft; or Gemma's change failed a rule and the draft was kept |
| `thap` | a new translation (the pair had no draft); or the value was changed by Qwen |

**Name level:** `cao` when the name is fixed or the draft was kept, `vua` when Gemma changed it, `thap` for a new name.

| Label | Pairs | Share | Products (lowest pair label) | Share |
|---|---:|---:|---:|---:|
| `cao` | 188,362 | 46.9 % | 0 | 0 % |
| `vua` | 188,086 | 46.8 % | 4,322 | 26.4 % |
| `thap` | 25,244 | 6.3 % | 12,020 | 73.6 % |

Labels by origin of the value:

| Origin of the value (`nguon_ban_dich`) | `cao` | `vua` | `thap` | Total |
|---|---:|---:|---:|---:|
| `tu_dien_quy_tac` | 129,330 | 40,526 | 2,482 | 172,338 |
| `giu_ban_dich_don_vi` | 59,032 | 135,252 | 12,389 | 206,673 |
| `gemma_sua` | 0 | 12,308 | 3,356 | 15,664 |
| `dich_moi` | 0 | 0 | 6,999 | 6,999 |
| `qwen_sua` | 0 | 0 | 18 | 18 |

### `nguon_ban_dich` (origin of the Vietnamese value)

| Value | Meaning | Pairs | Share |
|---|---|---:|---:|
| `tu_dien_quy_tac` | fixed entry: reviewed dictionary, yes/no table, copies of values without Chinese characters, brands, licence codes, dates, measurement ranges, and the values corrected by the consistency step | 172,338 | 42.9 % |
| `giu_ban_dich_don_vi` | the draft of `zh_to_vi` was kept | 206,673 | 51.5 % |
| `gemma_sua` | Gemma 4 changed the draft | 15,664 | 3.9 % |
| `dich_moi` | no draft existed and Gemma translated the value | 6,999 | 1.7 % |
| `qwen_sua` | Qwen3-14B changed the draft (used only when the Gemma change failed the rules) | 18 | 0.0 % |

The origin describes the value. Gemma also changed 942 attribute **names** (and 75 names that Gemma had changed according to the values were put back to the
usual translation of the Chinese name, see *How the data was built*, step 4).

## Statistics

| Category | Products | Pairs | `cao` | `vua` | `thap` |
|---|---:|---:|---:|---:|---:|
| shoes | 2,310 | 86,597 | 35,629 | 47,467 | 3,501 |
| mother_baby | 2,495 | 62,814 | 25,306 | 32,057 | 5,451 |
| home | 2,493 | 56,030 | 23,195 | 28,698 | 4,137 |
| bags | 2,878 | 52,797 | 32,266 | 18,456 | 2,075 |
| fashion | 1,350 | 46,626 | 20,495 | 21,986 | 4,145 |
| food | 1,420 | 28,586 | 13,805 | 13,493 | 1,288 |
| beauty | 1,317 | 26,904 | 14,682 | 9,479 | 2,743 |
| electronics | 1,108 | 21,260 | 11,759 | 8,560 | 941 |
| auto | 971 | 20,078 | 11,225 | 7,890 | 963 |
| **Total** | **16,342** | **401,692** | **188,362** | **188,086** | **25,244** |

- **Heavy repetition.** The 401,692 pairs contain only **61,674 distinct Chinese (name, value) combinations** (1,994 distinct names,
  54,466 distinct values): the same attributes occur in most products. Deduplicate or down-weight if you do not want frequent attributes to dominate training.
- **The same Chinese text can have several Vietnamese translations.** Because the translation depends on the product, 649 (name, value) combinations have
  more than one Vietnamese value (some only differ by a synonym such as `Cotton` / `Bông`). This differs from `zh_to_vi`, where one text always has one translation.
  15 Chinese attribute names have more than one Vietnamese name.
- Pair table, characters (median / 99th percentile / max): `gia_tri_zh` 3 / 144 / 5,111; `gia_tri_vi` 10 / 337 / 13,825.
- Product table: 24.6 / 22 / 53 attributes per product (mean, median, max); `description_zh` 399 / 1,187 / 5,488 and `description_vi` 1,044 / 3,143 / 14,889 characters (median / 99th percentile / max).
- **2,478 Vietnamese values keep Chinese characters on purpose.** They are licence and registration numbers (for example `粤G妆网备字2024456373 (Quảng Đông)`) that are
  copied as written with the province in brackets, as in `zh_to_vi`. They are not untranslated text. No attribute name contains Chinese characters in the Vietnamese column.

## How the data was built

### 1. Source, preparation and the draft

- **Input:** the pairs of `zh_to_vi` (394,662 released pairs) plus 9,162 pairs that `zh_to_vi` had set aside because no translation could be chosen with enough confidence,
  minus 1,326 very long values that had no draft (the 6 meaningless brand values `是` were not taken back). 402,498 pairs of 16,342 products went to the models.
  Personal data (shop names, phone numbers, contact details, URLs) had already been removed in the cleaning step before `zh_to_vi`; masked spans stay masked
  (`[电话]` → `[SĐT]`).
- **The draft:** the unit-level translation of `zh_to_vi` (build `20261022_2126`, parquet SHA256 `e8c5cfed878268e8f63cd8760f2dbd91f4c420f21107b76082d0378a42ffb9ba`), made attribute by attribute by Gemma 4 and Qwen3-14B with rules,
  reviewed dictionaries and a CometKiwi quality estimate. The pairs it had set aside have no draft.
- **Fixed entries:** values and names that need no review (reviewed name and value dictionaries, yes/no table, copies of values without Chinese characters, brand
  words, licence codes, dates, measurement ranges) were written into the input with their Vietnamese text and marked fixed. The models used them as context and could not change them.

### 2. Translation of the whole product

Each product (16,346 requests; 4 long products were cut in parts) is one request with: the product title (context only, never translated and **not released**; shop
names, long numbers and URLs removed), the category, and the attribute list as a JSON array. Each item carries the Chinese name and value and, per field, either the fixed
Vietnamese text or the draft. The model must keep a correct draft **exactly**, change it only when the other attributes or the title show that its meaning is wrong, when it leaves
Chinese characters or pinyin untranslated, when it writes `cân` for `斤`, or when it loses or adds meaning (never to change only the style), translate a field that has no draft,
and answer with a JSON array of the fields that are not fixed plus a short reason for every change. A reply that is not valid JSON, or that has missing or duplicated items, is rejected
and asked once more; a reply that still contains Chinese characters is also asked once more. The prompt (`phien_ban_cau_lenh`: `cum-zh_vi-v4b-2026-10-25`) contains a short glossary of terms that
the unit-level draft translated too literally and that were found systematically wrong in the pilots (for example `独立站` → `Website độc lập`, not `Trạm độc lập`; `低帮` → `Cổ thấp`;
`镜头` → `ống kính`, not `thấu kính`; `单排扣` → `Một hàng khuy`; `纯色` → `Màu trơn`). The full prompt is in `provenance.json`.

| | Model A | Model B (fallback only) |
|---|---|---|
| Model | Gemma 4 26B-A4B instruction-tuned (`gemma-4-26b-a4b-it`) | Qwen3-14B, 4-bit AWQ (`Qwen/Qwen3-14B-AWQ`, revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`) |
| Served by | Google AI Studio (Gemini API, OpenAI-compatible endpoint), 25-26 Oct 2026: 5 runs (four of them in parallel), 45.4 h of run time in total, 17,503 requests, 49.0 M input and 10.7 M output tokens | vLLM 0.32.0, offline, 1 × NVIDIA RTX PRO 6000 (96 GB), 26 Oct 2026, 1,345 s |
| Decoding | temperature 0, `reasoning_effort: minimal` | temperature 0, thinking mode off |
| Coverage | all 16,346 requests; 16,314 valid answers, 31 unusable after the retry (draft kept), 1 never answered | only the 1,733 requests where Gemma had a failed or unusable field; 1,718 valid answers, 15 unusable |

One Gemma run of 25 Oct 2026 was repeated by a later run over the same requests (its output is a subset of the later one and is not in this release).

### 3. Rules and selection

Every changed field must pass the rules of `zh_to_vi` (no leftover Chinese character, no added chatter, same numbers, same codes, same number of list items, `斤` never written as `cân`,
...), a length rule (the Vietnamese length must be within the range seen for Chinese texts of the same length in `zh_to_vi`) and two rules added for review: **`latin_moi`** (the edit
must not add a Latin word that is not a Vietnamese syllable, not in the Chinese source, not in the draft, not a common loanword and not the pinyin of a part of the source) and **`mat_viet`**
(the edit must not lose more than half of the Vietnamese words of the draft: the model went back to the Latin source or to pinyin). A Latin word that is a common Vietnamese loanword or a
platform or manufacturer name is allowed (`balo`, `vali`, `gile`, `cskh`, `navy`, `macaron`, `loafer`, `velcro`, `Amazon`, `AliExpress`, `Tmall`, ...); other English phrases (`dropshipping`, `flagship`,
`livestream`, `memory foam`) are still rejected. A rejected edit falls back to the draft.

The released value is the first of: fixed entry, Gemma's text if it passes the rules, Qwen's text if it passes the rules **and the pair has a draft**, the draft. A Qwen translation of a pair
that has no draft is not used (12 such values; in the pilot Qwen's new translations were clearly worse than Gemma's); that pair is then not released. The Gemma answer failed the rules for
814 values and 59 names; Qwen replaced the draft for 17 values and 1 name. 806 pairs were dropped because no model produced a usable
translation and they had no draft.

### 4. Consistency step (applied before the labels)

A check of the whole release (see *Quality checks*) found three kinds of systematic errors that are corrected by a fixed, reproducible step:

- **Attribute names** with at least 50 pairs and one Vietnamese translation in at least 98 % of them were put back to that translation when Gemma had changed them according to the *values*
  (for example `颜色` (colour) renamed `Kích thước` for size values, `储藏方法` renamed `Dịp sử dụng`): 75 names. The name of an attribute follows its Chinese name.
- **Known wrong renderings** in values kept from the draft or produced by Gemma: 34 values (`独立站` 7, `中筒` 10, `短筒` 7, `厚底` 3,
  `日式` 2, `无印风` 2, `镜头` 1, `拉链袋` 1, `套头` 1). A replacement was accepted only if it caused no new rule failure;
  these values have `nguon_ban_dich = tu_dien_quy_tac`.

## Quality checks

The checks below are small samples graded or read by an AI assistant (Claude), **not** independent human annotation. Read them as indicative.

1. **Blind pilot on 200 products** (24 Oct 2026, an earlier version of the prompt without the glossary; 137 items; the order of the versions was shuffled and the key opened only after
   grading). Only-Gemma edits of the draft (60 items): **Gemma better than the draft in 44, equal in 16, worse in 0**. Pairs where both models kept the draft (30 items): the draft was exactly
   correct in 23, a minor error in 7, clearly wrong in 0. New translations (20 items): **Gemma exactly correct in 15, minor error in 4, clearly wrong in 1; Qwen exactly correct in 5**,
   so Qwen's new translations are not used. Attribute names changed by Gemma (5 items): better in 3, worse in 1.
2. **Blind comparison of the final models** (26 Oct 2026, 20 new translations; Gemma and Qwen on the same items, the order shuffled): Gemma better in 14 (12 clearly), equal in 6, **Qwen better in 0**
   (Qwen errors: `荔枝` (lychee) → "mật lê"; `丝瓜络` (loofah) → "Sơ dừa"; `卡其` (khaki) → "màu nâu"; a Chinese brand left untranslated). This is the reason Qwen is only a fallback.
3. **Release audit** (26 Oct 2026, whole release; reading and rules): random sample of 45 Gemma edits of the draft (values up to 120 characters; no clearly wrong text; most are glossary corrections such as `Màu thuần` → `Màu trơn`,
   `Trạm độc lập` → `Website độc lập`, `Bông` → `Cotton`); random sample of 45 new translations (values up to 120 characters; no clearly wrong text found; mostly colour lists, platform names and product names); the 18 values changed by Qwen
   were read one by one (about 9 better than the draft, 5 equal, 3 worse); every value written by a model or by the consistency step was re-run through the rules (22,715 values re-checked; 2 still carry a rule flag, both inside very long lists kept from the draft).
   Structure checks: no duplicate `pair_id` or `product_id`, no empty cell, all 394,662 pairs of `zh_to_vi` present with the same id, product, category, duplicate group and Chinese text, the product label equals the lowest
   pair label, the counts `so_cap`, `so_cap_khong_cao`, `so_cap_thap` match the pair table, no Chinese attribute name in `name_vi`, no Vietnamese letter in `name_zh`.
   Privacy scan of every text column: 0 mobile phone numbers, 0 e-mail addresses, 0 URLs, 0 social-media handles after a contact label; strings that look like phone numbers are product or report codes.
4. **Difference from the draft:** compared with `zh_to_vi`, **15,705 of the 394,662 common values (4.0 %) and 930 names differ** (exact string comparison; the differences are mostly Gemma's
   edits, many of them glossary corrections such as `Màu thuần` → `Màu trơn` or small style changes), and this folder has 7,030 more pairs.

## Using the data

- **Split by `nhom_trung`** (or at least by `product_id`). Keep every row of a product (pairs and the product row) in the same split. There is no train/dev/test split in this folder.
- **Do not mix with the pairs of `zh_to_vi` for the same products.** About 4.0 % of the values have a different translation there.
- **Filter or weight by the pair label `do_tin_cay`.** `cao` pairs are the cleanest subset (46.9 % of the pairs). To drop products that have several low pairs, filter on `so_cap_thap`
  (`== 0`: 4,322 products; `<= 1`: 9,361; `<= 2`: 12,706). The product label alone is not useful for selection (see the note at the top).
- **Remove or keep new translations on purpose.** The 7,030 pairs that `zh_to_vi` did not release are `thap`; `nguon_ban_dich == "dich_moi"` identifies the ones translated by Gemma.
- **Down-weight repeated attributes** (42.9 % of the pairs are fixed entries and repeat across products).
- **Use the product table to train a model that translates a whole description,** and the pair table to train a model that translates single attributes. The title of a product
  was used as context only and is not released; it is in `data/1688/snapshot/bilingual_zh_vi.parquet`.

## Limitations

- **Machine translation, no human reference.** Do not use this data as a test set as it is.
- **Brand values** are romanised with pinyin or a reviewed brand-word table (`贝来` → `Beilai`), not verified official spellings. The rule is applied to every value of the attribute `品牌`
  (brand), also when the source value is not a brand but a word (for example `休闲` → `Xiuxian`, `广州` → `Guangzhou`, 3 values).
- **Licence and registration codes keep Chinese characters** (see *Statistics*).
- **Style is not normalised.** Synonyms (`Cotton` / `Bông`, `Tay ngắn` / `Ngắn tay`), spaces and the commas of lists follow the models; full-width punctuation of the source (`，（）`) is kept in
  about 318 Vietnamese values, as in `zh_to_vi`.
- **Context is partial.** The models saw the attribute list and the title, not the description text or the images. Errors remain, mostly colour and style names, product and place names, and ingredient names.
- **Qwen-family models.** Qwen3-14B wrote 17 values and 1 name; the draft itself has a small share of Qwen text (see `zh_to_vi`). `nguon_ban_dich` lets you remove `qwen_sua`.
- **Edits without a reason.** For 2,907 changed values or names the model gave no reason; they are in the data.

## Licences and terms of the tools used

- **Source text:** product listings on 1688.com, collected for non-commercial academic research.
- **Model A outputs** were produced through the **Gemini API (Google AI Studio)**. The Gemma 4 weights are Apache 2.0, but use of the API service is governed by the *Gemini API Additional
  Terms of Service*, which include a restriction on using the Services to develop models that compete with Google's services. Whether this applies to training a small domain-specific translation
  model is for each user to assess; please review those terms before using this data to train models, especially commercially.
- **Model B:** Qwen3-14B is Apache 2.0.
- **CometKiwi** (`Unbabel/wmt22-cometkiwi-da`) is CC BY-NC-SA 4.0. It was used to compute the quality labels of the draft in `zh_to_vi`, which enter `do_tin_cay` here; no scores are distributed.
- This folder follows the repository-level licence declaration, subject to the third-party terms above. Intended use: non-commercial research and education.

## Provenance file

`provenance.json` is the build card written by the pipeline (keys in Vietnamese), with API key names, test calls, error texts and local paths removed. Main keys:

| Key | Meaning |
|---|---|
| `ban_dung` | build tag and the two table file names |
| `nguon` | the draft this release started from and its SHA256 |
| `chuan_bi` | preparation counts: pairs, requests, origin of the fixed names and values |
| `cau_lenh_he_thong` | the full system prompt sent to both models |
| `chay` | run cards of the Gemma 4 runs and of the Qwen3-14B fallback run: decoding parameters, versions, token counts, retries, timing |
| `ghep` | assembly counts: origin of each value and name, rule failures, consistency step, label counts |

## Related files in this repository

- `data/1688/descriptions/zh_to_vi/` – the attribute-by-attribute translation of the same pairs (the draft of this folder).
- `data/1688/snapshot/bilingual_zh_vi.parquet` and `data/1688/cleaned/` – the 1688 product table and the cleaned layers (titles, shops, original attributes).
- `data/tiki/descriptions/vi_to_zh_context/` – the same kind of release for Tiki (Vietnamese → Chinese); the products tables have the same columns.

## Contact

Prepared by Nhật Anh (data processing, description pairs) for the team's thesis project. Please report errors through the Community tab of this dataset.
