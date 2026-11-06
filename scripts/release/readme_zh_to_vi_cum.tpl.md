# description_zh_to_vi_context: Chinese → Vietnamese product attributes, reviewed with product context

Release date: @@DATE@@ · Build tag: `@@NHAN@@` · Folder: `data/1688/descriptions/zh_to_vi_context/`

This folder holds the attribute lists of **@@N_PROD@@ products** listed on 1688.com (9 categories) with a Vietnamese translation that was **reviewed
one product at a time**: the models saw the whole attribute list of a product (and its title), not each attribute alone. The same translation is
shipped in **two tables**:

| Table | Rows | One row is | Use it to |
|---|---:|---|---|
| `description_pairs_zh_vi_ctx.parquet` | @@N_PAIRS@@ | one attribute (name + value) of one product | train or evaluate attribute-level translation |
| `description_products_zh_vi.parquet` | @@N_PROD@@ | one product, with its whole attribute list as one text | train or evaluate translation of a whole description in one pass |

The two tables are built from the same output, so they match character by character: the `description_*` text of a product is its pairs joined in
page order.

It is the context-aware companion of `data/1688/descriptions/zh_to_vi/` (same products and attributes, every attribute translated on its own). That
earlier translation was the **starting draft** of this one: the models only had to review it with the product in view. The @@N_SET1@@ pairs of
`zh_to_vi` are all here with the same id, product and Chinese text; this folder adds @@N_NEW@@ pairs that `zh_to_vi` did not release (there, a translation
could not be chosen with enough confidence). **If you train on this folder, do not also train on the pairs of `zh_to_vi` for the same products** (see *Using the data*).

The data was built for a student thesis project on Vietnamese–Chinese machine translation for cross-border e-commerce. It is
**machine-generated, silver-quality data**: suitable for training, fine-tuning and analysis, but **not a human reference** and not suitable as a
test set without human checking (see *Limitations*).

**Read this before using the product label.** No product is labelled `cao` in this folder. The label of a product is the lowest label of its pairs,
and a 1688 product has about @@PAIRS_PER_PRODUCT@@ attributes, so every product has at least one pair that is not `cao` (a pair that was not `cao` in `zh_to_vi`,
a change by a model, or one of the @@N_NEW@@ new translations). To select clean data, use the **pair** label `do_tin_cay`, or the counts
`so_cap_khong_cao` and `so_cap_thap` of the product table (see *Using the data*).

## Files

| File | Rows | Columns | Size (bytes) | SHA256 |
|---|---:|---:|---:|---|
| `description_pairs_zh_vi_ctx.parquet` | @@N_PAIRS@@ | 10 | @@SIZE_PAIRS@@ | `@@SHA_PAIRS@@` |
| `description_products_zh_vi.parquet` | @@N_PROD@@ | 8 | @@SIZE_PROD@@ | `@@SHA_PROD@@` |
| `provenance.json` | – | – | @@SIZE_PROV@@ | `@@SHA_PROV@@` |
| `SHA256SUMS.txt` | – | – | @@SIZE_SUMS@@ | checksums of the three files above |

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

The product table does not repeat the attribute list as separate fields. When you need the attributes one by one (a value can itself contain `;` or
`: `, so splitting the text is not safe), take the rows of the pair table with the same `product_id`, sorted by `pair_id`, which is page order:
`pairs.sort_values("pair_id").groupby("product_id")`.

The column names are Vietnamese (`gia_tri` = value, `do_tin_cay` = confidence, `nguon_ban_dich` = translation source, `so_cap` = number of pairs,
`nhom_trung` = duplicate group) to match the other files of this repository. No cell is empty, `pair_id` and `product_id` have no duplicates, and
`so_cap` sums to the number of rows of the pair table.

### Examples (pairs)

@@EXAMPLES@@

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
| `cao` | @@PAIRS_CAO@@ | @@PAIRS_CAO_P@@ | 0 | 0 % |
| `vua` | @@PAIRS_VUA@@ | @@PAIRS_VUA_P@@ | @@PROD_VUA@@ | @@PROD_VUA_P@@ |
| `thap` | @@PAIRS_THAP@@ | @@PAIRS_THAP_P@@ | @@PROD_THAP@@ | @@PROD_THAP_P@@ |

Labels by origin of the value:

@@XTAB@@

### `nguon_ban_dich` (origin of the Vietnamese value)

| Value | Meaning | Pairs | Share |
|---|---|---:|---:|
@@SRC_TABLE@@

The origin describes the value. Gemma also changed @@N_NAMES_GEMMA@@ attribute **names** (and @@N_NAMES_RESTORED@@ names that Gemma had changed according to the values were put back to the
usual translation of the Chinese name, see *How the data was built*, step 4).

## Statistics

@@CAT_TABLE@@

- **Heavy repetition.** The @@N_PAIRS@@ pairs contain only **@@N_COMBOS@@ distinct Chinese (name, value) combinations** (@@N_NAMES_DISTINCT@@ distinct names,
  @@N_VALUES_DISTINCT@@ distinct values): the same attributes occur in most products. Deduplicate or down-weight if you do not want frequent attributes to dominate training.
- **The same Chinese text can have several Vietnamese translations.** Because the translation depends on the product, @@MULTI_COMBOS@@ (name, value) combinations have
  more than one Vietnamese value (some only differ by a synonym such as `Cotton` / `Bông`). This differs from `zh_to_vi`, where one text always has one translation.
  @@MULTI_NAMES@@ Chinese attribute names have more than one Vietnamese name.
- Pair table, characters (median / 99th percentile / max): `gia_tri_zh` @@LEN_ZH@@; `gia_tri_vi` @@LEN_VI@@.
- Product table: @@ATTR_PER_PRODUCT@@ attributes per product (mean, median, max); `description_zh` @@DESC_ZH@@ and `description_vi` @@DESC_VI@@ characters (median / 99th percentile / max).
- **@@N_HAN_VI@@ Vietnamese values keep Chinese characters on purpose.** They are licence and registration numbers (for example `粤G妆网备字2024456373 (Quảng Đông)`) that are
  copied as written with the province in brackets, as in `zh_to_vi`. They are not untranslated text. No attribute name contains Chinese characters in the Vietnamese column.

## How the data was built

### 1. Source, preparation and the draft

- **Input:** the pairs of `zh_to_vi` (@@N_SET1@@ released pairs) plus @@N_BACK@@ pairs that `zh_to_vi` had set aside because no translation could be chosen with enough confidence,
  minus @@N_DROP_PREP@@ very long values that had no draft (the 6 meaningless brand values `是` were not taken back). @@N_PAIRS_IN@@ pairs of @@N_PROD@@ products went to the models.
  Personal data (shop names, phone numbers, contact details, URLs) had already been removed in the cleaning step before `zh_to_vi`; masked spans stay masked
  (`[电话]` → `[SĐT]`).
- **The draft:** the unit-level translation of `zh_to_vi` (build `20261022_2126`, parquet SHA256 `@@SHA_SET1@@`), made attribute by attribute by Gemma 4 and Qwen3-14B with rules,
  reviewed dictionaries and a CometKiwi quality estimate. The pairs it had set aside have no draft.
- **Fixed entries:** values and names that need no review (reviewed name and value dictionaries, yes/no table, copies of values without Chinese characters, brand
  words, licence codes, dates, measurement ranges) were written into the input with their Vietnamese text and marked fixed. The models used them as context and could not change them.

### 2. Translation of the whole product

Each product (@@N_CLUSTERS@@ requests; 4 long products were cut in parts) is one request with: the product title (context only, never translated and **not released**; shop
names, long numbers and URLs removed), the category, and the attribute list as a JSON array. Each item carries the Chinese name and value and, per field, either the fixed
Vietnamese text or the draft. The model must keep a correct draft **exactly**, change it only when the other attributes or the title show that its meaning is wrong, when it leaves
Chinese characters or pinyin untranslated, when it writes `cân` for `斤`, or when it loses or adds meaning (never to change only the style), translate a field that has no draft,
and answer with a JSON array of the fields that are not fixed plus a short reason for every change. A reply that is not valid JSON, or that has missing or duplicated items, is rejected
and asked once more; a reply that still contains Chinese characters is also asked once more. The prompt (`phien_ban_cau_lenh`: `@@PROMPT_VERSION@@`) contains a short glossary of terms that
the unit-level draft translated too literally and that were found systematically wrong in the pilots (for example `独立站` → `Website độc lập`, not `Trạm độc lập`; `低帮` → `Cổ thấp`;
`镜头` → `ống kính`, not `thấu kính`; `单排扣` → `Một hàng khuy`; `纯色` → `Màu trơn`). The full prompt is in `provenance.json`.

| | Model A | Model B (fallback only) |
|---|---|---|
| Model | Gemma 4 26B-A4B instruction-tuned (`gemma-4-26b-a4b-it`) | Qwen3-14B, 4-bit AWQ (`Qwen/Qwen3-14B-AWQ`, revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`) |
| Served by | Google AI Studio (Gemini API, OpenAI-compatible endpoint), @@GEMMA_DATES@@: @@GEMMA_RUNS@@ runs (four of them in parallel), @@GEMMA_HOURS@@ h of run time in total, @@GEMMA_REQ@@ requests, @@GEMMA_TOK_IN@@ M input and @@GEMMA_TOK_OUT@@ M output tokens | vLLM 0.32.0, offline, 1 × NVIDIA RTX PRO 6000 (96 GB), @@QWEN_DATE@@, @@QWEN_SECONDS@@ s |
| Decoding | temperature 0, `reasoning_effort: minimal` | temperature 0, thinking mode off |
| Coverage | all @@N_CLUSTERS@@ requests; @@GEMMA_OK@@ valid answers, @@GEMMA_BAD@@ unusable after the retry (draft kept), @@GEMMA_MISSING@@ never answered | only the @@QWEN_N@@ requests where Gemma had a failed or unusable field; @@QWEN_OK@@ valid answers, @@QWEN_BAD@@ unusable |

One Gemma run of 25 Oct 2026 was repeated by a later run over the same requests (its output is a subset of the later one and is not in this release).

### 3. Rules and selection

Every changed field must pass the rules of `zh_to_vi` (no leftover Chinese character, no added chatter, same numbers, same codes, same number of list items, `斤` never written as `cân`,
...), a length rule (the Vietnamese length must be within the range seen for Chinese texts of the same length in `zh_to_vi`) and two rules added for review: **`latin_moi`** (the edit
must not add a Latin word that is not a Vietnamese syllable, not in the Chinese source, not in the draft, not a common loanword and not the pinyin of a part of the source) and **`mat_viet`**
(the edit must not lose more than half of the Vietnamese words of the draft: the model went back to the Latin source or to pinyin). A Latin word that is a common Vietnamese loanword or a
platform or manufacturer name is allowed (`balo`, `vali`, `gile`, `cskh`, `navy`, `macaron`, `loafer`, `velcro`, `Amazon`, `AliExpress`, `Tmall`, ...); other English phrases (`dropshipping`, `flagship`,
`livestream`, `memory foam`) are still rejected. A rejected edit falls back to the draft.

The released value is the first of: fixed entry, Gemma's text if it passes the rules, Qwen's text if it passes the rules **and the pair has a draft**, the draft. A Qwen translation of a pair
that has no draft is not used (@@QWEN_NEW_NOT_USED@@ such values; in the pilot Qwen's new translations were clearly worse than Gemma's); that pair is then not released. The Gemma answer failed the rules for
@@GEMMA_FAIL_VALUES@@ values and @@GEMMA_FAIL_NAMES@@ names; Qwen replaced the draft for @@QWEN_VALUES@@ values and @@QWEN_NAMES@@ name. @@N_PAIRS_DROPPED@@ pairs were dropped because no model produced a usable
translation and they had no draft.

### 4. Consistency step (applied before the labels)

A check of the whole release (see *Quality checks*) found three kinds of systematic errors that are corrected by a fixed, reproducible step:

- **Attribute names** with at least 50 pairs and one Vietnamese translation in at least 98 % of them were put back to that translation when Gemma had changed them according to the *values*
  (for example `颜色` (colour) renamed `Kích thước` for size values, `储藏方法` renamed `Dịp sử dụng`): @@N_NAMES_RESTORED@@ names. The name of an attribute follows its Chinese name.
- **Known wrong renderings** in values kept from the draft or produced by Gemma: @@N_VALUE_FIXES@@ values (`独立站` @@F_DOC_LAP@@, `中筒` @@F_ZHONG@@, `短筒` @@F_DUAN@@, `厚底` @@F_HOUDI@@,
  `日式` @@F_RISHI@@, `无印风` @@F_WUYIN@@, `镜头` @@F_JINGTOU@@, `拉链袋` @@F_LALIAN@@, `套头` @@F_TAOTOU@@). A replacement was accepted only if it caused no new rule failure;
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
   were read one by one (about 9 better than the draft, 5 equal, 3 worse); every value written by a model or by the consistency step was re-run through the rules (@@RULE_RERUN@@).
   Structure checks: no duplicate `pair_id` or `product_id`, no empty cell, all @@N_SET1@@ pairs of `zh_to_vi` present with the same id, product, category, duplicate group and Chinese text, the product label equals the lowest
   pair label, the counts `so_cap`, `so_cap_khong_cao`, `so_cap_thap` match the pair table, no Chinese attribute name in `name_vi`, no Vietnamese letter in `name_zh`.
   Privacy scan of every text column: 0 mobile phone numbers, 0 e-mail addresses, 0 URLs, 0 social-media handles after a contact label; strings that look like phone numbers are product or report codes.
4. **Difference from the draft:** compared with `zh_to_vi`, **@@N_DIFF_VALUES@@ of the @@N_SET1@@ common values (@@N_DIFF_VALUES_P@@) and @@N_DIFF_NAMES@@ names differ** (exact string comparison; the differences are mostly Gemma's
   edits, many of them glossary corrections such as `Màu thuần` → `Màu trơn` or small style changes), and this folder has @@N_NEW@@ more pairs.

## Using the data

- **Split by `nhom_trung`** (or at least by `product_id`). Keep every row of a product (pairs and the product row) in the same split. There is no train/dev/test split in this folder.
- **Do not mix with the pairs of `zh_to_vi` for the same products.** About @@N_DIFF_VALUES_P@@ of the values have a different translation there.
- **Filter or weight by the pair label `do_tin_cay`.** `cao` pairs are the cleanest subset (@@PAIRS_CAO_P@@ of the pairs). To drop products that have several low pairs, filter on `so_cap_thap`
  (`== 0`: @@PROD_THAP0@@ products; `<= 1`: @@PROD_THAP1@@; `<= 2`: @@PROD_THAP2@@). The product label alone is not useful for selection (see the note at the top).
- **Remove or keep new translations on purpose.** The @@N_NEW_KEPT@@ pairs that `zh_to_vi` did not release are `thap`; `nguon_ban_dich == "dich_moi"` identifies the ones translated by Gemma.
- **Down-weight repeated attributes** (@@FIXED_P@@ of the pairs are fixed entries and repeat across products).
- **Use the product table to train a model that translates a whole description,** and the pair table to train a model that translates single attributes. The title of a product
  was used as context only and is not released; it is in `data/1688/snapshot/bilingual_zh_vi.parquet`.

## Limitations

- **Machine translation, no human reference.** Do not use this data as a test set as it is.
- **Brand values** are romanised with pinyin or a reviewed brand-word table (`贝来` → `Beilai`), not verified official spellings. The rule is applied to every value of the attribute `品牌`
  (brand), also when the source value is not a brand but a word (for example `休闲` → `Xiuxian`, `广州` → `Guangzhou`, 3 values).
- **Licence and registration codes keep Chinese characters** (see *Statistics*).
- **Style is not normalised.** Synonyms (`Cotton` / `Bông`, `Tay ngắn` / `Ngắn tay`), spaces and the commas of lists follow the models; full-width punctuation of the source (`，（）`) is kept in
  about @@N_FULLWIDTH@@ Vietnamese values, as in `zh_to_vi`.
- **Context is partial.** The models saw the attribute list and the title, not the description text or the images. Errors remain, mostly colour and style names, product and place names, and ingredient names.
- **Qwen-family models.** Qwen3-14B wrote @@QWEN_VALUES@@ values and @@QWEN_NAMES@@ name; the draft itself has a small share of Qwen text (see `zh_to_vi`). `nguon_ban_dich` lets you remove `qwen_sua`.
- **Edits without a reason.** For @@N_NO_REASON@@ changed values or names the model gave no reason; they are in the data.

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
