# Rubric for checking e-commerce translations (Chinese <-> Vietnamese), version kiem_v1

You check translations of product listings from a shopping site. Each item has a SOURCE text and ONE translation. Judge the translation on its own, against the SOURCE only. There is no reference translation, and you are not told who or what produced the translation.

Direction per item: `zh->vi` means the SOURCE is Chinese and the translation is Vietnamese; `vi->zh` means the SOURCE is Vietnamese and the translation is Chinese.

Item types:
- `title`: a product title.
- `mo_ta`: a product attribute list. The SOURCE is numbered `[1] name: value ⏎ [2] name: value ...` (`⏎` marks a line break). The translation is one block; its attributes may be separated by `; ` or by `⏎` and are not numbered. Match each numbered source attribute to its translation yourself.
- `manh`: one single attribute `name: value`.

## Grades

Give each `title` and `manh` item exactly one grade. For a `mo_ta` item, decide for EACH numbered attribute whether it has a severe error, and report the numbers of those attributes.

- `Nang` (severe): the translation would mislead a buyer or is unusable. Any of:
  1. Wrong meaning: wrong or reversed product type, material, colour, size, quantity, function, target user (gender, age), compatibility (for example "for iPhone 17"), or any attribute value; an attribute name translated as a different attribute (for example material translated as colour).
  2. Numbers and units: a number missing, changed or added; a wrong unit; a wrong unit conversion.
  3. Brand, model, product name or code: changed into a DIFFERENT name or code (an invented or incorrect transliteration counts), or dropped.
  4. Missing or added information that changes what a buyer understands about the product: a feature, an attribute value, a number, a brand or a code that is absent from the translation, or present in the translation but not in the SOURCE. For a `mo_ta` item, a numbered source attribute with no translation at all is severe for that number.
  5. Broken output: SOURCE-language text left untranslated where a translation is required (for example Chinese characters inside a Vietnamese translation, when they are not a brand or code), a wrong output language, garbled or repeated text, leaked instructions or comments.
- `Nhe` (minor): a defect that does not mislead a buyer. For example a non-standard but understandable term, awkward or unnatural wording, a dropped or added marketing filler word (such as "hot", "new style", "wholesale" in a title), punctuation or capitalisation problems.
- `Dat` (pass): no defect, or only the differences listed below.

## Never a defect

These differences are never errors, never even `Nhe`:
- keeping a Latin-script brand, model or product name as in the SOURCE, or writing the SAME brand in the target script (transliteration, official local brand name);
- keeping or translating ordinary foreign (for example English) words that already appear in that foreign form in the SOURCE and are not names or codes;
- synonyms of equal standing, including a shortened and a full form of the same term, and equally standard attribute labels;
- unit and number formatting (for example L and a local unit word, kg and kilogram, decimal point and decimal comma, spacing, punctuation, slashes, brackets);
- word order that keeps the meaning, and being more or less formal.

The one exception to the first rule: turning a brand, model or product name into a DIFFERENT name is always `Nang`.

## How to work

- Check every key element of the SOURCE one by one: product type, each attribute name and value, each number and unit, each brand, model and code.
- Do not claim an omission or an addition unless you can point to the exact words.
- If the SOURCE itself is wrong, odd or contains a typo, judge whether the translation renders what the SOURCE says; do not penalise the translation for the SOURCE's own problems.
- Values such as "Không"/"否"/"无", "Có"/"是"/"有" are correct when they keep the yes/no or none meaning.
- Use these error-type codes (several allowed, joined with `|`): `sai_nghia` (wrong meaning), `so_don_vi` (numbers/units), `thuong_hieu_ma` (brand/model/code), `thieu_thua` (missing/added information), `sot_chu_nguon` (untranslated source text, wrong language, broken output), `thuat_ngu` (terminology), `van_phong` (wording/style). Leave it empty for `Dat`.
- Notes (`ghi_chu`): at most 25 words, quoting the exact wrong words for every `Nang`.
