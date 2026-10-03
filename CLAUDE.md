# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Project-Specific: e-commerce - EDA (EDA + data processing)

This folder is the EDA and data-processing phase of the Vietnamese–Chinese e-commerce MT thesis (team: Quang PM, Nhật Anh, Huy). Crawling is finished and lives in the separate `ecom_crawler` folder. Derived from `ecom_crawler/CLAUDE.md` on 01/10/2026 (sections 1-4 unchanged). Before any task read `WALKTHROUGH.md`, especially section 0 and the open-items list at the end of the log.

### 5.1 Working with Nhật Anh

- Reply in Vietnamese: call the user "anh", yourself "em", natural plain language.
- Every prompt and every change, in every session, gets logged in `WALKTHROUGH.md`: append to the "NHẬT KÝ THEO NGÀY" section, never rewrite history, fix a mistake by adding a correction note. End the reply with the exact line "ĐÃ CẬP NHẬT VÀO file WALKTHROUGH", and only after the log is really written. WALKTHROUGH headings stay UPPERCASE; leave content you were not asked to change untouched.
- Logging in `WALKTHROUGH.md` is the one exception to "don't edit or run anything yet": always do it, even when the user says to hold off. Before appending, read the last log entry (another session may have written since) and continue its numbering; never keep two sessions editing the log at the same time.
- Be honest: say what is unverified, what failed, what you got wrong. Report a finished or failed background job as soon as you notice, and at the start of a session check for jobs left running or finished (a run that finished on 28/09 went unreported for two days).
- When asked for an opinion or a proposal, give it and stop; do not act until told.
- When the user defers something ("hoãn", "từ từ", "đợi tui"), do not start it.
- Text meant for the supervisor (thesis, report, slides): written as "em", natural spoken Vietnamese, no em/en dashes, no emphasis quotes, never invent or drop a number, quote only numbers from output that was actually run.

### 5.2 Evidence and review workflow (agreed)

- A claimed data problem comes with evidence: affected `product_id`s, count and %, ranked from most to least severe, exported as CSV. A proposed solution comes with sources and its risks.
- Nhật Anh reviews by hand. Propose, then export a separate file (e.g. `*_editted.csv`) with empty decision columns (`anh_duyet`, `anh_ghi_chu`) and wait. Never overwrite the input file; never apply a fix to the data before the table is signed off. Unsure about an item: mark it `XEM_TAY`, do not guess.
- Review by sampling (decided 02/10/2026): Claude may pre-fill a table, then Nhật Anh checks a sample instead of every row, under five conditions. (1) The sample is random, drawn by the notebook with a fixed seed and saved to a file; nobody hand-picks it. (2) "Rule of three": 0 errors in n random items bounds the true error rate below about 3/n at 95% confidence (n = 30 → ~10%, 50 → ~6%, 100 → ~3%); use 30-50 items for low-impact tables (role tags, numbering samples, dropped pairs). (3) High-impact tables (attribute-name and value dictionaries, where one wrong entry spreads to thousands of rows): Nhật Anh reads every entry Claude changed from the majority translation, and the rest is sampled weighted by row count. (4) Errors found: fix the rule or suggestion behind them, then draw a NEW sample (1 error → fix and draw 30 new; 2 or more in 30 → the table fails, redo that part). (5) Record per gate: items, sample size, errors, what was fixed, so the report can state it. Safe actions (masking personal data, tagging, dropping a pair) run by default without review; only content edits (replacing a translation) need sign-off. The test set (500-1,000 pairs) is still reviewed item by item (charter §7).
- Similarity is a detector, not a judge. Thresholds come from mismatched-pair baselines (hard = median of same-category wrong pairs, grey zone = median to p95), per category, never a fixed 0.5. Lists below a threshold are for reading and never trigger automatic deletion. Short strings (attribute values of 1-3 words) score badly (AUC 0.82): use regex flags there.
- Write CSVs with `utf-8-sig` so they open correctly in Excel.

### 5.3 Cleaning policy (decided by Nhật Anh, 28/09/2026)

- `title_vi` keeps English only for brand names, units (cm, ml, g, inch...) and technical terms (wifi, bluetooth, USB, PP/PPSU...). Other English words are replaced by Vietnamese.
- Chinese place names, tea names and product terms left as pinyin become Sino-Vietnamese (Hán Việt), e.g. Wuyishan → Vũ Di Sơn. Brand names written in pinyin stay as they are.
- Loanwords Vietnamese sellers really use stay: cotton, vali, retro, vintage, size, mini, tote, sneaker, collagen...
- For the "needs review" English group the AI decides word by word; unsure → `XEM_TAY`.
- Data scope is 9 categories: the 8 core (`fashion, electronics, shoes, bags, beauty, mother_baby, food, home`) plus `auto` (supplementary, 971 rows in the bilingual snapshot). Never add or drop a category without sign-off.

### 5.3b Description pipeline decisions (Nhật Anh, 02/10/2026)

Pipeline P0-P13: `docs/design/description_pipeline.html`, notebook `notebooks/desc-pipeline.ipynb`, solutions `data/eda/description/pro_solu_descriptions.csv`.

- Fix errors that follow a pattern (rules, reviewed dictionaries); a pair that cannot be fixed gets a reason code and is excluded from training (`dung_cho_train = 0`), never deleted from the processed files.
- LLMs only suggest; a person approves before anything is applied. Dictionary suggestions (~1,100 entries) and the top ~3,000 long-tail pairs are done in Claude Code sessions (Claude Pro plan, no paid API). The rest of the long tail (130,616 distinct pairs in the 02/10 run; the earlier estimate of ~47,600 counted only single-value pairs) is screened by Qwen2.5-7B-Instruct on Colab as a flagger only (03/10: loaded 4-bit with bitsandbytes through transformers, because the float16 model does not fit a T4 and recent vLLM may not support T4; labels read from the A/B/C answer probabilities, results saved batch by batch to Google Drive; notebook `notebooks/desc-qwen-tang2-colab.ipynb`), after a 500-pair trial; flagged pairs are dropped only if the flag precision measured on ~300 labelled pairs is about 80% or higher, otherwise they are tagged `chua_kiem_nghia` (kept in train, never used for test).
- 03/10/2026 update (Nhật Anh chose option A): Qwen2.5-7B did not work as a flagger (precision 28-32% at every threshold on 100 calibration pairs, about the 25% base error rate), so it is not used (`DUNG_QWEN = False`). Pairs holding a tier-2 (rare long-tail) value get reason code `T2_CHUA_KIEM` and are excluded from training but kept in the files, to be re-checked later if a better method appears (Claude labelled 300 random tier-2 pairs: ~26% meaning errors).
- Output in two forms: a clean attribute-pair table (with reason codes and tags) and description_zh/vi rebuilt from the kept pairs.
- 1688 trade attributes (是否跨境出口专供货源, 主要下游平台, ...) are kept and tagged with a role column.
- Chinese brand names (品牌) become pinyin (pypinyin), not Hán Việt; generic words (通用, 厂家, 中性...) are excluded via a list; Nhật Anh reads 200 random conversions before they are applied.
- Decimal numbers in measurements use a dot like the Chinese side ("1.5m"); only numbers with 1-2 digits after the separator are changed, and only when the Chinese side has exactly that a.b, so thousands separators ("1.000") and lists ("9,12,24") are left alone. (02/10: the earlier "must have a unit" condition was dropped by Nhật Anh.)
- Manufacturer names/addresses (生产厂家, 产品厂家厂址, 产品厂家厂名, 生产企业, 厂址, 厂名, 生产地址, 公司, 制造商...) are excluded with reason code `PII_NSX` (charter §6); the published version drops the `shop` column (the internal file keeps it for `nhom_trung`). Nhật Anh, 02/10/2026.
- The 200-sample brand check passes with at most 10 wrong (5%), like the DS06B check.
- Glossary terms chosen by Claude on Nhật Anh's request (02/10): 一件代发 → "dropship", 补水 → "cấp nước" (保湿 stays "dưỡng ẩm"), 老爹鞋 → "giày bố"; only listed wrong phrases are replaced (see `CUM_RIENG` in P7.7).
- Real codes are copied from the Chinese side verbatim: a value is a real code when (role is code/certificate and the Chinese value has no Chinese characters) or (the attribute name is a licence/registration number, i.e. contains 许可证, 备案, 批准文号, 注册证, 标准号, 证号 or 准字, and the value has 4+ consecutive digits). Descriptive text under 型号/货号 is translated as usual. 03/10/2026 (Nhật Anh): a copied code that still contains Chinese characters gets a Vietnamese gloss after it, naming the province (from its abbreviation) and/or "quốc gia" for 国 codes, e.g. `粤妆20170110 (Quảng Đông)`, `国妆网备进字（沪）2024007457 (quốc gia, Thượng Hải)`; a date like `2023年05月24日` becomes `24/05/2023`.
- Symbols (D16, applies to title and description): delete nothing. The Chinese side stays as is; the Vietnamese side turns 【】 into [ ] and keeps ~ | *.
- Train/dev/test split is postponed until the team chooses the model; the pipeline only assigns a duplicate-group id (`nhom_trung`).
- English handling is skipped (Nhật Anh, 02/10/2026): Quang will handle it later. Step P7.6 (replacing English/pinyin tokens with the title token table) is switched off (`XU_LY_TIENG_ANH = False` in P0); do not add new English rules, and when labelling the long tail do not mark a pair wrong only because English was left untranslated. Brand rules (Latin brand names copied from the Chinese side) stay.

### 5.4 Data handling

- `data/snapshot/*.parquet` is immutable (hashes in `data/snapshot/SNAPSHOT.md`). A refresh is a new dated copy, made only on request. Never analyze from `data/raw/` or HF `bronze/` (raw backups, not in this folder).
- Processing output goes to `data/processed/`; analysis output to `data/eda/` (keep its existing sub-paths, WALKTHROUGH refers to them); regenerable caches to `data/interim/`.
- Scripts use paths relative to the repo root (`Path(__file__).resolve().parents[2]`): no hard-coded drive, user or temp paths. Use only what is in `requirements.txt`; do not `pip install` ad hoc into `.venv`; if a package is needed, ask and add it to `requirements.txt`.
- After changing a script, re-run it and compare with the previous output. The four audit scripts (`attr_split_check`, `title_audit`, `build_tokens_edited`, `data_issues`) reproduce their outputs byte for byte; keep it that way or explain the difference.
- Notebook edits: back up first, change only the intended cells, leave the others byte-identical. Numbers computed locally (CPU / Ollama bge-m3, strings cut to 480 characters) are for iteration only: before they go into a report, say what they ran on and re-run with the notebook's real `transformers` model on Colab GPU.
- `src/textnorm.py` is what was already applied to every crawled record (NFC, full-width letters and digits folded, HTML and invisible characters removed, full-width CJK punctuation kept). Do not redo it differently.

### 5.5 Proposed in WALKTHROUGH sections 5-6, NOT yet signed off (do not treat as agreed)

- Order of the cleaning steps for TITLE (WALKTHROUGH 6.1); replacements conditioned on the Chinese source text; keeping removed rows with a reason code (D-codes) instead of deleting them. (For description these are decided, see 5.3b.)
- Train/dev/test split: postponed by Nhật Anh (02/10/2026) until the team chooses the model. Recommended when it comes: group by (title_zh + shop + identical/near-duplicate description), largest group 283 rows (1.73%), stratified by category. Needs a decision before any split.
- Tentative choices awaiting a decision: 回力 (Huili / Warrior / Pull-back) → "Warrior"; "dropship" → "giao hộ".
- D16 is decided (02/10/2026, see 5.3b): delete no symbols; Vietnamese side 【】 → [ ].
- Marketing loanwords: the charter treats `freeship` as the correct Vietnamese for 包邮, while the 28/09 policy only lists brands, units, technical terms and loanwords sellers use. "hot" (323 rows) is currently planned as `THEO_CUM` → "bán chạy". Decide whether marketing loanwords (freeship, hot, sale...) stay.

### 5.6 Machine and environment

- Windows 11, PowerShell 5.1, Python 3.13.7. Machine: i7-12700H, RTX 3050 4 GB, but the global `torch` is CPU-only: check `torch.cuda.is_available()` before estimating anything on GPU.
- Use `127.0.0.1`, never `localhost` (about 2 s extra per request). Ollama: `bge-m3` at `127.0.0.1:11434`; very long strings return HTTP 400 (cut at 480 characters or split the batch). Models live in `D:\ollama-models`; after a restart the server may not see them (empty list, HTTP 404 on `/api/embed`, seen 01/10). Never kill the user's Ollama: ask them to restart it from the tray. Workaround used 03/10 when the tray server still saw no models: start a separate server with `$env:OLLAMA_MODELS="D:\ollama-models"; $env:OLLAMA_HOST="127.0.0.1:11435"; ollama serve` as a background task, point scripts at it (`OLLAMA_URL=http://127.0.0.1:11435`), and stop that task when done.
- Long jobs: detach with WMI `Win32_Process.Create` and `ShowWindow=0` (`Start-Process` dies with the Claude session; a visible console window that the user closes kills the job). After launching, check that the parent is `WmiPrvSE` and no new Windows Terminal window appeared; log to a file; make it resumable; tell the user what kills it (sleep, shutdown, update restart). Never kill the user's running processes.
- PowerShell: `git commit -m` with a `<<'EOF'` heredoc does not work, use `git commit -F <file>`. `git` was once not found on PATH (it showed `D:\Tr?nh\Git\cmd`): call `D:\Trịnh\Git\cmd\git.exe`.
- A regex containing the `\u` escape is rejected by pandas' Arrow/RE2 strings inside `.str.contains` / `.str.fullmatch`: use `re.compile(...)` with `.map()`. This depends on the pinned pandas 3.0.5.
- Console output of Vietnamese or Chinese can look garbled; that is display only. Read files with the Read tool or set `PYTHONIOENCODING=utf-8`.
- Secrets: never expose `.env` contents anywhere. This folder has no `.env` and needs none (the HF dataset is public).
- Git: this folder is **not** a git repository (decided 01/10/2026). `github.com/ntquang-0410/eda_ecom` holds one independent branch per person (`main` = Quang, `giahuy` = Huy, `nhatanh_updating` = Nhật Anh, a snapshot of 28/09). Ask before publishing anything there; never push to `main` or `giahuy`; every branch needs its own `WALKTHROUGH.md`; push with the personal account `nhatanh20022005`, not `trinhnhatanh`.
- If more crawling is ever needed it happens in `ecom_crawler`, with the official crawl settings kept, and only after asking.

### 5.7 Team charter (15/09/2026): rules that stay binding

The team's original charter is `docs/PROJECT_CHARTER_team_CLAUDE.md` (written for the crawl phase; its banner lists what is stale). Read its §1, §4 (data folder principles), §6 and §7 before any cleaning or split work. Where a later decision of Nhật Anh differs (5.3), the later decision wins; say so when it happens.

- Goal: a Vi↔Zh MT model for e-commerce titles. Data quality caps model quality. The charter names Sailor2-8B (comparison branch Seed-X-7B), but on 01/10/2026 Nhật Anh took the fine-tune model out of the pipeline until the team decides again: do not design analysis or processing around a specific model or tokenizer. The charter's training GPU (RTX PRO 6000, 96 GB) is for a later phase, not this laptop.
- 1688's Vietnamese is Alibaba machine translation: usable for TRAIN (silver data, flagged as MT origin in the report) and as a terminology source, **never for the TEST set**. The test set must be human-translated or human-reviewed (500-1,000 pairs).
- Split train/dev/test by `product_id` or shop, never randomly by row (D10 adds: identical `title_zh` must stay in one split).
- Deduplication with MinHash/LSH (`datasketch`, Jaccard about 0.8) cuts the most data. Add `datasketch` to `requirements.txt` only when that step is approved.
- Do not wipe the domain's decorative characters and emoji; only normalize control characters and repeated whitespace. (No emoji exist in the current snapshot; the rule matters for symbols such as 【】 ~ |.)
- Personal data (shop names, phone numbers, addresses inside descriptions) must be removed before anything is published.
- Layout from the charter: `data/processed/` holds one new Parquet per cleaning step (never overwrite the previous step); `data/final/` holds `train.jsonl`, `dev.jsonl`, `test.jsonl` (create it only when needed).

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
