---
name: obsidian-semantic
description: Search, navigate, and audit an Obsidian vault via the `obsidian-semantic` CLI. Use whenever a task needs vault content beyond exact-string lookup — semantic search by meaning, finding related notes, locating sections by heading, suggesting missing wikilinks, or detecting duplicates. Prefer this tool over `grep`/`find` across `.md` files.
---

# obsidian-semantic

A CLI for semantic search over an Obsidian vault, backed by a Lance index of chunk-level embeddings. The configured vault and embedder come from `~/.config/obsidian-semantic/config.yaml` (or a `.obsidian-semantic.yaml` in the vault root). Run `obsidian-semantic configure --show` to see the active config.

## Read this first: how to interpret scores

Score interpretation is the single biggest source of friction. Bands below are empirical from a 378-file / ~2200-chunk personal vault on `ollama+qwen3-embedding:8b`; the same bands held closely on the same vault under `ollama+nomic-embed-text`. Treat them as starting points and re-calibrate against your own corpus by running 3-4 known-good queries.

### `search` (chunk-level cosine, post-dedup)

| Score | Meaning | What to do |
|---|---|---|
| ≥ 0.6 | Strong anchor — title-level match or central section | Trust it; treat as canonical |
| 0.5 – 0.6 | Solid topical match | Read or `show` it; usually genuine |
| 0.45 – 0.5 | Weak; could be vocabulary overlap | Verify with `show`; don't quote without checking |
| 0.4 – 0.45 | Noise floor | Skip unless query was very abstract |
| < 0.4 | Irrelevant | Ignore |

**Topic-absent heuristic.** If the top 5 results are all < 0.45 *and* the snippets drift off-topic, the vault is silent on that topic — declare it absent rather than mining lower. This pattern is more reliable than any single threshold.

**Short notes under-score.** Terse notes (a few hundred words, no long-form expansion) routinely cap 0.05–0.10 below these bands even when they're the canonical match. When repeated paraphrases converge on the same note, trust the convergence over the absolute score.

### `suggest-links` (note-level averaged cosine — different scale!)

Note-level embeddings are smoothed averages of chunk vectors, so the scale sits much higher and tighter:

| Score | Meaning |
|---|---|
| ≥ 0.95 | Likely duplicate — read both notes before merging |
| 0.88 – 0.94 | Probable missing wikilink |
| 0.85 – 0.87 | Possibly related; verify by reading |
| < 0.85 | Mostly shared vocabulary; ignore |

Do not use `search` thresholds for `suggest-links` (or vice versa). They live on different scales.

### `--score-min` is a noise filter, not a result-expander

Lowering `--score-min` does not surface long-tail results — it only truncates. To see more results, raise `--limit` (`-n`). To quiet noisy output, raise `--score-min`.

## Core workflows

### 1. Search → `show` a section

The most common pattern. Avoid dumping whole notes — pull just the section you need.

```
search "<query>" --limit 5
  → pick a hit; note its file_path and headers
show "<file_path>#<headers joined with #>"
```

Worked example:

```bash
obsidian-semantic search "ridge regression solvers" --limit 5
# Top hit:  Statistics/Ridge Regression Solvers.md  §  Overview  (0.68)
obsidian-semantic show "Statistics/Ridge Regression Solvers.md#Overview"
```

`show Note#A` matches a single heading; `show Note#A#B` traverses a breadcrumb. The match is on the suffix of the breadcrumb and is case-insensitive. Ambiguous headings are listed with line numbers and the command exits non-zero — disambiguate by adding more of the path.

### 2. Cluster discovery via paraphrase + `related`

Single-keyword queries underperform on these embedders. Fire 2-3 paraphrases before pivoting:

```
search "TDD"                            → 0.50 top
search "test driven development"         → 0.52 top, same anchor notes
search "writing tests before code"       → 0.49 top, same anchor notes
                                            ↓ converged → trust it
related "<top anchor>" --limit 8         → expand the cluster
```

Multi-query overlap (the same notes surfacing under different phrasings) is a stronger relevance signal than any single absolute score.

### 3. Suggest-links audit

Walk the highest-similarity unlinked pairs and decide which deserve linking, which are duplicates, and which are coincidence.

```bash
obsidian-semantic suggest-links --threshold 0.85 --limit 30
# To exclude folders that legitimately have many similar notes
# (e.g. a date-stamped daily log), pass --exclude-same-folder
# (repeatable) or set suggest_links.exclude_same_folder in config.
```

Always `show` both notes before declaring a duplicate. Two patterns to distinguish in the 0.88–0.94 grey zone:
- **Shared content sections** (overlapping prose, same headings) → merge candidate
- **Shared terminology with complementary roles** (producer/consumer, V6/V7, before/after) → link, don't merge

The default `--limit 20` is too narrow for a vault-wide audit; bump to 30-50.

**Folder name must match exactly.** `--exclude-same-folder "daily-log"` will not match a folder named `Daily Log` (with the space). The CLI warns to stderr if a value isn't a real top-level vault folder.

### 4. Longitudinal mining of a date-stamped folder

The tool has no `--since`/`--until` flag. Scope by folder, then sort manually by filename if your notes are named `YYYY-MM-DD.md`:

```bash
obsidian-semantic search "<topic>" \
  --folder "<date-stamped folder>" --limit 15 --score-min 0.5 --json \
  | jq -r '.[] | "\(.file_path)\t\(.score)"' | sort
```

Daily-log entries tend to be long and miscellaneous, so the noise floor is higher than for topical notes. Use ≥ 0.55 as a confidence floor for that corpus specifically, and `show` borderline hits before quoting.

### 5. Topic-absent detection

When asked "does the vault cover X?", the goal is a confident negative. Procedure:

1. Run 2-3 paraphrases of the query
2. If every top hit is < 0.45 *and* snippets are off-topic → declare absent
3. Don't lower `--score-min` to "find more" — there's nothing to find

## JSON output for automation

`search` and `related` accept `--json`. Each hit is a flat object:

```json
{
  "chunk_id": "<file_path>#<slug-or-chunk_N>",
  "file_path": "Statistics/Ridge Regression Solvers.md",
  "title":     "Ridge Regression Solvers",
  "headers":   ["Why Cholesky Is Faster in Practice", "When Eigendecomposition Wins"],
  "text":      "...",
  "score":     0.6027,
  "start_line": 142
}
```

`headers` is **already a list breadcrumb** (1 element for H2-boundary chunks, 2 for H3-split chunks). It is feedable directly to `show`:

```bash
obsidian-semantic search "<q>" --limit 1 --json \
  | jq -r '.[0] | "\(.file_path)#\(.headers | join("#"))"' \
  | xargs -I{} obsidian-semantic show "{}"
```

Caveats:
- `tags` is **not** present in `--json` output (gap to be aware of, no current workaround)
- `headers` may be empty `[]` for short notes that weren't chunked (just `show <file_path>`)
- `show <Note>#<Heading>` includes the markdown `## Heading` line followed by a blank line. Strip with `tail -n +3` (or equivalent) before computing offsets into the body.

## Tag filtering and discovery

`--tag` filters by frontmatter or inline tags. **There is no `list-tags` command.** To discover what tags exist, brute-force with a semantically vacuous query as the oracle:

```bash
obsidian-semantic search "asdf qwerty" --tag "<candidate>" --json
#   []  → tag doesn't exist (or no notes have it)
#   non-empty → tag exists
```

Do **not** probe with a meaningful query — the results conflate tag-filtering with topical match, making it ambiguous whether the tag is real or whether the query merely scored above zero on its own.

Common starter candidates worth probing first in a personal Obsidian vault: `project`, `moc`, `daily`, `meeting`, `book`, `person`, `idea`, plus any obvious domain words from the user's stated context (e.g. `bioinformatics`, `statistics`, `programming`). Treat these as a seed list, not a comprehensive set.

## Known limitations to recognize

| Limitation | Symptom | Workaround |
|---|---|---|
| **Hub notes** dominate broad queries | Long notes, MOC indexes, or auto-generated catalogs surface across unrelated topics | Treat repeat appearances under varied queries as a hub-warning, not a relevance signal |
| **`related` quality varies by seed** | Densely-connected topical seeds → great; outlier or auto-generated seeds → wandering tangents | Use `related` after one solid `search` hit, not as primary discovery |
| **Stub boilerplate causes false positives** | Templated text in §Related sections (e.g. "Use obsidian-semantic related ... to populate") matches tool-related queries | Recognize and filter the boilerplate when you see it in snippets |
| **Short-stub vs long-parent blindspot** | A short note absorbed by a longer parent (e.g. transcluded stub) scores low on `suggest-links` because the parent's averaged embedding is dominated by content the stub doesn't have | Run `related <stub-path>` directly when auditing a short note that should be reviewed for redundancy |
| **No time / date filters** | Can't query "since 2026-04-01" | Scope with `--folder` and sort by filename |
| **Snippets truncate mid-sentence** | `...` cuts off context | Use `show` to verify before quoting |
| **`status` doesn't show embedder model** | Can't tell from `status` whether index is nomic vs qwen3 | Read `~/.config/obsidian-semantic/config.yaml`; index size also disambiguates (4096-dim ≈ 16 KB/chunk, 768-dim ≈ 3 KB/chunk) |

## Command quick-reference

| Command | Purpose | Most useful flags |
|---|---|---|
| `status` | Confirm index is fresh and not pending | (none) |
| `search QUERY` | Semantic search over chunks | `--limit/-n`, `--folder`, `--tag/-t`, `--score-min`, `--per-file`, `--json` |
| `show NOTE[#H[#H]]` | Print full note or section by heading-breadcrumb | (no flags) |
| `related NOTE` | Find notes similar to a given note | `--limit/-n`, `--json` |
| `suggest-links` | Find unlinked pairs and likely duplicates | `--threshold/-t` (default 0.8), `--limit/-n`, `--exclude-same-folder` |
| `index` | Refresh the vector index | `--full` for a clean rebuild after embedder changes |

`--per-file 1` (default) deduplicates to the top chunk per file. Pass `--per-file 0` to allow every matching chunk through (useful for breadth on a single note); `--score-min` thresholds need re-calibration after dedup because the second-best file's surviving chunk often scores 0.05–0.10 lower than the duplicate chunks it displaced.

## Quick checks before declaring "done"

- `status` shows `Pending: 0 new, 0 modified` (otherwise the index is stale; run `index` first)
- For a synthesis answer: every claim is anchored to a `file_path` you cited
- For a topic-absent claim: at least 2 paraphrased queries tried and every top hit < 0.45
- For a `suggest-links` recommendation: both notes verified by `show`, not just trusted on score
