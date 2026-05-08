---
name: obsidian-semantic
description: Search, navigate, and audit an Obsidian vault via the `obsidian-semantic` CLI. Use whenever a task needs vault content beyond exact-string lookup — semantic search by meaning, finding related notes, locating sections by heading, suggesting missing wikilinks, or detecting duplicates. Prefer this tool over `grep`/`find` across `.md` files.
---

# obsidian-semantic

A CLI for semantic search over an Obsidian vault, backed by a Lance index of chunk-level embeddings. The configured vault and embedder come from `~/.config/obsidian-semantic/config.yaml` (or a `.obsidian-semantic.yaml` in the vault root). Run `obsidian-semantic configure --show` to see the active config. Run `status` before trusting results if the user just edited notes — the index may be stale.

## The default workflow: research a topic

Most vault tasks fit one shape: ask a question, run **one** `search`, then make a decision based on what came back. Read the score-band tables below before the first query — that's where most agents waste commands.

After one query, fork on the result:

- **Top score ≥ 0.6 with a snippet matching the question** → strong anchor. Pull the relevant section with `show "<file_path>#<headers joined with #>"` rather than dumping the whole note.
- **Top scores between 0.45 and 0.6, several plausible notes** → query is ambiguous or the answer is distributed. Fire 2-3 paraphrases (different vocabulary, not synonyms) and look for convergence: if the same notes top-rank across phrasings, trust them. Use `related <top-anchor>` to expand the cluster.
- **Top 5 results all < 0.45 *and* snippets drift off-topic** → vault is silent. Declare absent rather than mining lower; the floor is the floor.

**File diversity is a signal.** With the default `--per-file 1`, each result row is one file. If `-n 10` returns 3 distinct files, those *are* the candidate set — bumping `-n` further won't help. If hits cluster in a single file, that file is the canonical note; `show` it and stop searching.

## Score interpretation

Score-band cheat-sheets are empirical from a 378-file / ~2200-chunk personal vault on `ollama+qwen3-embedding:8b`; the same bands held closely under `ollama+nomic-embed-text` on the same vault. Treat them as starting points and re-calibrate against the local corpus by running 3-4 known-good queries.

### `search` (chunk-level cosine, post-dedup)

| Score | Meaning | What to do |
|---|---|---|
| ≥ 0.6 | Strong anchor — title-level match or central section | Trust it; treat as canonical |
| 0.5 – 0.6 | Solid topical match | Read or `show` it; usually genuine |
| 0.45 – 0.5 | Weak; could be vocabulary overlap | Verify with `show`; don't quote without checking |
| 0.4 – 0.45 | Noise floor | Skip unless the query was very abstract |
| < 0.4 | Irrelevant | Ignore |

**Short notes under-score.** Terse notes (a few hundred words, no long-form expansion) routinely cap 0.05–0.10 below these bands even when they're the canonical match. When repeated paraphrases converge on the same note, trust the convergence over the absolute score.

### `suggest-links` (note-level averaged cosine — different scale!)

Note-level embeddings are smoothed averages of chunk vectors, so the scale sits much higher and tighter. Do not reuse `search` thresholds here.

| Score | Meaning |
|---|---|
| ≥ 0.95 | Likely duplicate — read both notes before merging |
| 0.88 – 0.94 | Probable missing wikilink |
| 0.85 – 0.87 | Possibly related; verify by reading |
| < 0.85 | Mostly shared vocabulary; ignore |

### `--score-min` is a noise filter, not a result-expander

Lowering `--score-min` does not surface long-tail results — it only truncates. To see more results, raise `--limit` (`-n`). To quiet noisy output, raise `--score-min`. Note: `--score-min` thresholds need re-calibration after per-file dedup, because the second-best file's surviving chunk often scores 0.05–0.10 lower than the duplicate chunks it displaced.

## Pulling sections instead of whole notes

`show` accepts either a vault-relative path or a bare filename. If the basename is unique in the vault, `show "Unit Testing.md"` resolves automatically — no need to paste full paths from search output for unambiguous notes.

`show <Note>#<Heading>` prints just the named section. `show <Note>#<H2>#<H3>` traverses a breadcrumb. The match is on the suffix of the breadcrumb and is case-insensitive; ambiguous headings are listed with line numbers and the command exits non-zero — disambiguate by adding more of the path.

```bash
obsidian-semantic search "ridge regression solvers" --limit 5
# Top hit:  Statistics/Ridge Regression Solvers.md  §  Overview  (0.68)
obsidian-semantic show "Statistics/Ridge Regression Solvers.md#Overview"
```

`show <Note>#<Heading>` emits the `## Heading` line and a blank line before the body.

## Specialized workflows

### Auditing for missing wikilinks and duplicates

`suggest-links` walks the note-level cosine matrix. The default `--threshold 0.8` is too noisy for a vault-wide audit — raise to `0.85`, and bump `--limit` from its default of 20 to 30-50 so real candidates aren't capped. Pass `--exclude-same-folder <name>` (repeatable) for folders with many topically-similar notes (e.g. a daily log).

Always `show` both notes before declaring a duplicate. In the 0.88–0.94 grey zone:

- **Shared content sections** (overlapping prose, same headings) → merge candidate
- **Shared terminology with complementary roles** (producer/consumer, V6/V7, before/after) → link, don't merge

`--exclude-same-folder` matches folder names exactly; `daily-log` will not match `Daily Log`. The CLI warns on unknown values.

### Filtering by tag

`obsidian-semantic list-tags` enumerates every tag in the index with the number of files carrying it. Use it to discover what's available, then pass `--tag <name>` to `search`.

```bash
obsidian-semantic list-tags --json | jq 'to_entries | sort_by(-.value) | .[:20]'
obsidian-semantic search "<query>" --tag <name> --limit 10
```

`--tag` can be passed multiple times; results match any of the tags.

### JSON automation

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

`headers` is a breadcrumb list (1 element for H2-boundary chunks, 2 for H3-split chunks). It feeds directly into `show`:

```bash
obsidian-semantic search "<q>" --limit 1 --json \
  | jq -r '.[0] | "\(.file_path)#\(.headers | join("#"))"' \
  | xargs -I{} obsidian-semantic show "{}"
```

`headers` may be empty `[]` for short notes that weren't chunked — `show <file_path>` works without an anchor.

`tags` is **not** present in `--json` output of `search`; use `list-tags --json` for that surface.

## Known limitations

| Limitation | Symptom | Workaround |
|---|---|---|
| **Hub notes** dominate broad queries | Long notes, MOC indexes, or auto-generated catalogs surface across unrelated topics | Treat repeat appearances under varied queries as a hub-warning, not a relevance signal |
| **`related` quality varies by seed** | Densely-connected topical seeds → great; outlier or auto-generated seeds → wandering tangents | Use `related` after one solid `search` hit, not as primary discovery |
| **Stub boilerplate causes false positives** | Templated text in §Related sections matches tool-related queries | Recognize and filter the boilerplate when it appears in snippets |
| **Short-stub vs long-parent blindspot** | A short note absorbed by a longer parent (e.g. transcluded stub) scores low on `suggest-links` because the parent's averaged embedding is dominated by content the stub doesn't have | Run `related <stub-path>` directly when auditing a short note for redundancy |
| **No time / date filters** | Can't query "since 2026-04-01" | Scope with `--folder` and sort by filename if notes are `YYYY-MM-DD.md` |
| **Daily-log noise floor is higher** | Long miscellaneous date-stamped notes match on stray vocabulary | Raise the search confidence floor to ≥ 0.55 for that corpus and `show` borderline hits before quoting |
| **Snippets truncate mid-sentence** | `...` cuts off context | Use `show` to verify before quoting |
| **`status` doesn't show embedder model** | Can't tell from `status` whether the index is nomic vs qwen3 | Read `~/.config/obsidian-semantic/config.yaml`; index size also disambiguates (4096-dim ≈ 16 KB/chunk, 768-dim ≈ 3 KB/chunk) |

## Command quick-reference

| Command | Purpose | Most useful flags |
|---|---|---|
| `status` | Confirm index is fresh and not pending | (none) |
| `search QUERY` | Semantic search over chunks | `--limit/-n`, `--folder`, `--tag/-t`, `--score-min`, `--per-file`, `--json` |
| `show NOTE[#H[#H]]` | Print full note or section by heading-breadcrumb | (no flags) |
| `related NOTE` | Find notes similar to a given note | `--limit/-n`, `--json` |
| `list-tags` | Enumerate tags with per-file counts | `--json` |
| `suggest-links` | Find unlinked pairs and likely duplicates | `--threshold/-t` (default 0.8), `--limit/-n`, `--exclude-same-folder` |
| `index` | Refresh the vector index | `--full` for a clean rebuild after embedder changes |

`--per-file 1` (default) deduplicates to the top chunk per file. Pass `--per-file 0` for every matching chunk (useful for breadth on a single note).

## Done-checks before reporting back

- For a synthesis: every claim cites a `file_path` actually read with `show` — scores told you which notes to look at, the contents are still the source of truth
- For a topic-absent claim: at least 2 paraphrased queries tried and every top hit < 0.45
- For a `suggest-links` recommendation: both notes verified by `show`, not just trusted on score
- If the user just edited notes: confirm `status` shows `Pending: 0 new, 0 modified`
