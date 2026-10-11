# scripts

Dataset-build pipeline: collect repo URLs → resolve/dedupe → fetch GitHub
metadata → filter → clone → analyze → combine into the final dataset JSONL.
Generation and evaluation (what happens *after* the dataset exists) live in
`src/reverse_se/generation` and `src/reverse_se/evaluation`, not here.

Run `make help` from the repo root for the full target list. `make pipeline`
runs every stage below in order; each stage is also runnable on its own.

## Pipeline stages

Each script's default input path is the previous stage's default output
path, so most stages need no arguments for the common case — override with
the shown flags (or the Makefile's `TSV=`/`VERSION=`/`DATASET=` variables)
to run on a different slice of data.

| # | Stage | Script | Default input | Default output |
|---|---|---|---|---|
| 1 | Collect repo URLs | `python/extract_jgurry_urls.py`, `python/extract_awesome_urls.py`, `python/gitranking2csv.py` | (external HTML/MD/JSON sources under `data/raw/`) | `data/raw/{jgujerry_repos,repo-lists/repos,github_projects}_github.csv` (+ `*_webpage.csv`) |
| 1b | Resolve non-GitHub URLs (manual) | `python/repofinder/` (Scrapy spider, run via `scrapy crawl`) | a `*_webpage.csv` from stage 1 | resolved GitHub URLs, merge back in by hand |
| 2 | Merge + dedupe | `python/deduplicate.py` | the `*_github.csv` files from stage 1 | `data/raw/full_list_deduped.csv` |
| 3 | Fetch GitHub metadata | `python/fetch_github_info.py` | `data/raw/full_list_deduped.csv` | `data/raw/full_list_info.jsonl` |
| 4 | Filter by language/stars/activity | `python/filter_dataset.py` | `data/raw/full_list_info.jsonl` | `data/raw/filter_<stars>_<year>_list.tsv` + `data/raw/full_list_info.csv` |
| 5 | Clone repos | `bash/prefetch_repo.sh <tsv>` | a filter TSV from stage 4 | `data/repository/<owner>\|<repo>/` |
| 6 | Arcan dependency analysis | `bash/run_arcan_docker.sh <tsv>` (set `BUDGET_GB` to throttle by memory) | a filter TSV from stage 4 + cloned repos | `data/raw/arcanOutput/<owner>\|<repo>/` |
| 7 | Scan context files | `python/context_file_extractor.py <repos_dir> --language-tsv <tsv> --out <jsonl>` | cloned repos + a filter TSV | `data/interim/repo_context_files_all.jsonl` |
| 8 | Combine into dataset | `python/combine_arcan_w_context.py --arcanout_dir --context_file --info_file --version` | outputs of stages 3, 6, 7 | the final dataset JSONL (e.g. `reverse-se_<version>_..._dataset.jsonl`) |
| 8' | Combine, local-only variant | `python/build_test_dataset.py --repos_dir --arcanout_dir --language_tsv --output_file` | local checkouts + Arcan output (skips GitHub metadata) | same dataset schema as stage 8 |

Stage 8 and 8' share the graph-building code (`build_graph`/`get_graph_hash`
in `combine_arcan_w_context.py`) — they're two legitimate entry points
(with vs. without GitHub metadata), not a duplicate to reconcile.

## Standalone utilities (no strict order)

Consume the outputs above for stats/debugging; not part of the critical path:

- `python/dataset_stats.py`, `python/jsonl_token_stats.py`, `python/context_files_stats.py`, `python/repo_token_count.py`, `python/compare_lang.py`, `python/log_analysis.py`
- `R/graph_stats.R`, `R/token_stats.R`, `R/dataset_context_docs_stats.R` — plot the CSVs the scripts above produce
- `R/heatmap_dataset_filter_star_year.R` — sweeps star/year cutoffs; run this *before* stage 4 to pick `--min-year`/`--min-stars`
- `bash/render_c4_pdf.sh` — renders generated C4 markdown docs to PDF (`make pdf-docs`), independent of dataset-building

## Commit0 benchmark (A/B requirements+architecture study)

Measures whether generated requirements/architecture docs help a coding
agent implement a library from scratch, compared to intent alone
(`intent_only` vs `req_arch`). Built on
[Commit0](https://github.com/commit-0/commit0): its libraries, stubbed
starter repos, prebuilt pinned Docker images (`wentingzhao/<repo>:v0`),
test-ID lists and eval command. See `src/reverse_se/benchmark/commit0.py`
for exactly what is reused and what is added (human docs, docstrings and
git history removed; tests split visible/held-out). Needs Docker only.

```sh
uv run scripts/python/commit0_list.py                    # instances + TSV
./scripts/bash/prefetch_commit0.sh data/raw/commit0/commit0_instances.jsonl
# ... Arcan/context/combine + doc generation on data/raw/commit0 (docs land
#     in data/out/llm_results/<model>/ctx95/{intent,req,c4}/commit-0|<name>.md)
uv run scripts/python/benchmark_commit0.py prepare --lite  # scaffolds, agent images, gates
uv run scripts/python/benchmark_commit0.py run --lite \
    --docs-dir data/out/llm_results/<doc-model>/ctx95 \
    --model hosted_vllm/<model> --api_base http://<vllm-service>:8000/v1 \
    --replicates 3
```

`--scaffold` picks how much of the library's source the agent starts from
(all levels keep the same visible 30% of test functions and hide the rest,
and remove test data files only held-out tests use, `removed_test_data`):
`stubs` — Commit0's starter, every module/class/signature with `pass`
bodies; `api` — only the modules/names the tests use, still with
signatures (`api_scaffold.py`); `contract` — no source files at all, only
`API_CONTRACT.txt` listing the dotted names the tests import. Outputs are
kept per level (`scaffolds/<level>/`, `runs/<model>/<level>/`).

Grading adds one thing to Commit0's eval command: a per-test limit
(`--test-timeout`, default 60 s, pytest-timeout) so a single hanging test
fails alone instead of zeroing the run. Some hangs ignore every signal
inside pytest, so pytest runs under `grade_watchdog.py`: a test still
silent a minute past the limit is killed, counted as failed (`hung_tests`)
and skipped on a rerun of the rest; pytest kept alive by leftover processes
after writing its report is killed too. If pytest still dies before writing
its report, outcomes are salvaged from its verbose output (`salvaged`).
A single agent terminal command is killed after `--command-limit`
(default 1200 s, counted in `commands_killed`), so one command that never
returns can't freeze a run.
`--replicates N` runs replicates `01..N`; with `--skip-existing`, a run is
reused only if it completed (not interrupted), with the same settings and
scaffold. `results.jsonl` is rebuilt from the runs' `summary.json` files.
`uv run scripts/python/analyze_commit0.py [--trajectories] [--exclude chardet]` prints the
per-library and paired comparison (Wilcoxon, sign test, bootstrap CI),
efficiency and steps-to-threshold.

The agent runs *inside* the library's Commit0 image (plus OpenHands), so it
can run the visible tests against the preinstalled environment, attached
only to `--network` (default `$AGENT_SANDBOX_NETWORK`, the compose
`agent-sandbox` internal network) and with only its workspace and run
directory mounted. Grading copies just the implementation files into a
fresh, network-less container of the untouched image and scores Commit0's
full test-ID list (`all_*`), split into `visible_*` / `held_out_*`.

## Vendored copy

`collect-kit/collect_repo.py` (outside `scripts/`) is an intentionally
standalone, dependency-free reimplementation of stages 7–8′ for Docker
distribution. It's a deliberate fork, not drift — keep it in sync by hand if
you change the context-scan or combine logic here.
