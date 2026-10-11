# prompts

Jinja2 prompt templates used to evaluate LLM-based reverse software
engineering tasks (C4 architecture, requirements, and stakeholder intent
recovery). This folder holds templates only — the dataset and the renderer
script live elsewhere (see below).

## Contents

```
prompts/
├── c4_sys_prompt.jinja                  # System prompt: C4 architecture generation
├── c4_user_prompt.jinja                 # User prompt: C4 architecture generation
├── requirments_sys_prompt.jinja         # System prompt: requirements generation
├── requirments_user_prompt.jinja        # User prompt: requirements generation
├── stakeholder_intent_sys_prompt.jinja  # System prompt: stakeholder intent
├── stakeholder_intent_user_prompt.jinja # User prompt: stakeholder intent
├── nodes_template.jinja                 # Renders dependency graph nodes (CSV)
├── edges_template.jinja                 # Renders dependency graph edges (CSV)
├── context_template.jinja               # Renders contextual repository files
└── package_template.jinja               # Renders package-to-file mappings
```

## Dataset

The dataset (e.g. `reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000.jsonl` —
1000 GitHub repositories, one JSON object per line) lives under
`data/out/`, a generated/gitignored directory, not here.

Key fields per record:

| Field | Description |
|---|---|
| `repo_id` | `owner\|repo` identifier, used as the output filename |
| `package_only_graph` | Compressed dependency graph (package-level nodes and edges) |
| `original_graph` | Full dependency graph before compression |
| `package_files` | Per-package file listings |
| `context_files` | High-value repository files (README, config, manifests, etc.) |
| `node_compression_ratio` | Ratio of package nodes to original nodes |
| `edge_compression_ratio` | Ratio of package edges to original edges |

## Rendering prompts

`scripts/python/render_prompts.py` (built on `reverse_se.generation.utils`,
the same prompt-rendering code the generation and evaluation pipelines use)
assembles a full prompt (system + user) for each repository in the dataset
and writes it to `prompts_out/<repo_id>.txt`.

```bash
# render all three tasks
uv run scripts/python/render_prompts.py \
  --input_file data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000.jsonl \
  --output_dir prompts_out

# render a single task (c4 | req | intent)
uv run scripts/python/render_prompts.py \
  --input_file data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000.jsonl \
  --output_dir prompts_out \
  --task c4
```

Output is written to one subfolder per task: `prompts_out/c4/`, `prompts_out/req/`, `prompts_out/intent/`.

Templates are looked up by keyword in the filename (`c4_sys`, `c4_user`,
`requirments_sys`, etc.) under `prompts/` by default, overridable with
`--prompt_dir`.

## Prompt templates

Each task (C4, requirements, stakeholder intent) has a `sys` and `user` prompt pair. The user prompt is assembled from four sub-templates:

- **nodes** — graph nodes as CSV (id, name, type, LOC, fan-in, fan-out)
- **edges** — graph edges as CSV (from, to, weight)
- **package** — package name → file list mapping
- **context** — selected repository files with path, category, and content
