# LLM-judge rubrics

One folder per rubric version, one template per artefact (`c4`, `req`,
`intent`). `src/reverse_se/evaluation/judge/rubrics.py` renders them and pairs
each with its output schema from `judge/schemas.py`. Every judged row records
the version it was scored with (`rubric_version`).

**Default: `v3`** (`RUBRIC_VERSION` in `rubrics.py`), used by `llm_as_judge`
and `judge/aggregate.py` unless `--rubric_version` says otherwise. Never edit
a folder a run has already used; copy it to a new version instead.

| Version | Status | What it is |
|---|---|---|
| `v3` | **default**, frozen | Seven common criteria plus artefact-specific ones, numeric 1–5 scores, no defect lists. |
| `v2` | frozen, comparisons only | Five criteria, numeric 1–5 scores, plus a document-level issue list with categories. |

## Cross-artefact alignment (`align/`)

Separate from the per-artefact rubrics: one template per adjacent pair of the
chain intent → requirements → architecture (`intent_req`, `req_c4`), sharing
`trace_policy.jinja`. The judge extracts the intent items or architecture
elements and traces every requirement ID to them; coverage, orphan and
contradiction metrics are computed from the traces in
`src/reverse_se/evaluation/judge/alignment.py`, and
`reverse_se.evaluation.alignment_judge` runs them on matched, cross-generator,
shuffled and planted pairs. Same freezing rule: copy to a new version rather
than editing one a run has used.

| Version | Status | What it is |
|---|---|---|
| `align/v1` | current | Extract items, trace every requirement, list contradictions. |
