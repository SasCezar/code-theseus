"""OpenAI Batch API support for the judge.

Submits every pending judge call as Batch API jobs instead of one live request
per document, split into several batches when the input exceeds OpenAI's
per-file limit. The Batch API is ~50% cheaper than live calls, at the
cost of turnaround time (up to 24h, usually much faster).
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

import litellm
from litellm.utils import type_to_response_format_param

from reverse_se.evaluation.judge.client import render_judge_messages
from reverse_se.evaluation.judge.rubrics import RUBRIC_VERSION

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

# (model, ctx, task, repo_id, input_prompt, generated_doc) — the same shape
# llm_as_judge.py builds its `pending` list in.
PendingItem = tuple[str, str, str, str, str, str]


def resolve_batch_model(judge_model: str) -> str:
    """The Batch API is only wired up here for OpenAI; other providers have
    different upload/submit semantics that aren't implemented."""
    if "/" not in judge_model or judge_model.split("/", 1)[0] != "openai":
        raise ValueError(
            f"--batch only supports openai judge models, got {judge_model!r}"
        )
    return judge_model.split("/", 1)[1]


def build_batch_lines(
    judge_model: str,
    pending: list[PendingItem],
    rubric_version: str = RUBRIC_VERSION,
    sampling: dict | None = None,
) -> list[dict]:
    """One /v1/chat/completions batch line per pending item, in list order —
    results are matched back to `pending` by custom_id == str(index)."""
    model = resolve_batch_model(judge_model)
    lines = []
    for idx, (_model, _ctx, task, _repo_id, input_prompt, generated_doc) in enumerate(
        pending
    ):
        messages, schema = render_judge_messages(
            task, input_prompt, generated_doc, rubric_version
        )
        lines.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    **(sampling or {}),
                    "response_format": type_to_response_format_param(schema),
                },
            }
        )
    return lines


# OpenAI caps a batch input file at 200 MB and 50,000 requests; stay well
# below both.
MAX_BATCH_BYTES = 150 * 1024 * 1024
MAX_BATCH_REQUESTS = 40_000


def write_batch_inputs(
    judge_model: str,
    pending: list[PendingItem],
    batch_dir: Path,
    rubric_version: str = RUBRIC_VERSION,
    sampling: dict | None = None,
    lines: list[dict] | None = None,
) -> list[Path]:
    """Writes the batch input as one or more JSONL files, each under the
    per-file limits. custom_ids index the whole pending list, so results from
    any of the files map back to the same item. Prebuilt `lines` (e.g. the
    alignment judge's) are written as given instead of building them from
    `pending`."""
    resolve_batch_model(judge_model)  # validate early, before writing anything
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    chunk: list[bytes] = []
    size = 0

    def flush() -> None:
        paths.append(batch_dir / f"batch_input_{len(paths):02d}.jsonl")
        with open(paths[-1], "wb") as out:
            out.writelines(chunk)

    if lines is None:
        lines = build_batch_lines(judge_model, pending, rubric_version, sampling)
    for line in lines:
        data = (json.dumps(line) + "\n").encode("utf-8")
        if chunk and (
            size + len(data) > MAX_BATCH_BYTES or len(chunk) >= MAX_BATCH_REQUESTS
        ):
            flush()
            chunk, size = [], 0
        chunk.append(data)
        size += len(data)
    if chunk:
        flush()
    return paths


UPLOAD_ATTEMPTS = 5


def _with_retries(what: str, call: Callable[[], object]) -> object:
    """Retries a flaky upload or create call (e.g. a dropped connection on a
    slow network) with increasing pauses before giving up."""
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        try:
            return call()
        except Exception as e:
            if attempt == UPLOAD_ATTEMPTS:
                raise
            wait = 15 * attempt
            print(
                f"  {what} failed ({e}); retry {attempt}/{UPLOAD_ATTEMPTS - 1} in {wait}s"
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def submit_batches(
    judge_model: str,
    pending: list[PendingItem],
    batch_dir: Path,
    rubric_version: str = RUBRIC_VERSION,
    sampling: dict | None = None,
    lines: list[dict] | None = None,
) -> list[str]:
    """Writes the batch input files, uploads them, and creates one batch job
    per file. Returns the batch ids, also written one per line to
    batch_dir/batch_ids.txt so a crashed run can be resumed with --batch_id.

    If batch_ids.txt already lists batches from an interrupted submission,
    their files are skipped and only the remaining ones are submitted: the
    input files are rebuilt identically from the same pending list, so the
    k-th id always belongs to the k-th file."""
    ids_path = batch_dir / "batch_ids.txt"
    ids: list[str] = (
        ids_path.read_text(encoding="utf-8").split() if ids_path.exists() else []
    )
    paths = write_batch_inputs(
        judge_model, pending, batch_dir, rubric_version, sampling, lines
    )
    if len(ids) > len(paths):
        raise SystemExit(
            f"{ids_path} lists {len(ids)} batches but only {len(paths)} input files "
            "were built; the pending list changed since the first submission."
        )
    if ids:
        print(
            f"  {len(ids)} of {len(paths)} batch(es) already submitted "
            f"({ids_path}); submitting the remaining {len(paths) - len(ids)}",
            flush=True,
        )
    for path in paths[len(ids) :]:

        def upload(path: Path = path):
            with open(path, "rb") as f:
                return litellm.create_file(
                    file=f, purpose="batch", custom_llm_provider="openai"
                )

        uploaded = _with_retries(f"upload of {path.name}", upload)
        batch = _with_retries(
            f"batch creation for {path.name}",
            lambda file_id=uploaded.id: litellm.create_batch(
                completion_window="24h",
                endpoint="/v1/chat/completions",
                input_file_id=file_id,
                custom_llm_provider="openai",
            ),
        )
        ids.append(batch.id)
        ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"  submitted {path.name} -> {batch.id}", flush=True)
    return ids


def poll_batch(
    batch_id: str,
    poll_interval: int = 30,
    on_update: Callable[[object], None] | None = None,
):
    """Blocks until the batch reaches a terminal status and returns it."""
    while True:
        batch = litellm.retrieve_batch(batch_id=batch_id, custom_llm_provider="openai")
        if on_update is not None:
            on_update(batch)
        if batch.status in TERMINAL_STATUSES:
            return batch
        time.sleep(poll_interval)


def fetch_batch_results(batch) -> dict[int, tuple[str | None, dict, str | None]]:
    """Returns {index: (raw_content, usage, error)}, keyed by the custom_id
    index assigned in build_batch_lines. error is set (raw/usage None) for
    lines the API itself failed to complete."""
    results: dict[int, tuple[str | None, dict, str | None]] = {}

    if batch.error_file_id:
        error_content = litellm.file_content(
            file_id=batch.error_file_id, custom_llm_provider="openai"
        )
        for line in error_content.text.strip().splitlines():
            row = json.loads(line)
            results[int(row["custom_id"])] = (None, {}, json.dumps(row))

    if batch.output_file_id:
        output_content = litellm.file_content(
            file_id=batch.output_file_id, custom_llm_provider="openai"
        )
        for line in output_content.text.strip().splitlines():
            row = json.loads(line)
            idx = int(row["custom_id"])
            if row.get("error"):
                results[idx] = (None, {}, json.dumps(row["error"]))
                continue
            body = row["response"]["body"]
            raw = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage") or {}
            results[idx] = (raw, usage, None)

    return results
