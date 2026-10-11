"""Generation loops shared by the API-client (LiteLLM) and local
(transformers / llama.cpp) backends — one (task, repo) pair per request,
either one at a time or concurrently across a thread pool.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from reverse_se.generation.utils import render_messages

GenerateFn = Callable[[list[dict]], tuple[str, dict]]
# generate_fn(messages) -> (answer_text, meta), where meta may carry
# prompt_tokens, completion_tokens, finish_reason, elapsed_seconds, reasoning.


@dataclass
class GenerationRow:
    repo_id: str
    model: str
    model_slug: str
    task: str
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    elapsed_seconds: float | None = None
    reasoning: str | None = None

    def write(self, out_base: Path) -> None:
        """Write three sibling files from out_base (which has no extension
        of its own, so this is safe even when repo_id itself contains a dot):
          out_base.md            the answer, as raw markdown — directly
                                  copy-pasteable/renderable, no JSON-string
                                  escaping to undo
          out_base.meta.json     everything else (model, tokens, timing, ...)
          out_base.reasoning.txt the reasoning/thinking trace, when present
        """
        out_base.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            k: v
            for k, v in vars(self).items()
            if k not in ("answer", "reasoning") and v is not None
        }
        out_base.with_name(out_base.name + ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )
        out_base.with_name(out_base.name + ".md").write_text(
            self.answer, encoding="utf-8"
        )

        if self.reasoning:
            out_base.with_name(out_base.name + ".reasoning.txt").write_text(
                self.reasoning, encoding="utf-8"
            )


def _process_one(
    task: str,
    repo_id: str,
    out_base: Path,
    dataset: dict,
    prompt_dir: Path,
    model_name: str,
    slug: str,
    context_budget: str,
    generate_fn: GenerateFn,
) -> tuple[GenerationRow | None, Exception | None]:
    """Render + generate a single (task, repo) pair and write its result.
    Returns (row, None) on success or (None, exception) on failure — never
    raises, so callers (sequential loop or thread-pool futures) can report
    the failure without special-casing render vs. generate errors."""
    try:
        messages = render_messages(prompt_dir, task, dataset[repo_id])
    except Exception as exc:
        return None, exc

    t0 = time.perf_counter()
    try:
        text, meta = generate_fn(messages)
    except Exception as exc:
        return None, exc
    elapsed = meta.get("elapsed_seconds", round(time.perf_counter() - t0, 2))

    row = GenerationRow(
        repo_id=repo_id,
        model=model_name,
        model_slug=slug,
        task=task,
        answer=text,
        prompt_tokens=meta.get("prompt_tokens", 0),
        completion_tokens=meta.get("completion_tokens", 0),
        finish_reason=meta.get("finish_reason"),
        elapsed_seconds=elapsed,
        reasoning=meta.get("reasoning"),
    )
    row.write(out_base)
    return row, None


def run_sequential(
    work: list[tuple[str, str, Path]],
    dataset: dict,
    prompt_dir: Path,
    model_name: str,
    slug: str,
    context_budget: str,
    generate_fn: GenerateFn,
) -> tuple[int, int]:
    """Render + generate one (task, repo) pair at a time, writing each result
    as soon as it completes so partial runs remain resumable."""
    done = errors = 0
    with tqdm(work, desc=f"{slug}/{context_budget}", unit="doc") as pbar:
        for task, repo_id, out_path in pbar:
            row, exc = _process_one(
                task,
                repo_id,
                out_path,
                dataset,
                prompt_dir,
                model_name,
                slug,
                context_budget,
                generate_fn,
            )
            if exc is not None:
                tqdm.write(f"ERROR {context_budget}/{task}/{repo_id}: {exc}")
                errors += 1
            else:
                tqdm.write(
                    f"ok {context_budget}/{task}/{repo_id} "
                    f"({row.completion_tokens} tokens, {row.elapsed_seconds}s)"
                )
                done += 1
            pbar.set_postfix(done=done, errors=errors)

    return done, errors


def run_concurrent(
    work: list[tuple[str, str, Path]],
    dataset: dict,
    prompt_dir: Path,
    model_name: str,
    slug: str,
    context_budget: str,
    generate_fn: GenerateFn,
    max_workers: int,
) -> tuple[int, int]:
    """Like run_sequential, but issues up to max_workers requests at once —
    the right way to drive a server with continuous batching (e.g. vLLM)
    instead of feeding it one request at a time and leaving most of its
    batch capacity idle. Not useful for the local transformers/llama.cpp
    backends, which are single-GPU and compute-bound either way.

    All printing/counting happens in the main thread as futures complete, so
    output stays interleaved-but-readable without needing a print lock."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = errors = 0
    with (
        ThreadPoolExecutor(max_workers=max_workers) as pool,
        tqdm(total=len(work), desc=f"{slug}/{context_budget}", unit="doc") as pbar,
    ):
        futures = {
            pool.submit(
                _process_one,
                task,
                repo_id,
                out_path,
                dataset,
                prompt_dir,
                model_name,
                slug,
                context_budget,
                generate_fn,
            ): (task, repo_id)
            for task, repo_id, out_path in work
        }
        for future in as_completed(futures):
            task, repo_id = futures[future]
            row, exc = future.result()
            if exc is not None:
                tqdm.write(f"ERROR {context_budget}/{task}/{repo_id}: {exc}")
                errors += 1
            else:
                tqdm.write(
                    f"ok {context_budget}/{task}/{repo_id} "
                    f"({row.completion_tokens} tokens, {row.elapsed_seconds}s)"
                )
                done += 1
            pbar.set_postfix(done=done, errors=errors)
            pbar.update(1)

    return done, errors
