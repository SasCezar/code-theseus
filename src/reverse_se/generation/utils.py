"""Shared helpers for the generation scripts: dataset loading, prompt
rendering, output-path bookkeeping, and reasoning/answer splitting."""

import json
import os
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TASKS: dict[str, tuple[str, str]] = {
    "c4": ("c4_sys", "c4_user"),
    "req": ("requirments_sys", "requirments_user"),
    "intent": ("stakeholder_intent_sys", "stakeholder_intent_user"),
}

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"


# ── Dataset ──────────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> dict[str, dict]:
    """Load a JSONL dataset keyed by repo_id."""
    records: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rid := rec.get("repo_id"):
                records[rid] = rec
    return records


# ── Prompt rendering ─────────────────────────────────────────────────────────


def load_template(prompt_dir: Path, file_type: str):
    env = Environment(loader=FileSystemLoader(str(prompt_dir)))
    for fname in os.listdir(prompt_dir):
        if fname.endswith((".jinja", ".j2", ".txt")) and file_type in fname:
            return env.get_template(fname)
    raise ValueError(f"No template found for: {file_type}")


def render_template(prompt_dir: Path, file_type: str, **kwargs) -> str:
    return load_template(prompt_dir, file_type).render(**kwargs)


def render_user_content(prompt_dir: Path, user_template, record: dict) -> str:
    """Render the user-turn prompt: graph nodes/edges plus context/package files."""
    graph = record.get("package_only_graph", {})
    nodes = render_template(prompt_dir, "node", graph=graph)
    edges = render_template(prompt_dir, "edge", graph=graph)
    documents = render_template(
        prompt_dir, "context", context_files=record.get("context_files", [])
    )
    packages = render_template(
        prompt_dir, "package", package_files=record.get("package_files", [])
    )
    return user_template.render(
        nodes=nodes, edges=edges, documents=documents, package_files=packages
    )


def render_messages(prompt_dir: Path, task: str, record: dict) -> list[dict]:
    """Return [system, user] chat messages for a given task and dataset record."""
    sys_key, user_key = TASKS[task]
    sys_template = load_template(prompt_dir, sys_key)
    user_template = load_template(prompt_dir, user_key)
    return [
        {"role": "system", "content": sys_template.render()},
        {
            "role": "user",
            "content": render_user_content(prompt_dir, user_template, record),
        },
    ]


def build_full_prompt(prompt_dir: Path, task: str, record: dict) -> str:
    """Render a task's system+user prompt as a single string — for plain-text
    dumps, or for reconstructing the judge's view of the generation input."""
    system, user = render_messages(prompt_dir, task, record)
    return f"{system['content']}\n\n{user['content']}"


# ── Model naming ─────────────────────────────────────────────────────────────


def slug_from_model(model: str) -> str:
    return model.replace("/", "--").lower()


def resolve_slug(
    model: str, model_slug: str | None, gguf_path: Path | None = None
) -> str:
    if model_slug:
        return model_slug
    if gguf_path:
        return Path(gguf_path).stem
    return slug_from_model(model)


# ── Reasoning / answer splitting ─────────────────────────────────────────────


def extract_reasoning(message) -> str | None:
    """Pull a reasoning/thinking trace out of an OpenAI- or LiteLLM-shaped
    chat message object, checking the attribute names used by the various
    reasoning-capable providers (DeepSeek, gpt-oss, extended thinking, ...)."""
    for attr in ("reasoning_content", "thinking"):
        if val := getattr(message, attr, None):
            return val
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning_summary", "thinking"):
        if val := extra.get(key):
            return val
    return None


_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_HARMONY_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>\s*"
    r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>",
    re.DOTALL,
)


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Best-effort split of raw generated text into (answer, reasoning), for
    backends that inline the reasoning instead of returning it as a separate
    field (offline vLLM without a reasoning parser, transformers, llama.cpp).

    Recognises the two most common open-weight formats — DeepSeek-R1/QwQ
    <think> tags and gpt-oss harmony channels — and returns the text
    unchanged (reasoning=None) if neither is found.
    """
    if match := _HARMONY_ANALYSIS_RE.search(text):
        return text[match.end() :].strip(), match.group(1).strip()
    if match := _THINK_TAG_RE.search(text):
        return text[match.end() :].strip(), match.group(1).strip()
    return text, None


# ── Work scheduling ──────────────────────────────────────────────────────────


def shard(items: list, idx: int, count: int) -> list:
    return [x for i, x in enumerate(items) if i % count == idx]


def pending_pairs(
    dataset: dict,
    tasks: list[str],
    output_dir: Path,
    slug: str,
    context_budget: str,
    limit: int | None,
    skip_existing: bool,
    shard_index: int | None,
    shard_count: int | None,
) -> list[tuple[str, str, Path]]:
    """Return (task, repo_id, out_base) triples still to generate. out_base
    has no extension — GenerationRow.write() derives out_base.md (the
    answer), out_base.meta.json, and out_base.reasoning.txt from it."""
    work = []
    for task in tasks:
        repo_ids = sorted(dataset.keys())
        if shard_index is not None and shard_count:
            repo_ids = shard(repo_ids, shard_index, shard_count)
        if limit:
            repo_ids = repo_ids[:limit]
        for repo_id in repo_ids:
            out_base = output_dir / slug / context_budget / task / repo_id
            if skip_existing and out_base.with_name(out_base.name + ".md").exists():
                continue
            work.append((task, repo_id, out_base))
    return work
