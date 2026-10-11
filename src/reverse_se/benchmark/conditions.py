"""Build the A/B task instructions handed to the coding agent.

Condition A: intent only. Condition B: intent + requirements + architecture.
Both conditions share the same preamble (commit0.task_preamble) -- the
scaffold and visible tests are identical between conditions and are the
workspace the agent is launched into, not part of this text.
"""

from __future__ import annotations

from pathlib import Path

CONDITION_A = "intent_only"
CONDITION_B = "req_arch"
CONDITIONS = (CONDITION_A, CONDITION_B)


def _read_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def build_task(preamble: str, docs: dict[str, Path], condition: str) -> str:
    """Return the full task text; docs maps intent/req/c4 to their files."""
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        )
    parts = [preamble, "## Intent\n\n" + _read_doc(docs["intent"])]
    if condition == CONDITION_B:
        parts.append("## Requirements\n\n" + _read_doc(docs["req"]))
        parts.append("## Architecture\n\n" + _read_doc(docs["c4"]))
    return "\n\n".join(parts) + "\n"
