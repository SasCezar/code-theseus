"""LLM-as-judge scoring for generated SE artefacts.

Submodules:
  rubrics    Loads rubric prompts (prompts/judge/<version>/) with their schemas
  schemas    Structured-output schemas per rubric version
  client     Judge invocation via LiteLLM
  discovery  Locating result files to judge, across the layouts the pipeline has used
  report     Score summary table + tidy CSV export
"""

from reverse_se.evaluation.judge.client import call_judge
from reverse_se.evaluation.judge.rubrics import RUBRIC_VERSION, TASKS, resolve_rubric

__all__ = ["RUBRIC_VERSION", "TASKS", "call_judge", "resolve_rubric"]
