"""Judge rubrics: the prompt text lives in prompts/judge/<version>/<task>.jinja,
the output schema in judge/schemas.py; resolve_rubric pairs the two.

RUBRIC_VERSION is the rubric every run uses unless --rubric_version says
otherwise (and the one judge/aggregate.py reads by default); every judged
row records its version. What each version
changed is listed in prompts/judge/README.md. To add a version, copy the
current folder, edit the templates, and register its schemas in SCHEMAS.
"""

from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from reverse_se.evaluation.judge import schemas

RUBRIC_VERSION = "v3"
TASKS = ("intent", "req", "c4")
PROMPT_DIR = Path(__file__).resolve().parents[4] / "prompts" / "judge"

# version -> task -> output schema.
SCHEMAS: dict[str, dict[str, type[BaseModel]]] = {
    "v3": {
        "intent": schemas.IntentEvaluation,
        "req": schemas.ReqEvaluation,
        "c4": schemas.C4Evaluation,
    },
    "v2": {
        "intent": schemas.IntentEvaluationV2,
        "req": schemas.ReqEvaluationV2,
        "c4": schemas.C4EvaluationV2,
    },
}
RUBRIC_VERSIONS = tuple(SCHEMAS)

_env = Environment(
    loader=FileSystemLoader(PROMPT_DIR), undefined=StrictUndefined, autoescape=False
)


@cache
def rubric_text(task: str, rubric_version: str = RUBRIC_VERSION) -> str:
    return _env.get_template(f"{rubric_version}/{task}.jinja").render()


def resolve_rubric(
    task: str, rubric_version: str = RUBRIC_VERSION
) -> tuple[str, type[BaseModel]]:
    return rubric_text(task, rubric_version), SCHEMAS[rubric_version][task]
