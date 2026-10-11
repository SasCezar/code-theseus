"""Judge model invocation via LiteLLM — so judging speaks the same client
library as reverse_se.generation.inference_litellm, against any provider."""

import json

import litellm
from pydantic import BaseModel

from reverse_se.evaluation.judge.rubrics import RUBRIC_VERSION, resolve_rubric
from reverse_se.evaluation.judge.schemas import (
    C4Views,
    CriterionScore,
    RequirementsIdentifiability,
    RequirementsSections,
    ViewIdentifiability,
)


def flatten_criterion(flat: dict, name: str, criterion: CriterionScore | None) -> None:
    if criterion is None:
        flat[name] = None
        return
    flat[name] = criterion.score
    flat[f"{name}_justification"] = criterion.justification


def flatten_structured(evaluation: BaseModel) -> dict:
    """Flatten a structured-output evaluation (e.g. C4Evaluation) into the
    flat {metric: score} row shape the rest of the pipeline expects. A
    criterion the judge left null (e.g. diagram_quality with no diagrams)
    is stored as None, never as a score."""
    flat: dict = {}
    for field_name, value in evaluation:
        if isinstance(value, CriterionScore):
            flatten_criterion(flat, field_name, value)
        elif isinstance(value, ViewIdentifiability):
            flat["system_context_view"] = value.system_context
            flat["container_level_view"] = value.container_level
            flat["component_level_view"] = value.component_level
        elif isinstance(value, RequirementsIdentifiability):
            flat["fr_identifiability"] = value.fr
            flat["nfr_identifiability"] = value.nfr
        elif isinstance(value, list):
            flat[field_name] = json.dumps([item.model_dump() for item in value])
        elif isinstance(value, (C4Views, RequirementsSections)):
            for section, assessment in value:
                flat[f"{section}_identifiability"] = assessment.identifiability
                # An absent section has no correctness to score, whatever the
                # judge returned.
                criterion = (
                    None
                    if assessment.identifiability == "absent"
                    else assessment.correctness
                )
                flatten_criterion(flat, f"{section}_correctness", criterion)
        elif field_name == "overall_score":
            flat["overall"] = value
        else:
            flat[field_name] = value
    return flat


def render_judge_messages(
    task: str,
    input_prompt: str,
    generated_doc: str,
    rubric_version: str = RUBRIC_VERSION,
) -> tuple[list[dict], type[BaseModel]]:
    """Builds the judge chat messages and resolves the response schema for a
    task. Shared by the live and batch call paths so both speak the exact
    same rubric."""
    rubric, schema = resolve_rubric(task, rubric_version)
    messages = [
        {"role": "system", "content": rubric},
        {
            "role": "user",
            "content": (
                "## Input given to the generation model\n\n"
                f"{input_prompt}\n\n"
                "---\n\n"
                "## Generated document to evaluate\n\n"
                f"{generated_doc}"
            ),
        },
    ]
    return messages, schema


def parse_judge_response(
    raw: str, schema: type[BaseModel], usage: dict | None = None
) -> dict:
    scores = flatten_structured(schema.model_validate_json(raw))
    scores["_usage"] = usage or {}
    return scores


def call_judge(
    judge_model: str,
    task: str,
    input_prompt: str,
    generated_doc: str,
    api_base: str | None = None,
    rubric_version: str = RUBRIC_VERSION,
    sampling: dict | None = None,
    timeout: float = 600.0,
) -> dict:
    """sampling holds the parameters to send (e.g. temperature, top_p,
    reasoning_effort); anything absent is left to the model's default."""
    messages, schema = render_judge_messages(
        task, input_prompt, generated_doc, rubric_version
    )
    response = litellm.completion(
        model=judge_model,
        messages=messages,
        response_format=schema,
        api_base=api_base,
        timeout=timeout,
        **(sampling or {}),
    )
    raw = response.choices[0].message.content.strip()
    usage = (
        json.loads(json.dumps(dict(response.usage), default=vars))
        if response.usage
        else {}
    )
    return parse_judge_response(raw, schema, usage)
