"""Structured-output schemas the judge answers with, one set per rubric
version. The prompt text for each version is in prompts/judge/<version>/;
judge/rubrics.py pairs the two.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ── v3 (default) ───────────────────────────────────────────────────────────
# Unsuffixed: the API is sent the schema under the class name, so these keep
# the names of the original v3 runs.


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str = Field(
        description="Concise evidence-based explanation of the score."
    )


class SectionAssessment(BaseModel):
    identifiability: Literal["clear", "partial", "absent"]
    correctness: CriterionScore | None = Field(
        description="Accuracy of this section alone. Null if and only if "
        "identifiability is 'absent'."
    )


class C4Views(BaseModel):
    system_context: SectionAssessment
    container_level: SectionAssessment
    component_level: SectionAssessment


class RequirementsSections(BaseModel):
    fr: SectionAssessment
    nfr: SectionAssessment


class C4Evaluation(BaseModel):
    correctness: CriterionScore
    completeness: CriterionScore
    architectural_quality: CriterionScore
    consistency: CriterionScore
    comprehensibility: CriterionScore
    conciseness: CriterionScore
    usefulness: CriterionScore

    relationships: CriterionScore
    diagram_quality: CriterionScore | None = Field(
        description="Null if the document contains no diagrams."
    )
    c4_views: C4Views

    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Overall architectural-document quality score.",
    )

    overall_assessment: str


class ReqEvaluation(BaseModel):
    correctness: CriterionScore
    completeness: CriterionScore
    requirements_quality: CriterionScore
    consistency: CriterionScore
    comprehensibility: CriterionScore
    conciseness: CriterionScore
    usefulness: CriterionScore

    requirements_sections: RequirementsSections

    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Overall requirements-document quality score.",
    )

    overall_assessment: str


class IntentEvaluation(BaseModel):
    correctness: CriterionScore
    completeness: CriterionScore
    intent_quality: CriterionScore
    consistency: CriterionScore
    comprehensibility: CriterionScore
    conciseness: CriterionScore
    usefulness: CriterionScore

    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Overall intent-document quality score.",
    )

    overall_assessment: str


# ── v2 (comparisons only) ──────────────────────────────────────────────────


# v2 reuses the v3 CriterionScore, as the original v2 runs did.


class IssueCategory(StrEnum):
    SCOPE_BOUNDARY = "scope_boundary"
    ELEMENT_IDENTIFICATION = "element_identification"
    MEANING_ROLE = "meaning_role"
    RELATIONSHIP_DEPENDENCY = "relationship_dependency"
    CLASSIFICATION_TYPE = "classification_type"
    ABSTRACTION_GRANULARITY = "abstraction_granularity"
    MODALITY_PRIORITY = "modality_priority"
    UNSUPPORTED_INFERENCE = "unsupported_inference"
    REPRESENTATION = "representation"


class IssueSeverity(StrEnum):
    MAJOR = "major"
    MINOR = "minor"


class Issue(BaseModel):
    category: IssueCategory
    severity: IssueSeverity
    description: str = Field(
        description="Concrete, evidence-based description of the issue."
    )


class ViewIdentifiability(BaseModel):
    system_context: Literal["clear", "partial", "absent"]
    container_level: Literal["clear", "partial", "absent"]
    component_level: Literal["clear", "partial", "absent"]


class RequirementsIdentifiability(BaseModel):
    fr: Literal["clear", "partial", "absent"]
    nfr: Literal["clear", "partial", "absent"]


class C4EvaluationV2(BaseModel):
    correctness: CriterionScore
    completeness: CriterionScore
    architectural_quality: CriterionScore
    consistency: CriterionScore
    comprehensibility: CriterionScore

    c4_view_identifiability: ViewIdentifiability

    issues: list[Issue]

    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Overall architectural-document quality score.",
    )

    overall_assessment: str


class ReqEvaluationV2(BaseModel):
    correctness: CriterionScore
    completeness: CriterionScore
    requirements_quality: CriterionScore
    consistency: CriterionScore
    comprehensibility: CriterionScore

    requirements_identifiability: RequirementsIdentifiability

    issues: list[Issue]

    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Overall requirements-document quality score.",
    )

    overall_assessment: str


class IntentEvaluationV2(BaseModel):
    correctness: CriterionScore
    completeness: CriterionScore
    intent_quality: CriterionScore
    consistency: CriterionScore
    comprehensibility: CriterionScore

    issues: list[Issue]

    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Overall intent-document quality score.",
    )

    overall_assessment: str
