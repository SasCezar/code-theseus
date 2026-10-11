"""Cross-artefact alignment: does each artefact trace to the next one in the
chain intent -> requirements -> architecture?

The judge sees one adjacent pair per call (intent_req or req_c4), extracts the
upstream items (intent items) or the architecture elements, and traces every
requirement ID to them; coverage, orphan and contradiction metrics are then
computed here from the trace lists, not scored by the judge. Rubric text lives
in prompts/judge/align/<version>/.

Conditions, so the absolute numbers can be read against baselines:
- matched:  both artefacts from the same generator and repository;
- cross:    same repository, different generators;
- shuffled: same generator, the downstream artefact of another repository;
- planted:  matched, with requirements transplanted from another repository
            into the requirements document (the judge should leave them
            untraced; the share it does is the detection rate).
"""

import random
import re
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

ALIGN_VERSION = "v3"
CONDITIONS = ("matched", "cross", "shuffled", "planted")
CHAIN_CONDITIONS = ("matched", "cross", "shuffled")
PAIRS = ("intent_req", "req_c4")
PROMPT_DIR = Path(__file__).resolve().parents[4] / "prompts" / "judge"

# ── Output schemas ─────────────────────────────────────────────────────────
# v1: the 30-repo pilot that also listed contradictions.


class IntentItem(BaseModel):
    id: str = Field(description="I1, I2, ... in document order.")
    kind: Literal["goal", "capability", "constraint", "quality"]
    text: str = Field(description="The item, paraphrased in one sentence.")


class RequirementTrace(BaseModel):
    req_id: str = Field(description="Requirement ID exactly as in the document.")
    refines: list[str] = Field(
        description="IDs of the intent items it refines; empty if none."
    )
    note: str


class ArchitectureElement(BaseModel):
    id: str = Field(description="E1, E2, ... in document order.")
    name: str
    level: Literal["container", "component"]


class RequirementAllocation(BaseModel):
    req_id: str = Field(description="Requirement ID exactly as in the document.")
    allocated_to: list[str] = Field(
        description="IDs of the elements that realise or are constrained by "
        "it; empty if none."
    )
    note: str


class Contradiction(BaseModel):
    upstream_excerpt: str
    downstream_excerpt: str
    explanation: str


class IntentReqAlignment(BaseModel):
    intent_items: list[IntentItem]
    requirement_traces: list[RequirementTrace]
    contradictions: list[Contradiction]


class ReqC4Alignment(BaseModel):
    architecture_elements: list[ArchitectureElement]
    requirement_allocations: list[RequirementAllocation]
    contradictions: list[Contradiction]


# ── v2: no contradictions (they did not separate matched from shuffled pairs
# in the v1 pilot), notes, item kinds or element levels, to cut output tokens.


class IntentItemV2(BaseModel):
    id: str = Field(description="I1, I2, ... in document order.")
    text: str = Field(description="At most ten words identifying the item.")


class RequirementTraceV2(BaseModel):
    req_id: str = Field(description="Requirement ID exactly as in the document.")
    refines: list[str] = Field(
        description="IDs of the intent items it refines; empty if none."
    )


class ArchitectureElementV2(BaseModel):
    id: str = Field(description="E1, E2, ... in document order.")
    name: str


class RequirementAllocationV2(BaseModel):
    req_id: str = Field(description="Requirement ID exactly as in the document.")
    allocated_to: list[str] = Field(
        description="IDs of the elements that realise or are constrained by "
        "it; empty if none."
    )


class IntentReqAlignmentV2(BaseModel):
    intent_items: list[IntentItemV2]
    requirement_traces: list[RequirementTraceV2]


class ReqC4AlignmentV2(BaseModel):
    architecture_elements: list[ArchitectureElementV2]
    requirement_allocations: list[RequirementAllocationV2]


# version -> pair -> output schema.
SCHEMAS: dict[str, dict[str, type[BaseModel]]] = {
    "v3": {"chain": None},  # set below, after ChainAlignment
    "v2": {"intent_req": IntentReqAlignmentV2, "req_c4": ReqC4AlignmentV2},
    "v1": {"intent_req": IntentReqAlignment, "req_c4": ReqC4Alignment},
}

_env = Environment(
    loader=FileSystemLoader(PROMPT_DIR), undefined=StrictUndefined, autoescape=False
)


@cache
def rubric_text(pair: str, version: str = ALIGN_VERSION) -> str:
    return _env.get_template(f"align/{version}/{pair}.jinja").render()


def render_messages(
    pair: str, upstream: str, downstream: str, version: str = ALIGN_VERSION
) -> list[dict]:
    names = {
        "intent_req": ("Stakeholder intent", "Requirements"),
        "req_c4": ("Requirements", "C4 architecture"),
    }[pair]
    return [
        {"role": "system", "content": rubric_text(pair, version)},
        {
            "role": "user",
            "content": (
                f"## {names[0]} document\n\n{upstream}\n\n---\n\n"
                f"## {names[1]} document\n\n{downstream}"
            ),
        },
    ]


# ── Requirements parsing and transplanting ─────────────────────────────────

ID_LINE = re.compile(r"^\*\*ID:\*\*\s*(\S+)", re.M)
HEADING = re.compile(r"^#", re.M)


def requirement_ids(doc: str) -> list[str]:
    return ID_LINE.findall(doc)


def is_nfr(req_id: str) -> bool:
    return bool(re.search(r"NFUNC|NFR|NON", req_id, re.I))


def requirement_blocks(doc: str) -> list[tuple[str, int, int]]:
    """(id, start, end) of each requirement block: from its **ID:** line to
    the next **ID:** line or heading."""
    starts = [(m.group(1), m.start()) for m in ID_LINE.finditer(doc)]
    bounds = sorted(
        [s for _, s in starts] + [m.start() for m in HEADING.finditer(doc)] + [len(doc)]
    )
    blocks = []
    for req_id, start in starts:
        end = next(b for b in bounds if b > start)
        blocks.append((req_id, start, end))
    return blocks


def transplant(
    host: str, donor: str, k: int, rng: random.Random
) -> tuple[str, list[str]]:
    """Insert k functional requirements of the donor document into the host
    document, each after a random functional requirement of the host and
    renumbered to continue the host's functional series. Returns the new
    document and the planted IDs."""
    host_fr = [b for b in requirement_blocks(host) if not is_nfr(b[0])]
    donor_fr = [b for b in requirement_blocks(donor) if not is_nfr(b[0])]
    if not host_fr or len(donor_fr) < k:
        return host, []
    numbers = [int(m.group(1)) for b in host_fr if (m := re.search(r"(\d+)$", b[0]))]
    prefix = re.sub(r"\d+$", "", host_fr[0][0]) or "REQ-FUNC-"
    width = len(re.search(r"(\d*)$", host_fr[0][0]).group(1)) or 3
    next_n = max(numbers, default=0) + 1

    inserts = []  # (position in host, text)
    planted = []
    anchors = rng.sample(host_fr, k) if len(host_fr) >= k else rng.choices(host_fr, k=k)
    for donor_block, anchor in zip(rng.sample(donor_fr, k), anchors, strict=False):
        new_id = f"{prefix}{next_n:0{width}d}"
        next_n += 1
        text = donor[donor_block[1] : donor_block[2]].rstrip()
        text = ID_LINE.sub(f"**ID:** {new_id}", text, count=1)
        inserts.append((anchor[2], "\n\n" + text + "\n\n"))
        planted.append(new_id)
    for pos, text in sorted(inserts, key=lambda x: -x[0]):
        host = host[:pos].rstrip("\n") + text + host[pos:].lstrip("\n")
    return host, planted


# ── Pair building ──────────────────────────────────────────────────────────


@dataclass
class PairJob:
    pair: str
    condition: str
    up_model: str
    up_repo: str
    down_model: str
    down_repo: str
    planted_ids: list[str] = field(default_factory=list)
    # The requirements text actually sent, when it differs from the file
    # (planted condition). Not written to the output.
    req_override: str | None = None

    @property
    def key(self) -> tuple:
        return (
            self.pair,
            self.condition,
            self.up_model,
            self.up_repo,
            self.down_model,
            self.down_repo,
        )

    def record(self) -> dict:
        d = asdict(self)
        d.pop("req_override")
        return d


def doc_path(results_dir: Path, model: str, task: str, repo: str) -> Path:
    return results_dir / model / "ctx95" / task / f"{repo}.md"


def derangement(
    repos: list[str], groups: dict[str, str], rng: random.Random
) -> dict[str, str]:
    """Partner for each repo, never itself, preferably sharing its group
    (language) so the shuffled baseline is not trivially easy."""
    while True:
        partner = {}
        by_group: dict[str, list[str]] = {}
        for r in repos:
            by_group.setdefault(groups.get(r, ""), []).append(r)
        for r in repos:
            pool = [p for p in by_group[groups.get(r, "")] if p != r]
            pool = pool or [p for p in repos if p != r]
            partner[r] = rng.choice(pool)
        if all(partner[r] != r for r in repos):
            return partner


def build_jobs(
    results_dir: Path,
    models: list[str],
    repos: list[str],
    groups: dict[str, str],
    seed: int = 0,
    k_planted: int = 2,
    conditions: tuple[str, ...] = CONDITIONS,
) -> list[PairJob]:
    """Jobs for the given conditions. The partner and transplant draws do not
    depend on which conditions are requested, so a subset is the same pairs
    as in a full build."""
    rng = random.Random(seed)
    partner = derangement(repos, groups, rng)
    read = lambda m, t, r: doc_path(results_dir, m, t, r).read_text()  # noqa: E731
    jobs: list[PairJob] = []
    for repo in repos:
        for m in models:
            jobs.append(PairJob("intent_req", "matched", m, repo, m, repo))
            jobs.append(PairJob("req_c4", "matched", m, repo, m, repo))
            jobs.append(PairJob("intent_req", "shuffled", m, repo, m, partner[repo]))
            jobs.append(PairJob("req_c4", "shuffled", m, repo, m, partner[repo]))
            for other in models:
                if other != m:
                    jobs.append(PairJob("intent_req", "cross", m, repo, other, repo))
                    jobs.append(PairJob("req_c4", "cross", m, repo, other, repo))
            planted_doc, planted = transplant(
                read(m, "req", repo), read(m, "req", partner[repo]), k_planted, rng
            )
            if planted:
                jobs.append(
                    PairJob(
                        "intent_req", "planted", m, repo, m, repo, planted, planted_doc
                    )
                )
                jobs.append(
                    PairJob("req_c4", "planted", m, repo, m, repo, planted, planted_doc)
                )
    return [j for j in jobs if j.condition in conditions]


def job_documents(results_dir: Path, job: PairJob) -> tuple[str, str, str]:
    """(upstream text, downstream text, requirements text) for a job."""
    if job.pair == "intent_req":
        up = doc_path(results_dir, job.up_model, "intent", job.up_repo).read_text()
        down = (
            job.req_override
            or doc_path(results_dir, job.down_model, "req", job.down_repo).read_text()
        )
        return up, down, down
    up = (
        job.req_override
        or doc_path(results_dir, job.up_model, "req", job.up_repo).read_text()
    )
    down = doc_path(results_dir, job.down_model, "c4", job.down_repo).read_text()
    return up, down, up


# ── Metrics ────────────────────────────────────────────────────────────────


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


def alignment_metrics(
    pair: str, evaluation: BaseModel, req_doc: str, planted: list[str]
) -> dict:
    """Coverage metrics from the judge's trace lists. Requirement IDs come
    from parsing the document, so a requirement the judge skipped counts as
    omitted, not as traced."""
    if pair == "intent_req":
        items = [i.id for i in evaluation.intent_items]
        links = {t.req_id: t.refines for t in evaluation.requirement_traces}
    else:
        items = [e.id for e in evaluation.architecture_elements]
        links = {a.req_id: a.allocated_to for a in evaluation.requirement_allocations}
    out = trace_metrics(items, links, req_doc, planted)
    if hasattr(evaluation, "contradictions"):
        out["n_contradictions"] = len(evaluation.contradictions)
    return out


def trace_metrics(
    items: list[str], links: dict[str, list[str]], req_doc: str, planted: list[str]
) -> dict:
    parsed = requirement_ids(req_doc)
    item_set = set(items)
    links = {r: [i for i in v if i in item_set] for r, v in links.items()}
    judged = [r for r in parsed if r in links]
    original = [r for r in judged if r not in planted]
    fr = [r for r in original if not is_nfr(r)]
    nfr = [r for r in original if is_nfr(r)]
    linked_items = {i for r in judged if r not in planted for i in links[r]}
    traced = lambda rs: _share(sum(bool(links[r]) for r in rs), len(rs))  # noqa: E731
    out = {
        "n_items": len(items),
        "n_requirements": len(parsed),
        "omitted_share": _share(len(parsed) - len(judged), len(parsed)),
        # Downstream coverage: upstream items reached by >= 1 requirement.
        "item_coverage": _share(len(linked_items), len(items)),
        # Upstream coverage: requirements traced to >= 1 item.
        "req_traced": traced(original),
        "fr_traced": traced(fr),
        "nfr_traced": traced(nfr),
        "mean_links": _share(sum(len(links[r]) for r in original), len(original)),
    }
    if planted:
        seen = [r for r in planted if r in links]
        out["planted_detected"] = _share(sum(not links[r] for r in seen), len(seen))
    return out


# ── v3: the whole chain in one call ────────────────────────────────────────
# One call per (intent, requirements, architecture) triple returns both trace
# lists, so the requirements document is sent once and both pairs come from
# one call. Three call types per repository and generator:
# - matched:  all three from the same generator and repository, with
#             requirements of another repository planted in the requirements
#             (planted IDs are excluded from the coverage metrics, so these
#             calls give the matched scores and the planted detection rate);
# - cross:    intent, requirements and architecture from three different
#             generators, rotated (A, B, C), (B, C, A), (C, A, B);
# - shuffled: intent and architecture of the repository, requirements of
#             its partner repository (same generator).


class RequirementChainTrace(BaseModel):
    req_id: str = Field(description="Requirement ID exactly as in the document.")
    refines: list[str] = Field(
        description="IDs of the intent items it refines; empty if none."
    )
    allocated_to: list[str] = Field(
        description="IDs of the elements that realise or are constrained by "
        "it; empty if none."
    )


class ChainAlignment(BaseModel):
    intent_items: list[IntentItemV2]
    architecture_elements: list[ArchitectureElementV2]
    requirement_traces: list[RequirementChainTrace]


SCHEMAS["v3"]["chain"] = ChainAlignment


@dataclass
class ChainJob:
    condition: str
    intent: tuple[str, str]  # (model, repo)
    req: tuple[str, str]
    c4: tuple[str, str]
    planted_ids: list[str] = field(default_factory=list)
    req_override: str | None = None

    @property
    def key(self) -> tuple:
        return (self.condition, *self.intent, *self.req, *self.c4)

    def documents(self, results_dir: Path) -> tuple[str, str, str]:
        return (
            doc_path(results_dir, *self.intent_path()).read_text(),
            self.req_override or doc_path(results_dir, *self.req_path()).read_text(),
            doc_path(results_dir, *self.c4_path()).read_text(),
        )

    def intent_path(self):
        return self.intent[0], "intent", self.intent[1]

    def req_path(self):
        return self.req[0], "req", self.req[1]

    def c4_path(self):
        return self.c4[0], "c4", self.c4[1]


def render_chain_messages(
    intent: str, req: str, c4: str, version: str = ALIGN_VERSION
) -> list[dict]:
    return [
        {"role": "system", "content": rubric_text("chain", version)},
        {
            "role": "user",
            "content": (
                f"## Stakeholder intent document\n\n{intent}\n\n---\n\n"
                f"## Requirements document\n\n{req}\n\n---\n\n"
                f"## C4 architecture document\n\n{c4}"
            ),
        },
    ]


def build_chain_jobs(
    results_dir: Path,
    models: list[str],
    repos: list[str],
    groups: dict[str, str],
    seed: int = 0,
    k_planted: int = 2,
    conditions: tuple[str, ...] = CHAIN_CONDITIONS,
) -> list[ChainJob]:
    rng = random.Random(seed)
    partner = derangement(repos, groups, rng)
    read = lambda m, t, r: doc_path(results_dir, m, t, r).read_text()  # noqa: E731
    jobs: list[ChainJob] = []
    n = len(models)
    for repo in repos:
        for i, m in enumerate(models):
            planted_doc, planted = transplant(
                read(m, "req", repo), read(m, "req", partner[repo]), k_planted, rng
            )
            jobs.append(
                ChainJob(
                    "matched",
                    (m, repo),
                    (m, repo),
                    (m, repo),
                    planted,
                    planted_doc if planted else None,
                )
            )
            if n >= 3:
                b, c = models[(i + 1) % n], models[(i + 2) % n]
                jobs.append(ChainJob("cross", (m, repo), (b, repo), (c, repo)))
            jobs.append(ChainJob("shuffled", (m, repo), (m, partner[repo]), (m, repo)))
    return [j for j in jobs if j.condition in conditions]


def chain_rows(job: ChainJob, evaluation: ChainAlignment, req_doc: str) -> list[dict]:
    """One row per pair, in the same shape as the pair-wise judge's rows, so
    the same summary reads both."""
    intent_items = [i.id for i in evaluation.intent_items]
    elements = [e.id for e in evaluation.architecture_elements]
    refines = {t.req_id: t.refines for t in evaluation.requirement_traces}
    allocated = {t.req_id: t.allocated_to for t in evaluation.requirement_traces}
    common = {"chain_key": list(job.key), "planted_ids": job.planted_ids}
    return [
        {
            "pair": "intent_req",
            "condition": job.condition,
            "up_model": job.intent[0],
            "up_repo": job.intent[1],
            "down_model": job.req[0],
            "down_repo": job.req[1],
            **common,
            **trace_metrics(intent_items, refines, req_doc, job.planted_ids),
        },
        {
            "pair": "req_c4",
            "condition": job.condition,
            "up_model": job.req[0],
            "up_repo": job.req[1],
            "down_model": job.c4[0],
            "down_repo": job.c4[1],
            **common,
            **trace_metrics(elements, allocated, req_doc, job.planted_ids),
        },
    ]
