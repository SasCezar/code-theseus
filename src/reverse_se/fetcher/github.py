from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
from urllib.parse import ParseResult, urlparse

import requests
from tqdm import tqdm  # pip install tqdm

from reverse_se.core.types import Repository


@dataclass
class FetchStats:
    total_urls: int = 0
    total_valid_repos: int = 0  # number of owner/repo pairs after parsing
    processed_repos: int = 0
    success: int = 0
    failures: Counter = None
    failure_examples: dict[str, list[dict[str, Any]]] = None

    def __post_init__(self):
        self.failures = Counter()
        self.failure_examples = defaultdict(list)

    def add_failure(
        self,
        kind: str,
        owner: str | None,
        repo: str | None,
        detail: Any = None,
        max_examples: int = 5,
    ):
        self.failures[kind] += 1
        if len(self.failure_examples[kind]) < max_examples:
            self.failure_examples[kind].append(
                {"owner": owner, "repo": repo, "detail": detail}
            )


class GitHubRepoBatchFetcher:
    """
    Fetches repository metadata in batches using GitHub GraphQL API,
    with live progress + failure breakdown.
    """

    GRAPHQL_URL = "https://api.github.com/graphql"

    def __init__(self, github_token: str, batch_size: int = 50, timeout_s: int = 30):
        if not github_token:
            raise ValueError("github_token is required for GraphQL batching.")

        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "github-repo-batch-fetcher",
            }
        )

    def fetch_projects(
        self,
        repo_urls: list[str],
        *,
        show_progress: bool = True,
        return_stats: bool = False,
    ) -> list[Repository] | tuple[list[Repository], FetchStats]:
        """
        Fetch projects for repo URLs, batching requests to GitHub GraphQL API.
        """
        stats = FetchStats(total_urls=len(repo_urls))

        # Parse URLs
        owner_repos: list[tuple[str, str]] = []
        for u in repo_urls:
            owner, repo = self._extract_owner_repo(u)
            if not owner or not repo:
                stats.add_failure("invalid_url", owner, repo, detail=u)
                continue
            owner_repos.append((owner, repo))

        stats.total_valid_repos = len(owner_repos)
        projects: list[Repository] = []

        pbar = tqdm(
            total=stats.total_valid_repos,
            disable=not show_progress,
            unit="repo",
            dynamic_ncols=True,
            desc="Fetching GitHub repos",
        )

        try:
            for chunk in self._chunks(owner_repos, self.batch_size):
                query, variables, alias_to_input = self._build_query(chunk)

                try:
                    resp = self.session.post(
                        self.GRAPHQL_URL,
                        json={"query": query, "variables": variables},
                        timeout=self.timeout_s,
                    )
                except requests.RequestException as e:
                    # whole-batch transport error -> count each repo in chunk as http_error
                    for owner, repo in chunk:
                        stats.add_failure("http_error", owner, repo, detail=str(e))
                    stats.processed_repos += len(chunk)
                    pbar.update(len(chunk))
                    pbar.set_postfix(
                        success=stats.success,
                        failures=sum(stats.failures.values()),
                        **self._top_failures(stats, top=3),
                    )
                    continue

                if resp.status_code != 200:
                    # count per repo in chunk as http_error (status code)
                    detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    for owner, repo in chunk:
                        stats.add_failure("http_error", owner, repo, detail=detail)
                    stats.processed_repos += len(chunk)
                    pbar.update(len(chunk))
                    pbar.set_postfix(
                        success=stats.success,
                        failures=sum(stats.failures.values()),
                        **self._top_failures(stats, top=3),
                    )
                    continue

                try:
                    payload = resp.json()
                except ValueError as e:
                    detail = f"json_decode_error: {e}"
                    for owner, repo in chunk:
                        stats.add_failure("parse_error", owner, repo, detail=detail)
                    stats.processed_repos += len(chunk)
                    pbar.update(len(chunk))
                    pbar.set_postfix(
                        success=stats.success,
                        failures=sum(stats.failures.values()),
                        **self._top_failures(stats, top=3),
                    )
                    continue

                # GraphQL top-level errors (may or may not accompany partial data)
                if payload.get("errors"):
                    # track once per query, but also keep moving
                    stats.failures["graphql_top_level_error"] += 1
                    if len(stats.failure_examples["graphql_top_level_error"]) < 5:
                        stats.failure_examples["graphql_top_level_error"].append(
                            {"detail": payload["errors"][:3]}
                        )

                data = payload.get("data") or {}

                # For each repo in the chunk, decide success/failure
                for alias, (owner, repo) in alias_to_input.items():
                    node = data.get(alias)

                    if not node:
                        # Repo not found / no access / invalid name
                        stats.add_failure(
                            "repo_null", owner, repo, detail="repository node is null"
                        )
                        continue

                    try:
                        projects.append(
                            Repository(
                                name=node["nameWithOwner"].replace("/", "|"),
                                url=node["url"],
                                description=node.get("description"),
                                stargazers_count=node["stargazerCount"],
                                language=(node.get("primaryLanguage") or {}).get(
                                    "name"
                                ),
                                archived=node["isArchived"],
                                pushed_at=node["pushedAt"],
                            )
                        )
                        stats.success += 1
                    except KeyError as e:
                        stats.add_failure(
                            "missing_field", owner, repo, detail=f"missing {e}"
                        )

                stats.processed_repos += len(chunk)
                pbar.update(len(chunk))
                pbar.set_postfix(
                    success=stats.success,
                    failures=sum(stats.failures.values()),
                    **self._top_failures(stats, top=3),
                )

        finally:
            pbar.close()

        if return_stats:
            return projects, stats
        return projects

    @staticmethod
    def _top_failures(stats: FetchStats, top: int = 3) -> dict[str, int]:
        # show only the largest few failure categories in the live postfix
        most_common = stats.failures.most_common(top)
        return dict(most_common)

    def _build_query(
        self, owner_repo_list: list[tuple[str, str]]
    ) -> tuple[str, dict, dict]:
        """
        Build a single GraphQL query with aliased repository fields.
        """
        fields = []
        variables: dict[str, str] = {}
        alias_to_input: dict[str, tuple[str, str]] = {}

        for i, (owner, repo) in enumerate(owner_repo_list):
            o_var = f"owner{i}"
            r_var = f"name{i}"
            alias = f"r{i}"

            variables[o_var] = owner
            variables[r_var] = repo
            alias_to_input[alias] = (owner, repo)

            fields.append(
                f"""
                {alias}: repository(owner: ${o_var}, name: ${r_var}) {{
                    nameWithOwner
                    url
                    description
                    stargazerCount
                    isArchived
                    pushedAt
                    primaryLanguage {{ name }}
                }}
                """
            )

        var_defs = ", ".join([f"${k}: String!" for k in variables])

        query = f"""
        query({var_defs}) {{
            {"".join(fields)}
        }}
        """

        return query, variables, alias_to_input

    def _extract_owner_repo(self, repo_url: str) -> tuple[str | None, str | None]:
        parsed: ParseResult = urlparse(repo_url)
        parts: list[str] = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            return None, None
        return parts[0], parts[1]

    def _chunks(self, items: list[tuple[str, str]], n: int):
        for i in range(0, len(items), n):
            yield items[i : i + n]
