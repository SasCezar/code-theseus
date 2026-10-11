"""Commit0-backed variant of the doc-assisted coding benchmark.

Replaces the act/CI-mining pipeline (HANDOVER.md: low yield, dependency
drift) with the Commit0 benchmark (arxiv 2412.01769), reusing as much of it
as possible:
  - the 54 libraries, their reference/base commits and test_dir
    (data/raw/commit0/commit0_instances.jsonl, scripts/python/commit0_list.py);
  - Commit0's stubbed starter repo (public function bodies -> `pass`) as the
    scaffold, taken straight from `/testbed` of its prebuilt, pinned Docker
    image `wentingzhao/<repo>:v0` -- dependencies are installed once in that
    image and never re-resolved, so ground truth can't drift;
  - Commit0's canonical per-repo test-ID lists and its eval command/metric
    (`<test_cmd> --json-report ... --continue-on-collection-errors
    <test_dir>`, pass rate = passed / len(test_ids), missing = failed).

Our only additions on top of Commit0's starter repo, applied identically in
both conditions so the generated docs are the only prose spec:
  - remove human-written documentation (README/docs/changelog/examples,
    Commit0's own spec.pdf.bz2, package long-description metadata);
  - remove docstrings from implementation files;
  - drop git history (the image's .git contains the reference commit);
  - hide a held-out majority of test functions (test_split.py) -- grading
    always runs the complete, original suite in a fresh container.
"""

from __future__ import annotations

import ast
import bz2
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import urllib.request
from pathlib import Path

from reverse_se.benchmark.api_scaffold import (
    build_api_scaffold,
    build_contract_scaffold,
    module_map,
)
from reverse_se.benchmark.test_paths import TEST_DIR_NAMES, is_test_path
from reverse_se.benchmark.test_split import split_tests_in_repo, test_function_names

PLATFORM = "linux/amd64"  # Commit0 images are amd64-only
COMMIT0_SHA = "1123db33219ee1bfcc735304cb4308663e329aeb"  # commit-0/commit0 @ main
TEST_IDS_URL = (
    "https://raw.githubusercontent.com/commit-0/commit0/"
    f"{COMMIT0_SHA}/commit0/data/test_ids/{{name}}.bz2"
)
OPENHANDS_VERSION = "1.16.0"  # same as the `benchmark` image's uv tool
AGENT_OPENHANDS_PYTHON = "/root/.local/share/uv/tools/openhands/bin/python"
AGENT_SHELL = "/opt/agent-shell"
# uid the agent's commands run as when the workspace is root-owned, and the
# uid find_leaks checks readability as.
AGENT_FALLBACK_UID = 1000
TESTBED = "/testbed"
# "stubs": Commit0's starter as-is (every module/class/signature kept);
# "api": only the public interface the tests use (api_scaffold.py);
# "contract": no implementation files at all, only API_CONTRACT.txt listing
# the dotted names the tests import (no signatures).
SCAFFOLD_LEVELS = ("stubs", "api", "contract")
CONTRACT_FILE = "API_CONTRACT.txt"

_DOC_DIRS = {"docs", "doc", "examples", "example", ".github"}
_DOC_FILE_RE = re.compile(
    r"^(readme|changelog|changes|history|news|authors|contributing|"
    r"code_of_conduct|spec\.pdf)(\..*)?$",
    re.IGNORECASE,
)
_DOC_EXTENSIONS = {".md", ".rst"}
_JUNK_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def load_instances(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def commit0_image(name: str) -> str:
    return f"wentingzhao/{name.lower()}:v0"


def agent_image(name: str) -> str:
    return f"reverse-se-commit0-{name.lower()}:v0"


def fetch_test_ids(name: str) -> list[str]:
    url = TEST_IDS_URL.format(name=name.lower().replace(".", "-"))
    with urllib.request.urlopen(url) as resp:
        text = bz2.decompress(resp.read()).decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _docker(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kwargs)


def _identity(path: Path) -> Path:
    return path


def remove_tree(path: Path, host_path=_identity) -> None:
    """rmtree that also handles files written by our (root) containers --
    the agent and grading containers leave root-owned files behind on a
    Linux host, which a non-root user can't delete directly."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except PermissionError:
        _docker(
            [
                "run",
                "--rm",
                "-v",
                f"{host_path(path.parent.resolve())}:/p",
                "--entrypoint",
                "rm",
                "alpine",
                "-rf",
                f"/p/{path.name}",
            ]
        )
    if path.exists():
        raise RuntimeError(f"could not remove {path}")


# ── Scaffold ──────────────────────────────────────────────────────────────────


_PYTEST_FILE_RE = re.compile(r"^(test_.*|.*_test|conftest)\.py$")


def is_test_file(rel: str, instance: dict) -> bool:
    """Test/fixture file: anything under the instance's own test_dir when
    that's a dedicated directory (e.g. `portalocker_tests/`, whose files
    don't all look like tests by name); inside the package, only test
    directories and pytest-style file names -- a module named `tests.py` or
    `testing.py` there is library code (jinja2.tests, xarray.testing);
    elsewhere, test_paths.is_test_path."""
    test_dir = instance["test"]["test_dir"].strip("/")
    src_dir = instance["src_dir"].strip("/")
    if test_dir not in (".", "", src_dir) and (rel + "/").startswith(test_dir + "/"):
        return True
    path = Path(rel)
    if (rel + "/").startswith(src_dir + "/"):
        return any(p.lower() in TEST_DIR_NAMES for p in path.parts[:-1]) or bool(
            _PYTEST_FILE_RE.match(path.name)
        )
    return is_test_path(path)


def is_impl_file(rel: str, instance: dict) -> bool:
    src_dir = instance["src_dir"].strip("/")
    return (rel + "/").startswith(src_dir + "/") and not is_test_file(rel, instance)


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or (
                [] if isinstance(node, ast.Module) else [ast.Pass()]
            )
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def strip_docstrings(source: str) -> str | None:
    """Drop module/class/function docstrings (and, via ast.unparse, comments)."""
    try:
        tree = _DocstringStripper().visit(ast.parse(source))
        return ast.unparse(ast.fix_missing_locations(tree))
    except (SyntaxError, ValueError, RecursionError):
        return None


def strip_long_description(path: Path) -> None:
    """Keep only a PKG-INFO/METADATA file's header block, minus Description:
    -- the body/Description is the project README verbatim."""
    text = path.read_text(encoding="utf-8", errors="replace")
    headers = text.split("\n\n", 1)[0].splitlines()
    kept, in_description = [], False
    for line in headers:
        if line.startswith("Description:"):
            in_description = True
            continue
        if in_description and (line.startswith((" ", "\t")) or line.startswith("|")):
            continue
        in_description = False
        kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def remove_human_docs(repo: Path, instance: dict) -> list[str]:
    removed = []
    for path in sorted(repo.iterdir()):
        name = path.name
        if path.is_dir() and name.lower() in _DOC_DIRS:
            shutil.rmtree(path)
        elif path.is_file() and (
            _DOC_FILE_RE.match(name)
            or (
                path.suffix.lower() in _DOC_EXTENSIONS
                and not name.upper().startswith("LICENSE")
            )
        ):
            path.unlink()
        else:
            continue
        removed.append(name)
    for path in list(repo.rglob("*")):
        if (
            path.name == "SOURCES.txt"
            and path.parent.name.endswith(".egg-info")
            and path.is_file()
        ):
            # Build metadata listing every file of the original implementation
            # (e.g. 67 babel/*.py paths): the module layout an architecture doc
            # is supposed to supply. PKG-INFO/top_level.txt stay -- the
            # installed package's metadata points at this egg-info.
            removed.append(path.relative_to(repo).as_posix())
            path.unlink()
        elif path.is_dir() and path.name in _JUNK_DIRS:
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file() and (path.suffix == ".pyc" or path.name == "PKG-INFO"):
            if path.suffix == ".pyc":
                path.unlink()
            else:
                strip_long_description(path)
    # Stray prose inside the package itself (e.g. src/pkg/README.md).
    src = repo / instance["src_dir"]
    for path in list(src.rglob("*")) if src.is_dir() else []:
        if path.is_file() and path.suffix.lower() in _DOC_EXTENSIONS:
            removed.append(path.relative_to(repo).as_posix())
            path.unlink()
    return removed


# Test-tree config pytest reads; never test data.
_TEST_CONFIG_RE = re.compile(r"^(.*\.(ini|cfg|toml)|requirements.*\.txt)$")


def remove_held_out_data(repo: Path, instance: dict) -> list[str]:
    """Remove test data files no visible test refers to, after the split.

    Held-out tests' data stays otherwise, and can hand over the answers --
    observed: chardet's only test detects the encoding of each file under
    tests/<ENCODING>/, all of it held out; the agent built its language
    models from those very files and scored 90% held-out with 0 visible
    tests. A file is kept if the remaining test code mentions its name (or
    stem, 4+ characters), or the name of a folder it's in below the test
    root -- deliberately conservative, since a removed file a visible test
    needs only costs the agent a local test run: grading uses the image's
    untouched tests and data."""
    test_dir = instance["test"]["test_dir"].strip("/")
    if test_dir in (".", "", instance["src_dir"].strip("/")):
        test_dir = "\0"  # not a dedicated test directory
    tests, data = [], []
    for path in repo.rglob("*"):
        rel = path.relative_to(repo)
        if not path.is_file() or rel.parts[0] in (".git", ".venv"):
            continue
        if path.suffix == ".py":
            if is_test_file(rel.as_posix(), instance):
                tests.append(rel)
        # Data only inside a test directory: a name-based match (e.g. babel's
        # cldr/.../GraphemeBreakTest.html) is library data.
        elif (rel.as_posix() + "/").startswith(test_dir + "/") or any(
            p.lower() in TEST_DIR_NAMES for p in rel.parts[:-1]
        ):
            data.append(rel)
    text = "\n".join(
        (repo / rel).read_text(encoding="utf-8", errors="replace") for rel in tests
    )
    removed = []
    for rel in sorted(data):
        if _TEST_CONFIG_RE.match(rel.name):
            continue
        folders = rel.parts[:-1]
        roots = [i for i, p in enumerate(folders) if p.lower() in TEST_DIR_NAMES]
        # Only folders below the test root: e.g. the package name (in every
        # import) must not keep src/pkg/tests/data/*.
        below = folders[roots[-1] + 1 :] if roots else folders[1:]
        names = {rel.name, *(p for p in below if len(p) >= 3)}
        if len(rel.stem) >= 4:
            names.add(rel.stem)
        if not any(name in text for name in names):
            (repo / rel).unlink()
            removed.append(rel.as_posix())
    return removed


def export_testbed(instance: dict, dest: Path) -> None:
    """Copy Commit0's starter repo out of its image, without .venv and .git.

    Taken from the image rather than a fresh clone so build artifacts the
    image's setup produced in-tree (babel's CLDR data, compiled extensions,
    submodules, generated _version.py) are kept.
    """
    image = commit0_image(instance["name"])
    if _docker(["image", "inspect", image]).returncode != 0:
        pull = _docker(["pull", "--platform", PLATFORM, image])
        if pull.returncode != 0:
            raise RuntimeError(f"docker pull {image} failed: {pull.stderr[-2000:]}")
    dest.mkdir(parents=True)
    tar = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            "--network",
            "none",
            "--entrypoint",
            "tar",
            image,
            "-C",
            TESTBED,
            "--exclude=./.venv",
            "--exclude=./.git",
            "-cf",
            "-",
            ".",
        ],
        stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-xf", "-", "-C", str(dest)], stdin=tar.stdout, check=True)
    if tar.wait() != 0:
        raise RuntimeError(f"exporting {TESTBED} from {image} failed")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=benchmark",
            "-c",
            "user.email=benchmark@local",
            # workspaces are written by the agent's container (root)
            "-c",
            "safe.directory=*",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def build_scaffold(
    instance: dict,
    dest: Path,
    test_ids: list[str],
    visible_fraction: float,
    seed: int,
    level: str = "stubs",
    host_path=_identity,
) -> dict:
    """Build the frozen agent workspace at dest/repo; returns its manifest."""
    repo = dest / "repo"
    remove_tree(dest, host_path)
    export_testbed(instance, repo)

    removed_docs = remove_human_docs(repo, instance)
    docstring_files, unparsable = 0, []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        if not is_impl_file(rel, instance):
            continue
        stripped = strip_docstrings(path.read_text(encoding="utf-8", errors="replace"))
        if stripped is None:
            unparsable.append(rel)
            continue
        path.write_text(stripped + "\n", encoding="utf-8")
        docstring_files += 1

    all_py = sorted(p.relative_to(repo).as_posix() for p in repo.rglob("*.py"))
    base_impl = [rel for rel in all_py if is_impl_file(rel, instance)]
    (dest / "base_impl_files.json").write_text(json.dumps(base_impl), encoding="utf-8")
    import_roots = sorted({m.split(".")[0] for m in module_map(repo, base_impl)})
    api_stats = {}
    if level == "api":
        # Before the split: the surface must cover held-out tests too.
        tests = [rel for rel in all_py if is_test_file(rel, instance)]
        api_stats = build_api_scaffold(repo, base_impl, tests)
    elif level == "contract":
        tests = [rel for rel in all_py if is_test_file(rel, instance)]
        contract, api_stats = build_contract_scaffold(repo, base_impl, tests)
        (repo / CONTRACT_FILE).write_text(
            "# Names the test suite imports; each must be importable at exactly\n"
            f"# this dotted path. Package source goes under {instance['src_dir']}\n"
            + "".join(f"{name}\n" for name in contract),
            encoding="utf-8",
        )
    elif level != "stubs":
        raise ValueError(f"unknown scaffold level {level!r}")

    # Split where each test function is *defined*: a test ID names the file
    # it runs in, but its function may be inherited from a mixin/base class
    # elsewhere (observed: cachetools' tests/__init__.py mixin -- splitting
    # only the ID's own file left every inherited test visible, 63% of IDs).
    defined = {
        rel: test_function_names(
            (repo / rel).read_text(encoding="utf-8", errors="replace")
        )
        for rel in all_py
        if is_test_file(rel, instance)
    }

    def owner(test_id: str) -> tuple[str, str]:
        rel, *_, func = test_id.split("::")
        func = func.split("[")[0]
        if func in defined.get(rel, ()):
            return rel, func
        candidates = sorted(f for f, names in defined.items() if func in names)
        same_dir = [f for f in candidates if Path(f).parent == Path(rel).parent]
        return (same_dir or candidates or [rel])[0], func

    owners = {tid: owner(tid) for tid in test_ids}
    weights: dict[str, dict[str, int]] = {}
    for rel, func in owners.values():
        weights.setdefault(rel, {})
        weights[rel][func] = weights[rel].get(func, 0) + 1
    split = split_tests_in_repo(
        repo,
        dest / "held_out_originals",
        sorted(weights),
        visible_fraction,
        seed,
        weights,
    )
    held_out_ids = sorted(
        tid
        for tid, (rel, func) in owners.items()
        if func in split.per_file.get(rel, {}).get("held_out", ())
    )
    removed_test_data = remove_held_out_data(repo, instance)

    _git(["init", "-q", "-b", "main"], repo)
    (repo / ".git/info/exclude").write_text(".venv\n__pycache__/\n*.pyc\n")
    _git(["add", "-A", "-f", "."], repo)
    commit = _git(["commit", "-q", "-m", "scaffold"], repo)
    if commit.returncode != 0:
        raise RuntimeError(f"scaffold commit failed: {commit.stderr}")
    scaffold_commit = _git(["rev-parse", "HEAD"], repo).stdout.strip()

    held_out = {
        "held_out_ids": held_out_ids,
        "per_file": {rel: names["held_out"] for rel, names in split.per_file.items()},
    }
    manifest = {
        "project_id": instance["project_id"],
        "name": instance["name"],
        "lite": instance["lite"],
        "image": commit0_image(instance["name"]),
        "base_commit": instance["base_commit"],
        "reference_commit": instance["reference_commit"],
        "src_dir": instance["src_dir"],
        "test_dir": instance["test"]["test_dir"],
        "scaffold": level,
        "import_roots": import_roots,
        "scaffold_commit": scaffold_commit,
        "visible_fraction": visible_fraction,
        "seed": seed,
        "num_test_ids": len(test_ids),
        "num_held_out_test_ids": len(held_out_ids),
        "removed_docs": removed_docs,
        "docstring_stripped_files": docstring_files,
        "unparsable_impl_files": unparsable,
        "unsplit_test_files": split.unsplit_files,
        "removed_test_data": removed_test_data,
        **api_stats,
    }
    (dest / "test_ids.json").write_text(json.dumps(test_ids), encoding="utf-8")
    (dest / "held_out.json").write_text(
        json.dumps(held_out, indent=1), encoding="utf-8"
    )
    return manifest


def is_held_out(test_id: str, held_out: dict) -> bool:
    """held_out.json: {"held_out_ids": [...]} (explicit, handles inherited
    tests), or the older {test_file: [held-out function names]}."""
    if "held_out_ids" in held_out:
        ids = held_out.get("_id_set")
        if ids is None:
            ids = held_out["_id_set"] = set(held_out["held_out_ids"])
        return test_id in ids
    rel, *_, func = test_id.split("::")
    return func.split("[")[0] in held_out.get(rel, ())


# ── Agent task ────────────────────────────────────────────────────────────────

_SCAFFOLD_DESCRIPTION = {
    "stubs": """\
Every public function and method in the library's source (`{src_dir}`) has
had its body replaced by `pass`; private helpers, docstrings and all
documentation have been removed.""",
    "api": """\
The library's source (`{src_dir}`) contains only its public interface -- the
modules, classes and functions the tests use -- with every function and
method body replaced by `pass`. All internal modules, helpers, docstrings
and documentation have been removed: design and write the internals
yourself, adding whatever modules you need.""",
    "contract": """\
The library's source code has been removed entirely: there are no
implementation files under `{src_dir}`. `API_CONTRACT.txt` lists every
name the tests import, at the dotted path they import it from -- those
paths are fixed, everything else (modules, classes, signatures, internal
structure) is yours to design and write.""",
}

TASK_PREAMBLE = """\
You are implementing a Python library from scratch inside this repository.

{scaffold_description} Your job is to implement the library so
that its test suite passes. Only part of the test suite is present in
`{test_dir}` -- a larger hidden test suite is used for grading, so implement
the full behavior described below, not just what the visible tests check.
Do not modify the tests.

The Python environment with all dependencies is installed and active: run
the tests with `{test_cmd} {test_dir}`. There is no network access.

The original implementation of this library is not available anywhere in
this environment (no installed copy, no git history, no network). Do not
search for it: writing the implementation yourself is the task. Use only
this repository, the documents below, and the visible tests.
"""


def task_preamble(instance: dict, level: str = "stubs") -> str:
    test = instance["test"]
    description = _SCAFFOLD_DESCRIPTION[level].format(src_dir=instance["src_dir"])
    return TASK_PREAMBLE.format(
        scaffold_description=description,
        test_dir=test["test_dir"],
        test_cmd=test["test_cmd"],
    )


# ── Agent image ───────────────────────────────────────────────────────────────

# Scrubs the library's own installed metadata (long description == README)
# from the image's venv; the agent sees this venv, not the image's /testbed.
_SANITIZE_PY = r"""
import json, pathlib, sys
sys.path.insert(0, "/opt")
from strip_meta import strip_long_description
for direct_url in pathlib.Path("/testbed/.venv").glob("lib/python*/site-packages/*.dist-info/direct_url.json"):
    if "/testbed" in json.loads(direct_url.read_text()).get("url", ""):
        meta = direct_url.with_name("METADATA")
        if meta.is_file():
            strip_long_description(meta)
"""

# Terminal commands of the agent run through this (openhands_runner.
# _confined_agent): as the workspace owner's uid, never root, so OpenHands'
# own environment under /root -- which holds copies of benchmarked libraries
# (jinja2, requests, ...) -- is unreadable.
_AGENT_SHELL_SCRIPT = f"""\
#!/bin/sh
uid=$(stat -c %u {TESTBED}); gid=$(stat -c %g {TESTBED})
if [ "$uid" = 0 ]; then uid={AGENT_FALLBACK_UID}; gid={AGENT_FALLBACK_UID}; fi
exec setpriv --reuid="$uid" --regid="$gid" --clear-groups \\
    env HOME=/home/agent /bin/bash "$@"
"""

# Deletes other copies of the library's packages the agent could read:
# system Python / pip's vendored packages / the library's own venv. Not
# OpenHands' env (it needs them; hidden by permissions instead).
_SCRUB_PY = r"""
import pathlib, shutil, subprocess, sys
roots = sys.argv[1:]
names = [a for r in roots for a in ("-name", r, "-o", "-name", r + ".py", "-o")][:-1]
out = subprocess.run(
    ["find", "/", "-xdev", "(", "-path", "/proc", "-o", "-path", "/sys", "-o",
     "-path", "/testbed", "-o", "-path", "/root/.local/share/uv/tools", ")",
     "-prune", "-o", "(", *names, ")", "-print"]
    + [],
    capture_output=True, text=True).stdout.split()
out += subprocess.run(["find", "/testbed/.venv", "(", *names, ")", "-print"],
                      capture_output=True, text=True).stdout.split()
for p in map(pathlib.Path, out):
    if p.parent.name in {"site-packages", "dist-packages", "_vendor"} or (
        p.is_dir() and (p / "__init__.py").exists()
    ):
        print("scrub", p)
        shutil.rmtree(p) if p.is_dir() else p.unlink()
"""

_AGENT_DOCKERFILE = """\
FROM --platform={platform} {base}
RUN uv tool install openhands=={openhands} --python 3.12 && rm -rf /root/.cache
COPY strip_meta.py sanitize.py scrub.py agent-shell /opt/
RUN python3 /opt/sanitize.py && python3 /opt/scrub.py {roots} \\
    && chmod 755 /opt/agent-shell && mkdir -p /home/agent && chmod 777 /home/agent \\
    && chmod 711 /root /root/.local /root/.local/share /root/.local/share/uv \\
    && chmod 700 /root/.local/share/uv/tools
ENV VIRTUAL_ENV=/testbed/.venv PATH=/testbed/.venv/bin:$PATH \\
    LITELLM_LOCAL_MODEL_COST_MAP=True OPENHANDS_SUPPRESS_BANNER=1
WORKDIR /testbed
"""


def build_agent_image(instance: dict, build_dir: Path, import_roots: list[str]) -> str:
    """Commit0 image + OpenHands SDK, so the agent runs *inside* the pinned
    environment and can execute the visible tests (no network needed) --
    confined: see _AGENT_SHELL_SCRIPT, _SCRUB_PY and find_leaks."""
    import inspect

    image = agent_image(instance["name"])
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    (build_dir / "strip_meta.py").write_text(
        "from pathlib import Path\n\n\n" + inspect.getsource(strip_long_description)
    )
    (build_dir / "sanitize.py").write_text(_SANITIZE_PY)
    (build_dir / "scrub.py").write_text(_SCRUB_PY)
    (build_dir / "agent-shell").write_text(_AGENT_SHELL_SCRIPT)
    (build_dir / "Dockerfile").write_text(
        _AGENT_DOCKERFILE.format(
            platform=PLATFORM,
            base=commit0_image(instance["name"]),
            openhands=OPENHANDS_VERSION,
            roots=" ".join(import_roots),
        )
    )
    build = _docker(["build", "--platform", PLATFORM, "-t", image, str(build_dir)])
    if build.returncode != 0:
        raise RuntimeError(f"docker build {image} failed: {build.stderr[-3000:]}")
    return image


# ── Grading ───────────────────────────────────────────────────────────────────


def build_overlay(
    workspace: Path,
    scaffold_commit: str,
    instance: dict,
    overlay: Path,
    base_impl_files: list[str] = (),
) -> dict:
    """Collect the agent's implementation (non-test files under src_dir) for
    grading. Tests, docs and config always come from the untouched image."""
    if overlay.exists():
        shutil.rmtree(overlay)
    (overlay / "files").mkdir(parents=True)
    src = instance["src_dir"]
    current = _git(
        ["ls-files", "--cached", "--others", "--exclude-standard", "--", src], workspace
    )
    original = _git(
        ["ls-tree", "-r", "--name-only", scaffold_commit, "--", src], workspace
    )
    # Plus git-ignored .py files: e.g. virtualenv ignores its generated
    # src/virtualenv/version.py -- the agent writes it, and without it every
    # test fails to import (observed: 0/296 in 5 of 6 runs).
    ignored = _git(
        ["ls-files", "--others", "--ignored", "--exclude-standard", "--", src],
        workspace,
    )
    current_files = {
        p for p in current.stdout.splitlines() if is_impl_file(p, instance)
    } | {
        p
        for p in ignored.stdout.splitlines()
        if p.endswith(".py") and "__pycache__" not in p and is_impl_file(p, instance)
    }
    original_files = {
        p for p in original.stdout.splitlines() if is_impl_file(p, instance)
    }
    for rel in sorted(current_files):
        if (workspace / rel).is_file():
            target = overlay / "files" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace / rel, target)
    # base_impl_files: the starter's .py files, which the grading container
    # still has even where the scaffold removed them (level "api").
    deleted = sorted((original_files | set(base_impl_files)) - current_files)
    (overlay / "deleted.txt").write_text("".join(f"{p}\n" for p in deleted))
    return {"impl_files": len(current_files), "deleted_files": len(deleted)}


# Per-test time limit (our one change to Commit0's eval command). Without it
# a single hanging test -- observed: agent-written prompt code looping
# forever on EOF, stuck at 78% of cookiecutter's suite -- runs into the
# whole-suite timeout, pytest never writes its report, and every test
# (including the ~180 that had passed) counts as failed. pytest-timeout is
# installed offline into the grading container from a pinned wheel.
PYTEST_TIMEOUT_WHEEL = (
    "https://files.pythonhosted.org/packages/py3/p/pytest-timeout/"
    "pytest_timeout-2.3.1-py3-none-any.whl"
)
WHEEL_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "wheels"
_WATCHDOG_SCRIPT = Path(__file__).resolve().with_name("grade_watchdog.py")


def _pytest_timeout_wheel_dir() -> Path:
    wheel = WHEEL_CACHE / PYTEST_TIMEOUT_WHEEL.rsplit("/", 1)[1]
    if not wheel.is_file():
        WHEEL_CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PYTEST_TIMEOUT_WHEEL) as resp:
            wheel.write_bytes(resp.read())
    return WHEEL_CACHE


def grade(
    instance: dict,
    out_dir: Path,
    host_path,
    overlay: Path | None = None,
    reference: bool = False,
    timeout: int = 1800,
    test_timeout: int = 0,
) -> dict:
    """Run Commit0's eval command in a fresh, network-less container of the
    unmodified image. overlay=None and reference=False grades the untouched
    Commit0 starter repo; reference=True grades the reference commit.
    host_path maps a local path to the path the Docker daemon sees.
    test_timeout > 0 adds a per-test limit (pytest-timeout; see above) and
    runs pytest under grade_watchdog.py, which kills and skips tests that
    hang past it anyway; 0 is Commit0's command exactly."""
    remove_tree(out_dir, host_path)
    out_dir.mkdir(parents=True)
    test = instance["test"]
    steps = [
        f"cd {TESTBED}",
        "source .venv/bin/activate",
        f"git reset --hard {instance['base_commit']} >/dev/null",
    ]
    mounts = ["-v", f"{host_path(out_dir)}:/grade"]
    if reference:
        steps.append(f"git reset --hard {instance['reference_commit']} >/dev/null")
    elif overlay is not None:
        mounts += ["-v", f"{host_path(overlay)}:/overlay:ro"]
        steps += [
            "xargs -r -d '\\n' rm -f -- < /overlay/deleted.txt",
            "cp -a /overlay/files/. .",
        ]
    # Verbose per-test lines let score() salvage outcomes if pytest dies
    # before writing its report; -vv so a project's addopts "-q" can't
    # cancel it out. Output only -- outcomes are unaffected.
    pytest_cmd = f"{test['test_cmd']} --json-report --continue-on-collection-errors -vv"
    if test_timeout:
        mounts += [
            "-v",
            f"{host_path(_pytest_timeout_wheel_dir())}:/wheels:ro",
            "-v",
            f"{host_path(_WATCHDOG_SCRIPT)}:/opt/grade_watchdog/grade_watchdog.py:ro",
        ]
        steps += [
            "uv pip install -q --offline --no-index --find-links /wheels "
            "--python .venv/bin/python pytest-timeout",
            # A test that outlives pytest-timeout (by a margin) is hung in a
            # way no signal inside pytest stops: see grade_watchdog.py.
            "python /opt/grade_watchdog/grade_watchdog.py --out /grade "
            f"--idle {test_timeout + 60} --deadline {max(60, timeout - 120)} "
            + shlex.quote(f"{pytest_cmd} --timeout={test_timeout} {test['test_dir']}"),
        ]
    else:
        steps.append(
            f"{pytest_cmd} --json-report-file=/grade/report.json {test['test_dir']} "
            "> /grade/test_output.txt 2>&1"
        )
    steps.append("echo $? > /grade/pytest_exit_code.txt")
    name = (
        "commit0-grade-"
        + hashlib.sha1(str(out_dir.resolve()).encode()).hexdigest()[:16]
    )
    _docker(["rm", "-fv", name])
    try:
        proc = _docker(
            [
                "run",
                "--rm",
                "--init",
                "--name",
                name,
                "--platform",
                PLATFORM,
                "--network",
                "none",
                *mounts,
                "--entrypoint",
                "bash",
                commit0_image(instance["name"]),
                "-c",
                "\n".join(steps),
            ],
            timeout=timeout,
        )
        error = None if proc.returncode == 0 else proc.stderr[-2000:]
    except subprocess.TimeoutExpired:
        _docker(["kill", name])
        error = f"grading timed out after {timeout}s"
    exit_code = out_dir / "pytest_exit_code.txt"
    if (
        error is None
        and test_timeout
        and exit_code.is_file()
        and exit_code.read_text().strip() == "3"
    ):
        error = f"grading deadline ({max(60, timeout - 120)}s) reached"
    watchdog = {}
    if (out_dir / "report.json").is_file():
        with (out_dir / "report.json").open(encoding="utf-8") as f:
            watchdog = json.load(f).get("watchdog") or {}
    return {
        "error": error,
        "test_timeout": test_timeout,
        # Tests the watchdog killed (counted as failed) and its pytest runs.
        "hung_tests": watchdog.get("hung_tests", []),
        "grade_attempts": watchdog.get("attempts", []),
    }


# pytest -v per-test result lines: "tests/test_x.py::test_y[p] PASSED  [ 10%]",
# for an inherited test "tests/t.py::T::test_y <- tests/base.py PASSED".
_VERBOSE_RESULT_RE = re.compile(
    r"^(\S+?::\S.*?)(?: <- \S+)? (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)


def _salvage_outcomes(output_path: Path) -> dict[str, str]:
    """Per-test outcomes from pytest's verbose output, for when pytest died
    before writing its JSON report (observed: agent code killing the pytest
    process mid-suite) -- otherwise every test that passed first is lost."""
    outcomes: dict[str, str] = {}
    if not output_path.is_file():
        return outcomes
    for line in output_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _VERBOSE_RESULT_RE.match(line.strip())
        if match:
            outcomes[match.group(1)] = match.group(2).lower()
    return outcomes


def score(
    report_path: Path, test_ids: list[str], held_out: dict[str, list[str]]
) -> dict:
    """Commit0's metric (passed / len(test_ids); a test that never ran counts
    as failed), also split into visible vs held-out test IDs. Without a JSON
    report, outcomes are salvaged from the verbose test output next to it
    (flagged `salvaged`)."""
    outcomes: dict[str, str] = {}
    salvaged = False
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for t in report.get("tests", []):
            if t.get("call"):
                outcomes[t["nodeid"]] = t["call"]["outcome"]
    else:
        outcomes = _salvage_outcomes(report_path.with_name("test_output.txt"))
        salvaged = bool(outcomes)
    counts = {"all": [0, 0], "visible": [0, 0], "held_out": [0, 0]}
    ran = sum(tid in outcomes for tid in test_ids)
    for tid in test_ids:
        passed = outcomes.get(tid) == "passed"
        for key in ("all", "held_out" if is_held_out(tid, held_out) else "visible"):
            counts[key][0] += passed
            counts[key][1] += 1
    # ran < all_total with a report present = collection/import errors.
    result = {"report_found": report_path.is_file(), "salvaged": salvaged, "ran": ran}
    for key, (passed, total) in counts.items():
        result[f"{key}_passed"] = passed
        result[f"{key}_total"] = total
        result[f"{key}_pass_rate"] = passed / total if total else 0.0
    return result


# ── Leakage ───────────────────────────────────────────────────────────────────

# Other installed copies of the library the agent could read: OpenHands'
# own tool env (it depends on e.g. jinja2, requests, click, pydantic), pip's
# vendored packages, the system Python. Confirmed in a pilot run: the agent
# `find`s /root/.local/share/uv/tools/openhands/.../jinja2 and copies it.
# The agent image scrubs or hides them (build_agent_image); find_leaks
# verifies nothing readable is left.
_PACKAGE_DIRS = {"site-packages", "dist-packages", "_vendor"}


def find_leaks(image: str, import_roots: list[str]) -> list[str]:
    """Copies of the library's top-level packages/modules (outside the
    workspace) that the agent's unprivileged uid can read in `image`: a
    package dir anywhere, or a module/package in a site-packages,
    dist-packages or pip _vendor dir."""
    names = " -o ".join(f"-name {r} -o -name {r}.py" for r in import_roots)
    script = (
        "{ find / -xdev \\( -path /proc -o -path /sys -o -path /testbed \\) -prune "
        f"-o \\( {names} \\) -print; find /testbed/.venv \\( {names} \\) -print; }} "
        "2>/dev/null | while read -r p; do "
        '[ -e "$p/__init__.py" ] && echo "P $p" || echo "F $p"; done'
    )
    uid = str(AGENT_FALLBACK_UID)
    proc = _docker(
        [
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            "--network",
            "none",
            "--entrypoint",
            "setpriv",
            image,
            f"--reuid={uid}",
            f"--regid={uid}",
            "--clear-groups",
            "sh",
            "-c",
            script,
        ]
    )
    leaks = []
    for line in proc.stdout.splitlines():
        kind, _, path = line.partition(" ")
        if kind == "P" or Path(path).parent.name in _PACKAGE_DIRS:
            leaks.append(path)
    return sorted(set(leaks))


_TOOL_ERROR_MARKERS = (
    "No such file",
    "cannot access",
    "Permission denied",
    "not found",
    "does not exist",
    "Access denied",
)


def _observation_texts(trajectory: Path):
    """Tool outputs (what the agent actually saw), one string per event."""
    for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "ObservationEvent":
            continue
        observation = event.get("observation") or {}
        content = observation.get("content") if isinstance(observation, dict) else None
        if isinstance(content, list):
            yield "\n".join(
                str(c.get("text", "")) for c in content if isinstance(c, dict)
            )
        elif content:
            yield str(content)


def _readable_in_image(image: str, paths: list[str]) -> list[str]:
    """The subset of `paths` that exist and are readable by the agent's
    unprivileged uid in `image`."""
    if not paths:
        return []
    uid = str(AGENT_FALLBACK_UID)
    proc = _docker(
        [
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            "--network",
            "none",
            "--entrypoint",
            "setpriv",
            image,
            f"--reuid={uid}",
            f"--regid={uid}",
            "--clear-groups",
            "sh",
            "-c",
            'for p in "$@"; do [ -r "$p" ] && echo "$p"; done',
            "_",
            *paths,
        ]
    )
    return sorted(set(proc.stdout.split()) & set(paths))


def audit_trajectory(
    trajectory: Path, import_roots: list[str], image: str | None = None
) -> list[str]:
    """Paths to other copies of the library (see find_leaks) that appear in
    the agent's tool *outputs* -- i.e. that it actually saw.

    Its own commands don't count, nor do error lines ("No such file",
    "Permission denied", ...): agents search for such paths (counted by
    search_attempts), which isn't a leak unless something was there. Only
    copies of the library's own packages count -- `find / -name "*marsh*"`
    listing unrelated files of OpenHands' env is not a leak. With `image`,
    a path counts only if it really exists and is readable by the agent
    there -- agents also write code naming imagined copies (observed:
    `_REAL_INIT = "/testbed/.venv/.../site-packages/jinja2/__init__.py"`)."""
    if not trajectory.is_file() or not import_roots:
        return []
    roots = "|".join(re.escape(r) for r in import_roots)
    pattern = re.compile(
        rf"/[\w./-]*(?:site-packages|dist-packages|_vendor)/(?:{roots})(?:/|\.py)[\w./-]*"
    )
    hits = set()
    for text in _observation_texts(trajectory):
        for line in text.splitlines():
            if not any(marker in line for marker in _TOOL_ERROR_MARKERS):
                hits.update(pattern.findall(line))
    hits = sorted(hits)[:200]
    if image:
        hits = _readable_in_image(image, hits)
    return hits[:50]


# Terminal commands that look for the reference implementation outside the
# workspace rather than build it. Pilot agents ran e.g. `git fsck
# --lost-found`, `find / -name "*.whl" | grep jinja`, `find / -name
# codegen.py`. Counted per run, reported as a metric (not a validity check:
# confinement + audit_trajectory are that). Library-specific where it can
# be: a first version also counted `ls .venv/.../site-packages` (listing
# dependencies) and babel's own `locale-data/root.dat` (matched "/root"),
# inflating babel's count to 67.
_GENERIC_SEARCH_RE = re.compile(
    r"\bfind\s+/(?!testbed)(?:\s|$)|\blocate\s|\bgit\s+(?:fsck|count-objects|reflog"
    r"|cat-file|rev-list\s+--all)|\bpip\d*\s+(?:download|install)\b|\.whl\b"
    r"|(?:^|[\s'\"=(:])/root(?:/|\s|$)"
)


def search_attempts(trajectory: Path, import_roots: list[str] = ()) -> list[str]:
    """The agent's terminal commands hunting for the reference: generic
    forensics (_GENERIC_SEARCH_RE), or looking up the library's own package
    (`pip show <pkg>`, `site-packages/<pkg>`, `find ... -name <pkg>`)."""
    hits = []
    if not trajectory.is_file():
        return hits
    library_re = None
    if import_roots:
        roots = "|".join(re.escape(r) for r in import_roots)
        library_re = re.compile(
            rf"\bpip\d*\s+show\s+(?:\S+\s+)*(?:{roots})\b"
            rf"|(?:site|dist)-packages/+(?:{roots})\b"
            rf"|\bfind\s+\S+.*-i?name\s+['\"]?\*?(?:{roots})\b"
        )
    for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        action = event.get("action")
        if event.get("tool_name") != "terminal" or not isinstance(action, dict):
            continue
        command = str(action.get("command") or "")
        if _GENERIC_SEARCH_RE.search(command) or (
            library_re and library_re.search(command)
        ):
            hits.append(command[:300])
    return hits


# OpenHands sends this when a reply has neither a tool call nor a message --
# observed when a reasoning model spends its whole max_output_tokens budget
# on thinking and gets cut off (~62k chars of reasoning, no action). Such
# steps produce nothing; if they hit one condition more (req_arch's longer
# prompt invites longer planning) the comparison is confounded, so count them.
_EMPTY_REPLY_NUDGE = "did not include a function call or a message"


def empty_replies(trajectory: Path) -> int:
    """How many of the agent's replies produced no action (see above)."""
    if not trajectory.is_file():
        return 0
    count = 0
    for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
        if _EMPTY_REPLY_NUDGE not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("source") == "user":
            count += 1
    return count


def task_tokens(text: str) -> int:
    """Size of a run's task text (the treatment: intent vs intent + req +
    c4), in tiktoken o200k_base tokens -- a model-agnostic proxy."""
    import tiktoken

    return len(tiktoken.get_encoding("o200k_base").encode(text))
