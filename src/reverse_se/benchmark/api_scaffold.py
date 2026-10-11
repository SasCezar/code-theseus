"""Scaffold level "api": prune Commit0's stubbed starter repo down to the
public interface the test suite actually uses.

Commit0's starter keeps every module, class and signature -- i.e. the whole
architecture -- which leaves an architecture doc little to add. This keeps
only what tests reference (so held-out tests still import and are gradable
on behavior, not on guessing names) and deletes everything else: internal
modules, helper functions/classes, and module-level imports that would
reveal them. The agent has to design the internals itself.

References are collected statically from *every* test file (visible and
held-out, conftest and test helpers included):
  - `import a.b [as x]`, `from a.b import c [as d]`, `from a import *`;
  - attribute chains on those bindings (`x.Name.method` -> a.b.Name);
  - dotted string constants naming project objects (mock.patch targets,
    importorskip, ...).
Each reference resolves to (module, top-level name), following re-exports
(`from .x import Name` in a package __init__) to the defining module. A kept
class keeps its full (stubbed) body: which methods tests call on instances
isn't statically knowable, and a class's method list is its public API.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


def module_map(repo: Path, impl_files: list[str]) -> dict[str, str]:
    """dotted module name -> repo-relative .py path, for implementation files.

    The import root of a file is the first ancestor directory without an
    __init__.py (handles both `pkg/` and `src/pkg/` layouts)."""
    modules = {}
    for rel in impl_files:
        path = repo / rel
        parts = [path.stem] if path.name != "__init__.py" else []
        parent = path.parent
        while (parent / "__init__.py").is_file() and parent != repo:
            parts.insert(0, parent.name)
            parent = parent.parent
        if parts:
            modules[".".join(parts)] = rel
    return modules


class _RefCollector(ast.NodeVisitor):
    def __init__(self, roots: set[str]):
        self.roots = roots
        self.bindings: dict[str, str] = {}
        self.refs: set[str] = set()
        # Dotted strings (mock.patch targets, ...) -- noisier: a string like
        # "pkg.json" may be a file name, so only trusted once resolved.
        self.string_refs: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in self.roots:
                self.refs.add(alias.name)
                if alias.asname:
                    self.bindings[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    self.bindings[root] = root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module or node.module.split(".")[0] not in self.roots:
            return
        for alias in node.names:
            ref = f"{node.module}.{alias.name}"
            self.refs.add(ref)
            if alias.name != "*":
                self.bindings[alias.asname or alias.name] = ref

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain, cur = [], node
        while isinstance(cur, ast.Attribute):
            chain.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id in self.bindings:
            self.refs.add(".".join([self.bindings[cur.id], *chain]))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if (
            isinstance(value, str)
            and "." in value
            and value.split(".")[0] in self.roots
            and all(part.isidentifier() for part in value.split("."))
        ):
            self.string_refs.add(value)


def collect_test_refs(
    repo: Path, test_files: list[str], roots: set[str]
) -> tuple[set[str], set[str]]:
    """(import/attribute refs, dotted-string refs) over all test files."""
    refs: set[str] = set()
    string_refs: set[str] = set()
    for rel in test_files:
        try:
            tree = ast.parse((repo / rel).read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        collector = _RefCollector(roots)
        # Two passes: bindings first (imports can come after use in functions).
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                collector.visit(node)
        collector.visit(tree)
        refs |= collector.refs
        string_refs |= collector.string_refs
    return refs, string_refs


def _parse(path: Path) -> ast.Module | None:
    """Some Commit0 starter files don't parse (e.g. an `except:` whose only
    statement was a private helper Commit0 deleted)."""
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return None


def _defined_names(stmt: ast.stmt) -> set[str]:
    """Top-level names a statement binds (recursing into if/try blocks)."""
    if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return {
            n.id for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)
        }
    if isinstance(stmt, ast.AnnAssign | ast.AugAssign) and isinstance(
        stmt.target, ast.Name
    ):
        return {stmt.target.id}
    if isinstance(stmt, ast.Import | ast.ImportFrom):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    if isinstance(stmt, ast.If | ast.Try | ast.With):
        names = set()
        for block in ("body", "orelse", "finalbody"):
            for sub in getattr(stmt, block, []):
                names |= _defined_names(sub)
        for handler in getattr(stmt, "handlers", []):
            for sub in handler.body:
                names |= _defined_names(sub)
        return names
    return set()


def _resolve_from(module: str, node: ast.ImportFrom, is_package: bool) -> str | None:
    if not node.level:
        return node.module
    base = module.split(".") if is_package else module.split(".")[:-1]
    if node.level > 1:
        base = base[: -(node.level - 1)] or None
    if base is None:
        return None
    return ".".join(base + ([node.module] if node.module else []))


@dataclass
class ApiSurface:
    keep: dict[str, set[str]] = field(default_factory=dict)  # module -> names
    keep_all: set[str] = field(default_factory=set)  # modules star-imported
    unresolved: set[str] = field(default_factory=set)


def resolve_surface(repo: Path, modules: dict[str, str], refs: set[str]) -> ApiSurface:
    surface = ApiSurface()
    trees = {m: _parse(repo / rel) or ast.Module(body=[]) for m, rel in modules.items()}

    def keep_module(module: str) -> None:
        parts = module.split(".")
        for i in range(1, len(parts) + 1):  # ancestor packages too
            if ".".join(parts[:i]) in modules:
                surface.keep.setdefault(".".join(parts[:i]), set())

    def keep_name(module: str, name: str, seen: set) -> bool:
        """Keep `name` of `module` (following re-exports); False if not found."""
        if (module, name) in seen:
            return name in surface.keep.get(module, ())
        seen.add((module, name))
        keep_module(module)
        if f"{module}.{name}" in modules:  # a submodule, not an object
            keep_module(f"{module}.{name}")
            # keeps a `from . import name` in the package, if it has one, so
            # `import pkg; pkg.name.X` still works
            surface.keep[module].add(name)
            return True
        is_package = modules[module].endswith("__init__.py")
        found = False
        for stmt in trees[module].body:
            if isinstance(stmt, ast.ImportFrom):
                target = _resolve_from(module, stmt, is_package)
                for alias in stmt.names:
                    if alias.name == "*":
                        if target in modules and keep_name(target, name, seen):
                            found = True
                    elif (alias.asname or alias.name) == name:
                        found = True
                        if target in modules:
                            keep_name(target, alias.name, seen)
                        elif target and f"{target}.{alias.name}" in modules:
                            keep_module(f"{target}.{alias.name}")
            elif name in _defined_names(stmt):
                found = True
        if found:
            surface.keep[module].add(name)
        return found

    seen: set = set()
    for ref in sorted(refs):
        parts = ref.split(".")
        for i in range(len(parts), 0, -1):
            module = ".".join(parts[:i])
            if module in modules:
                break
        else:
            surface.unresolved.add(ref)
            continue
        rest = parts[i:]
        if not rest:
            keep_module(module)
        elif rest[0] == "*":
            keep_module(module)
            surface.keep_all.add(module)
        elif not keep_name(module, rest[0], seen):
            surface.unresolved.add(ref)

    # Closure: keep module-level names that kept code itself references
    # (base classes, decorators, TypeVars, constants, ...) -- otherwise the
    # kept interface can't even be imported. Follows imports across modules.
    changed = True
    while changed:
        changed = False
        for module, names in list(surface.keep.items()):
            for stmt in trees[module].body:
                if not (_defined_names(stmt) & names) or isinstance(
                    stmt, ast.Import | ast.ImportFrom
                ):
                    continue
                for node in ast.walk(stmt):
                    if (
                        isinstance(node, ast.Name)
                        and node.id not in names
                        and (module, node.id) not in seen
                        and any(
                            node.id in _defined_names(s) for s in trees[module].body
                        )
                    ):
                        changed |= keep_name(module, node.id, seen)
    return surface


def _prune_module(
    tree: ast.Module,
    module: str,
    keep: set[str],
    keep_all: bool,
    kept: set[str],
    roots: set[str],
    is_package: bool,
) -> ast.Module:
    body = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            target = _resolve_from(module, stmt, is_package) or ""
            if stmt.module == "__future__" or target.split(".")[0] not in roots:
                body.append(stmt)  # stdlib / third-party / __future__
                continue
            # Project import: keep only re-exports of kept names (and star
            # re-exports of kept modules, through which kept names resolve).
            names = [
                a
                for a in stmt.names
                if (a.asname or a.name) in keep or (a.name == "*" and target in kept)
            ]
            if names:
                stmt.names = names
                body.append(stmt)
        elif isinstance(stmt, ast.Import):
            names = [a for a in stmt.names if a.name.split(".")[0] not in roots]
            if names:
                stmt.names = names
                body.append(stmt)
        elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if stmt.name in keep or (keep_all and not stmt.name.startswith("_")):
                body.append(stmt)
        elif isinstance(stmt, ast.Assign | ast.AnnAssign):
            names = _defined_names(stmt)
            if names & keep or (keep_all and any(not n.startswith("_") for n in names)):
                body.append(stmt)
            elif names == {"__all__"} and isinstance(stmt.value, ast.List | ast.Tuple):
                stmt.value.elts = [
                    e
                    for e in stmt.value.elts
                    if isinstance(e, ast.Constant) and (e.value in keep or keep_all)
                ]
                body.append(stmt)
            elif all(n.startswith("__") and n.endswith("__") for n in names):
                body.append(stmt)  # __version__, __author__, ...
        elif isinstance(stmt, ast.If | ast.Try | ast.With):
            if _defined_names(stmt) & keep:
                body.append(stmt)
    tree.body = body
    return tree


def build_api_scaffold(
    repo: Path, impl_files: list[str], test_files: list[str]
) -> dict:
    """Prune repo in place; returns stats for the scaffold manifest."""
    modules = module_map(repo, impl_files)
    roots = {m.split(".")[0] for m in modules}
    refs, string_refs = collect_test_refs(repo, test_files, roots)
    surface = resolve_surface(repo, modules, refs | string_refs)

    removed = []
    for module, rel in sorted(modules.items()):
        path = repo / rel
        if module not in surface.keep:
            path.unlink()
            removed.append(rel)
            continue
        tree = _parse(path)
        if tree is None:  # left as Commit0 shipped it
            continue
        tree = _prune_module(
            tree,
            module,
            surface.keep[module],
            module in surface.keep_all,
            set(surface.keep),
            roots,
            rel.endswith("__init__.py"),
        )
        path.write_text(
            ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8"
        )
    return {
        "api_modules_total": len(modules),
        "api_modules_kept": len(surface.keep),
        "api_names_kept": sum(len(v) for v in surface.keep.values()),
        "api_removed_files": removed,
        "api_unresolved_refs": sorted(surface.unresolved),
    }


def build_contract_scaffold(
    repo: Path, impl_files: list[str], test_files: list[str]
) -> tuple[list[str], dict]:
    """Scaffold level "contract": delete every implementation .py file and
    return the names-only API contract -- each dotted name the tests import
    or reference, at the path they use it (not where the reference defines
    it), with no signatures. Import/attribute references are listed even
    when the Commit0 starter lacks them (tests still need them); dotted
    strings only when they resolve to a real module/name."""
    modules = module_map(repo, impl_files)
    roots = {m.split(".")[0] for m in modules}
    refs, string_refs = collect_test_refs(repo, test_files, roots)
    surface = resolve_surface(repo, modules, refs | string_refs)

    contract = set()
    for ref in refs | string_refs:
        if ref not in refs and ref in surface.unresolved:
            continue
        parts = ref.split(".")
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in modules:
                break
        else:
            if ref in refs:  # a module the starter lacks entirely
                contract.add(ref)
            continue
        entry = ".".join(parts[: i + 1]) if i < len(parts) else ".".join(parts[:i])
        if not entry.endswith(".*"):
            contract.add(entry)

    for rel in impl_files:
        (repo / rel).unlink(missing_ok=True)
    # Drop package dirs left empty (their names would still hint at layout).
    for rel in sorted({str(Path(r).parent) for r in impl_files}, key=len, reverse=True):
        d = repo / rel
        while d != repo and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent
    return sorted(contract), {
        "contract_names": len(contract),
        "contract_removed_files": len(impl_files),
    }
