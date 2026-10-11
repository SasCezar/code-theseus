import sys
import traceback
from pathlib import Path

import pandas as pd
import tiktoken
from pandas import DataFrame
from tiktoken import Encoding
from tqdm import tqdm

from reverse_se.graph.arcan import ArcanCSV

encoding: Encoding = tiktoken.encoding_for_model("gpt-5-mini")


def graph_to_token_counts(
    project: str,
    graph_path: Path,
    package_only: bool,
) -> dict:
    """
    Builds the same text payload as `simple_architecture` (nodes/edges/package-files),
    then computes token counts for:
      - total payload
      - nodes-only
      - edges-only
      - package-files-only
    """
    nodes, edges, package_files = ArcanCSV(
        name=project,
        path=graph_path,
        package_only=package_only,
    ).graph

    str_nodes: str = nodes.to_csv(
        index=False,
        float_format="%.2e",
        sep=" ",
    )

    str_edges: str = edges.to_csv(
        index=False,
        sep=" ",
        columns=["fromId", "toId", "Weight"],
    )

    str_package_files = ""
    if package_files is not None:
        df: DataFrame = package_files.copy()
        df["files"] = df["files"].apply(
            lambda x: ";".join(x) if isinstance(x, list) else ""
        )
        str_package_files: str = df.to_csv(index=False, sep=" ")

    payload = f"{str_nodes}\n{str_edges}\n{str_package_files}"
    n_tokens: int = len(encoding.encode(payload))
    n_nodes_tokens: int = len(encoding.encode(str_nodes))
    n_edges_tokens: int = len(encoding.encode(str_edges))
    n_package_files_tokens: int = (
        len(encoding.encode(str_package_files)) if str_package_files else 0
    )

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "package_only": package_only,
        "n_tokens": n_tokens,
        "n_nodes_tokens": n_nodes_tokens,
        "n_edges_tokens": n_edges_tokens,
        "n_package_files_tokens": n_package_files_tokens,
    }


def compute_repo_size(repos_path: Path, out_path: Path):
    project_dirs: list[Path] = [p for p in repos_path.iterdir() if p.is_dir()]

    results: list[dict] = []
    success = 0
    failed = 0

    with tqdm(project_dirs, desc="Processing repos", unit="repo") as pbar:
        for dirpath in pbar:
            project = dirpath.name
            repo_url = "https://github.com/" + project.replace("|", "/")

            try:
                for package_only in (True, False):
                    row: dict = graph_to_token_counts(
                        project=project,
                        graph_path=dirpath,
                        package_only=package_only,
                    )
                    row["repo"] = repo_url
                    results.append(row)

                success += 1

            except Exception as e:
                print(traceback.format_exc())
                failed += 1

            pbar.set_postfix(success=success, failed=failed)

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Done. Success={success}, Failed={failed}")


if __name__ == "__main__":
    compute_repo_size(Path(sys.argv[1]), Path(sys.argv[2]))
