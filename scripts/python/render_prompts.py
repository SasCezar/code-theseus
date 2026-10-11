#!/usr/bin/env python3
"""Dump the exact system+user prompt for each (task, repo) pair to a .txt
file, for manual inspection without calling any model."""

import argparse
import json
from pathlib import Path

from reverse_se.generation.utils import DEFAULT_PROMPT_DIR, TASKS, build_full_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--prompt_dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default=None,
        help="Task to render (default: all)",
    )
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())

    for task in tasks_to_run:
        (args.output_dir / task).mkdir(parents=True, exist_ok=True)

    with open(args.input_file) as f:
        for line in f:
            repo = json.loads(line)
            repo_id = repo.get("repo_id")
            if not repo_id:
                print("Skipping object without repo_id")
                continue

            print(f"Saving prompts for {repo_id}")
            for task in tasks_to_run:
                full_prompt = build_full_prompt(args.prompt_dir, task, repo)
                (args.output_dir / task / f"{repo_id}.txt").write_text(full_prompt)
                print(f"  [{task}] saved")


if __name__ == "__main__":
    main()
