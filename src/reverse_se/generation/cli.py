"""Shared argparse group builders for the generation CLIs, so
inference_litellm.py / inference_transformers.py define the data/run/SLURM
flags identically instead of each re-declaring them."""

import argparse
import os
from pathlib import Path

from reverse_se.generation.utils import DEFAULT_PROMPT_DIR, TASKS


def add_data_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("data / prompts")
    g.add_argument("--dataset", required=True, type=Path, help="Path to JSONL dataset")
    g.add_argument("--prompt_dir", type=Path, default=DEFAULT_PROMPT_DIR)
    g.add_argument("--output_dir", type=Path, default=Path("data/out/llm_results"))


def add_run_args(
    parser: argparse.ArgumentParser, *, max_new_tokens_default: int = 16384
) -> None:
    g = parser.add_argument_group("run")
    g.add_argument(
        "--tasks", nargs="+", choices=list(TASKS.keys()), default=list(TASKS.keys())
    )
    g.add_argument(
        "--context_budget",
        default="ctx95",
        help="Label written into the output path, e.g. ctx95",
    )
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--skip_existing", action="store_true", default=True)
    g.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    g.add_argument("--max_new_tokens", type=int, default=max_new_tokens_default)


def add_slurm_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("SLURM sharding")
    g.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="0-based shard index (auto-read from $SLURM_ARRAY_TASK_ID)",
    )
    g.add_argument(
        "--shard_count",
        type=int,
        default=None,
        help="Total shards (auto-read from $SLURM_ARRAY_TASK_COUNT)",
    )


def resolve_slurm_sharding(args: argparse.Namespace) -> None:
    """Fill --shard_index/--shard_count from SLURM env vars when not passed explicitly."""
    if args.shard_index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        args.shard_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.shard_count is None and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        args.shard_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
