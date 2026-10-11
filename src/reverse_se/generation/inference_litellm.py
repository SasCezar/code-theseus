#!/usr/bin/env python3
"""
API-based generation via LiteLLM — one HTTP request per (task, repo) pair
against any LiteLLM-supported model: a self-hosted vLLM server (see
../../../compose.yaml), OpenAI, DeepSeek, Anthropic, etc. This is the
standard way to drive the 2x-GPU vLLM containers.

Model strings follow LiteLLM's `<provider>/<model>` convention, e.g.:
  hosted_vllm/mistral-medium-3.5-128b   (self-hosted, needs --api_base)
  openai/gpt-4.1
  deepseek/deepseek-reasoner

Output layout: llm_results/{model_slug}/{context_budget}/{task}/{repo_id}.json
A sibling {repo_id}.reasoning.txt is written whenever the response carries a
reasoning/thinking trace (DeepSeek-R1, gpt-oss, extended thinking, ...).

Usage:
    # Self-hosted vLLM (docker compose --profile gpt-oss up), port 8002:
    python -m reverse_se.generation.inference_litellm \\
        --dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000.jsonl \\
        --model hosted_vllm/gpt-oss-120b \\
        --api_base http://localhost:8002/v1 \\
        --context_budget ctx95

    # Hosted provider, with reasoning effort:
    python -m reverse_se.generation.inference_litellm \\
        --dataset ... --model openai/o3 --reasoning_effort high

    # Concurrent requests against a self-hosted vLLM server, to actually use
    # its continuous batching instead of one request at a time:
    python -m reverse_se.generation.inference_litellm \\
        --dataset ... --model hosted_vllm/gpt-oss-120b \\
        --api_base http://localhost:8002/v1 --concurrency 8
"""

import argparse

from reverse_se.generation.cli import (
    add_data_args,
    add_run_args,
    add_slurm_args,
    resolve_slurm_sharding,
)
from reverse_se.generation.runner import GenerateFn, run_concurrent, run_sequential
from reverse_se.generation.utils import (
    extract_reasoning,
    load_dataset,
    pending_pairs,
    resolve_slug,
)


def make_generate_fn(args: argparse.Namespace) -> GenerateFn:
    import litellm

    # OpenAI's reasoning models reject max_tokens; LiteLLM only translates
    # it for models in its own list (not e.g. gpt-6-sol), and OpenAI accepts
    # max_completion_tokens for every chat model.
    tokens_key = (
        "max_completion_tokens" if args.model.startswith("openai/") else "max_tokens"
    )
    completion_kwargs: dict = {
        "model": args.model,
        tokens_key: args.max_new_tokens,
        "timeout": args.request_timeout,
    }
    if not args.default_temperature:
        completion_kwargs["temperature"] = 0.0
    if args.api_base:
        completion_kwargs["api_base"] = args.api_base
    if args.api_key:
        completion_kwargs["api_key"] = args.api_key
    if args.reasoning_effort:
        completion_kwargs["reasoning_effort"] = args.reasoning_effort

    def generate(messages: list[dict]) -> tuple[str, dict]:
        response = litellm.completion(messages=messages, **completion_kwargs)
        choice = response.choices[0]
        usage = response.usage
        return choice.message.content or "", {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "finish_reason": choice.finish_reason,
            "reasoning": extract_reasoning(choice.message),
        }

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SE artefacts via LiteLLM")

    add_data_args(parser)

    g_model = parser.add_argument_group("model")
    g_model.add_argument(
        "--model",
        required=True,
        help="LiteLLM model string, e.g. hosted_vllm/mistral-medium-3.5-128b or openai/gpt-4.1",
    )
    g_model.add_argument("--model_slug", default=None)
    g_model.add_argument(
        "--api_base",
        default=None,
        help="Base URL for a self-hosted OpenAI-compatible server (e.g. a vLLM docker container)",
    )
    g_model.add_argument(
        "--api_key",
        default=None,
        help="API key (default: provider env var / EMPTY for vLLM)",
    )
    g_model.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Reasoning effort, for models that support it",
    )
    g_model.add_argument(
        "--default_temperature",
        action="store_true",
        help="Send no temperature (the model's default) instead of 0.0 -- for "
        "models that only accept their default, e.g. OpenAI reasoning models "
        "LiteLLM doesn't know yet",
    )
    g_model.add_argument(
        "--request_timeout",
        type=float,
        default=1800.0,
        help=(
            "Per-request timeout in seconds passed to litellm.completion(). "
            "LiteLLM's own default is 600s, which a dense model under load "
            "can exceed well before hitting --max_new_tokens."
        ),
    )
    g_model.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of requests to issue at once. >1 lets a server with "
            "continuous batching (vLLM) work on several documents "
            "simultaneously instead of idling most of its batch capacity "
            "waiting on one sequence at a time. Has no effect against "
            "providers that rate-limit hard on concurrency."
        ),
    )

    add_run_args(parser)
    add_slurm_args(parser)

    args = parser.parse_args()
    resolve_slurm_sharding(args)

    from dotenv import load_dotenv

    load_dotenv()

    slug = resolve_slug(args.model, args.model_slug)

    print(f"Loading dataset from {args.dataset} ...", flush=True)
    dataset = load_dataset(args.dataset)
    print(f"  {len(dataset)} repos loaded", flush=True)

    work = pending_pairs(
        dataset,
        args.tasks,
        args.output_dir,
        slug,
        args.context_budget,
        args.limit,
        args.skip_existing,
        args.shard_index,
        args.shard_count,
    )
    print(f"  {len(work)} (task, repo) pairs to generate", flush=True)

    generate_fn = make_generate_fn(args)
    if args.concurrency > 1:
        done, errors = run_concurrent(
            work,
            dataset,
            args.prompt_dir,
            args.model,
            slug,
            args.context_budget,
            generate_fn,
            max_workers=args.concurrency,
        )
    else:
        done, errors = run_sequential(
            work,
            dataset,
            args.prompt_dir,
            args.model,
            slug,
            args.context_budget,
            generate_fn,
        )

    print(f"\nDone. {done} generated, {errors} errors.")
    print(f"Output root: {args.output_dir / slug / args.context_budget}")


if __name__ == "__main__":
    main()
