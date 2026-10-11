#!/usr/bin/env python3
"""
Local in-process generation with HuggingFace Transformers or llama.cpp.

One request at a time (no batching) — for SLURM/HPC nodes without vLLM, or
for GGUF quantised checkpoints via llama-cpp-python. For everything else
(including a self-hosted vLLM server, see ../../../compose.yaml), use
inference_litellm.py.

Output layout: llm_results/{model_slug}/{context_budget}/{task}/{repo_id}.json
A sibling {repo_id}.reasoning.txt is written whenever the raw output
contains a recognised reasoning block (see reverse_se.generation.utils.split_reasoning).

SLURM sharding: set --shard_index / --shard_count, or rely on
$SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT being set by the scheduler.

Usage:
    # BF16, all GPUs, all tasks:
    python -m reverse_se.generation.inference_transformers \\
        --dataset data/out/reverse-se_v0.6_p10-95_ctx95_dataset_sample_1000.jsonl \\
        --model mistralai/Mistral-Medium-3.5-128B \\
        --context_budget ctx95

    # 4-bit quantised:
    python -m reverse_se.generation.inference_transformers ... --quantize 4bit

    # GGUF (unsloth):
    python -m reverse_se.generation.inference_transformers \\
        --dataset ... \\
        --gguf_path /scratch/models/Mistral-Medium-3.5-128B-Q4_K_M.gguf \\
        --model_slug mistral-medium-3.5-q4km \\
        --context_budget ctx95
"""

import argparse
import time
from pathlib import Path

from reverse_se.generation.cli import (
    add_data_args,
    add_run_args,
    add_slurm_args,
    resolve_slurm_sharding,
)
from reverse_se.generation.runner import GenerateFn, run_sequential
from reverse_se.generation.utils import (
    load_dataset,
    pending_pairs,
    resolve_slug,
    split_reasoning,
)

# ── Transformers backend ──────────────────────────────────────────────────────


def load_transformers(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "auto": "auto"}
    dtype = dtype_map[args.dtype]

    bnb_cfg = None
    if args.quantize == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantize == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    attn_impl = "flash_attention_2" if args.flash_attn else "sdpa"

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(
        f"Loading model (dtype={args.dtype}, quant={args.quantize}, attn={attn_impl}) ...",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if bnb_cfg is None else None,
        quantization_config=bnb_cfg,
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def make_transformers_generate_fn(
    tokenizer, model, args: argparse.Namespace
) -> GenerateFn:
    import torch

    def generate(messages: list[dict]) -> tuple[str, dict]:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        prompt_len = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        new_ids = out_ids[0, prompt_len:]
        text, reasoning = split_reasoning(
            tokenizer.decode(new_ids, skip_special_tokens=True)
        )
        return text, {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(new_ids.shape[0]),
            "elapsed_seconds": round(elapsed, 2),
            "reasoning": reasoning,
        }

    return generate


# ── llama-cpp backend (GGUF) ──────────────────────────────────────────────────


def load_llamacpp(args: argparse.Namespace):
    try:
        from llama_cpp import Llama  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(
            "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
        ) from exc

    print(f"Loading GGUF model: {args.gguf_path}", flush=True)
    return Llama(
        model_path=str(args.gguf_path),
        n_ctx=args.max_context_len,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
    )


def make_llamacpp_generate_fn(llm, args: argparse.Namespace) -> GenerateFn:
    def generate(messages: list[dict]) -> tuple[str, dict]:
        t0 = time.perf_counter()
        resp = llm.create_chat_completion(
            messages=messages, max_tokens=args.max_new_tokens, temperature=0.0
        )
        elapsed = time.perf_counter() - t0
        text, reasoning = split_reasoning(resp["choices"][0]["message"]["content"])
        usage = resp.get("usage", {})
        return text, {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed_seconds": round(elapsed, 2),
            "reasoning": reasoning,
        }

    return generate


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SE artefacts with HF Transformers or llama-cpp"
    )

    add_data_args(parser)

    g_model = parser.add_argument_group("model")
    g_model.add_argument(
        "--model",
        default="mistralai/Mistral-Medium-3.5-128B",
        help="HF repo ID or local path (transformers backend)",
    )
    g_model.add_argument(
        "--model_slug",
        default=None,
        help="Override directory name in output path (default: derived from --model or gguf filename)",
    )
    g_model.add_argument(
        "--gguf_path",
        type=Path,
        default=None,
        help="Path to GGUF file — activates the llama-cpp backend",
    )

    g_hf = parser.add_argument_group("transformers options")
    g_hf.add_argument(
        "--quantize",
        choices=["4bit", "8bit", "none"],
        default="none",
        help="BitsAndBytes quantisation (transformers only)",
    )
    g_hf.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "auto"],
        default="bf16",
        help="Model dtype. bf16 recommended for H200/H100; fp16 for A100",
    )
    g_hf.add_argument("--flash_attn", action="store_true", default=True)
    g_hf.add_argument("--no_flash_attn", dest="flash_attn", action="store_false")

    g_cpp = parser.add_argument_group("llama-cpp options")
    g_cpp.add_argument(
        "--n_gpu_layers", type=int, default=-1, help="-1 offloads all layers to GPU"
    )
    g_cpp.add_argument("--n_threads", type=int, default=8)
    g_cpp.add_argument(
        "--max_context_len",
        type=int,
        default=32768,
        help="KV-cache context window for llama-cpp",
    )

    add_run_args(parser)
    add_slurm_args(parser)

    args = parser.parse_args()
    resolve_slurm_sharding(args)

    slug = resolve_slug(args.model, args.model_slug, args.gguf_path)

    print(f"Loading dataset from {args.dataset} ...", flush=True)
    dataset = load_dataset(args.dataset)
    print(f"  {len(dataset)} repos loaded", flush=True)

    if args.gguf_path:
        backend = "llamacpp"
        llm_cpp = load_llamacpp(args)
        generate_fn = make_llamacpp_generate_fn(llm_cpp, args)
        model_name = str(args.gguf_path)
    else:
        backend = "transformers"
        tokenizer, model = load_transformers(args)
        generate_fn = make_transformers_generate_fn(tokenizer, model, args)
        model_name = args.model

    print(f"Backend: {backend} | slug: {slug}", flush=True)

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

    done, errors = run_sequential(
        work,
        dataset,
        args.prompt_dir,
        model_name,
        slug,
        args.context_budget,
        generate_fn,
    )

    print(f"\nDone. {done} generated, {errors} errors.")
    print(f"Output root: {args.output_dir / slug / args.context_budget}")


if __name__ == "__main__":
    main()
