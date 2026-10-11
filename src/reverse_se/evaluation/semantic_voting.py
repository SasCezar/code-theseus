#!/usr/bin/env python3
"""
Semantic Voting scorer.

Computes pairwise cosine similarities between corresponding artifacts
across model configurations and writes a fully denormalized CSV.

Usage:
    python -m reverse_se.evaluation.semantic_voting <root_dir> [--output scores.csv] [--glob "*.md"]

Expected layout:
    root/
      model1/
        repo_a.md
        repo_b.md
      model2/
        repo_a.md
        repo_b.md

The script only scores artifacts whose filename appears in ALL model dirs.
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL_ID = "microsoft/harrier-oss-v1-0.6b"
MAX_TOKENS = 32000
BATCH_SIZE = 8


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_embedder(model_id: str):
    print(f"Loading embedding model: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    print(f"  device: {device}", flush=True)
    return tokenizer, model, device


def embed(texts: list[str], tokenizer, model, device) -> np.ndarray:
    # On CUDA use batching; on MPS/CPU encode one at a time to avoid
    # padding-inflated allocations (e.g. 8 × 16 k tokens = OOM on MPS).
    batch_size = BATCH_SIZE if device == "cuda" else 1
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        vecs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        vecs = vecs.cpu().float().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-9)
        all_vecs.append(vecs)
        print(
            f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}",
            end="\r",
            flush=True,
        )
    print()
    return np.vstack(all_vecs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Voting pairwise scorer")
    parser.add_argument(
        "root", type=Path, help="Root dir containing one subdir per model config"
    )
    parser.add_argument("--output", type=Path, default=Path("semantic_voting.csv"))
    parser.add_argument(
        "--glob", default="*.md", help="Glob for artifact files (default: *.md)"
    )
    parser.add_argument(
        "--max-artifacts",
        type=int,
        default=None,
        help="Limit number of artifacts (debug)",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    # Discover model dirs (configs)
    configs = sorted(d for d in root.iterdir() if d.is_dir())
    if len(configs) < 2:
        sys.exit(f"Need >= 2 model dirs under {root}, found {len(configs)}")
    print(f"Configurations ({len(configs)}): {[c.name for c in configs]}")

    # Intersect artifact file names across all configs
    artifact_sets = [
        {f.name for f in cfg.glob(args.glob) if f.is_file()} for cfg in configs
    ]
    shared = sorted(artifact_sets[0].intersection(*artifact_sets[1:]))
    if not shared:
        sys.exit("No shared artifacts found across all configurations.")
    print(f"Shared artifacts: {len(shared)}")

    # Build flat list of (config_name, artifact_name, text) in a stable order
    # so we can slice the embedding matrix by index
    if args.max_artifacts is not None:
        shared = shared[: args.max_artifacts]

    keys: list[tuple[str, str]] = []
    texts: list[str] = []
    for cfg in configs:
        for art in shared:
            path = cfg / art
            keys.append((cfg.name, art))
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    print(f"Total texts to embed: {len(texts)}")
    tokenizer, model, device = load_embedder(EMBED_MODEL_ID)

    print("Embedding...")
    vecs_array = embed(texts, tokenizer, model, device)
    vecs: dict[tuple[str, str], np.ndarray] = {
        k: vecs_array[i] for i, k in enumerate(keys)
    }

    # Restrict to only configs/artifacts that were actually embedded (debug slice may truncate)
    embedded_configs = sorted({cfg for cfg, _ in vecs})
    embedded_arts = sorted(
        {art for _, art in vecs if all((cfg, art) in vecs for cfg in embedded_configs)}
    )

    # Compute all ordered pairs (config_a, config_b) with config_a < config_b (undirected)
    # Cosine sim is symmetric, but store both directions so R can filter by config easily
    print("Computing pairwise similarities...")
    rows: list[dict] = []
    for ca, cb in itertools.combinations(embedded_configs, 2):
        for art in embedded_arts:
            sim = float(np.dot(vecs[(ca, art)], vecs[(cb, art)]))
            row = {
                "config_a": ca,
                "config_b": cb,
                "artifact": art,
                "cosine_similarity": sim,
            }
            rows.append(row)
            # Mirror row so every config appears in config_a for easy per-config filtering
            rows.append(
                {
                    "config_a": cb,
                    "config_b": ca,
                    "artifact": art,
                    "cosine_similarity": sim,
                }
            )

    # Write CSV
    fieldnames = ["config_a", "config_b", "artifact", "cosine_similarity"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_pairs = len(list(itertools.combinations(configs, 2))) * len(shared)
    print(
        f"\nWrote {len(rows)} rows ({total_pairs} unique pairs × 2 directions) → {args.output}"
    )

    # Quick SV(c) summary  — Agreement(c,x) averaged over x, then over c' != c
    print(f"\n{'Config':<45} {'SV(c)':>8}")
    print("-" * 55)
    for cfg in configs:
        sims = [r["cosine_similarity"] for r in rows if r["config_a"] == cfg.name]
        sv = float(np.mean(sims)) if sims else float("nan")
        print(f"{cfg.name:<45} {sv:>8.4f}")


if __name__ == "__main__":
    main()
