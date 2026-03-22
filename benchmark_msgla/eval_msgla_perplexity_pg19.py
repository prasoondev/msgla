"""
eval_perplexity_pg19.py — Perplexity and forgetting-curve evaluation for MS-GLA checkpoints.

Evaluates on PG19 (fla-hub/pg19) using:
  - Full-document perplexity (bits-per-byte and NLL)
  - Forgetting curves: per-position NLL averaged across documents,
    showing how loss evolves as context length increases.

Examples:
    # Evaluate a single DCP checkpoint
    python eval_perplexity_pg19.py \\
        --checkpoint_path /path/to/exp/msgla-340M --step 10800 \\
        --model_ref /path/to/exp/msgla-340M

    # Evaluate multiple steps to compare forgetting across training
    python eval_perplexity_pg19.py \\
        --checkpoint_path /path/to/exp/msgla-340M \\
        --steps 5400,10800,21600 \\
        --model_ref /path/to/exp/msgla-340M \\
        --plot_output forgetting_curve.png

    # Quick smoke-test on 5 docs, 512-token chunks
    python eval_perplexity_pg19.py \\
        --checkpoint_path /path/to/checkpoint/step-10800 \\
        --model_ref /path/to/exp/msgla-340M \\
        --max_docs 5 --stride 512
"""

import argparse
import io
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.serialization
from datasets import load_dataset
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoTokenizer

# ── make custom_models importable ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLAME_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "flame")
if FLAME_ROOT not in sys.path:
    sys.path.insert(0, FLAME_ROOT)

from custom_models.ms_gla import MSGLAConfig, MSGLAForCausalLM  # noqa: E402


# ================================================================
# Checkpoint loading  (identical helpers to eval_dcp.py)
# ================================================================

def resolve_checkpoint_path(checkpoint_path: str, step: Optional[int]) -> str:
    base = Path(checkpoint_path)

    if base.is_file():
        if base.suffix != ".pt":
            raise ValueError(f"Expected a .pt checkpoint file, got: {base}")
        if step is not None:
            raise ValueError("Do not pass --step when --checkpoint_path already points to a .pt file.")
        return str(base)

    if step is not None:
        for candidate in [base / "checkpoint" / f"step-{step}", base / f"step-{step}"]:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"Could not find step-{step} under {base}."
        )

    if (base / ".metadata").exists() and any(base.glob("*.distcp")):
        return str(base)

    latest = None
    checkpoint_root = base / "checkpoint"
    if checkpoint_root.exists():
        for p in checkpoint_root.glob("step-*"):
            try:
                step_num = int(p.name.split("step-")[-1])
            except ValueError:
                continue
            if latest is None or step_num > latest[0]:
                latest = (step_num, p)

    if latest is not None:
        print(f"No --step provided. Using latest checkpoint: {latest[1]}")
        return str(latest[1])

    raise FileNotFoundError(
        f"Could not infer DCP checkpoint directory from {base}. "
        "Pass either a step directory directly or set --step."
    )


def load_state_from_checkpoint(checkpoint_path: str, tmp_dir: Optional[str]) -> dict:
    if checkpoint_path.endswith(".pt"):
        print(f"Loading converted checkpoint from {checkpoint_path} ...")
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        return torch.load(checkpoint_path, map_location="cpu")

    print(f"Loading DCP checkpoint from {checkpoint_path} ...")
    with tempfile.TemporaryDirectory(dir=tmp_dir) as workdir:
        checkpoint_pt = os.path.join(workdir, "checkpoint.pt")
        dcp_to_torch_save(checkpoint_path, checkpoint_pt)
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        return torch.load(checkpoint_pt, map_location="cpu")


def load_model_and_tokenizer(
    model_ref: str,
    checkpoint_path: str,
    device: str,
    tmp_dir: Optional[str],
):
    print(f"Loading tokenizer from {model_ref} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model config from {model_ref} ...")
    config = MSGLAConfig.from_pretrained(model_ref)
    model = MSGLAForCausalLM(config)

    state = load_state_from_checkpoint(checkpoint_path, tmp_dir)
    if "model" not in state:
        raise KeyError("Checkpoint does not contain 'model' in the top-level state dict.")
    model.load_state_dict(state["model"])

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {n/1e6:.1f}M params on {device}.")
    return model, tokenizer


# ================================================================
# Perplexity computation
# ================================================================

@torch.inference_mode()
def compute_doc_perplexity(
    model,
    input_ids: torch.Tensor,   # (1, T)
    stride: int,
    device: str,
    max_ctx: int,
) -> dict:
    """
    Sliding-window perplexity (Jelinek–Mercer style) as in the GPT-2 paper.
    Returns per-token NLL list (aligned to token positions) and aggregate stats.

    stride < max_ctx  →  overlapping windows; only the *new* tokens in each
    window contribute to the loss (avoids double-counting).
    stride == max_ctx →  non-overlapping chunks (faster, slightly higher PPL).
    """
    T = input_ids.shape[-1]
    nlls: list[float] = []
    prev_end = 0

    for begin in range(0, T, stride):
        end = min(begin + max_ctx, T)
        chunk = input_ids[:, begin:end].to(device)
        target_len = end - max(begin, prev_end)   # tokens not seen in last window

        if target_len <= 0:
            prev_end = end
            continue

        outputs = model(chunk, use_cache=False)
        logits = outputs.logits                    # (1, chunk_len, vocab)

        # Shift: predict token i+1 from token i
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )  # (chunk_len - 1,)

        # Keep only the *new* tokens (last target_len positions)
        new_token_losses = token_loss[-(target_len):].tolist()
        nlls.extend(new_token_losses)
        prev_end = end
        if end == T:
            break

    if not nlls:
        return {}

    mean_nll = sum(nlls) / len(nlls)
    ppl = math.exp(mean_nll)
    # Bits-per-byte: NLL (nats) / log(2) / avg_bytes_per_token
    # We approximate bytes_per_token = 1 for BPE; caller can override.
    bpb = mean_nll / math.log(2)

    return {
        "per_token_nll": nlls,
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "bpb": bpb,
        "num_tokens": len(nlls),
    }


# ================================================================
# Forgetting-curve helpers
# ================================================================

def bucket_nlls_by_position(
    per_token_nll: list[float],
    bucket_size: int,
) -> dict[int, float]:
    """
    Average NLL within non-overlapping position buckets.
    Returns {bucket_start_token: mean_nll}.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for i, nll in enumerate(per_token_nll):
        bucket = (i // bucket_size) * bucket_size
        buckets[bucket].append(nll)
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def merge_forgetting_curves(
    curves: list[dict[int, float]],
) -> dict[int, float]:
    """Average multiple per-document forgetting curves into one."""
    accum: dict[int, list[float]] = defaultdict(list)
    for curve in curves:
        for pos, nll in curve.items():
            accum[pos].append(nll)
    return {k: sum(v) / len(v) for k, v in sorted(accum.items())}


# ================================================================
# Plot (optional — skipped gracefully if matplotlib missing)
# ================================================================

MAX_PLOT_TOKENS = 32_768   # 32 K — benchmark upper bound

def maybe_plot(
    curves_by_label: dict[str, dict[int, float]],
    output_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot. Run: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    for label, curve in curves_by_label.items():
        # Keep only positions up to 32 K and convert NLL → perplexity
        xs_tokens = sorted(k for k in curve.keys() if k <= MAX_PLOT_TOKENS)
        xs_k      = [x / 1_000 for x in xs_tokens]          # convert to thousands
        ys_ppl    = [math.exp(curve[x]) for x in xs_tokens]  # NLL → perplexity
        ax.plot(xs_k, ys_ppl, marker="o", markersize=3, linewidth=1.5, label=label)

    # ── X axis: 0, 5, 10, 15, 20, 25, 30 (K) ──────────────────────
    ax.set_xlim(0, MAX_PLOT_TOKENS / 1_000)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Position Bucket (K tokens)", fontsize=12)

    # ── Y axis: 0, 5, 10, 15, … ────────────────────────────────────
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylabel("Perplexity", fontsize=12)

    ax.set_title("PG19 Perplexity vs. Position Bucket (up to 32 K)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Perplexity-vs-position plot saved to {output_path}")
    plt.close(fig)


# ================================================================
# Core evaluation loop
# ================================================================

def evaluate_pg19(
    model,
    tokenizer,
    device: str,
    max_docs: Optional[int],
    max_ctx: int,
    stride: int,
    bucket_size: int,
    min_doc_tokens: int,
    split: str,
    label: str,
) -> dict:
    print(f"\n{'─' * 60}")
    print(f"  Evaluating : {label}")
    print(f"  Dataset    : fla-hub/pg19  [{split}]")
    print(f"  max_ctx    : {max_ctx}   stride: {stride}   bucket: {bucket_size}")
    print(f"{'─' * 60}")

    dataset = load_dataset("fla-hub/pg19", split=split, trust_remote_code=True)
    print(f"  Total docs in split: {len(dataset)}")

    if max_docs and max_docs < len(dataset):
        dataset = dataset.select(range(max_docs))
    print(f"  Docs to evaluate   : {len(dataset)}")

    all_nlls: list[float] = []
    all_curves: list[dict[int, float]] = []
    doc_ppls: list[float] = []
    skipped = 0
    t_start = time.perf_counter()

    for i, example in enumerate(dataset):
        text = example.get("text") or example.get("book_text") or ""
        if not text:
            skipped += 1
            continue

        enc = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"]   # (1, T)
        T = input_ids.shape[-1]

        if T < min_doc_tokens:
            skipped += 1
            continue

        # Cap at 32 K to benchmark the full position range of interest
        max_eval_tokens = MAX_PLOT_TOKENS  # 32_768
        if T > max_eval_tokens:
            input_ids = input_ids[:, :max_eval_tokens]
            T = max_eval_tokens

        t0 = time.perf_counter()
        result = compute_doc_perplexity(model, input_ids, stride, device, max_ctx)
        elapsed = time.perf_counter() - t0

        if not result:
            skipped += 1
            continue

        all_nlls.extend(result["per_token_nll"])
        doc_ppls.append(result["perplexity"])
        curve = bucket_nlls_by_position(result["per_token_nll"], bucket_size)
        all_curves.append(curve)

        if (i + 1) % 5 == 0 or i == 0:
            running_ppl = math.exp(sum(all_nlls) / len(all_nlls))
            print(
                f"  [{i+1:4d}/{len(dataset)}] "
                f"doc_tokens={T:6d}  doc_ppl={result['perplexity']:7.2f}  "
                f"running_ppl={running_ppl:7.2f}  elapsed={elapsed:.2f}s",
                flush=True,
            )

    total_elapsed = time.perf_counter() - t_start
    print(f"\n  Docs skipped (too short / empty): {skipped}")
    print(f"  Total evaluation time: {total_elapsed:.1f}s")

    if not all_nlls:
        print("  No documents evaluated!")
        return {}

    corpus_mean_nll = sum(all_nlls) / len(all_nlls)
    corpus_ppl = math.exp(corpus_mean_nll)
    corpus_bpb = corpus_mean_nll / math.log(2)
    median_doc_ppl = sorted(doc_ppls)[len(doc_ppls) // 2]
    mean_doc_ppl = sum(doc_ppls) / len(doc_ppls)
    forgetting_curve = merge_forgetting_curves(all_curves)

    print(f"\n  ── Results for [{label}] ──")
    print(f"     Corpus perplexity (token-weighted) : {corpus_ppl:.3f}")
    print(f"     Mean doc perplexity                : {mean_doc_ppl:.3f}")
    print(f"     Median doc perplexity              : {median_doc_ppl:.3f}")
    print(f"     Bits-per-byte (corpus)             : {corpus_bpb:.4f}")
    print(f"     Total tokens evaluated             : {len(all_nlls):,}")
    print(f"     Documents evaluated                : {len(doc_ppls)}")

    return {
        "label": label,
        "corpus_perplexity": corpus_ppl,
        "mean_doc_perplexity": mean_doc_ppl,
        "median_doc_perplexity": median_doc_ppl,
        "bits_per_byte": corpus_bpb,
        "total_tokens": len(all_nlls),
        "num_docs": len(doc_ppls),
        "forgetting_curve": forgetting_curve,
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Perplexity + forgetting-curve evaluation on PG19 for MS-GLA checkpoints"
    )
    # ── checkpoint args ──────────────────────────────────────────
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Step directory, experiment path, or .pt file. "
             "For multi-step forgetting curves, pass the experiment root and use --steps.",
    )
    parser.add_argument("--step", type=int, default=None,
                        help="Single checkpoint step (mutually exclusive with --steps)")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated list of steps for forgetting-curve comparison, "
                             "e.g. '5400,10800,21600'")
    parser.add_argument(
        "--model_ref",
        default=SCRIPT_DIR,
        help="Path/repo containing config+tokenizer (defaults to script directory)",
    )
    parser.add_argument("--tmp_dir", default=None,
                        help="Scratch directory for DCP→pt conversion")

    # ── dataset args ─────────────────────────────────────────────
    parser.add_argument("--split", default="test",
                        help="HuggingFace split to use (default: test)")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Max documents to evaluate (None = full split)")
    parser.add_argument("--min_doc_tokens", type=int, default=256,
                        help="Skip documents shorter than this many tokens")

    # ── evaluation args ──────────────────────────────────────────
    parser.add_argument("--max_ctx", type=int, default=2048,
                        help="Maximum context window passed to the model per forward pass")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding-window stride. Defaults to max_ctx // 2 (50%% overlap).")
    parser.add_argument("--bucket_size", type=int, default=128,
                        help="Token-position bucket width for forgetting curves")

    # ── output args ──────────────────────────────────────────────
    parser.add_argument("--results_json", default=None,
                        help="If set, dump all results to this JSON file")
    parser.add_argument("--plot_output", default=None,
                        help="If set, save perplexity-vs-position plot (up to 32 K) to this path (.png/.pdf)")

    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.step is not None and args.steps is not None:
        raise ValueError("Provide --step or --steps, not both.")

    # Build list of (label, resolved_path) pairs
    eval_targets: list[tuple[str, str]] = []

    if args.steps is not None:
        raw_steps = [int(s.strip()) for s in args.steps.split(",")]
        for s in raw_steps:
            resolved = resolve_checkpoint_path(args.checkpoint_path, s)
            eval_targets.append((f"step-{s}", resolved))
    else:
        resolved = resolve_checkpoint_path(args.checkpoint_path, args.step)
        label = Path(args.checkpoint_path).name if args.step is None else f"step-{args.step}"
        eval_targets.append((label, resolved))

    stride = args.stride if args.stride is not None else args.max_ctx // 2

    print(f"\nDevice     : {args.device}")
    print(f"max_ctx    : {args.max_ctx}   stride: {stride}   bucket: {args.bucket_size}")
    print(f"Eval steps : {[lbl for lbl, _ in eval_targets]}")

    all_results: list[dict] = []
    curves_by_label: dict[str, dict[int, float]] = {}

    for label, ckpt_path in eval_targets:
        model, tokenizer = load_model_and_tokenizer(
            model_ref=args.model_ref,
            checkpoint_path=ckpt_path,
            device=args.device,
            tmp_dir=args.tmp_dir,
        )

        result = evaluate_pg19(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_docs=args.max_docs,
            max_ctx=args.max_ctx,
            stride=stride,
            bucket_size=args.bucket_size,
            min_doc_tokens=args.min_doc_tokens,
            split=args.split,
            label=label,
        )

        if result:
            all_results.append(result)
            curves_by_label[label] = result["forgetting_curve"]

        # Free VRAM between checkpoints when comparing multiple steps
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY  —  PG19 Perplexity")
    print("=" * 72)
    print(f"  {'Label':<20} {'Corpus PPL':>11} {'Mean Doc PPL':>13} {'BPB':>8} {'Tokens':>10}")
    print("-" * 72)
    for r in all_results:
        print(
            f"  {r['label']:<20} "
            f"{r['corpus_perplexity']:>11.3f} "
            f"{r['mean_doc_perplexity']:>13.3f} "
            f"{r['bits_per_byte']:>8.4f} "
            f"{r['total_tokens']:>10,}"
        )
    print("=" * 72)

    # ── Forgetting-curve summary ─────────────────────────────────
    if curves_by_label:
        print("\n  FORGETTING CURVES  (mean NLL per position bucket)")
        # Print first 10 buckets for a quick inline view
        all_positions = sorted(
            set(pos for curve in curves_by_label.values() for pos in curve)
        )[:10]
        header = f"  {'Position':>10}" + "".join(f"  {lbl:>12}" for lbl in curves_by_label)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for pos in all_positions:
            row = f"  {pos:>10}"
            for curve in curves_by_label.values():
                val = curve.get(pos, float("nan"))
                row += f"  {val:>12.4f}"
            print(row)
        if len(all_positions) < len(set(pos for c in curves_by_label.values() for pos in c)):
            print("  ... (truncated — see --results_json for full curves)")

    # ── Optional outputs ─────────────────────────────────────────
    if args.results_json and all_results:
        # forgetting_curve keys are ints → convert to str for JSON
        serialisable = []
        for r in all_results:
            rc = dict(r)
            rc["forgetting_curve"] = {str(k): v for k, v in rc["forgetting_curve"].items()}
            serialisable.append(rc)
        with open(args.results_json, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nFull results written to {args.results_json}")

    if args.plot_output and curves_by_label:
        maybe_plot(curves_by_label, args.plot_output)


if __name__ == "__main__":
    main()