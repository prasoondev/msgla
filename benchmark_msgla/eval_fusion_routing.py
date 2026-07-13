"""
eval_fusion_routing.py — Investigates MS-GLA fusion routing weights for
extended context lengths and Out-of-Distribution (OOD) sequences.

Examples:
    # All MS-GLA variants under pt_checkpoints/ + config/ (default layout)
    python eval_fusion_routing.py --all_variants

    # Single checkpoint
    python eval_fusion_routing.py \
        --model_ref config/msgla-12 \
        --checkpoint_path pt_checkpoints/msgla-12.pt \
        --mode length_extrap \
        --max_seq_len 65536

Outputs (per model, under --output_dir/<variant>/):
    length_extrapolation_routing.csv  — per-token weights (layer-averaged)
    length_extrapolation_summary.csv  — first/middle/last 10% + overall means
    ood_routing_comparison.csv        — steady-state weights per OOD condition
"""

import argparse
import csv
import io
import os
import sys
import tempfile
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.serialization
from datasets import load_dataset
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoTokenizer

# ── Import custom models ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLAME_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "flame")
if FLAME_ROOT not in sys.path:
    sys.path.insert(0, FLAME_ROOT)

from custom_models.ms_gla import MSGLAConfig, MSGLAForCausalLM


# ================================================================
# Checkpoint loading
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
        raise FileNotFoundError(f"Could not find step-{step} under {base}.")

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


@dataclass(frozen=True)
class MSGLAVariant:
    name: str
    checkpoint_path: Path
    config_dir: Path


def discover_msgla_variants(
    checkpoints_dir: Path,
    configs_dir: Path,
    variant_filter: Optional[set[str]] = None,
) -> list[MSGLAVariant]:
    """
    Pair msgla-*.pt checkpoints with config/<variant>/ directories.

    Expects pt_checkpoints/msgla-12.pt -> config/msgla-12/config.json.
    """
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    variants: list[MSGLAVariant] = []
    for ckpt_path in sorted(checkpoints_dir.glob("msgla-*.pt")):
        name = ckpt_path.stem
        if variant_filter is not None and name not in variant_filter:
            continue

        config_dir = configs_dir / name
        config_json = config_dir / "config.json"
        if not config_json.is_file():
            print(f"Skipping {ckpt_path.name}: missing {config_json}")
            continue

        variants.append(MSGLAVariant(name=name, checkpoint_path=ckpt_path, config_dir=config_dir))

    if not variants:
        raise RuntimeError(
            f"No MS-GLA variants found in {checkpoints_dir} with configs under {configs_dir}."
        )
    return variants


def load_model_and_tokenizer(
    model_ref: str,
    checkpoint_path: str,
    device: str,
    tmp_dir: Optional[str],
    tokenizer_ref: Optional[str] = None,
):
    tok_ref = tokenizer_ref or model_ref
    print(f"Loading tokenizer from {tok_ref} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_ref, trust_remote_code=True)
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
# Forward Hook Manager for Routing Weights
# ================================================================
class FusionRoutingCollector:
    """
    Captures per-token fusion routing weights from MultiScaleGatedLinearAttention.

    MS-GLA uses `self.fuse` (nn.Linear) + functional softmax, not nn.Softmax modules.
    See flame/custom_models/ms_gla/ms_gla_layer.py.
    """

    def __init__(self, model, fusion_module_name: str = "fuse", fuse_mode: str = "softmax"):
        self.hooks = []
        self.routing_weights = []  # [SeqLen, NumScales] per hooked MS-GLA layer
        self.fusion_module_name = fusion_module_name
        self.fuse_mode = fuse_mode
        self._register_hooks(model)

    def _is_fusion_linear(self, name: str, module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        # Match `...attn.fuse` (MultiScaleGatedLinearAttention routing head).
        suffix = f".{self.fusion_module_name}"
        return name.endswith(suffix) or name == self.fusion_module_name

    def _register_hooks(self, model) -> None:
        if self.fuse_mode != "softmax":
            print(
                f"Warning: model fuse_mode={self.fuse_mode!r}; routing weights are only "
                "defined for fuse_mode='softmax'."
            )
            return

        for name, module in model.named_modules():
            if not self._is_fusion_linear(name, module):
                continue
            hook = module.register_forward_hook(self._get_hook(name))
            self.hooks.append(hook)
            print(f"Registered routing hook on: {name} (Linear → softmax)")

        if not self.hooks:
            fuse_linears = [
                n for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and "fuse" in n.lower()
            ]
            raise RuntimeError(
                "No fusion routing hooks registered. MS-GLA uses nn.Linear named "
                f"'{self.fusion_module_name}' inside MultiScaleGatedLinearAttention "
                "(e.g. model.layers.0.attn.fuse), with torch.softmax on its output. "
                f"Linear modules containing 'fuse': {fuse_linears or '(none)'}"
            )

    def _get_hook(self, name):
        def hook(module, inp, out):
            # out: fusion logits [Batch, SeqLen, NumScales]
            if out.dim() == 3:
                weights = torch.softmax(out, dim=-1)
                self.routing_weights.append(weights[0].detach().float().cpu().numpy())
        return hook

    def clear(self):
        self.routing_weights.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


# ================================================================
# Data Generators
# ================================================================
def get_long_text(tokenizer, min_length=65536):
    ds = load_dataset("fla-hub/pg19", split="test", streaming=True, trust_remote_code=True)
    for row in ds:
        text = row.get("text", "")
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        if tokens.shape[-1] >= min_length:
            return tokens[:, :min_length]
    raise ValueError(f"Could not find a document of length {min_length}")

def get_ood_sequences(tokenizer, length=4096):
    vocab_size = tokenizer.vocab_size
    return {
        "In-Distribution (PG19)": get_long_text(tokenizer, length),
        "OOD (Random Uniform Noise)": torch.randint(0, vocab_size, (1, length)),
        "OOD (Repeated 10-grams)": torch.randint(0, vocab_size, (1, 10)).repeat(1, length // 10 + 1)[:, :length]
    }


# ================================================================
# Result printing
# ================================================================

def print_scale_weights(title: str, weights: np.ndarray) -> None:
    print(title)
    for i, w in enumerate(weights):
        print(f"  scale {i}: {w:.4f}")
    dominant = int(np.argmax(weights))
    print(f"  dominant scale: {dominant} (w={weights[dominant]:.4f})")


def print_region_summaries(
    avg_weights: np.ndarray,
    num_layers: int,
    region_fraction: float = 0.1,
) -> None:
    """avg_weights: [SeqLen, NumScales]"""
    seq_len, num_scales = avg_weights.shape
    region_len = max(1, int(seq_len * region_fraction))

    print("\n=== Length extrapolation routing summary ===")
    print(f"Layers averaged: {num_layers}, scales: {num_scales}, seq_len: {seq_len}")

    regions = [
        ("first 10%", 0, region_len),
        ("middle 10%", seq_len // 2 - region_len // 2, seq_len // 2 + region_len // 2),
        ("last 10%", seq_len - region_len, seq_len),
    ]
    for name, start, end in regions:
        region_mean = avg_weights[start:end].mean(axis=0)
        print_scale_weights(f"\n{name} (tokens {start}:{end}):", region_mean)

    print_scale_weights("\nOverall sequence mean:", avg_weights.mean(axis=0))


def _scale_column_names(num_scales: int, prefix: str = "weight_scale") -> list[str]:
    return [f"{prefix}_{i}" for i in range(num_scales)]


def write_routing_timeseries_csv(path: str, avg_weights: np.ndarray, num_layers: int) -> None:
    """avg_weights: [SeqLen, NumScales], layer-averaged."""
    num_scales = avg_weights.shape[1]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["token_position", "num_layers_averaged", *_scale_column_names(num_scales)]
        )
        for t in range(avg_weights.shape[0]):
            writer.writerow(
                [t, num_layers, *[f"{avg_weights[t, s]:.8f}" for s in range(num_scales)]]
            )


def write_region_summary_csv(
    path: str,
    avg_weights: np.ndarray,
    num_layers: int,
    region_fraction: float = 0.1,
) -> None:
    seq_len, num_scales = avg_weights.shape
    region_len = max(1, int(seq_len * region_fraction))
    regions = [
        ("first_10pct", 0, region_len),
        ("middle_10pct", seq_len // 2 - region_len // 2, seq_len // 2 + region_len // 2),
        ("last_10pct", seq_len - region_len, seq_len),
        ("overall", 0, seq_len),
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "region",
                "token_start",
                "token_end",
                "num_layers_averaged",
                "seq_len",
                *_scale_column_names(num_scales),
                "dominant_scale",
                "dominant_weight",
            ]
        )
        for name, start, end in regions:
            region_mean = avg_weights[start:end].mean(axis=0)
            dominant = int(np.argmax(region_mean))
            writer.writerow(
                [
                    name,
                    start,
                    end,
                    num_layers,
                    seq_len,
                    *[f"{region_mean[s]:.8f}" for s in range(num_scales)],
                    dominant,
                    f"{region_mean[dominant]:.8f}",
                ]
            )


def write_ood_routing_csv(path: str, results: dict[str, np.ndarray], num_layers: int, tail_fraction: float) -> None:
    num_scales = len(next(iter(results.values())))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "num_layers_averaged",
                "tail_fraction",
                *_scale_column_names(num_scales),
                "dominant_scale",
                "dominant_weight",
            ]
        )
        for condition, weights in results.items():
            dominant = int(np.argmax(weights))
            writer.writerow(
                [
                    condition,
                    num_layers,
                    tail_fraction,
                    *[f"{weights[s]:.8f}" for s in range(num_scales)],
                    dominant,
                    f"{weights[dominant]:.8f}",
                ]
            )


# ================================================================
# Experiments
# ================================================================
@torch.inference_mode()
def run_length_extrapolation(model, tokenizer, collector, args):
    print(f"\n--- Running Length Extrapolation (Up to {args.max_seq_len} tokens) ---")
    input_ids = get_long_text(tokenizer, args.max_seq_len).to(args.device)
    
    _ = model(input_ids, use_cache=False)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not collector.routing_weights:
        print("No routing weights captured. Check module names.")
        return

    # Average the routing weights across all fusion layers in the network
    # Shape of stacked: [NumLayers, SeqLen, NumScales]
    stacked_weights = np.stack(collector.routing_weights)
    avg_weights_over_layers = stacked_weights.mean(axis=0) # [SeqLen, NumScales]
    
    num_layers = len(collector.routing_weights)

    print_region_summaries(avg_weights_over_layers, num_layers=num_layers)

    timeseries_path = os.path.join(args.output_dir, "length_extrapolation_routing.csv")
    summary_path = os.path.join(args.output_dir, "length_extrapolation_summary.csv")
    write_routing_timeseries_csv(timeseries_path, avg_weights_over_layers, num_layers)
    write_region_summary_csv(summary_path, avg_weights_over_layers, num_layers)
    print(f"Saved per-token routing to {timeseries_path}")
    print(f"Saved region summary to {summary_path}")

@torch.inference_mode()
def run_ood_evaluation(model, tokenizer, collector, args):
    print(f"\n--- Running OOD Sequence Evaluation ---")
    sequences = get_ood_sequences(tokenizer, length=args.ood_seq_len)
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    for label, seq in sequences.items():
        print(f"Evaluating: {label}")
        collector.clear()
        _ = model(seq.to(args.device), use_cache=False)
        
        if collector.routing_weights:
            stacked = np.stack(collector.routing_weights) # [Layers, SeqLen, Scales]
            # Take the mean across layers and across the final 10% of the sequence to see "steady state" saturation
            tail_idx = int(args.ood_seq_len * 0.9)
            steady_state_weights = stacked[:, tail_idx:, :].mean(axis=(0, 1)) # [Scales]
            results[label] = steady_state_weights
                
    if not results:
        print("No routing weights captured. Check module names.")
        return

    num_layers = len(collector.routing_weights)
    tail_fraction = 0.1

    print("\n=== OOD steady-state routing (last 10% of context) ===")
    print(f"Layers averaged: {num_layers}, scales: {len(next(iter(results.values())))}")
    for label, weights in results.items():
        print_scale_weights(f"\n{label}:", weights)

    save_path = os.path.join(args.output_dir, "ood_routing_comparison.csv")
    write_ood_routing_csv(save_path, results, num_layers, tail_fraction)
    print(f"Saved OOD routing comparison to {save_path}")


def evaluate_variant(args, variant_name: str, model_ref: str, checkpoint_path: str) -> str:
    """Run routing evaluation for one variant. Returns output subdirectory path."""
    output_dir = os.path.join(args.output_dir, variant_name)
    os.makedirs(output_dir, exist_ok=True)

    variant_args = argparse.Namespace(**{**vars(args), "output_dir": output_dir})
    ckpt_path = resolve_checkpoint_path(checkpoint_path, args.step)
    model, tokenizer = load_model_and_tokenizer(
        model_ref=model_ref,
        checkpoint_path=ckpt_path,
        device=args.device,
        tmp_dir=args.tmp_dir,
        tokenizer_ref=args.tokenizer_ref,
    )

    fuse_mode = getattr(model.config, "fuse_mode", "softmax")
    collector = FusionRoutingCollector(
        model,
        fusion_module_name=args.fusion_module_name,
        fuse_mode=fuse_mode,
    )

    try:
        if args.mode in ["length_extrap", "both"]:
            run_length_extrapolation(model, tokenizer, collector, variant_args)
        if args.mode in ["ood", "both"]:
            run_ood_evaluation(model, tokenizer, collector, variant_args)
    finally:
        collector.remove_hooks()
        del model, tokenizer, collector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output_dir


def write_variants_manifest(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_all_variants(args) -> None:
    checkpoints_dir = Path(args.checkpoints_dir)
    configs_dir = Path(args.configs_dir)
    variant_filter = None
    if args.variants:
        variant_filter = {v.strip() for v in args.variants.split(",") if v.strip()}

    variants = discover_msgla_variants(checkpoints_dir, configs_dir, variant_filter)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Found {len(variants)} MS-GLA variant(s) to evaluate.")
    manifest_rows: list[dict] = []

    for variant in variants:
        print(f"\n{'=' * 72}\nVariant: {variant.name}\n{'=' * 72}")
        print(f"  checkpoint : {variant.checkpoint_path}")
        print(f"  config     : {variant.config_dir}")
        try:
            out_dir = evaluate_variant(
                args,
                variant_name=variant.name,
                model_ref=str(variant.config_dir),
                checkpoint_path=str(variant.checkpoint_path),
            )
            status = "ok"
        except Exception as exc:
            print(f"ERROR evaluating {variant.name}: {exc}")
            out_dir = ""
            status = f"error: {exc}"

        manifest_rows.append(
            {
                "variant": variant.name,
                "checkpoint": str(variant.checkpoint_path),
                "config_dir": str(variant.config_dir),
                "status": status,
                "output_dir": out_dir,
            }
        )

    manifest_path = os.path.join(args.output_dir, "variants_manifest.csv")
    write_variants_manifest(manifest_path, manifest_rows)
    print(f"\nWrote run manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all_variants",
        action="store_true",
        help="Evaluate every msgla-*.pt in --checkpoints_dir with config in --configs_dir",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default=os.path.join(SCRIPT_DIR, "pt_checkpoints"),
        help="Directory containing msgla-*.pt checkpoints (default: benchmark_msgla/pt_checkpoints)",
    )
    parser.add_argument(
        "--configs_dir",
        default=os.path.join(SCRIPT_DIR, "config"),
        help="Directory containing per-variant config subdirs (default: benchmark_msgla/config)",
    )
    parser.add_argument(
        "--tokenizer_ref",
        default=SCRIPT_DIR,
        help="Tokenizer path shared across variants (default: benchmark_msgla/)",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated subset of variant names (e.g. msgla-12,msgla-1248)",
    )
    parser.add_argument("--model_ref", default=None, help="Config directory for a single variant")
    parser.add_argument("--checkpoint_path", default=None, help="Checkpoint .pt or DCP path")
    parser.add_argument("--step", type=int, default=None, help="Training step for DCP checkpoints")
    parser.add_argument("--tmp_dir", default=None, help="Temp dir for DCP → .pt conversion")
    parser.add_argument("--mode", choices=["length_extrap", "ood", "both"], default="both")
    parser.add_argument(
        "--fusion_module_name",
        default="fuse",
        help="Name of the fusion Linear in MultiScaleGatedLinearAttention (default: fuse)",
    )
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--ood_seq_len", type=int, default=4096)
    parser.add_argument("--output_dir", default="./investigation_output")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.all_variants:
        if args.model_ref or args.checkpoint_path:
            parser.error("Do not pass --model_ref/--checkpoint_path with --all_variants.")
        run_all_variants(args)
        return

    if not args.model_ref or not args.checkpoint_path:
        parser.error("Provide --model_ref and --checkpoint_path, or use --all_variants.")

    variant_name = Path(args.model_ref).name
    evaluate_variant(
        args,
        variant_name=variant_name,
        model_ref=args.model_ref,
        checkpoint_path=args.checkpoint_path,
    )

if __name__ == "__main__":
    main()
