"""
convert_checkpoint_to_hf.py — Convert a GLA-family checkpoint (.pt or DCP) to a Hugging Face folder.

Examples:
    python convert_checkpoint_to_hf.py \
        --checkpoint_path benchmark_msgla/checkpoint/7B/124/step-106813/tmp/106.pt \
        --config flame/configs/ms_gla_340M.json \
        --tokenizer benchmark_msgla \
        --output_dir /tmp/msgla-7b-124-hf

    python convert_checkpoint_to_hf.py \
        --checkpoint_path benchmark_gla \
        --step 106813 \
        --config flame/configs/gla_340M.json \
        --tokenizer benchmark_gla \
        --output_dir /tmp/gla-7b-hf
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
FLAME_ROOT = REPO_ROOT / "flame"

for path in (FLAME_ROOT,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import fla  # noqa: F401,E402
import custom_models  # noqa: F401,E402


def normalize_tied_weight_keys(module: torch.nn.Module) -> None:
    """
    Newer transformers builds expect `_tied_weights_keys` to be dict-like and call `.keys()`.
    FLA/custom models currently define it as a list, so normalize those attributes before
    calling `save_pretrained`.
    """
    for submodule in module.modules():
        tied = getattr(submodule, "_tied_weights_keys", None)
        if isinstance(tied, list):
            submodule._tied_weights_keys = {key: None for key in tied}


def resolve_checkpoint_path(checkpoint_path: str, step: int | None) -> Path:
    base = Path(checkpoint_path)

    if base.is_file():
        if base.suffix != ".pt":
            raise ValueError(f"Expected a .pt file or DCP directory, got file: {base}")
        if step is not None:
            raise ValueError("Do not pass --step when --checkpoint_path already points to a .pt file.")
        return base

    if step is not None:
        candidate1 = base / "checkpoint" / f"step-{step}"
        candidate2 = base / f"step-{step}"
        if candidate1.exists():
            return candidate1
        if candidate2.exists():
            return candidate2
        raise FileNotFoundError(
            f"Could not find step-{step} under {base}. Tried {candidate1} and {candidate2}."
        )

    if (base / ".metadata").exists() and any(base.glob("*.distcp")):
        return base

    latest = None
    checkpoint_root = base / "checkpoint"
    if checkpoint_root.exists():
        for path in checkpoint_root.glob("step-*"):
            try:
                step_num = int(path.name.split("step-")[-1])
            except ValueError:
                continue
            if latest is None or step_num > latest[0]:
                latest = (step_num, path)

    if latest is not None:
        print(f"No --step provided. Using latest checkpoint: {latest[1]}")
        return latest[1]

    raise FileNotFoundError(
        f"Could not infer checkpoint path from {base}. Pass a .pt file, a DCP step directory, or set --step."
    )


def load_checkpoint_state(resolved_checkpoint: Path) -> dict:
    torch.serialization.add_safe_globals([timedelta, io.BytesIO])

    if resolved_checkpoint.is_file():
        print(f"Loading .pt checkpoint from {resolved_checkpoint} ...")
        return torch.load(resolved_checkpoint, map_location="cpu")

    print(f"Loading DCP checkpoint from {resolved_checkpoint} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_pt = Path(tmpdir) / "checkpoint.pt"
        dcp_to_torch_save(str(resolved_checkpoint), str(checkpoint_pt))
        return torch.load(checkpoint_pt, map_location="cpu")


@torch.inference_mode()
def convert_checkpoint_to_hf(
    checkpoint_path: str,
    step: int | None,
    config_path: str,
    tokenizer_path: str,
    output_dir: str,
) -> None:
    resolved_checkpoint = resolve_checkpoint_path(checkpoint_path, step)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {config_path} ...")
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

    print(f"Loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print("Initializing model from config ...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    state = load_checkpoint_state(resolved_checkpoint)
    if "model" not in state:
        raise KeyError("Checkpoint does not contain a top-level 'model' key.")

    print("Loading model weights ...")
    missing_keys, unexpected_keys = model.load_state_dict(state["model"], strict=False)
    if missing_keys:
        print(f"Warning: missing keys when loading state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: unexpected keys when loading state dict: {unexpected_keys}")

    normalize_tied_weight_keys(model)

    print(f"Saving Hugging Face model to {output_path} ...")
    config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, safe_serialization=True)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GLA/MS-GLA checkpoint (.pt or DCP) into a Hugging Face save_pretrained folder."
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to a .pt checkpoint, a DCP step directory, or an experiment root containing checkpoint/step-*",
    )
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step if --checkpoint_path is an experiment root")
    parser.add_argument("--config", required=True, help="Path to the model config JSON or HF config directory")
    parser.add_argument("--tokenizer", required=True, help="Path to the tokenizer directory")
    parser.add_argument("--output_dir", required=True, help="Output folder for Hugging Face save_pretrained files")
    args = parser.parse_args()

    convert_checkpoint_to_hf(
        checkpoint_path=args.checkpoint_path,
        step=args.step,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
