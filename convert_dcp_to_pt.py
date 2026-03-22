"""
convert_dcp_to_pt.py — Convert a distributed checkpoint (DCP) step into a single .pt file.

Examples:
    python convert_dcp_to_pt.py --checkpoint_path benchmark_gla --step 68664
    python convert_dcp_to_pt.py --checkpoint_path benchmark_msgla/checkpoint/step-68664 --output_path /tmp/msgla_step68664.pt
"""

import argparse
from pathlib import Path

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


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
        f"Could not infer DCP checkpoint directory from {base}. "
        "Pass either a step directory directly or set --step."
    )


def default_output_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file():
        return checkpoint_path
    return checkpoint_path / "checkpoint.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a DCP checkpoint step into a single .pt file.")
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Experiment path or step directory in DCP format",
    )
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (optional)")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Where to write the .pt file (defaults to <resolved_step_dir>/checkpoint.pt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path, args.step)
    if checkpoint_path.suffix == ".pt":
        raise ValueError("Input already points to a .pt file; no conversion is needed.")

    output_path = Path(args.output_path) if args.output_path else default_output_path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to overwrite it.")

    print(f"Converting DCP checkpoint from {checkpoint_path} ...")
    print(f"Writing .pt checkpoint to {output_path} ...")
    dcp_to_torch_save(str(checkpoint_path), str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()
