"""
run_lm_eval_custom.py — Launch lm-evaluation-harness with local GLA/MS-GLA registrations preloaded.

Usage:
    /home/prasoon/Documents/research/msgla/.venv/bin/python run_lm_eval_custom.py \
        --model hf \
        --model_args pretrained=/home/prasoon/Documents/research/msgla/hf-checkpoints/msgla_124,trust_remote_code=True \
        --tasks hellaswag,piqa,winogrande,lambada_openai \
        --device cuda:0 \
        --batch_size auto
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
FLAME_ROOT = REPO_ROOT / "flame"
FLA_ROOT = REPO_ROOT / "3rd_party" / "flash-linear-attention"

for path in (FLAME_ROOT, FLA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Import side effects register custom config/model types with transformers.
import fla  # noqa: F401,E402
import custom_models  # noqa: F401,E402

from lm_eval.__main__ import cli_evaluate  # noqa: E402


if __name__ == "__main__":
    sys.exit(cli_evaluate())
