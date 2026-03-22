"""
eval_gla_benchmarks.py — Benchmark evaluation for GLA-family checkpoints in DCP format.

Benchmarks:
  - WikiText   : ppl
  - LAMBADA    : ppl, acc
  - PIQA       : acc
  - HellaSwag  : acc_norm
  - WinoGrande : acc

Examples:
    python eval_gla_benchmarks.py \
        --checkpoint_path /path/to/exp/gla-340M --step 10800 \
        --model_ref /path/to/exp/gla-340M

    python eval_gla_benchmarks.py \
        --checkpoint_path /path/to/checkpoint/step-10800 \
        --model_ref /path/to/exp/gla-340M \
        --tasks wikitext,lambada,piqa \
        --max_samples 200
"""

import argparse
import io
import math
import os
import re
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.serialization
from datasets import load_dataset
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoTokenizer

# ── make local model packages importable ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FLAME_ROOT = os.path.join(REPO_ROOT, "flame")
FLA_ROOT = os.path.join(REPO_ROOT, "3rd_party", "flash-linear-attention")
for path in (FLAME_ROOT, FLA_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Importing these packages triggers AutoConfig / AutoModelForCausalLM registration.
from custom_models.ms_gla import MSGLAConfig, MSGLAForCausalLM  # noqa: E402
from fla.models.gla import GLAConfig, GLAForCausalLM  # noqa: E402


MODEL_REGISTRY = {
    "gla": (GLAConfig, GLAForCausalLM),
    "ms_gla": (MSGLAConfig, MSGLAForCausalLM),
}


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
        candidate1 = base / "checkpoint" / f"step-{step}"
        candidate2 = base / f"step-{step}"
        if candidate1.exists():
            return str(candidate1)
        if candidate2.exists():
            return str(candidate2)
        raise FileNotFoundError(
            f"Could not find step-{step} under {base}. Tried {candidate1} and {candidate2}."
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


def load_model_and_tokenizer_from_dcp(
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
    auto_config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    model_type = getattr(auto_config, "model_type", None)
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_type {model_type!r} in {model_ref}. Supported types: {supported}."
        )
    config_cls, model_cls = MODEL_REGISTRY[model_type]
    config = config_cls.from_pretrained(model_ref)
    model = model_cls(config)
    print(f"Instantiated model_type={model_type!r}.")

    state = load_state_from_checkpoint(checkpoint_path, tmp_dir)
    if "model" not in state:
        raise KeyError("Checkpoint does not contain 'model' in top-level state dict.")
    model.load_state_dict(state["model"])

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {n/1e6:.1f}M params ({model_type}) on {device}.")
    return model, tokenizer, model_type


# ================================================================
# Scoring helpers
# ================================================================

def safe_exp(x: float) -> float:
    if x > 80:
        return float("inf")
    return math.exp(x)


@torch.inference_mode()
def continuation_loglikelihood(
    model,
    tokenizer,
    context: str,
    continuation: str,
    device: str,
    max_ctx: int,
) -> tuple[float, int, bool]:
    if max_ctx < 2:
        raise ValueError(f"max_ctx must be >= 2 for continuation scoring, got {max_ctx}")

    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]

    if not cont_ids:
        return 0.0, 0, True

    max_cont_tokens = max_ctx - 1
    if len(cont_ids) > max_cont_tokens:
        cont_ids = cont_ids[-max_cont_tokens:]

    max_ctx_tokens = max_ctx - len(cont_ids)
    if len(ctx_ids) >= max_ctx_tokens:
        ctx_ids = ctx_ids[-max_ctx_tokens:]

    if not ctx_ids:
        prefix_id = tokenizer.bos_token_id
        if prefix_id is None:
            prefix_id = tokenizer.eos_token_id
        if prefix_id is None:
            prefix_id = tokenizer.pad_token_id
        if prefix_id is None:
            raise ValueError("Tokenizer must define one of bos/eos/pad token ids for log-likelihood scoring.")
        ctx_ids = [prefix_id]
        if len(ctx_ids) + len(cont_ids) > max_ctx:
            cont_ids = cont_ids[-(max_ctx - 1):]

    full_ids = ctx_ids + cont_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    logits = model(input_ids=input_ids, use_cache=False).logits[0]  # (L, V)
    shift_logits = logits[:-1, :].float()                           # predicts tokens 1..L-1

    cont_start = len(ctx_ids) - 1
    cont_end = cont_start + len(cont_ids)
    choice_logits = shift_logits[cont_start:cont_end, :]

    targets = torch.tensor(cont_ids, dtype=torch.long, device=device)
    log_probs = torch.log_softmax(choice_logits, dim=-1)
    token_log_probs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    sum_log_prob = float(token_log_probs.sum().item())

    greedy = choice_logits.argmax(dim=-1)
    exact = bool(torch.equal(greedy, targets))
    return sum_log_prob, len(cont_ids), exact


@torch.inference_mode()
def compute_doc_nll_sliding(
    model,
    input_ids: torch.Tensor,  # (1, T)
    max_ctx: int,
    stride: int,
    device: str,
) -> tuple[float, int]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape (1, T), got {tuple(input_ids.shape)}")

    T = int(input_ids.shape[-1])
    if T < 2:
        return 0.0, 0

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    total_nll = 0.0
    total_tokens = 0
    prev_covered = 0

    for begin in range(0, T - 1, stride):
        end = min(begin + max_ctx, T)
        chunk = input_ids[:, begin:end].to(device)

        logits = model(input_ids=chunk, use_cache=False).logits[0]
        shift_logits = logits[:-1, :].contiguous().float()
        shift_labels = chunk[0, 1:].contiguous()
        token_losses = loss_fct(shift_logits, shift_labels)

        chunk_first_pred_pos = begin + 1
        chunk_last_pred_pos = end - 1
        new_from = max(prev_covered + 1, chunk_first_pred_pos)
        if new_from <= chunk_last_pred_pos:
            idx_start = new_from - chunk_first_pred_pos
            new_losses = token_losses[idx_start:]
            total_nll += float(new_losses.sum().item())
            total_tokens += int(new_losses.numel())
            prev_covered = chunk_last_pred_pos

        if end == T:
            break

    return total_nll, total_tokens


def _clean_hellaswag_text(text: str) -> str:
    cleaned = text.replace(" [title]", ". ")
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ================================================================
# Task evaluators
# ================================================================

def evaluate_wikitext(
    model,
    tokenizer,
    device: str,
    max_ctx: int,
    stride: int,
    max_samples: Optional[int],
    max_wikitext_tokens: Optional[int],
    split: str,
) -> dict:
    print(f"\n{'─' * 60}")
    print("  Task   : WIKITEXT")
    print(f"  Metric : ppl")
    print(f"  Dataset: Salesforce/wikitext (wikitext-2-raw-v1) [{split}]")
    print(f"{'─' * 60}")

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    texts = [row["text"] for row in dataset if row.get("text", "").strip()]
    if not texts:
        raise RuntimeError("No non-empty WikiText rows found.")

    corpus = "\n\n".join(texts)
    enc = tokenizer(corpus, return_tensors="pt", truncation=False)
    input_ids = enc["input_ids"]
    if max_wikitext_tokens and input_ids.shape[-1] > max_wikitext_tokens:
        input_ids = input_ids[:, :max_wikitext_tokens]
        print(f"  Token cap: {max_wikitext_tokens}")

    print(f"  Total tokens to score: {input_ids.shape[-1]:,}")
    t0 = time.perf_counter()
    nll_sum, n_tokens = compute_doc_nll_sliding(
        model=model,
        input_ids=input_ids,
        max_ctx=max_ctx,
        stride=stride,
        device=device,
    )
    elapsed = time.perf_counter() - t0

    ppl = safe_exp(nll_sum / n_tokens) if n_tokens else float("inf")
    print(f"  FINAL WIKITEXT ppl: {ppl:.4f}  (tokens={n_tokens:,}, time={elapsed:.1f}s)")
    return {
        "task": "wikitext",
        "metric": "ppl",
        "score": ppl,
        "n": n_tokens,
        "higher_is_better": False,
    }


def evaluate_lambada(
    model,
    tokenizer,
    device: str,
    max_ctx: int,
    max_samples: Optional[int],
    split: str,
) -> list[dict]:
    print(f"\n{'─' * 60}")
    print("  Task   : LAMBADA")
    print("  Metrics: ppl, acc")
    print(f"{'─' * 60}")

    dataset = load_dataset("cimec/lambada", "plain_text", split=split)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    print(f"  Samples: {len(dataset)}")
    print("  Source : cimec/lambada (plain_text)")

    total_nll = 0.0
    total_target_tokens = 0
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        context = None
        target = None

        if "context" in example:
            context = str(example["context"])
            if "target_word" in example:
                target = str(example["target_word"])
            elif "answer" in example:
                target = str(example["answer"])
            elif "target" in example:
                target = str(example["target"])
        elif "text" in example:
            text = str(example["text"]).strip()
            if " " in text:
                context, target = text.rsplit(" ", 1)
        elif "sentence" in example:
            text = str(example["sentence"]).strip()
            if " " in text:
                context, target = text.rsplit(" ", 1)

        if not context or not target:
            continue

        continuation = " " + target
        ll, tok_count, exact = continuation_loglikelihood(
            model=model,
            tokenizer=tokenizer,
            context=context,
            continuation=continuation,
            device=device,
            max_ctx=max_ctx,
        )
        total_nll += -ll
        total_target_tokens += tok_count
        correct += int(exact)
        total += 1

        if (i + 1) % 50 == 0 or i == 0:
            running_acc = 100.0 * correct / total if total else 0.0
            running_ppl = safe_exp(total_nll / total_target_tokens) if total_target_tokens else float("inf")
            print(f"  [{i+1:4d}/{len(dataset)}]  running ppl={running_ppl:.4f}  acc={running_acc:.2f}")

    ppl = safe_exp(total_nll / total_target_tokens) if total_target_tokens else float("inf")
    acc = 100.0 * correct / total if total else 0.0
    print(f"  FINAL LAMBADA ppl: {ppl:.4f}")
    print(f"  FINAL LAMBADA acc: {acc:.2f}")

    return [
        {
            "task": "lambada",
            "metric": "ppl",
            "score": ppl,
            "n": total_target_tokens,
            "higher_is_better": False,
        },
        {
            "task": "lambada",
            "metric": "acc",
            "score": acc,
            "n": total,
            "higher_is_better": True,
        },
    ]


def evaluate_piqa(
    model,
    tokenizer,
    device: str,
    max_ctx: int,
    max_samples: Optional[int],
    split: str,
) -> dict:
    print(f"\n{'─' * 60}")
    print("  Task   : PIQA")
    print("  Metric : acc")
    print(f"  Dataset: baber/piqa [{split}]")
    print(f"{'─' * 60}")

    dataset = load_dataset("baber/piqa", split=split)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    print(f"  Samples: {len(dataset)}")

    correct = 0
    total = 0
    for i, ex in enumerate(dataset):
        context = f"Question: {ex['goal']}\nAnswer:"
        choices = [f" {ex['sol1']}", f" {ex['sol2']}"]
        label = int(ex["label"])

        scores = []
        for c in choices:
            ll, _, _ = continuation_loglikelihood(
                model=model,
                tokenizer=tokenizer,
                context=context,
                continuation=c,
                device=device,
                max_ctx=max_ctx,
            )
            scores.append(ll)

        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == label)
        total += 1

        if (i + 1) % 100 == 0 or i == 0:
            running = 100.0 * correct / total
            print(f"  [{i+1:4d}/{len(dataset)}]  running acc={running:.2f}")

    acc = 100.0 * correct / total if total else 0.0
    print(f"  FINAL PIQA acc: {acc:.2f}")
    return {"task": "piqa", "metric": "acc", "score": acc, "n": total, "higher_is_better": True}


def evaluate_hellaswag(
    model,
    tokenizer,
    device: str,
    max_ctx: int,
    max_samples: Optional[int],
    split: str,
) -> dict:
    print(f"\n{'─' * 60}")
    print("  Task   : HELLASWAG")
    print("  Metric : acc_norm")
    print(f"  Dataset: allenai/hellaswag [{split}]")
    print(f"{'─' * 60}")

    dataset = load_dataset("allenai/hellaswag", split=split)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    print(f"  Samples: {len(dataset)}")

    correct = 0
    total = 0
    for i, ex in enumerate(dataset):
        context = _clean_hellaswag_text(str(ex.get("ctx", "")))
        if not context:
            ctx_a = _clean_hellaswag_text(str(ex.get("ctx_a", "")))
            ctx_b = _clean_hellaswag_text(str(ex.get("ctx_b", "")))
            activity = _clean_hellaswag_text(str(ex.get("activity_label", "")))
            context = f"{activity}: {ctx_a} {ctx_b}".strip()
            context = re.sub(r"\s+", " ", context)

        choices = [f" {_clean_hellaswag_text(c)}" for c in ex["endings"]]
        label_str = str(ex.get("label", "")).strip()
        if label_str not in {"0", "1", "2", "3"}:
            continue
        label = int(label_str)

        norm_scores = []
        for c in choices:
            ll, tok_count, _ = continuation_loglikelihood(
                model=model,
                tokenizer=tokenizer,
                context=context,
                continuation=c,
                device=device,
                max_ctx=max_ctx,
            )
            norm_scores.append(ll / max(tok_count, 1))

        pred = int(torch.tensor(norm_scores).argmax().item())
        correct += int(pred == label)
        total += 1

        if (i + 1) % 100 == 0 or i == 0:
            running = 100.0 * correct / total
            print(f"  [{i+1:4d}/{len(dataset)}]  running acc_norm={running:.2f}")

    acc_norm = 100.0 * correct / total if total else 0.0
    print(f"  FINAL HELLASWAG acc_norm: {acc_norm:.2f}")
    return {
        "task": "hellaswag",
        "metric": "acc_norm",
        "score": acc_norm,
        "n": total,
        "higher_is_better": True,
    }


def evaluate_winogrande(
    model,
    tokenizer,
    device: str,
    max_ctx: int,
    max_samples: Optional[int],
    split: str,
) -> dict:
    print(f"\n{'─' * 60}")
    print("  Task   : WINOGRANDE")
    print("  Metric : acc")
    print(f"  Dataset: allenai/winogrande (winogrande_xl) [{split}]")
    print(f"{'─' * 60}")

    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    print(f"  Samples: {len(dataset)}")

    correct = 0
    total = 0
    for i, ex in enumerate(dataset):
        sentence = str(ex["sentence"])
        if "_" not in sentence:
            continue
        prefix, suffix = sentence.split("_", 1)
        choices = [f"{ex['option1']}{suffix}", f"{ex['option2']}{suffix}"]
        answer = str(ex.get("answer", "")).strip()
        if answer not in {"1", "2"}:
            continue
        label = int(answer) - 1

        scores = []
        for c in choices:
            ll, _, _ = continuation_loglikelihood(
                model=model,
                tokenizer=tokenizer,
                context=prefix,
                continuation=c,
                device=device,
                max_ctx=max_ctx,
            )
            scores.append(ll)

        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == label)
        total += 1

        if (i + 1) % 100 == 0 or i == 0:
            running = 100.0 * correct / total if total else 0.0
            print(f"  [{i+1:4d}/{len(dataset)}]  running acc={running:.2f}")

    acc = 100.0 * correct / total if total else 0.0
    print(f"  FINAL WINOGRANDE acc: {acc:.2f}")
    return {
        "task": "winogrande",
        "metric": "acc",
        "score": acc,
        "n": total,
        "higher_is_better": True,
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GLA-family checkpoints on WikiText/LAMBADA/PIQA/HellaSwag/WinoGrande"
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Step directory, experiment path, or converted .pt checkpoint file",
    )
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (optional)")
    parser.add_argument(
        "--model_ref",
        default=SCRIPT_DIR,
        help="Path/repo containing config+tokenizer (defaults to script directory)",
    )
    parser.add_argument(
        "--tmp_dir",
        default=None,
        help="Directory for temporary checkpoint.pt during DCP conversion (unused for .pt inputs)",
    )
    parser.add_argument(
        "--tasks",
        default="wikitext,lambada,piqa,hellaswag,winogrande",
        help="Comma-separated tasks from: wikitext,lambada,piqa,hellaswag,winogrande",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap per task split")
    parser.add_argument("--max_ctx", type=int, default=2048, help="Maximum context window")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding-window stride for perplexity tasks (default: max_ctx // 2)",
    )
    parser.add_argument(
        "--max_wikitext_tokens",
        type=int,
        default=None,
        help="Optional hard token cap for WikiText (useful for quick smoke tests)",
    )
    parser.add_argument("--wikitext_split", default="test")
    parser.add_argument("--lambada_split", default="test")
    parser.add_argument("--piqa_split", default="validation")
    parser.add_argument("--hellaswag_split", default="validation")
    parser.add_argument("--winogrande_split", default="validation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.max_ctx < 2:
        raise ValueError(f"--max_ctx must be >= 2, got {args.max_ctx}")

    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    supported = {"wikitext", "lambada", "piqa", "hellaswag", "winogrande"}
    for t in tasks:
        if t not in supported:
            raise ValueError(f"Unknown task '{t}'. Choose from: {sorted(supported)}")

    stride = args.stride if args.stride is not None else max(args.max_ctx // 2, 1)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path, args.step)
    model, tokenizer, model_type = load_model_and_tokenizer_from_dcp(
        model_ref=args.model_ref,
        checkpoint_path=checkpoint_path,
        device=args.device,
        tmp_dir=args.tmp_dir,
    )

    print(f"\nDevice: {args.device}  |  Tasks: {tasks}  |  Max ctx: {args.max_ctx}  |  Stride: {stride}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} per task")

    results = []
    for task in tasks:
        if task == "wikitext":
            results.append(
                evaluate_wikitext(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    max_ctx=args.max_ctx,
                    stride=stride,
                    max_samples=args.max_samples,
                    max_wikitext_tokens=args.max_wikitext_tokens,
                    split=args.wikitext_split,
                )
            )
        elif task == "lambada":
            results.extend(
                evaluate_lambada(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    max_ctx=args.max_ctx,
                    max_samples=args.max_samples,
                    split=args.lambada_split,
                )
            )
        elif task == "piqa":
            results.append(
                evaluate_piqa(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    max_ctx=args.max_ctx,
                    max_samples=args.max_samples,
                    split=args.piqa_split,
                )
            )
        elif task == "hellaswag":
            results.append(
                evaluate_hellaswag(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    max_ctx=args.max_ctx,
                    max_samples=args.max_samples,
                    split=args.hellaswag_split,
                )
            )
        elif task == "winogrande":
            results.append(
                evaluate_winogrande(
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    max_ctx=args.max_ctx,
                    max_samples=args.max_samples,
                    split=args.winogrande_split,
                )
            )

    avg_score_metrics = [r["score"] for r in results if r["higher_is_better"]]
    avg_ppl_metrics = [r["score"] for r in results if not r["higher_is_better"]]
    avg_score = sum(avg_score_metrics) / len(avg_score_metrics) if avg_score_metrics else 0.0
    avg_ppl = sum(avg_ppl_metrics) / len(avg_ppl_metrics) if avg_ppl_metrics else float("nan")

    print("\n" + "=" * 76)
    print("  RESULTS SUMMARY")
    print("=" * 76)
    print(f"  {'Model':<26} {'Task':<12} {'Metric':<10} {'Score':>12} {'N':>10}")
    print("-" * 76)
    model_ref_name = Path(args.model_ref).name
    model_name = f"{model_type}:{model_ref_name}" if model_ref_name else model_type
    for r in results:
        if r["metric"] == "ppl":
            score_text = f"{r['score']:.4f}"
        else:
            score_text = f"{r['score']:.2f}"
        print(f"  {model_name:<26} {r['task']:<12} {r['metric']:<10} {score_text:>12} {r['n']:>10,}")
    print("-" * 76)
    print(f"  {'Average Score (acc/acc_norm)':<50} {avg_score:>12.2f}")
    if avg_ppl == avg_ppl:
        print(f"  {'Average PPL (ppl metrics)':<50} {avg_ppl:>12.4f}")
    print("=" * 76)

    one_liner = " | ".join(f"{r['task']}:{r['metric']}={r['score']:.4f}" if r["metric"] == "ppl" else
                           f"{r['task']}:{r['metric']}={r['score']:.2f}" for r in results)
    print(f"\nOne-liner: {model_name} >> {one_liner} >> AvgScore={avg_score:.2f}\n")


if __name__ == "__main__":
    main()
