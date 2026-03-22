"""
eval_sfs.py — Zero-shot recall evaluation for MS-GLA checkpoints in DCP format.
Evaluates on SWDE (Accuracy), FDA (F1), and SQUAD (F1).

Examples:
    python eval_sfs.py --checkpoint_path /path/to/exp/msgla-340M --step 10800 --model_ref /path/to/exp/msgla-340M
    python eval_sfs.py --checkpoint_path /path/to/checkpoint/step-10800 --model_ref /path/to/exp/msgla-340M --tasks squad --max_samples 10
"""

import argparse
import csv
import io
import json
import os
import re
import string
import sys
import tempfile
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

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

# Importing the package triggers AutoConfig / AutoModelForCausalLM registration
from custom_models.ms_gla import MSGLAConfig, MSGLAForCausalLM  # noqa: E402


# ================================================================
# Checkpoint loading
# ================================================================

def resolve_checkpoint_path(checkpoint_path: str, step: int | None) -> str:
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


def load_state_from_checkpoint(checkpoint_path: str, tmp_dir: str | None) -> dict:
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
    tmp_dir: str | None,
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
        raise KeyError("Checkpoint does not contain 'model' in top-level state dict.")
    model.load_state_dict(state["model"])

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {n/1e6:.1f}M params on {device}.")
    return model, tokenizer


# ================================================================
# Generation
# ================================================================

@torch.inference_mode()
def generate_hf(model, tokenizer, prompt: str, max_new_tokens=32, device="cuda", max_ctx=2048) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded.split("\n")[0].strip()


def _sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _apply_repetition_penalty(scores: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return scores
    adjusted = scores.clone()
    for batch_idx in range(scores.shape[0]):
        token_ids = torch.unique(generated_ids[batch_idx])
        token_scores = adjusted[batch_idx, token_ids]
        adjusted[batch_idx, token_ids] = torch.where(
            token_scores < 0,
            token_scores * penalty,
            token_scores / penalty,
        )
    return adjusted


@torch.inference_mode()
def generate_cached(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens=32,
    device="cuda",
    max_ctx=2048,
    repetition_penalty=1.1,
    return_stats=False,
):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = input_ids.shape[-1]

    _sync_if_needed(device)
    t0 = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=None,
    )
    _sync_if_needed(device)
    prefill_time = time.perf_counter() - t0

    generated = input_ids.clone()
    past_key_values = outputs.past_key_values
    next_scores = outputs.logits[:, -1, :]
    decode_step_times = []

    for _ in range(max_new_tokens):
        adjusted_scores = _apply_repetition_penalty(next_scores, generated, repetition_penalty)
        next_token = adjusted_scores.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        if tokenizer.eos_token_id is not None and bool((next_token == tokenizer.eos_token_id).all().item()):
            break

        _sync_if_needed(device)
        t0 = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        _sync_if_needed(device)
        decode_step_times.append(time.perf_counter() - t0)

        past_key_values = outputs.past_key_values
        next_scores = outputs.logits[:, -1, :]

    new_tokens = generated[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).split("\n")[0].strip()
    stats = {
        "prompt_tokens": int(input_len),
        "generated_tokens": int(new_tokens.shape[0]),
        "prefill_time_s": prefill_time,
        "decode_total_time_s": float(sum(decode_step_times)),
        "decode_step_avg_time_s": float(sum(decode_step_times) / len(decode_step_times)) if decode_step_times else 0.0,
    }
    if return_stats:
        return decoded, stats
    return decoded


@torch.inference_mode()
def warmup_generate(model, tokenizer, device="cuda", max_ctx=2048) -> None:
    warmup_prompt = "Warmup prompt."
    inputs = tokenizer(warmup_prompt, return_tensors="pt", truncation=True, max_length=max_ctx)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )


# ================================================================
# Metrics
# ================================================================

def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p)
    rec = n / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def contains_match(pred: str, gold: str) -> float:
    """Accuracy metric for FDA: check (case-insensitively) if the gold value
    is contained within the generation, as described in Arora et al. (2023b)."""
    return float(normalize(gold) in normalize(pred))


# ================================================================
# Prompt builder
# ================================================================

def build_prompt(example: dict) -> tuple[str, str]:
    return example["text"], str(example["value"])


# ================================================================
# Task config
# ================================================================

TASK_CONFIG = {
    "swde": {
        "hf_name": "hazyresearch/based-swde",
        "hf_split": "validation",
        "metric_fn": token_f1,
        "metric_name": "Accuracy",
    },
    "fda": {
        "hf_name": "hazyresearch/based-fda",
        "hf_split": "validation",
        "metric_fn": token_f1,
        "metric_name": "F1",
    },
    "squad": {
        "hf_name": "hazyresearch/based-squad",
        "hf_split": "validation",
        "metric_fn": token_f1,
        "metric_name": "F1",
    },
}


# ================================================================
# Evaluation loop
# ================================================================

def evaluate_task(
    task_name,
    model,
    tokenizer,
    device,
    max_samples,
    max_new_tokens,
    max_ctx,
    profile_generation,
    generation_backend,
):
    cfg = TASK_CONFIG[task_name]
    print(f"\n{'─' * 60}")
    print(f"  Task   : {task_name.upper()}")
    print(f"  Dataset: {cfg['hf_name']}  [{cfg['hf_split']}]")
    print(f"{'─' * 60}")

    dataset = load_dataset(cfg["hf_name"], split=cfg["hf_split"])
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    print(f"  Samples: {len(dataset)}")

    scores = []
    for i, example in enumerate(dataset):
        prompt, gold = build_prompt(example)
        prompt_tokens = len(tokenizer(prompt, truncation=True, max_length=max_ctx)["input_ids"])
        if i == 0:
            print(
                f"  First sample prompt tokens: {prompt_tokens}  |  max_new_tokens: {max_new_tokens}",
                flush=True,
            )
            print("  Starting first generation...", flush=True)
        t0 = time.perf_counter()
        profile_stats = None
        if profile_generation:
            pred, profile_stats = generate_cached(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                max_ctx=max_ctx,
                return_stats=True,
            )
        elif generation_backend == "hf":
            pred = generate_hf(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                max_ctx=max_ctx,
            )
        else:
            pred = generate_cached(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                device=device,
                max_ctx=max_ctx,
            )
        dt = time.perf_counter() - t0
        score = cfg["metric_fn"](pred, gold)
        scores.append(score)

        if (i + 1) % 10 == 0 or i == 0:
            running = sum(scores) / len(scores) * 100
            if i == 0:
                if profile_stats is not None:
                    print(
                        "  First generation breakdown: "
                        f"prefill={profile_stats['prefill_time_s']:.2f}s, "
                        f"decode_total={profile_stats['decode_total_time_s']:.2f}s, "
                        f"decode_avg_step={profile_stats['decode_step_avg_time_s']:.2f}s, "
                        f"generated_tokens={profile_stats['generated_tokens']}",
                        flush=True,
                    )
                print(f"  First generation time: {dt:.2f}s", flush=True)
            print(f"  [{i + 1:4d}/{len(dataset)}]  running {cfg['metric_name']}: {running:.2f}", flush=True)

        if i < 3:
            print(f"\n  --- Example {i}  (key: {example.get('key', '')!r}) ---")
            print(f"  GOLD : {gold!r}")
            print(f"  PRED : {pred!r}")
            print(f"  Score: {score:.3f}\n")

    final = sum(scores) / len(scores) * 100 if scores else 0.0
    print(f"\n  FINAL {task_name.upper()} {cfg['metric_name']}: {final:.2f}")
    return {"task": task_name, "metric": cfg["metric_name"], "score": final, "n": len(scores)}


# ================================================================
# Export
# ================================================================

def save_json(output_json: str, payload: dict) -> None:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON results to: {output_json}")


def save_csv(output_csv: str, results: list[dict], metadata: dict, summary: dict) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_name",
        "model_type",
        "checkpoint_path",
        "model_ref",
        "task_list",
        "device",
        "max_ctx",
        "max_new_tokens",
        "max_samples",
        "generation_backend",
        "warmup",
        "profile_generation",
        "splits",
        "avg_score",
        "task",
        "metric",
        "score",
        "n",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model_name": metadata["model_name"],
                    "model_type": metadata["model_type"],
                    "checkpoint_path": metadata["checkpoint_path"],
                    "model_ref": metadata["model_ref"],
                    "task_list": ",".join(metadata["tasks"]),
                    "device": metadata["device"],
                    "max_ctx": metadata["max_ctx"],
                    "max_new_tokens": metadata["max_new_tokens"],
                    "max_samples": metadata["max_samples"],
                    "generation_backend": metadata["generation_backend"],
                    "warmup": metadata["warmup"],
                    "profile_generation": metadata["profile_generation"],
                    "splits": json.dumps(metadata["splits"], sort_keys=True),
                    "avg_score": summary["avg_score"],
                    **result,
                }
            )
    print(f"Saved CSV results to: {output_csv}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate MS-GLA checkpoint on SWDE/FDA/SQUAD")
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
    parser.add_argument("--tasks", default="swde,fda,squad")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_ctx", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", action="store_true", help="Run one tiny generation before evaluation.")
    parser.add_argument(
        "--profile_generation",
        action="store_true",
        help="Measure first-sample prefill and decode separately using explicit cached decoding.",
    )
    parser.add_argument(
        "--generation_backend",
        choices=("cached", "hf"),
        default="cached",
        help="Generation path to use. `cached` uses explicit cached decoding and is the recommended fast path.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save structured results as JSON",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save flat results as CSV",
    )
    args = parser.parse_args()

    tasks = [t.strip().lower() for t in args.tasks.split(",")]
    for t in tasks:
        if t not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{t}'. Choose from: {list(TASK_CONFIG.keys())}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path, args.step)
    model, tokenizer = load_model_and_tokenizer_from_dcp(
        model_ref=args.model_ref,
        checkpoint_path=checkpoint_path,
        device=args.device,
        tmp_dir=args.tmp_dir,
    )

    if args.warmup:
        print("Running warmup generation...", flush=True)
        t0 = time.perf_counter()
        warmup_generate(model, tokenizer, device=args.device, max_ctx=args.max_ctx)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        print(f"Warmup done in {time.perf_counter() - t0:.2f}s", flush=True)

    print(f"\nDevice: {args.device}  |  Tasks: {tasks}  |  Max ctx: {args.max_ctx}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} per task")

    results = []
    for task in tasks:
        r = evaluate_task(
            task,
            model,
            tokenizer,
            args.device,
            args.max_samples,
            args.max_new_tokens,
            args.max_ctx,
            args.profile_generation,
            args.generation_backend,
        )
        results.append(r)

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0

    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    print(f"  {'Model':<34} {'Task':<8} {'Metric':<10} {'Score':>6}")
    print("-" * 62)
    model_name = Path(args.model_ref).name or "msgla"
    for r in results:
        print(f"  {model_name:<34} {r['task'].upper():<8} {r['metric']:<10} {r['score']:>6.2f}")
    print("-" * 62)
    print(f"  {'Average':<52} {avg:>6.2f}")
    print("=" * 62)

    model_type = "ms_gla"
    payload = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "model_ref": args.model_ref,
        "device": args.device,
        "tasks": tasks,
        "max_ctx": args.max_ctx,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "generation_backend": args.generation_backend,
        "warmup": args.warmup,
        "profile_generation": args.profile_generation,
        "splits": {task_name: TASK_CONFIG[task_name]["hf_split"] for task_name in tasks},
        "summary": {"avg_score": avg},
        "results": results,
    }
    if args.output_json:
        save_json(args.output_json, payload)
    if args.output_csv:
        save_csv(
            args.output_csv,
            results,
            metadata={
                "model_name": model_name,
                "model_type": model_type,
                "checkpoint_path": checkpoint_path,
                "model_ref": args.model_ref,
                "tasks": tasks,
                "device": args.device,
                "max_ctx": args.max_ctx,
                "max_new_tokens": args.max_new_tokens,
                "max_samples": args.max_samples,
                "generation_backend": args.generation_backend,
                "warmup": args.warmup,
                "profile_generation": args.profile_generation,
                "splits": payload["splits"],
            },
            summary=payload["summary"],
        )


if __name__ == "__main__":
    main()
