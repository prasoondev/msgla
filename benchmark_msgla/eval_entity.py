"""
eval_entity_tracking.py — Multi-entity tracking probe for GLA-family checkpoints.

Tests whether a model's recurrent state can simultaneously track multiple named entities
that update at different rates across a long context window. Both models were trained
at 2k context — queries are placed in the OOD region (>2k tokens) to stress long-range
memory under out-of-distribution conditions.

Architecture hypothesis
-----------------------
MS-GLA with [1, 2, 4] scales has three branches:
  - Scale 1 (fine)   : sensitive to local updates, ~50-token effective horizon
  - Scale 2 (mid)    : aggregates over spans, ~200-token effective horizon
  - Scale 4 (coarse) : temporally pooled, ~800-token effective horizon

GLA has a single scale-1 branch that must track all entity types with the same
state. Prediction: MS-GLA outperforms GLA most on the slowest entities, moderately
on mid-rate ones, and least on the fastest one.

Entity design
-------------
  Entity A "score"  : integer 1-99,  updates every FREQ_A tokens (~50)   [fine branch]
  Entity B "rank"   : integer 1-20,  updates every FREQ_B tokens (~200)  [mid branch]
  Entity C "code"   : word,          updates every FREQ_C tokens (~800)   [coarse branch]
  Entity D "status" : word,          updates every FREQ_D tokens (~1600)  [extra coarse]
  Entity E "signal" : word,          updates every FREQ_E tokens (~3200)  [extra coarse]

Sequence structure
------------------
  [intro]   one sentence per entity with the current value
  [body]    filler text with entity updates injected at their respective frequencies
  [queries] placed at query_gap tokens from start (OOD region):
            "What is Alice's score?" -> score NLL of answer
            "What is Bob's rank?"    -> score NLL of answer
            "What is Carol's code?"  -> score NLL of answer
            "What is Dylan's status?" -> score NLL of answer
            "What is Eve's signal?"   -> score NLL of answer

Metric
------
  retention_NLL = NLL(wrong_answer) - NLL(correct_answer)
    > 0  : model assigns lower NLL to correct answer -> it knows the current value
    ~ 0  : model can't distinguish correct from wrong -> no memory
    < 0  : model actively prefers wrong answer -> interference

  exact_match_% = top-1 logit prediction == correct answer token

Examples
--------
    # Smoke test:
    python eval_entity_tracking.py \\
        --checkpoint_path checkpoint/step-68664/tmp/68k.pt \\
        --n_probes 50 --query_gaps 2048,4096

    # Full run:
    python eval_entity_tracking.py \\
        --checkpoint_path checkpoint/step-68664/tmp/68k.pt \\
        --n_probes 200 --query_gaps 2048,3072,4096,6144,8192 \\
        --output_csv results/msgla_entity.csv

    # GLA baseline:
    python eval_entity_tracking.py \\
        --checkpoint_path /path/to/gla.pt --model_ref /path/to/gla-ref \\
        --n_probes 200 --query_gaps 2048,3072,4096,6144,8192 \\
        --output_csv results/gla_entity.csv
"""

import argparse
import csv
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.serialization
from datasets import load_dataset
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoTokenizer

# ── make local model packages importable ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
FLAME_ROOT = os.path.join(REPO_ROOT, "flame")
FLA_ROOT   = os.path.join(REPO_ROOT, "3rd_party", "flash-linear-attention")
for _p in (FLAME_ROOT, FLA_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from custom_models.ms_gla import MSGLAConfig, MSGLAForCausalLM  # noqa: E402
from fla.models.gla import GLAConfig, GLAForCausalLM             # noqa: E402

MODEL_REGISTRY = {
    "gla":    (GLAConfig,   GLAForCausalLM),
    "ms_gla": (MSGLAConfig, MSGLAForCausalLM),
}

# ================================================================
# Entity configuration — frequencies map to MS-GLA branch scales
# ================================================================

FREQ_A = 50      # fast       — scale-1 branch handles this, both models similar
FREQ_B = 200     # mid        — scale-2 branch specialises here, MS-GLA advantage
FREQ_C = 800     # coarse     — scale-4 branch specialises here
FREQ_D = 1600    # coarser    — extra long-range stress test
FREQ_E = 3200    # coarsest   — extra long-range stress test

SCORE_VALUES = list(range(1, 100))       # Entity A: integer score
RANK_VALUES  = list(range(1, 21))        # Entity B: integer rank
CODE_VALUES  = [                         # Entity C: single-token code words
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON",
    "ZETA", "THETA", "KAPPA", "SIGMA", "OMEGA",
    "PHOENIX", "NEXUS", "COBALT", "VORTEX", "ZENITH",
    "AXIOM", "HELIOS", "CYGNUS", "BOREAL", "STRATUM",
]
STATUS_VALUES = [                       # Entity D: status words
    "ACTIVE", "IDLE", "READY", "HIDDEN", "STABLE",
    "PRIME", "LUCID", "STEADY", "AMBER", "SCARLET",
    "SILVER", "GOLD", "FROST", "EMBER", "RAPID",
    "MELLOW", "BRISK", "SOLID", "CLEAR", "QUIET",
]
SIGNAL_VALUES = [                       # Entity E: signal words
    "AURORA", "PULSAR", "QUASAR", "RADAR", "BEACON",
    "VECTOR", "ORBIT", "NOVA", "COMET", "SOLAR",
    "LUNAR", "COSMOS", "PRISM", "SONIC", "IONIC",
    "ATLAS", "LYRIC", "MATRIX", "NEBULA", "APEX",
]

ENTITY_SPECS = [
    {
        "key": "A",
        "subject": "Alice",
        "attribute": "score",
        "freq": FREQ_A,
        "values": SCORE_VALUES,
        "band": "fast",
    },
    {
        "key": "B",
        "subject": "Bob",
        "attribute": "rank",
        "freq": FREQ_B,
        "values": RANK_VALUES,
        "band": "mid",
    },
    {
        "key": "C",
        "subject": "Carol",
        "attribute": "code",
        "freq": FREQ_C,
        "values": CODE_VALUES,
        "band": "coarse",
    },
    {
        "key": "D",
        "subject": "Dylan",
        "attribute": "status",
        "freq": FREQ_D,
        "values": STATUS_VALUES,
        "band": "coarser",
    },
    {
        "key": "E",
        "subject": "Eve",
        "attribute": "signal",
        "freq": FREQ_E,
        "values": SIGNAL_VALUES,
        "band": "coarsest",
    },
]


# ================================================================
# Checkpoint loading  (mirrors eval_dcp.py exactly)
# ================================================================

def resolve_checkpoint_path(checkpoint_path: str, step: int | None) -> str:
    base = Path(checkpoint_path)
    if base.is_file():
        if base.suffix != ".pt":
            raise ValueError(f"Expected a .pt checkpoint file, got: {base}")
        if step is not None:
            raise ValueError("Do not pass --step when --checkpoint_path points to a .pt file.")
        return str(base)
    if step is not None:
        for cand in [base / "checkpoint" / f"step-{step}", base / f"step-{step}"]:
            if cand.exists():
                return str(cand)
        raise FileNotFoundError(f"Could not find step-{step} under {base}.")
    if (base / ".metadata").exists() and any(base.glob("*.distcp")):
        return str(base)
    latest = None
    checkpoint_root = base / "checkpoint"
    if checkpoint_root.exists():
        for p in checkpoint_root.glob("step-*"):
            try:
                n = int(p.name.split("step-")[-1])
            except ValueError:
                continue
            if latest is None or n > latest[0]:
                latest = (n, p)
    if latest:
        print(f"No --step provided. Using latest checkpoint: {latest[1]}")
        return str(latest[1])
    raise FileNotFoundError(f"Could not infer DCP checkpoint directory from {base}.")


def load_state_from_checkpoint(checkpoint_path: str, tmp_dir: str | None) -> dict:
    if checkpoint_path.endswith(".pt"):
        print(f"Loading converted checkpoint: {checkpoint_path}")
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        return torch.load(checkpoint_path, map_location="cpu")
    print(f"Loading DCP checkpoint: {checkpoint_path}")
    with tempfile.TemporaryDirectory(dir=tmp_dir) as workdir:
        pt_path = os.path.join(workdir, "checkpoint.pt")
        dcp_to_torch_save(checkpoint_path, pt_path)
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        return torch.load(pt_path, map_location="cpu")


def load_model_and_tokenizer(
    model_ref: str, checkpoint_path: str, device: str, tmp_dir: str | None
):
    print(f"Loading tokenizer from {model_ref} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model config from {model_ref} ...")
    auto_config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    model_type  = getattr(auto_config, "model_type", None)
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_type {model_type!r} in {model_ref}. Supported: {supported}."
        )
    config_cls, model_cls = MODEL_REGISTRY[model_type]
    config = config_cls.from_pretrained(model_ref)
    model  = model_cls(config)
    print(f"Instantiated model_type={model_type!r}.")

    state = load_state_from_checkpoint(checkpoint_path, tmp_dir)
    if "model" not in state:
        raise KeyError("Checkpoint missing 'model' key.")
    model.load_state_dict(state["model"])

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {n / 1e6:.1f}M params on {device}.")
    return model, tokenizer, model_type


# ================================================================
# Filler corpus
# ================================================================

def load_filler_pool(tokenizer, min_tokens: int = 100_000) -> list[int]:
    print("Loading filler corpus (WikiText-103 test) ...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    eos_id = tokenizer.eos_token_id or 0
    flat: list[int] = []
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        flat.extend(ids)
        flat.append(eos_id)
        if len(flat) >= min_tokens * 4:
            break
    print(f"  Filler pool: {len(flat):,} tokens.")
    return flat


# ================================================================
# Sequence builder
# ================================================================

def enc(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def answer_ids(tokenizer, value) -> list[int]:
    """Tokenise " <value>" — leading space for natural subword tokenisation."""
    return tokenizer.encode(f" {value}", add_special_tokens=False)


def entity_state_text(spec: dict, value) -> str:
    return f"{spec['subject']}'s {spec['attribute']} is {value}. "


def entity_update_text(spec: dict, value) -> str:
    return f"{spec['subject']}'s {spec['attribute']} is updated to {value}. "


def entity_query_text(spec: dict) -> str:
    return f"What is {spec['subject']}'s {spec['attribute']}? "


def build_sequence(
    tokenizer,
    filler: list[int],
    filler_offset: int,
    initial_values: dict[str, object],
    query_gap: int,
    rng: random.Random,
) -> dict:
    """
    Build a single entity-tracking sequence.

    Returns a dict containing:
      tokens          : full token ID list
      final_<entity>  : ground-truth final values at query time
      <entity>_start  : token positions of each answer in tokens
      <entity>_end    : token positions of each answer in tokens
      <entity>_ids    : token IDs of each correct answer
    """
    fil     = filler[filler_offset:]
    fil_ptr = [0]   # mutable so inner function can update it

    def next_filler(n: int) -> list[int]:
        start = fil_ptr[0] % max(1, len(fil) - n - 1)
        chunk = fil[start : start + n]
        fil_ptr[0] = start + n
        if len(chunk) < n:
            extra = fil[:n - len(chunk)]
            chunk = chunk + extra
        return list(chunk)

    tokens: list[int] = []

    # Intro
    for spec in ENTITY_SPECS:
        tokens.extend(enc(tokenizer, entity_state_text(spec, initial_values[spec["key"]])))

    current_values = {spec["key"]: initial_values[spec["key"]] for spec in ENTITY_SPECS}

    # Body: inject updates at their frequencies, fill gaps with corpus
    body_target  = query_gap
    tok_in_body  = 0
    next_due = {spec["key"]: spec["freq"] for spec in ENTITY_SPECS}

    while tok_in_body < body_target:
        # Advance to the next event (or end of body)
        step = min(*next_due.values(), body_target - tok_in_body)
        step = max(step, 1)

        chunk = next_filler(step)
        tokens.extend(chunk)
        tok_in_body += len(chunk)
        for key in next_due:
            next_due[key] -= len(chunk)

        if tok_in_body >= body_target:
            break

        # Fire updates that are due
        for spec in ENTITY_SPECS:
            key = spec["key"]
            if next_due[key] <= 0:
                cur_val = current_values[key]
                new_val = rng.choice([v for v in spec["values"] if v != cur_val])
                upd = enc(tokenizer, entity_update_text(spec, new_val))
                tokens.extend(upd)
                tok_in_body += len(upd)
                current_values[key] = new_val
                next_due[key] = spec["freq"]

    # Queries — append all entity questions, record where each answer sits
    def append_query(query_text: str, value) -> tuple[int, int, list[int]]:
        tokens.extend(enc(tokenizer, query_text))
        ids   = answer_ids(tokenizer, value)
        start = len(tokens)
        tokens.extend(ids)
        end   = len(tokens)
        tokens.extend(enc(tokenizer, " "))   # separator
        return start, end, ids

    result = {"tokens": tokens}
    for spec in ENTITY_SPECS:
        key = spec["key"]
        lower = key.lower()
        start, end, ids = append_query(entity_query_text(spec), current_values[key])
        result[f"final_{lower}"] = current_values[key]
        result[f"{lower}_start"] = start
        result[f"{lower}_end"] = end
        result[f"{lower}_ids"] = ids

    return result


# ================================================================
# NLL scoring — single forward pass
# ================================================================

@torch.inference_mode()
def forward_logits(
    model,
    tokens: list[int],
    device: str,
) -> torch.Tensor:
    ids_t     = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    attn_mask = torch.ones_like(ids_t, device=device)

    outputs = model(
        input_ids=ids_t,
        attention_mask=attn_mask,
        use_cache=True,
        past_key_values=None,
    )
    return outputs.logits[0]   # [T, vocab]


def score_spans_from_logits(
    logits: torch.Tensor,
    tokens: list[int],
    score_spans: list[tuple[int, int, list[int]]],
) -> list[float]:
    """
    For each (start, end, ids) in score_spans, return mean NLL of ids at
    positions start..end-1. ids may differ from tokens[start:end] — this lets
    us score wrong answers at the same positions.
    """
    results: list[float] = []
    for start, end, score_ids in score_spans:
        if not score_ids or start == 0:
            results.append(float("nan"))
            continue
        total = 0.0
        for i, tok in enumerate(score_ids):
            pos = start + i
            if pos >= len(tokens):
                break
            lp = F.log_softmax(logits[pos - 1], dim=-1)
            total += -lp[tok].item()
        results.append(total / len(score_ids))

    return results


def top1_from_logits(logits: torch.Tensor, pred_pos: int) -> int:
    """Return argmax prediction at pred_pos (i.e. what follows tokens[:pred_pos])."""
    return int(logits[pred_pos - 1].argmax().item())


# ================================================================
# Core evaluation loop
# ================================================================

def evaluate_gap(
    query_gap: int,
    model,
    tokenizer,
    filler: list[int],
    n_probes: int,
    device: str,
    seed: int,
    verbose: bool,
) -> dict:
    rng = random.Random(seed)

    ret = {spec["key"]: [] for spec in ENTITY_SPECS}
    em  = {spec["key"]: [] for spec in ENTITY_SPECS}

    print(f"\n{'─' * 68}")
    print(f"  Query gap : {query_gap} tokens")
    freq_text = "  ".join(f"{spec['key']}:{spec['freq']}" for spec in ENTITY_SPECS)
    print(f"  Frequencies — {freq_text}")
    print(f"{'─' * 68}")

    t_start = time.perf_counter()

    for i in range(n_probes):
        initial_values = {
            spec["key"]: rng.choice(spec["values"])
            for spec in ENTITY_SPECS
        }
        foff = rng.randint(0, max(1, len(filler) - query_gap - 2000))

        seq = build_sequence(
            tokenizer, filler, foff,
            initial_values,
            query_gap, rng,
        )
        tokens = seq["tokens"]

        # Correct answer spans
        correct_spans = []
        for spec in ENTITY_SPECS:
            lower = spec["key"].lower()
            correct_spans.append(
                (seq[f"{lower}_start"], seq[f"{lower}_end"], seq[f"{lower}_ids"])
            )

        # Wrong answer spans — same positions, wrong value IDs
        wrong_spans = []
        for spec in ENTITY_SPECS:
            lower = spec["key"].lower()
            wrong_value = rng.choice(
                [v for v in spec["values"] if v != seq[f"final_{lower}"]]
            )
            wrong_spans.append(
                (
                    seq[f"{lower}_start"],
                    seq[f"{lower}_end"],
                    answer_ids(tokenizer, wrong_value),
                )
            )

        logits = forward_logits(model, tokens, device)
        correct_nlls = score_spans_from_logits(logits, tokens, correct_spans)
        wrong_nlls   = score_spans_from_logits(logits, tokens, wrong_spans)

        # Accumulate
        for idx, spec in enumerate(ENTITY_SPECS):
            key = spec["key"]
            lower = key.lower()
            nlls_c = correct_nlls[idx]
            nlls_w = wrong_nlls[idx]
            if not (math.isnan(nlls_c) or math.isnan(nlls_w)):
                ret[key].append(nlls_w - nlls_c)

            pred = top1_from_logits(logits, seq[f"{lower}_start"])
            em[key].append(int(pred == seq[f"{lower}_ids"][0]) if seq[f"{lower}_ids"] else 0)

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_start
            def _m(lst): return sum(lst)/len(lst) if lst else float("nan")
            ret_text = "  ".join(f"ret {k}={_m(ret[k]):.3f}" for k in ret)
            em_text = "  ".join(f"EM {k}={_m(em[k])*100:.1f}%" for k in em)
            print(
                f"  [{i+1:4d}/{n_probes}]  "
                f"{ret_text}  "
                f"{em_text}  "
                f"t={elapsed:.1f}s",
                flush=True,
            )

    def _mean(lst): return sum(lst)/len(lst) if lst else float("nan")
    def _pct(lst):  return _mean(lst) * 100

    result = {"query_gap": query_gap, "n_probes": n_probes}
    for spec in ENTITY_SPECS:
        key = spec["key"]
        result[f"ret_{key}"] = _mean(ret[key])
        result[f"em_{key}"] = _pct(em[key])

    label_width = 16
    print(f"\n  {'Entity':<{label_width}}  {'Freq':>6}  {'Retention NLL':>14}  {'Exact match':>12}")
    print(f"  {'-'*label_width}  {'-'*6}  {'-'*14}  {'-'*12}")
    for spec in ENTITY_SPECS:
        key = spec["key"]
        ret_val = result[f"ret_{key}"]
        em_val = result[f"em_{key}"]
        label = f"{key} ({spec['band']})"
        flag = "  **" if ret_val > 0.5 else ("  *" if ret_val > 0.1 else "")
        print(
            f"  {label:<{label_width}}  "
            f"{spec['freq']:>6}  {ret_val:>14.4f}{flag}  {em_val:>11.1f}%"
        )

    return result


# ================================================================
# Summary
# ================================================================

def print_summary(results: list[dict], model_type: str) -> None:
    print("\n\n" + "=" * 80)
    print(f"  MULTI-ENTITY TRACKING  —  {model_type.upper()}")
    print("=" * 80)
    print("  retention_NLL = NLL(wrong) - NLL(correct)  "
          "[>0 = model knows correct value]")
    freq_line = "  |  ".join(
        f"{spec['key']} every {spec['freq']} tok" for spec in ENTITY_SPECS
    )
    print(f"  {freq_line}")
    print()

    gaps = [r["query_gap"] for r in results]
    row_label_width = 20
    col  = 10

    def _row(label, key):
        vals = "  ".join(f"{r[key]:>{col}.4f}" for r in results)
        print(f"  {label:<{row_label_width}}  {vals}")

    def _pct_row(label, key):
        vals = "  ".join(f"{r[key]:>{col-1}.1f}%" for r in results)
        print(f"  {label:<{row_label_width}}  {vals}")

    header = f"  {'':{row_label_width}}  " + "  ".join(f"{'gap='+str(g):>{col}}" for g in gaps)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for spec in ENTITY_SPECS:
        _row(f"ret {spec['key']} ({spec['band']})", f"ret_{spec['key']}")
    print()
    print(header)
    print("  " + "─" * (len(header) - 2))
    for spec in ENTITY_SPECS:
        _pct_row(f"EM% {spec['key']} ({spec['band']})", f"em_{spec['key']}")

    print("=" * 80)
    print("\nComparing GLA vs MS-GLA:")
    print("  * ret_A should remain the easiest local-memory case.")
    print("  * ret_B should show a moderate branch-specialization advantage.")
    print("  * ret_C/ret_D/ret_E are the main coarse-memory stress tests.")
    print("  * If MS-GLA stays >0 on D/E at long gaps while GLA collapses toward 0,")
    print("    that is direct evidence that multi-resolution decomposition helps.")


# ================================================================
# Export
# ================================================================

def save_csv(output_csv: str, results: list[dict], model_type: str) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_type", "query_gap", "n_probes"]
    for prefix in ("ret", "em"):
        for spec in ENTITY_SPECS:
            fields.append(f"{prefix}_{spec['key']}")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({"model_type": model_type, **r})
    print(f"\nResults saved to: {output_csv}")


def save_json(output_json: str, results: list[dict], model_type: str) -> None:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"model_type": model_type, "results": results}, f, indent=2)
    print(f"Results also saved to: {output_json}")


# ================================================================
# Main
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-entity tracking probe for GLA-family checkpoints."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--step",      type=int,  default=None)
    parser.add_argument("--model_ref", default=SCRIPT_DIR)
    parser.add_argument("--tmp_dir",   default=None)
    parser.add_argument(
        "--query_gaps", default="2048,3072,4096,6144,8192",
        help="Token distances from context start to query. "
             "All should be >2048 to sit in OOD region. Default: 2048,3072,4096,6144,8192"
    )
    parser.add_argument(
        "--n_probes", type=int, default=100,
        help="Probe pairs per gap. >=100 for stable results, >=200 for paper quality."
    )
    parser.add_argument(
        "--min_filler_tokens", type=int, default=100_000,
        help="Minimum tokens to load as filler. Must be >> max(query_gaps)."
    )
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv",  default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-entity stats every 20 probes.")
    args = parser.parse_args()

    query_gaps = [int(x.strip()) for x in args.query_gaps.split(",")]

    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path, args.step)
    model, tokenizer, model_type = load_model_and_tokenizer(
        model_ref=args.model_ref,
        checkpoint_path=checkpoint_path,
        device=args.device,
        tmp_dir=args.tmp_dir,
    )
    print(f"Model type  : {model_type}")
    print(f"Query gaps  : {query_gaps}")
    print(f"n_probes    : {args.n_probes} per gap")
    freq_text = "  ".join(f"{spec['key']}={spec['freq']}" for spec in ENTITY_SPECS)
    print(f"Freqs       : {freq_text}")

    filler = load_filler_pool(
        tokenizer,
        min_tokens=max(args.min_filler_tokens, max(query_gaps) * 3),
    )

    results: list[dict] = []
    for gap in query_gaps:
        r = evaluate_gap(
            query_gap=gap,
            model=model,
            tokenizer=tokenizer,
            filler=filler,
            n_probes=args.n_probes,
            device=args.device,
            seed=args.seed + gap,
            verbose=args.verbose,
        )
        results.append(r)

    print_summary(results, model_type)

    if args.output_csv:
        save_csv(args.output_csv, results, model_type)
    if args.output_json:
        save_json(args.output_json, results, model_type)


if __name__ == "__main__":
    main()
