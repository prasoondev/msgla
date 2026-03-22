"""
eval_entity_tracking.py — Multi-entity tracking probe for GLA-family checkpoints.

Tests whether a model's recurrent state can simultaneously track three named entities
that update at different rates across a long context window. Both models were trained
at 2k context — queries are placed in the OOD region (>2k tokens) to stress long-range
memory under out-of-distribution conditions.

Architecture hypothesis
-----------------------
MS-GLA with [1, 2, 4] scales has three branches:
  - Scale 1 (fine)   : sensitive to local updates, ~50-token effective horizon
  - Scale 2 (mid)    : aggregates over spans, ~200-token effective horizon
  - Scale 4 (coarse) : temporally pooled, ~800-token effective horizon

GLA has a single scale-1 branch that must track all three entity types with the
same state. Prediction: MS-GLA outperforms GLA most on Entity C (slowest updates),
moderately on B, least on A.

Entity design
-------------
  Entity A "score"  : integer 1-99,  updates every FREQ_A tokens (~50)   [fine branch]
  Entity B "rank"   : integer 1-20,  updates every FREQ_B tokens (~200)  [mid branch]
  Entity C "code"   : word,          updates every FREQ_C tokens (~800)   [coarse branch]

Sequence structure
------------------
  [intro]   "Alice's score is X. Bob's rank is Y. Carol's code is Z."
  [body]    filler text with entity updates injected at their respective frequencies
  [queries] placed at query_gap tokens from start (OOD region):
            "What is Alice's score?" -> score NLL of answer
            "What is Bob's rank?"    -> score NLL of answer
            "What is Carol's code?"  -> score NLL of answer

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

FREQ_A = 50     # fast  — scale-1 branch handles this, both models similar
FREQ_B = 200    # mid   — scale-2 branch specialises here, MS-GLA advantage
FREQ_C = 800    # slow  — scale-4 branch specialises here, largest MS-GLA advantage

SCORE_VALUES = list(range(1, 100))       # Entity A: integer score
RANK_VALUES  = list(range(1, 21))        # Entity B: integer rank
CODE_VALUES  = [                         # Entity C: single-token code words
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON",
    "ZETA", "THETA", "KAPPA", "SIGMA", "OMEGA",
    "PHOENIX", "NEXUS", "COBALT", "VORTEX", "ZENITH",
    "AXIOM", "HELIOS", "CYGNUS", "BOREAL", "STRATUM",
]

INTRO_TEMPLATE = "Alice's score is {score}. Bob's rank is {rank}. Carol's code is {code}. "
UPDATE_A = "Alice's score is updated to {val}. "
UPDATE_B = "Bob's rank is updated to {val}. "
UPDATE_C = "Carol's code is updated to {val}. "
QUERY_A  = "What is Alice's score? "
QUERY_B  = "What is Bob's rank? "
QUERY_C  = "What is Carol's code? "


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


def build_sequence(
    tokenizer,
    filler: list[int],
    filler_offset: int,
    init_a: int, init_b: int, init_c: str,
    query_gap: int,
    rng: random.Random,
) -> dict:
    """
    Build a single entity-tracking sequence.

    Returns a dict containing:
      tokens          : full token ID list
      final_a/b/c     : ground-truth final values at query time
      ans_start/end_* : token positions of each answer in tokens
      ans_ids_*       : token IDs of each correct answer
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
    tokens.extend(enc(tokenizer, INTRO_TEMPLATE.format(
        score=init_a, rank=init_b, code=init_c
    )))

    cur_a, cur_b, cur_c = init_a, init_b, init_c

    # Body: inject updates at their frequencies, fill gaps with corpus
    body_target  = query_gap
    tok_in_body  = 0
    next_a = FREQ_A
    next_b = FREQ_B
    next_c = FREQ_C

    while tok_in_body < body_target:
        # Advance to the next event (or end of body)
        step = min(next_a, next_b, next_c, body_target - tok_in_body)
        step = max(step, 1)

        chunk = next_filler(step)
        tokens.extend(chunk)
        tok_in_body += len(chunk)
        next_a -= len(chunk)
        next_b -= len(chunk)
        next_c -= len(chunk)

        if tok_in_body >= body_target:
            break

        # Fire updates that are due
        if next_a <= 0:
            new_a = rng.choice([v for v in SCORE_VALUES if v != cur_a])
            upd   = enc(tokenizer, UPDATE_A.format(val=new_a))
            tokens.extend(upd)
            tok_in_body += len(upd)
            cur_a  = new_a
            next_a = FREQ_A

        if next_b <= 0:
            new_b = rng.choice([v for v in RANK_VALUES if v != cur_b])
            upd   = enc(tokenizer, UPDATE_B.format(val=new_b))
            tokens.extend(upd)
            tok_in_body += len(upd)
            cur_b  = new_b
            next_b = FREQ_B

        if next_c <= 0:
            new_c = rng.choice([v for v in CODE_VALUES if v != cur_c])
            upd   = enc(tokenizer, UPDATE_C.format(val=new_c))
            tokens.extend(upd)
            tok_in_body += len(upd)
            cur_c  = new_c
            next_c = FREQ_C

    # Queries — append all three, record where each answer sits
    def append_query(query_text: str, value) -> tuple[int, int, list[int]]:
        tokens.extend(enc(tokenizer, query_text))
        ids   = answer_ids(tokenizer, value)
        start = len(tokens)
        tokens.extend(ids)
        end   = len(tokens)
        tokens.extend(enc(tokenizer, " "))   # separator
        return start, end, ids

    a_start, a_end, a_ids = append_query(QUERY_A, cur_a)
    b_start, b_end, b_ids = append_query(QUERY_B, cur_b)
    c_start, c_end, c_ids = append_query(QUERY_C, cur_c)

    return {
        "tokens":    tokens,
        "final_a":   cur_a, "final_b": cur_b, "final_c": cur_c,
        "a_start":   a_start, "a_end": a_end, "a_ids": a_ids,
        "b_start":   b_start, "b_end": b_end, "b_ids": b_ids,
        "c_start":   c_start, "c_end": c_end, "c_ids": c_ids,
    }


# ================================================================
# NLL scoring — single forward pass
# ================================================================

@torch.inference_mode()
def forward_nll(
    model,
    tokens: list[int],
    score_spans: list[tuple[int, int, list[int]]],
    device: str,
) -> list[float]:
    """
    Single forward pass. For each (start, end, ids) in score_spans,
    return mean NLL of ids at positions start..end-1.
    ids may differ from tokens[start:end] — this lets us score wrong answers.
    """
    ids_t     = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    attn_mask = torch.ones_like(ids_t, device=device)

    outputs = model(
        input_ids=ids_t,
        attention_mask=attn_mask,
        use_cache=True,
        past_key_values=None,
    )
    logits = outputs.logits[0]   # [T, vocab]

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


@torch.inference_mode()
def top1_at(
    model,
    tokens: list[int],
    pred_pos: int,
    device: str,
) -> int:
    """Return argmax prediction at pred_pos (i.e. what follows tokens[:pred_pos])."""
    ids_t     = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    attn_mask = torch.ones_like(ids_t, device=device)
    outputs   = model(
        input_ids=ids_t,
        attention_mask=attn_mask,
        use_cache=True,
        past_key_values=None,
    )
    return int(outputs.logits[0, pred_pos - 1].argmax().item())


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

    ret_a: list[float] = []
    ret_b: list[float] = []
    ret_c: list[float] = []
    em_a:  list[int]   = []
    em_b:  list[int]   = []
    em_c:  list[int]   = []

    print(f"\n{'─' * 68}")
    print(f"  Query gap : {query_gap} tokens")
    print(f"  Frequencies — A:{FREQ_A}  B:{FREQ_B}  C:{FREQ_C}")
    print(f"{'─' * 68}")

    t_start = time.perf_counter()

    for i in range(n_probes):
        init_a = rng.choice(SCORE_VALUES)
        init_b = rng.choice(RANK_VALUES)
        init_c = rng.choice(CODE_VALUES)
        foff   = rng.randint(0, max(1, len(filler) - query_gap - 2000))

        seq = build_sequence(
            tokenizer, filler, foff,
            init_a, init_b, init_c,
            query_gap, rng,
        )
        tokens = seq["tokens"]

        # Correct answer spans
        correct_spans = [
            (seq["a_start"], seq["a_end"], seq["a_ids"]),
            (seq["b_start"], seq["b_end"], seq["b_ids"]),
            (seq["c_start"], seq["c_end"], seq["c_ids"]),
        ]

        # Wrong answer spans — same positions, wrong value IDs
        wrong_a = rng.choice([v for v in SCORE_VALUES if v != seq["final_a"]])
        wrong_b = rng.choice([v for v in RANK_VALUES  if v != seq["final_b"]])
        wrong_c = rng.choice([v for v in CODE_VALUES  if v != seq["final_c"]])

        wrong_spans = [
            (seq["a_start"], seq["a_end"], answer_ids(tokenizer, wrong_a)),
            (seq["b_start"], seq["b_end"], answer_ids(tokenizer, wrong_b)),
            (seq["c_start"], seq["c_end"], answer_ids(tokenizer, wrong_c)),
        ]

        # Two forward passes: one for correct NLL, one for wrong NLL
        correct_nlls = forward_nll(model, tokens, correct_spans, device)
        wrong_nlls   = forward_nll(model, tokens, wrong_spans,   device)

        # Exact match: does top-1 at each query position == correct answer?
        t1_a = top1_at(model, tokens, seq["a_start"], device)
        t1_b = top1_at(model, tokens, seq["b_start"], device)
        t1_c = top1_at(model, tokens, seq["c_start"], device)

        # Accumulate
        for nlls_c, nlls_w, ret_list in [
            (correct_nlls[0], wrong_nlls[0], ret_a),
            (correct_nlls[1], wrong_nlls[1], ret_b),
            (correct_nlls[2], wrong_nlls[2], ret_c),
        ]:
            if not (math.isnan(nlls_c) or math.isnan(nlls_w)):
                ret_list.append(nlls_w - nlls_c)

        em_a.append(int(t1_a == seq["a_ids"][0]) if seq["a_ids"] else 0)
        em_b.append(int(t1_b == seq["b_ids"][0]) if seq["b_ids"] else 0)
        em_c.append(int(t1_c == seq["c_ids"][0]) if seq["c_ids"] else 0)

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_start
            def _m(lst): return sum(lst)/len(lst) if lst else float("nan")
            print(
                f"  [{i+1:4d}/{n_probes}]  "
                f"ret A={_m(ret_a):.3f}  B={_m(ret_b):.3f}  C={_m(ret_c):.3f}  "
                f"EM A={_m(em_a)*100:.1f}%  B={_m(em_b)*100:.1f}%  C={_m(em_c)*100:.1f}%  "
                f"t={elapsed:.1f}s",
                flush=True,
            )

    def _mean(lst): return sum(lst)/len(lst) if lst else float("nan")
    def _pct(lst):  return _mean(lst) * 100

    result = {
        "query_gap": query_gap,
        "n_probes":  n_probes,
        "ret_A":     _mean(ret_a),  "ret_B": _mean(ret_b),  "ret_C": _mean(ret_c),
        "em_A":      _pct(em_a),    "em_B":  _pct(em_b),    "em_C":  _pct(em_c),
    }

    print(f"\n  {'Entity':<10}  {'Freq':>6}  {'Retention NLL':>14}  {'Exact match':>12}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*14}  {'-'*12}")
    for label, freq, ret, em in [
        ("A (fast)",  FREQ_A, result["ret_A"], result["em_A"]),
        ("B (mid)",   FREQ_B, result["ret_B"], result["em_B"]),
        ("C (slow)",  FREQ_C, result["ret_C"], result["em_C"]),
    ]:
        flag = "  **" if ret > 0.5 else ("  *" if ret > 0.1 else "")
        print(f"  {label:<10}  {freq:>6}  {ret:>14.4f}{flag}  {em:>11.1f}%")

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
    print(f"  A updates every {FREQ_A} tok  |  B every {FREQ_B} tok  |  C every {FREQ_C} tok")
    print()

    gaps = [r["query_gap"] for r in results]
    col  = 10

    def _row(label, key):
        vals = "  ".join(f"{r[key]:>{col}.4f}" for r in results)
        print(f"  {label:<14}  {vals}")

    def _pct_row(label, key):
        vals = "  ".join(f"{r[key]:>{col-1}.1f}%" for r in results)
        print(f"  {label:<14}  {vals}")

    header = f"  {'':14}  " + "  ".join(f"{'gap='+str(g):>{col}}" for g in gaps)
    print(header)
    print("  " + "─" * (len(header) - 2))
    _row("ret A (fast)",  "ret_A")
    _row("ret B (mid)",   "ret_B")
    _row("ret C (slow)",  "ret_C")
    print()
    print(header)
    print("  " + "─" * (len(header) - 2))
    _pct_row("EM%  A (fast)", "em_A")
    _pct_row("EM%  B (mid)",  "em_B")
    _pct_row("EM%  C (slow)", "em_C")

    print("=" * 80)
    print("\nComparing GLA vs MS-GLA:")
    print("  * ret_A should be similar — both models fine branch handles fast updates.")
    print("  * ret_B gap should be moderate — scale-2 branch advantage.")
    print("  * ret_C gap should be largest — scale-4 coarse branch advantage.")
    print("  * If ret_C > 0 for MS-GLA but ~0 for GLA at gaps >4k, that is direct")
    print("    evidence that multi-resolution decomposition improves temporal memory.")


# ================================================================
# Export
# ================================================================

def save_csv(output_csv: str, results: list[dict], model_type: str) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_type", "query_gap", "n_probes",
              "ret_A", "ret_B", "ret_C", "em_A", "em_B", "em_C"]
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
    print(f"Freqs       : A={FREQ_A}  B={FREQ_B}  C={FREQ_C}")

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