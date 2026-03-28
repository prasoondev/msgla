"""
msgla_lm_eval.py — lm-evaluation-harness wrapper for MS-GLA checkpoints.

Registers the model under the name "msgla" so it can be used directly
with lm_eval's CLI and evaluate() API.

Usage (CLI):
    lm_eval --model msgla \\
        --model_args "checkpoint_path=/path/to/checkpoint.pt,model_ref=/path/to/config_dir" \\
        --tasks lambada_openai,piqa,hellaswag,winogrande \\
        --device cuda \\
        --batch_size 8

Usage (Python):
    import lm_eval
    results = lm_eval.simple_evaluate(
        model="msgla",
        model_args={
            "checkpoint_path": "/path/to/checkpoint.pt",
            "model_ref": "/path/to/config_dir",
        },
        tasks=["lambada_openai", "piqa"],
    )

Registration:
    Either import this module before calling lm_eval (the @register_model
    decorator runs at import time), or point lm_eval to it via the
    --include_path / LMEVAL_INCLUDE_PATH mechanism.

Checkpoint formats accepted (mirrors eval_msgla_benchmarks.py):
    - A converted .pt file   → loaded directly with torch.load.
    - A DCP step directory   → converted to a temporary .pt on the fly
      via torch.distributed.checkpoint.format_utils.dcp_to_torch_save.
    - An experiment root     → the latest step-* checkpoint is auto-selected
      (same resolution logic as eval_msgla_benchmarks.py).
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.serialization
from tqdm import tqdm
from transformers import AutoTokenizer

# ── make custom MS-GLA models importable (mirrors eval_msgla_benchmarks.py) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLAME_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "flame")
if FLAME_ROOT not in sys.path:
    sys.path.insert(0, FLAME_ROOT)

from custom_models.ms_gla import MSGLAConfig, MSGLAForCausalLM  # noqa: E402

# ── lm-eval imports ───────────────────────────────────────────────────────────
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model


# =============================================================================
# Checkpoint resolution  (identical logic to eval_msgla_benchmarks.py)
# =============================================================================

def _resolve_checkpoint_path(checkpoint_path: str, step: Optional[int]) -> str:
    base = Path(checkpoint_path)

    if base.is_file():
        if base.suffix != ".pt":
            raise ValueError(f"Expected a .pt checkpoint file, got: {base}")
        if step is not None:
            raise ValueError(
                "Do not pass step= when checkpoint_path already points to a .pt file."
            )
        return str(base)

    if step is not None:
        candidate1 = base / "checkpoint" / f"step-{step}"
        candidate2 = base / f"step-{step}"
        if candidate1.exists():
            return str(candidate1)
        if candidate2.exists():
            return str(candidate2)
        raise FileNotFoundError(
            f"Could not find step-{step} under {base}. "
            f"Tried {candidate1} and {candidate2}."
        )

    if (base / ".metadata").exists() and any(base.glob("*.distcp")):
        return str(base)

    # Auto-select the latest checkpoint under <base>/checkpoint/step-*
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
        print(f"No step provided. Using latest checkpoint: {latest[1]}")
        return str(latest[1])

    raise FileNotFoundError(
        f"Could not infer DCP checkpoint directory from {base}. "
        "Pass either a step directory directly or set step=<int>."
    )


def _load_state_dict(checkpoint_path: str, tmp_dir: Optional[str]) -> dict:
    if checkpoint_path.endswith(".pt"):
        print(f"Loading converted checkpoint from {checkpoint_path} ...")
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        return torch.load(checkpoint_path, map_location="cpu")

    # DCP directory → convert to a temporary .pt file first
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    print(f"Converting DCP checkpoint at {checkpoint_path} ...")
    with tempfile.TemporaryDirectory(dir=tmp_dir) as workdir:
        checkpoint_pt = os.path.join(workdir, "checkpoint.pt")
        dcp_to_torch_save(checkpoint_path, checkpoint_pt)
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        return torch.load(checkpoint_pt, map_location="cpu")


# =============================================================================
# MSGLA lm-eval model
# =============================================================================

@register_model("msgla")
class MSGLAEvalModel(TemplateLM):
    """lm-evaluation-harness wrapper for MS-GLA causal language models.

    Constructor arguments (passed via --model_args or the model_args dict):

        checkpoint_path (str, required):
            Path to a .pt checkpoint file, a DCP step directory, or an
            experiment root (the latest checkpoint is auto-selected).

        model_ref (str, required):
            Path to the directory (or HuggingFace repo) that holds the model
            config (config.json) and tokenizer files.

        step (int, optional):
            Checkpoint step to load when checkpoint_path is an experiment root
            or a directory containing multiple step-* subdirectories.

        device (str, default "cuda" if available else "cpu"):
            Torch device string.

        max_length (int, default 2048):
            Maximum sequence length for scoring and generation.

        batch_size (int, default 1):
            Number of sequences processed per forward pass.

        dtype (str, default "bfloat16" on CUDA / "float32" on CPU):
            Model floating-point dtype. One of "float32", "float16", "bfloat16".

        tmp_dir (str, optional):
            Temporary directory used during DCP-to-.pt conversion.
            Defaults to the system temp directory.

        add_bos_token (bool, default False):
            Prepend the BOS token when tokenizing contexts. Set True if your
            tokenizer does not add BOS automatically and the model was trained
            with BOS.
    """

    # TemplateLM uses this to choose the causal scoring path.
    backend = "causal"

    def __init__(
        self,
        checkpoint_path: str,
        model_ref: str,
        step: Optional[int] = None,
        device: Optional[str] = None,
        max_length: int = 2048,
        batch_size: int = 1,
        dtype: Optional[str] = None,
        tmp_dir: Optional[str] = None,
        add_bos_token: bool = False,
        **kwargs,
    ):
        super().__init__()

        # ── device & dtype ────────────────────────────────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype is None:
            _dtype = (
                torch.bfloat16
                if str(self._device).startswith("cuda")
                else torch.float32
            )
        elif dtype in _dtype_map:
            _dtype = _dtype_map[dtype]
        else:
            raise ValueError(
                f"Unknown dtype '{dtype}'. Choose from: {list(_dtype_map.keys())}"
            )

        # ── tokenizer ─────────────────────────────────────────────────────────
        print(f"Loading tokenizer from {model_ref} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── model ─────────────────────────────────────────────────────────────
        resolved_path = _resolve_checkpoint_path(checkpoint_path, step)
        state = _load_state_dict(resolved_path, tmp_dir)

        if "model" not in state:
            raise KeyError(
                "Checkpoint does not contain a 'model' key in the top-level "
                f"state dict. Keys found: {list(state.keys())}"
            )

        print(f"Loading model config from {model_ref} ...")
        config = MSGLAConfig.from_pretrained(model_ref)
        self._model = MSGLAForCausalLM(config)
        self._model.load_state_dict(state["model"])
        self._model.to(device=self._device, dtype=_dtype).eval()

        n_params = sum(p.numel() for p in self._model.parameters())
        print(
            f"Model loaded: {n_params / 1e6:.1f}M params on "
            f"{self._device} ({_dtype})."
        )

        # ── settings ──────────────────────────────────────────────────────────
        self._max_length = max_length
        self.batch_size_per_gpu = int(batch_size)
        self._add_bos_token = add_bos_token
        self.vocab_size = config.vocab_size

    # =========================================================================
    # TemplateLM required properties
    # =========================================================================

    @property
    def eot_token_id(self) -> int:
        """End-of-text token ID (BOS/prefix fallback when context is empty)."""
        eot = self.tokenizer.eos_token_id
        if eot is None:
            eot = self.tokenizer.pad_token_id
        return eot

    @property
    def prefix_token_id(self) -> int:
        """Token used as a one-token context when the context string is empty.

        TemplateLM uses this instead of eot_token_id when BOS != EOS.
        Prefers BOS, falls back to EOT.
        """
        bos = self.tokenizer.bos_token_id
        return bos if bos is not None else self.eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        """Default maximum tokens to generate (can be overridden per-request
        via gen_kwargs["max_gen_toks"])."""
        return 256

    @property
    def batch_size(self) -> int:
        return self.batch_size_per_gpu

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer_name(self) -> str:
        """Used by lm-eval to fingerprint the request cache."""
        return self.tokenizer.name_or_path

    # =========================================================================
    # Tokenization helpers  (required by TemplateLM)
    # =========================================================================

    def tok_encode(
        self,
        string: str,
        add_special_tokens: Optional[bool] = None,
        **kwargs,
    ) -> list[int]:
        """Tokenize *string* and return a list of integer token IDs.

        When add_special_tokens is None (TemplateLM default when encoding the
        continuation), we honour the add_bos_token constructor flag so that
        BOS is only added to the context, not to continuations.
        """
        if add_special_tokens is None:
            add_special_tokens = self._add_bos_token
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(
        self, tokens: list[int], skip_special_tokens: bool = True
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    # =========================================================================
    # Core scoring  (required by TemplateLM)
    # =========================================================================

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        """Score a list of (context_enc, continuation_enc) token pairs.

        Each element of *requests* is a triple:
            ((context_str, continuation_str), context_enc, continuation_enc)

        Returns a list of (log_prob_sum, is_greedy) tuples — one per request —
        satisfying the TemplateLM / lm-eval contract.

        Implementation notes:
        - Sequences are left-padded to the batch maximum length.
        - If context + continuation exceeds max_length, the context is trimmed
          from the left; the continuation is always kept intact.
        - Attention mask is passed to the model so padded positions are ignored.
        """
        results: list[tuple[float, bool]] = []

        for batch_start in tqdm(
            range(0, len(requests), self.batch_size_per_gpu),
            desc="Scoring loglikelihoods",
            disable=disable_tqdm,
        ):
            batch = requests[batch_start : batch_start + self.batch_size_per_gpu]
            results.extend(self._score_batch(batch))

        return results

    @torch.inference_mode()
    def _score_batch(
        self,
        batch: list[tuple[tuple[str, str], list[int], list[int]]],
    ) -> list[tuple[float, bool]]:
        """Run a single batched forward pass and extract per-sample scores."""

        # Build full token sequences, truncating context from the left if needed
        full_seqs: list[list[int]] = []
        cont_lengths: list[int] = []

        for _, context_enc, continuation_enc in batch:
            full = context_enc + continuation_enc
            if len(full) > self._max_length:
                # Trim context; never trim the continuation
                full = full[-self._max_length :]
            full_seqs.append(full)
            cont_lengths.append(len(continuation_enc))

        # Left-pad to the longest sequence in this batch
        max_seq_len = max(len(s) for s in full_seqs)
        pad_id = self.tokenizer.pad_token_id or 0

        padded = torch.full(
            (len(full_seqs), max_seq_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=self._device,
        )
        for i, seq in enumerate(full_seqs):
            padded[i, max_seq_len - len(seq) :] = torch.tensor(
                seq, dtype=torch.long, device=self._device
            )

        attn_mask = (padded != pad_id).long()

        # Forward pass
        outputs = self._model(
            input_ids=padded,
            attention_mask=attn_mask,
            use_cache=False,
        )
        # Shift logits/tokens by 1 for next-token prediction alignment
        shift_logits = outputs.logits[:, :-1, :].float()   # (B, T-1, V)
        shift_tokens = padded[:, 1:]                        # (B, T-1)

        log_probs = torch.log_softmax(shift_logits, dim=-1)

        results: list[tuple[float, bool]] = []
        for i, cont_len in enumerate(cont_lengths):
            # Because we left-pad, the real tokens occupy the rightmost
            # positions.  In the shifted space (length max_seq_len - 1):
            #   continuation occupies the last cont_len positions.
            cont_start = max_seq_len - cont_len - 1  # inclusive
            cont_end   = max_seq_len - 1             # exclusive

            cont_lp  = log_probs[i, cont_start:cont_end, :]     # (cont_len, V)
            cont_tgt = shift_tokens[i, cont_start:cont_end]      # (cont_len,)

            token_lps = cont_lp.gather(1, cont_tgt.unsqueeze(-1)).squeeze(-1)
            sum_lp    = float(token_lps.sum().item())
            is_greedy = bool(torch.equal(cont_lp.argmax(dim=-1), cont_tgt))

            results.append((sum_lp, is_greedy))

        return results

    # =========================================================================
    # Rolling log-likelihood  (perplexity tasks, e.g. wikitext)
    # =========================================================================

    def loglikelihood_rolling(
        self,
        requests,
        disable_tqdm: bool = False,
    ) -> list[float]:
        """Compute whole-document log-likelihood with a sliding context window.

        Implements the same chunked sliding-window strategy as
        ``compute_doc_nll_sliding`` in eval_msgla_benchmarks.py:
          - Every token is predicted exactly once.
          - For the last chunk, the full max_length context is provided.
          - Stride defaults to max_length // 2.

        Args:
            requests: List of Instance objects, each with args = (string,).

        Returns:
            A list of total log-probabilities (negative NLL sums), one per doc.
        """
        results: list[float] = []
        stride = self._max_length // 2

        for req in tqdm(
            requests, desc="Rolling loglikelihood", disable=disable_tqdm
        ):
            (string,) = req.args
            token_ids = self.tok_encode(string, add_special_tokens=False)

            if not token_ids:
                results.append(0.0)
                continue

            total_nll, _ = self._sliding_window_nll(token_ids, stride)
            results.append(-total_nll)   # lm-eval expects log-prob (positive = better)

        return results

    @torch.inference_mode()
    def _sliding_window_nll(
        self,
        token_ids: list[int],
        stride: int,
    ) -> tuple[float, int]:
        """Sliding-window NLL for a single document.

        Returns:
            (total_nll, total_tokens): summed negative log-likelihood and the
            count of scored tokens.
        """
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        T = input_ids.shape[-1]

        if T < 2:
            return 0.0, 0

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        total_nll    = 0.0
        total_tokens = 0
        prev_covered = 0
        max_ctx      = self._max_length

        for begin in range(0, T - 1, stride):
            end   = min(begin + max_ctx, T)
            chunk = input_ids[:, begin:end].to(self._device)

            logits       = self._model(input_ids=chunk, use_cache=False).logits[0].float()
            shift_logits = logits[:-1].contiguous()
            shift_labels = chunk[0, 1:].contiguous()
            token_losses = loss_fct(shift_logits, shift_labels)

            # Only count tokens not yet covered by a previous window
            chunk_first_pred = begin + 1
            chunk_last_pred  = end - 1
            new_from         = max(prev_covered + 1, chunk_first_pred)

            if new_from <= chunk_last_pred:
                idx_start     = new_from - chunk_first_pred
                new_losses    = token_losses[idx_start:]
                total_nll    += float(new_losses.sum().item())
                total_tokens += int(new_losses.numel())
                prev_covered  = chunk_last_pred

            if end == T:
                break

        return total_nll, total_tokens

    # =========================================================================
    # Generation  (generative tasks, e.g. TriviaQA)
    # =========================================================================

    def generate_until(
        self,
        requests,
        disable_tqdm: bool = False,
    ) -> list[str]:
        """Greedy (or sampled) autoregressive generation until a stop sequence.

        Args:
            requests: List of Instance objects.  Each Instance.args is a
                (context_str, gen_kwargs) tuple.
                Recognised gen_kwargs keys:
                  until        (list[str] | str) — stop sequences
                  max_gen_toks (int)             — max tokens to generate
                  do_sample    (bool)            — sample vs greedy
                  temperature  (float)           — sampling temperature
                  top_p        (float)           — nucleus sampling threshold

        Returns:
            A list of generated strings (stop sequences stripped).
        """
        results: list[str] = []

        for req in tqdm(requests, desc="Generating", disable=disable_tqdm):
            context, gen_kwargs = req.args
            results.append(self._generate_single(context, gen_kwargs))

        return results

    @torch.inference_mode()
    def _generate_single(self, context: str, gen_kwargs: dict) -> str:
        """Token-by-token autoregressive generation for a single prompt."""
        until: list[str] = gen_kwargs.get(
            "until", [self.tok_decode([self.eot_token_id])]
        )
        if isinstance(until, str):
            until = [until]

        max_gen_toks: int   = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
        do_sample:    bool  = gen_kwargs.get("do_sample", False)
        temperature:  float = gen_kwargs.get("temperature", 1.0)
        top_p:        float = gen_kwargs.get("top_p", 1.0)

        # Encode context; trim from the left to leave room for generation
        context_ids = self.tok_encode(
            context, add_special_tokens=self._add_bos_token
        )
        max_ctx_len = self._max_length - max_gen_toks
        if max_ctx_len < 1:
            max_ctx_len = 1
        context_ids = context_ids[-max_ctx_len:]

        input_ids = torch.tensor(
            [context_ids], dtype=torch.long, device=self._device
        )
        generated: list[int] = []
        past_kv = None

        for _ in range(max_gen_toks):
            # On the first step pass the full context; afterwards pass only the
            # most recent token and rely on the KV cache.
            if past_kv is None:
                out = self._model(input_ids=input_ids, use_cache=True)
            else:
                out = self._model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                )

            logits = out.logits[:, -1, :].float()   # (1, V)
            past_kv = out.past_key_values

            # Sampling or greedy selection
            if do_sample and temperature > 0:
                logits = logits / temperature
                if top_p < 1.0:
                    logits = self._top_p_filter(logits, top_p)
                probs      = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id = int(next_token[0, 0].item())
            input_ids = next_token

            # Stop on EOT (don't include it in the output)
            if token_id == self.eot_token_id:
                break

            generated.append(token_id)

            # Check stop sequences after each token
            generated_text = self.tok_decode(
                generated, skip_special_tokens=False
            )
            for stop in until:
                if stop and generated_text.endswith(stop):
                    return generated_text[: -len(stop)]

        return self.tok_decode(generated, skip_special_tokens=True)

    # -------------------------------------------------------------------------
    # Nucleus (top-p) sampling helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Zero out logits whose cumulative softmax probability exceeds top_p.

        Args:
            logits: Shape (1, vocab_size).
            top_p:  Nucleus probability threshold in (0, 1].

        Returns:
            Filtered logits tensor (same shape, some entries set to -inf).
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        # Mask tokens whose *individual* probability pushes the cumulative sum
        # over the threshold (shift by one so we always keep the top token)
        remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove_mask] = float("-inf")
        return logits.scatter(-1, sorted_indices, sorted_logits)

    # -------------------------------------------------------------------------
    # Optional chat template support
    # -------------------------------------------------------------------------

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a chat history into a single prompt string.

        Delegates to the HuggingFace tokenizer's apply_chat_template.
        Raises NotImplementedError if the tokenizer has no chat template.
        """
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )