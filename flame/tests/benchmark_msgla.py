import argparse
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "/home/prasoon/Documents/research/flame")
import custom_models  # noqa: E402,F401


def build_config(args: argparse.Namespace):
    return AutoConfig.for_model(
        "ms_gla",
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_heads=args.num_heads,
        scales=args.scales,
        scale_num_heads=args.scale_num_heads,
        expand_k=args.expand_k,
        expand_v=args.expand_v,
        vocab_size=args.vocab_size,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        fuse_cross_entropy=False,
        use_cache=True,
    )


def random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    left_pad: int,
    device: str,
):
    input_ids = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    if left_pad > 0:
        left_pad = min(left_pad, seq_len - 1)
        attention_mask[:, :left_pad] = False
        input_ids[:, :left_pad] = 0
    return input_ids, attention_mask


@torch.inference_mode()
def run_prefill(model, input_ids, attention_mask):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=None,
    )


@torch.inference_mode()
def run_tokenwise_prefill(model, input_ids, attention_mask):
    out = model(
        input_ids=input_ids[:, :1],
        attention_mask=attention_mask[:, :1],
        use_cache=True,
        past_key_values=None,
    )
    past_key_values = out.past_key_values
    logits = [out.logits]
    for idx in range(1, input_ids.shape[1]):
        out = model(
            input_ids=input_ids[:, idx:idx + 1],
            attention_mask=attention_mask[:, :idx + 1],
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = out.past_key_values
        logits.append(out.logits)
    return torch.cat(logits, dim=1)


@torch.inference_mode()
def run_decode(model, input_ids, attention_mask, decode_steps: int):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=None,
    )
    past_key_values = out.past_key_values
    next_token = input_ids[:, -1:]

    for _ in range(decode_steps):
        next_mask = torch.ones(
            (attention_mask.shape[0], attention_mask.shape[1] + 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        out = model(
            input_ids=next_token,
            attention_mask=next_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1:].argmax(dim=-1)


def timed(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    return total / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark MS-GLA prefill and decode.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--left-pad", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--scale-num-heads", type=int, nargs="+", default=[2, 1, 1])
    parser.add_argument("--expand-k", type=float, default=0.5)
    parser.add_argument("--expand-v", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    if len(args.scales) != len(args.scale_num_heads):
        raise ValueError("`--scales` and `--scale-num-heads` must have the same length.")
    if sum(args.scale_num_heads) != args.num_heads:
        raise ValueError("`--scale-num-heads` must sum to `--num-heads`.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    torch.manual_seed(42)
    cfg = build_config(args)
    model = AutoModelForCausalLM.from_config(cfg).eval().to(args.device)
    input_ids, attention_mask = random_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=cfg.vocab_size,
        left_pad=args.left_pad,
        device=args.device,
    )

    prefill_ms = timed(lambda: run_prefill(model, input_ids, attention_mask), args.warmup, args.iters) * 1000
    tokenwise_prefill_ms = timed(
        lambda: run_tokenwise_prefill(model, input_ids, attention_mask), args.warmup, args.iters
    ) * 1000
    decode_ms = timed(
        lambda: run_decode(model, input_ids, attention_mask, args.decode_steps), args.warmup, args.iters
    ) * 1000

    prefill_tok_per_s = (args.batch_size * args.seq_len) / (prefill_ms / 1000.0)
    tokenwise_tok_per_s = (args.batch_size * args.seq_len) / (tokenwise_prefill_ms / 1000.0)
    decode_tok_per_s = (args.batch_size * args.decode_steps) / (decode_ms / 1000.0)

    print("ms_gla benchmark")
    print(f"device={args.device}")
    print(f"batch_size={args.batch_size} seq_len={args.seq_len} decode_steps={args.decode_steps}")
    print(f"scales={args.scales} scale_num_heads={args.scale_num_heads}")
    print(f"prefill_ms={prefill_ms:.3f} prefill_tok_per_s={prefill_tok_per_s:.2f}")
    print(
        f"tokenwise_prefill_ms={tokenwise_prefill_ms:.3f} "
        f"tokenwise_prefill_tok_per_s={tokenwise_tok_per_s:.2f}"
    )
    print(f"prefill_speedup_vs_tokenwise={tokenwise_prefill_ms / prefill_ms:.3f}x")
    print(f"decode_ms={decode_ms:.3f} decode_tok_per_s={decode_tok_per_s:.2f}")


if __name__ == "__main__":
    main()
