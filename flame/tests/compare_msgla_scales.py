import argparse
import copy
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "/home/prasoon/Documents/research/flame")
import custom_models  # noqa: E402,F401


CONFIG_MAP = {
    "s12": "/home/prasoon/Documents/research/flame/configs/ms_gla_340M_s12.json",
    "s124": "/home/prasoon/Documents/research/flame/configs/ms_gla_340M.json",
    "s1248": "/home/prasoon/Documents/research/flame/configs/ms_gla_340M_s1248.json",
}


def load_config(config_path: str, args: argparse.Namespace):
    cfg = AutoConfig.from_pretrained(config_path)
    cfg.hidden_size = args.hidden_size
    cfg.intermediate_size = args.intermediate_size
    cfg.num_hidden_layers = args.num_hidden_layers
    cfg.num_heads = args.num_heads
    cfg.vocab_size = args.vocab_size
    cfg.expand_k = args.expand_k
    cfg.expand_v = args.expand_v
    cfg.use_cache = True
    cfg.fuse_cross_entropy = False
    return cfg


def random_batch(batch_size: int, seq_len: int, vocab_size: int, left_pad: int, device: str):
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
    return (time.perf_counter() - start) / iters


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def benchmark_variant(name: str, config_path: str, args: argparse.Namespace):
    cfg = load_config(config_path, args)
    model = AutoModelForCausalLM.from_config(cfg).eval().to(args.device)
    input_ids, attention_mask = random_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=cfg.vocab_size,
        left_pad=args.left_pad,
        device=args.device,
    )

    prefill_s = timed(lambda: run_prefill(model, input_ids, attention_mask), args.warmup, args.iters)
    decode_s = timed(lambda: run_decode(model, input_ids, attention_mask, args.decode_steps), args.warmup, args.iters)

    result = {
        "name": name,
        "config_path": config_path,
        "scales": copy.deepcopy(cfg.scales),
        "scale_num_heads": copy.deepcopy(cfg.scale_num_heads),
        "params_m": count_parameters(model) / 1_000_000,
        "prefill_tok_per_s": (args.batch_size * args.seq_len) / prefill_s,
        "decode_tok_per_s": (args.batch_size * args.decode_steps) / decode_s,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def print_results(results):
    print("ms_gla scale sweep")
    print("variant | scales | heads | params_m | prefill_tok/s | decode_tok/s")
    for result in results:
        print(
            f"{result['name']} | {result['scales']} | {result['scale_num_heads']} | "
            f"{result['params_m']:.2f} | {result['prefill_tok_per_s']:.2f} | {result['decode_tok_per_s']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare MS-GLA scale variants.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", default=["s12", "s124", "s1248"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--left-pad", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--expand-k", type=float, default=0.5)
    parser.add_argument("--expand-v", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    unknown_variants = [variant for variant in args.variants if variant not in CONFIG_MAP]
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}. Valid options: {sorted(CONFIG_MAP)}")

    torch.manual_seed(42)
    results = [benchmark_variant(variant, CONFIG_MAP[variant], args) for variant in args.variants]
    print_results(results)


if __name__ == "__main__":
    main()
