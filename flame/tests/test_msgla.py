import sys

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "/home/prasoon/Documents/research/flame")
import custom_models  # noqa: E402,F401


def build_test_config(**overrides):
    params = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_heads=4,
        scales=[1, 2],
        scale_num_heads=[2, 2],
        expand_k=0.5,
        expand_v=1.0,
        vocab_size=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        fuse_cross_entropy=False,
        use_cache=True,
    )
    params.update(overrides)
    return AutoConfig.for_model("ms_gla", **params)


def test_msgla_registration_and_instantiation():
    cfg = AutoConfig.from_pretrained("/home/prasoon/Documents/research/flame/configs/ms_gla_340M.json")
    model = AutoModelForCausalLM.from_config(cfg)

    assert type(cfg).__name__ == "MSGLAConfig"
    assert type(model).__name__ == "MSGLAForCausalLM"
    assert len(model.model.layers) == 24
    assert cfg.scales == [1, 2, 4]
    assert cfg.scale_num_heads == [2, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MS-GLA runtime tests.")
def test_msgla_forward_backward_cuda():
    torch.manual_seed(42)
    cfg = build_test_config()
    model = AutoModelForCausalLM.from_config(cfg).to("cuda")

    input_ids = torch.randint(0, cfg.vocab_size, (1, 96), device="cuda")
    labels = torch.randint(0, cfg.vocab_size, (1, 96), device="cuda")
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()

    assert out.loss.isfinite()
    assert model.model.layers[0].attn.fuse.weight.grad is not None
    assert model.model.layers[0].attn.fuse.weight.grad.norm().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MS-GLA runtime tests.")
def test_msgla_cache_equivalence_cuda():
    torch.manual_seed(42)
    cfg = build_test_config()
    model = AutoModelForCausalLM.from_config(cfg).eval().to("cuda")

    batch_size, seq_len, chunk_size = 2, 32, 8
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device="cuda")
    seq_start = torch.tensor([1, 3], device="cuda")
    for i, start in enumerate(seq_start):
        attention_mask[i, :start] = False

    ref = torch.cat(
        [
            model(input_ids=input_ids[i:i + 1, start:], use_cache=False).logits
            for i, start in enumerate(seq_start.tolist())
        ],
        dim=1,
    )

    out = model(
        input_ids=input_ids[:, :chunk_size],
        attention_mask=attention_mask[:, :chunk_size],
        use_cache=True,
        past_key_values=None,
    )
    logits = [out.logits]
    past_key_values = out.past_key_values

    for j in range(chunk_size, seq_len):
        out = model(
            input_ids=input_ids[:, j:j + 1],
            attention_mask=attention_mask[:, :j + 1],
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits.append(out.logits)
        past_key_values = out.past_key_values

    gen = torch.cat(logits, dim=1)
    gen = torch.cat([gen[i:i + 1, start:] for i, start in enumerate(seq_start.tolist())], dim=1)

    assert ref.shape == gen.shape
    assert torch.allclose(ref, gen, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MS-GLA runtime tests.")
def test_msgla_prefill_cache_equivalence_cuda():
    torch.manual_seed(42)
    cfg = build_test_config(scales=[1, 2, 4], scale_num_heads=[2, 1, 1])
    model = AutoModelForCausalLM.from_config(cfg).eval().to("cuda")

    batch_size, seq_len = 2, 24
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device="cuda")
    seq_start = torch.tensor([0, 3], device="cuda")
    for i, start in enumerate(seq_start):
        attention_mask[i, :start] = False

    prefill = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=None,
    ).logits

    out = model(
        input_ids=input_ids[:, :1],
        attention_mask=attention_mask[:, :1],
        use_cache=True,
        past_key_values=None,
    )
    logits = [out.logits]
    past_key_values = out.past_key_values

    for j in range(1, seq_len):
        out = model(
            input_ids=input_ids[:, j:j + 1],
            attention_mask=attention_mask[:, :j + 1],
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits.append(out.logits)
        past_key_values = out.past_key_values

    tokenwise = torch.cat(logits, dim=1)
    prefill = torch.cat([prefill[i:i + 1, start:] for i, start in enumerate(seq_start.tolist())], dim=1)
    tokenwise = torch.cat([tokenwise[i:i + 1, start:] for i, start in enumerate(seq_start.tolist())], dim=1)

    assert prefill.shape == tokenwise.shape
    assert torch.allclose(prefill, tokenwise, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MS-GLA runtime tests.")
def test_msgla_generate_cuda():
    torch.manual_seed(42)
    cfg = build_test_config()
    model = AutoModelForCausalLM.from_config(cfg).eval().to("cuda")

    input_ids = torch.randint(3, cfg.vocab_size, (1, 12), device="cuda")
    output = model.generate(input_ids=input_ids, max_new_tokens=4, do_sample=False)

    assert output.shape == (1, 16)
