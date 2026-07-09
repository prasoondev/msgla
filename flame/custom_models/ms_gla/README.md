# MS-GLA Custom Model

This directory implements `model_type="ms_gla"`, the Multi-Scale Gated Linear Attention model used in the paper. The implementation is designed as a Hugging Face-compatible custom model inside the `flame` training framework.

## Files

| File | Purpose |
| --- | --- |
| `config_msgla.py` | defines `MSGLAConfig` and validates scales, head allocation, pooling, and fusion options |
| `ms_gla_layer.py` | implements temporal pooling, scale-specific GLA branches, causal hold upsampling, fusion, and cached decoding |
| `modeling_msgla.py` | defines `MSGLABlock`, `MSGLAModel`, and `MSGLAForCausalLM` |
| `__init__.py` | registers MS-GLA with `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` |

## Layer Computation

Each MS-GLA layer receives hidden states of shape `[batch, length, hidden_size]` and evaluates one GLA branch per temporal scale:

```text
X_s    = causal_average_pool(X, scale=s)
Y_s    = GLA_s(X_s)
Y_s_up = causal_hold_upsample(Y_s, scale=s)
O_t    = sum_s softmax(fuse(x_t))_s * Y_s_up[t]
```

The pooling operation is non-overlapping and mask-aware. For `scale=1`, the branch sees the original token sequence. For larger scales, each recurrent update summarizes a block of tokens, giving that branch a coarser temporal view of the same sequence.

Coarse branch outputs are repeated back to token resolution and shifted by `scale - 1` positions so that a pooled block is not exposed before all of its tokens have been observed. This preserves causal semantics.

## Head Budget

MS-GLA preserves the total GLA head budget. The `scale_num_heads` list partitions `num_heads` across branches and must sum to `num_heads`.

Common configurations:

| Scales | Heads | Use |
| --- | --- | --- |
| `[1, 2]` | `[2, 2]` | two-scale variant |
| `[1, 4]` | `[2, 2]` | fine plus coarse ablation |
| `[2, 4]` | `[2, 2]` | coarse-only ablation |
| `[1, 2, 4]` | `[2, 1, 1]` | main paper configuration |
| `[1, 2, 4, 8]` | `[1, 1, 1, 1]` | four-scale variant |

The branch key/value expansion ratios are derived from this head partition so that the total key/value state budget remains matched to the baseline GLA model.

## Configuration Fields

MS-GLA adds four fields to the base GLA configuration:

| Field | Description |
| --- | --- |
| `scales` | positive, unique temporal pooling factors |
| `scale_num_heads` | per-scale head allocation; if omitted, heads are divided as evenly as possible |
| `pool_mode` | currently supports `"avg"` |
| `fuse_mode` | supports learned `"softmax"` fusion or unlearned `"mean"` fusion |

The current implementation expects `num_kv_heads` to be unset or equal to `num_heads`.

## Cached Decoding

Generation uses a structured decode state per branch:

```text
past_key_values        branch-local GLA recurrent cache
pending_hidden_states  unpooled tokens waiting to fill a coarse block
last_output            most recent branch output used for causal hold
```

During prefill, full coarse groups are pooled and processed immediately, while incomplete groups are retained as pending state. During token-by-token decoding, a branch updates only when its pending buffer reaches the branch scale; otherwise it reuses the previous held output.

This design keeps cached generation causal for every scale while still allowing batched execution when branch cache states are aligned across batch elements.

## Registration

Importing `custom_models.ms_gla` registers the model classes with Hugging Face auto classes:

```python
from custom_models.ms_gla import MSGLAConfig, MSGLAForCausalLM
```

After registration, configs with `"model_type": "ms_gla"` can be loaded through `AutoConfig` and instantiated through `AutoModelForCausalLM`.
