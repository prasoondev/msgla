# MS-GLA

`MS-GLA` is a multiscale extension of Gated Linear Attention implemented as a custom `flame` model with `model_type = "ms_gla"`.

## Motivation

Standard GLA gives linear attention a selective memory through data-dependent gates, but a single recurrent state still has to compress short-range and long-range structure at the same time.

`MS-GLA` addresses that by splitting the sequence into multiple temporal resolutions:

* a fine branch for local detail,
* one or more coarser branches for longer-range structure,
* a learned fusion layer that mixes branch outputs back at token resolution.

The intended effect is to let each branch specialize to its own timescale while keeping the overall head/state budget fixed.

## File Layout

* `config_msgla.py`: configuration and validation for `MSGLAConfig`
* `ms_gla_layer.py`: multiscale GLA layer, pooling, fusion, and decode cache logic
* `modeling_msgla.py`: transformer block, base model, and causal LM wrapper
* `__init__.py`: Hugging Face auto-class registration

## Architecture

At each MS-GLA layer:

1. The input is pooled into one sequence per scale.
2. Each scale is processed by its own smaller `GatedLinearAttention` branch.
3. Coarse outputs are upsampled back to token resolution with causal hold behavior.
4. Branch outputs are fused token-wise.

In simplified form:

```text
x_s = pool(hidden_states, scale=s)
y_s = GLA_s(x_s)
y_s_up = causal_hold_upsample(y_s, scale=s)
y = fuse(y_1_up, y_2_up, ..., y_S_up)
```

The implementation keeps the total head budget fixed across scales using `scale_num_heads`. For example:

* `scales=[1, 2]`, `scale_num_heads=[2, 2]`
* `scales=[1, 2, 4]`, `scale_num_heads=[2, 1, 1]`
* `scales=[1, 2, 4, 8]`, `scale_num_heads=[1, 1, 1, 1]`

## Config Knobs

Main MS-GLA-specific config fields:

* `scales`: temporal resolutions to use, must start with `1`
* `scale_num_heads`: head allocation per scale, must sum to `num_heads`
* `pool_mode`: currently only `avg`
* `fuse_mode`: `softmax` or `mean`

Important validation rules:

* `scales` must be positive and unique
* `scale_num_heads` must match the number of scales
* `num_kv_heads` must be unset or equal to `num_heads`

## Cache And Generation

`MS-GLA` supports cached decoding and Hugging Face `generate()`.

Each scale keeps its own decode state:

* branch-local GLA cache,
* pending fine-resolution tokens that have not yet formed a coarse update,
* the most recent branch output used for causal hold behavior.

The current implementation has two decode paths:

* a correctness-first fallback path that updates branches sample-by-sample,
* a faster batched path used when branch states are aligned and `batch_size > 1`

This keeps left-padded and partially aligned batches correct while still improving common generation workloads.

## Tensor Parallel Support

`flame.models.parallelize_fla` includes a dedicated TP plan for `model_type="ms_gla"`.

The branch-local GLA projections are parallelized branch-by-branch.
The small fusion projection is kept replicated so fusion weights are computed from all scale logits locally on each rank.

## Config Variants

Provided 340M-style configs:

* `configs/ms_gla_340M_s12.json`
  * scales: `[1, 2]`
  * heads: `[2, 2]`
  * recommended first efficiency-oriented run
* `configs/ms_gla_340M.json`
  * scales: `[1, 2, 4]`
  * heads: `[2, 1, 1]`
  * recommended first multiscale research comparison
* `configs/ms_gla_340M_s1248.json`
  * scales: `[1, 2, 4, 8]`
  * heads: `[1, 1, 1, 1]`
  * richer timescale decomposition, but slower in local sweeps

## Testing

Main tests:

* `tests/test_msgla.py`
  * config/model registration
  * CUDA forward/backward
  * cache equivalence
  * prefill-cache equivalence
  * `generate()` smoke test
* `tests/test_msgla_tp.py`
  * TP plan registration and branch coverage
* `tests/test_msgla_scale_configs.py`
  * scale-config load coverage

Useful commands:

```bash
python -m pytest flame/tests/test_msgla.py -v
python -m pytest flame/tests/test_msgla_tp.py -v
python -m pytest flame/tests/test_msgla_scale_configs.py -v
```

## Benchmarking

Use:

* `tests/benchmark_msgla.py` for one variant
* `tests/compare_msgla_scales.py` for side-by-side scale sweeps

Example:

```bash
python flame/tests/compare_msgla_scales.py \
  --device cuda \
  --batch-size 2 \
  --seq-len 1024 \
  --decode-steps 128 \
  --hidden-size 512 \
  --intermediate-size 1536 \
  --num-hidden-layers 4 \
  --num-heads 4
```

In the local sweep used during development, `[1, 2]` was the fastest variant, `[1, 2, 4]` was the best balanced research candidate, and `[1, 2, 4, 8]` was the slowest.

## Known Limitations

* Only `avg` pooling is currently implemented.
* `output_attentions` is not supported.
* The strongest runtime gains so far are in cached prefill; decode is improved but still more complex than baseline GLA.
* More scales increase compute cost even when total head count is preserved.

## Suggested Experiment Order

1. Train `ms_gla_340M_s12.json` as the first efficiency baseline.
2. Train `ms_gla_340M.json` as the first multiscale comparison.
3. Only then decide whether `ms_gla_340M_s1248.json` is worth the extra latency.
