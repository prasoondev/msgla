# MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution

🎉 Accepted at COLM 2026. 🎉

This repository contains the implementation and evaluation code for **MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution**. MS-GLA extends Gated Linear Attention (GLA) by assigning separate groups of recurrent attention heads to different temporal resolutions, then recombining their outputs with a learnable input-dependent fusion layer.

The main idea is simple: a single GLA recurrent state has fixed capacity and must represent both local token-level structure and slower long-range semantic structure at the same temporal resolution. MS-GLA keeps the total head and state budget matched to a baseline GLA model, but redistributes that budget across multiple pooled views of the sequence. Fine-scale branches operate at token resolution; coarse branches operate on causally pooled token blocks and therefore update more slowly over longer spans.

The implementation depends on `flash-linear-attention` and the `flame` training framework. The model is registered as a Hugging Face model type named `ms_gla`.

## Background: Gated Linear Attention

Linear attention replaces quadratic softmax attention with a recurrent update. In its unnormalized form, the hidden memory matrix is updated as

```text
S_t = S_{t-1} + k_t^T v_t
o_t = q_t S_t
```

GLA adds a data-dependent forget gate so the model can decay stale information:

```text
S_t = Diag(alpha_t) S_{t-1} + k_t^T v_t
```

where `alpha_t` is produced from the current hidden state. This makes the recurrent memory selective, while preserving linear-time inference and chunkwise-parallel training.

The remaining bottleneck is that every head still operates at the same temporal resolution. Each head receives one token at a time and must use a fixed-size recurrent state to encode both high-frequency local syntax and low-frequency long-range structure. MS-GLA targets this bottleneck directly.

## How MS-GLA Works

For an input hidden-state sequence `X` of shape `[batch, length, hidden_size]`, each MS-GLA layer creates one branch per temporal scale. A scale `s` means that every non-overlapping block of `s` tokens is pooled into a single coarse token before being processed by its branch.

For example, with `scales=[1, 2, 4]`:

```text
scale 1: x_0, x_1, x_2, x_3, ...             native token resolution
scale 2: avg(x_0,x_1), avg(x_2,x_3), ...     two-token resolution
scale 4: avg(x_0..x_3), avg(x_4..x_7), ...   four-token resolution
```

Each pooled sequence is processed by an independent `GatedLinearAttention` branch. The coarse branch outputs are then repeated back to token resolution using a causal hold, and a learned fusion projection combines all branch outputs at every original timestep.

In simplified form:

```text
X_s      = causal_average_pool(X, scale=s)
Y_s      = GLA_s(X_s)
Y_s_up   = causal_hold_upsample(Y_s, scale=s, target_len=L)
w_t      = softmax(W_fuse x_t + b_fuse)
O_t      = sum_s w_{t,s} Y_s_up[t]
```

### Causal Average Pooling

The implementation in `flame/custom_models/ms_gla/ms_gla_layer.py` pads the sequence on the right when its length is not divisible by the scale, reshapes the sequence into non-overlapping blocks, and averages within each block. If an attention mask is present, masked tokens contribute zero to the numerator and the denominator is the number of valid tokens in the block.

For a block at scale `s`, the masked pooling operation is:

```text
X_i^(s) = sum_j X_{is+j} M_{is+j} / max(1, sum_j M_{is+j})
```

The pooled attention mask marks a coarse block as valid when at least one token in that block is valid.

### Scale-Specific GLA Branches

Each branch is a smaller instance of FLA's `GatedLinearAttention`. Branches do not share GLA projection or gating parameters. They share only the same architectural form.

The total number of heads is conserved. If the baseline GLA model has four heads, an MS-GLA model also has four total heads, partitioned across scales:

| Variant | `scales` | `scale_num_heads` | Interpretation |
| --- | ---: | ---: | --- |
| `s12` | `[1, 2]` | `[2, 2]` | equal split between token and two-token branches |
| `s14` | `[1, 4]` | `[2, 2]` | token branch plus coarser four-token branch |
| `s24` | `[2, 4]` | `[2, 2]` | coarse-only ablation with no native-resolution branch |
| `s124` | `[1, 2, 4]` | `[2, 1, 1]` | main three-scale model |
| `s1248` | `[1, 2, 4, 8]` | `[1, 1, 1, 1]` | four-scale model with one head per branch |

The helper `_compute_branch_expand_ratios` preserves the key/value state budget by splitting the baseline key and value dimensions proportionally to each branch's head count.

### Causal Hold Upsampling

Coarse branches should not reveal information from a pooled block before all tokens in that block have been observed. The implementation therefore repeats each coarse output `s` times and shifts it right by `s - 1` positions, inserting zeros at the beginning.

For scale `s=4`, the output for pooled block `(x_0, x_1, x_2, x_3)` first becomes available at position `3`, not at position `0`. This makes the pooled branch causal with respect to the original token sequence.

### Learnable Scale Fusion

By default, `fuse_mode="softmax"`. Each MS-GLA layer contains a small linear projection:

```python
self.fuse = nn.Linear(hidden_size, len(scales), bias=True)
```

At every timestep, the fusion layer reads the original unpooled hidden state and produces scale weights. These weights are softmax-normalized and used to combine the aligned branch outputs. The fusion weight and bias are initialized to zero except for `bias[0] = 1.0`, which initially favors the first listed scale. In the standard configs, the first scale is the finest branch.

The code also supports `fuse_mode="mean"` as a non-learned averaging ablation.

### Cached Decoding

MS-GLA supports cached generation, but its cache is more structured than baseline GLA because coarse branches update only after enough fine-resolution tokens have accumulated.

Each branch maintains:

```text
past_key_values        # branch-local FLA recurrent cache
pending_hidden_states  # fine-resolution tokens waiting to form a full coarse block
last_output            # most recent held branch output
```

During prefill, valid tokens are grouped by scale, full groups are pooled and sent through the branch, and any leftover tokens are stored as pending state. During token-by-token decoding, a branch either appends the current hidden state to its pending buffer or, when the buffer reaches the branch scale, pools the buffer, updates the branch recurrent cache, and refreshes the held output.

This logic is implemented in `MSGLADecodeState` and `MSGLABranchDecodeState` in `ms_gla_layer.py`. The code includes a batched fast path when branch cache states are aligned across batch elements, with a sample-wise fallback for partially aligned or left-padded batches.

## Model Integration

### Configuration

`MSGLAConfig` extends FLA's `GLAConfig` and adds:

| Field | Meaning |
| --- | --- |
| `scales` | temporal pooling factors |
| `scale_num_heads` | number of GLA heads assigned to each scale |
| `pool_mode` | currently only `"avg"` |
| `fuse_mode` | `"softmax"` or `"mean"` |

Validation rules in `config_msgla.py` enforce that scales are positive and unique, that head allocations are positive and sum to `num_heads`, and that `num_kv_heads` is either unset or equal to `num_heads`.

If `scale_num_heads` is omitted, the config divides heads as evenly as possible across the scales. For example, `num_heads=4` and `scales=[1,2,4]` gives `[2,1,1]`.

### Model Classes

The main classes are:

| File | Class | Role |
| --- | --- | --- |
| `flame/custom_models/ms_gla/config_msgla.py` | `MSGLAConfig` | validates scale and head allocation |
| `flame/custom_models/ms_gla/ms_gla_layer.py` | `MultiScaleGatedLinearAttention` | pooling, branch execution, causal hold, fusion, cache logic |
| `flame/custom_models/ms_gla/modeling_msgla.py` | `MSGLABlock` | RMSNorm, MS-GLA/attention mixer, Gated MLP |
| `flame/custom_models/ms_gla/modeling_msgla.py` | `MSGLAModel` | decoder-only backbone |
| `flame/custom_models/ms_gla/modeling_msgla.py` | `MSGLAForCausalLM` | language-model head and training loss |
| `flame/custom_models/ms_gla/__init__.py` | auto-registration | registers config/model classes with Hugging Face auto classes |

`MSGLABlock` mirrors the GLA block structure: attention norm, sequence mixer, MLP norm, and Gated MLP. If a config specifies softmax attention layers through `config.attn`, those layers use FLA's `Attention`; otherwise the block uses `MultiScaleGatedLinearAttention`.

### Tensor Parallel Plan

`flame/flame/models/parallelize_fla.py` registers `MSGLATPPlan` under `model_type="ms_gla"`. The plan parallelizes each branch's `q_proj`, `k_proj`, `v_proj`, `g_proj`, `gk_proj`, `g_norm`, and `o_proj` in the same style as GLA. The fusion projection is kept replicated so every tensor-parallel rank sees all branch logits before the softmax.

## Training Configurations

The 340M configurations used in the paper share the same backbone:

```text
num_hidden_layers = 24
hidden_size       = 1024
num_heads         = 4
expand_k          = 0.5
expand_v          = 1.0
hidden_ratio      = 4
vocab_size        = 32000
attn_mode         = "chunk"
use_short_conv    = false
fuse_norm         = true
```

Key configs:

```text
flame/configs/gla_340M.json             baseline GLA
flame/configs/ms_gla_340M_s12.json      MS-GLA [1,2], heads [2,2]
flame/configs/ms_gla_340M_s14.json      MS-GLA [1,4], heads [2,2]
flame/configs/ms_gla_340M_s24.json      MS-GLA [2,4], heads [2,2]
flame/configs/ms_gla_340M.json          MS-GLA [1,2,4], heads [2,1,1]
flame/configs/ms_gla_340M_s1248.json    MS-GLA [1,2,4,8], heads [1,1,1,1]
```

The paper trains 340M-parameter models from scratch on FineWeb-Edu with the tokenizer from `fla-hub/transformer-1.3B-100B`. Training uses sequence length 2048, batch size 32, gradient accumulation 1, AdamW with learning rate `3e-4`, epsilon `1e-15`, weight decay `0.1`, cosine decay to a minimum learning-rate ratio of `0.1`, 3400 warmup steps, max gradient norm `1.0`, and seed 42.

The launch wrapper is `flame/train.sh`. It copies the configs, custom model code, and FLA/torchtitan dependencies into the run directory for reproducibility, launches `flame.train` through `torchrun`, and converts the final DCP checkpoint to Hugging Face format.

Example:

```bash
cd flame

NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/ms_gla_340M-7B/batch32.seqlen2048.warmup3400.steps106813.lr3e-4 \
  --model.config configs/ms_gla_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 3400 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 106813 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset ../../fineweb-edu \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.interval 8096 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10
```

## Evaluation and Benchmarks

The repository includes three benchmark families: language modeling and multiple-choice evaluation, recall-intensive generation, and long-context perplexity-vs-position analysis. Each benchmark has MS-GLA and GLA versions under `benchmark_msgla/` and `benchmark_gla/`.

### 1. Language Modeling and Downstream Accuracy

Script:

```text
benchmark_msgla/eval_msgla_benchmarks.py
benchmark_gla/eval_gla_benchmarks.py
```

Tasks and metrics:

| Task | Dataset | Metric |
| --- | --- | --- |
| WikiText | `Salesforce/wikitext`, `wikitext-2-raw-v1` | perplexity |
| LAMBADA | `cimec/lambada`, `plain_text` | continuation perplexity and exact next-word accuracy |
| PIQA | `baber/piqa` | multiple-choice accuracy |
| HellaSwag | `allenai/hellaswag` | length-normalized multiple-choice accuracy |
| WinoGrande | `allenai/winogrande`, `winogrande_xl` | multiple-choice accuracy |

The script loads a DCP directory or a converted `.pt` checkpoint, builds `MSGLAForCausalLM`, and scores continuations by summing token log-likelihoods. WikiText uses sliding-window perplexity with configurable `max_ctx` and `stride`. HellaSwag uses average log-likelihood per continuation token.

Example:

```bash
python benchmark_msgla/eval_msgla_benchmarks.py \
  --checkpoint_path /path/to/exp/ms_gla_340M \
  --step 106813 \
  --model_ref benchmark_msgla \
  --tasks wikitext,lambada,piqa,hellaswag,winogrande \
  --max_ctx 2048 \
  --stride 1024 \
  --device cuda
```

By default, results are saved as JSON and CSV under `benchmark_msgla/results/benchmarks/` unless explicit output paths are provided.

### 2. Recall-Intensive Generation

Script:

```text
benchmark_msgla/eval_sfs.py
benchmark_gla/eval_sfs.py
```

Tasks and metrics:

| Task | Dataset | Metric |
| --- | --- | --- |
| SWDE | `hazyresearch/based-swde` | token-level F1 |
| FDA | `hazyresearch/based-fda` | token-level F1 |
| SQuAD | `hazyresearch/based-squad` | token-level F1 |

The script evaluates zero-shot retrieval-style prompts. Each example provides a prompt in `example["text"]` and a gold answer in `example["value"]`. The model generates greedily for up to `max_new_tokens`, and predictions are normalized by lowercasing, removing punctuation and articles, and whitespace-normalizing before F1 is computed.

The recommended generation backend is `cached`, which uses the explicit MS-GLA cache path rather than relying only on Hugging Face `generate`.

Example:

```bash
python benchmark_msgla/eval_sfs.py \
  --checkpoint_path /path/to/exp/ms_gla_340M \
  --step 106813 \
  --model_ref benchmark_msgla \
  --tasks swde,fda,squad \
  --max_samples 500 \
  --max_ctx 2048 \
  --max_new_tokens 32 \
  --generation_backend cached \
  --device cuda \
  --output_json benchmark_msgla/results/7B/124/sfs/msgla.json \
  --output_csv benchmark_msgla/results/7B/124/sfs/msgla.csv
```

`--profile_generation` can be used to report the first sample's prefill time, decode time, average decode-step time, and generated token count.

### 3. Long-Context Perplexity vs. Position

Script:

```text
benchmark_msgla/eval_msgla_perplexity.py
benchmark_gla/eval_gla_perplexity.py
plot_perplexity_vs_position.py
```

The long-context evaluation measures perplexity as a function of token position rather than only a single aggregate score. It evaluates documents from:

```text
fla-hub/pg19
fla-hub/slimpajama-test
```

Each document is tokenized without truncation, capped at 32768 tokens, and evaluated with a sliding context window. Token negative log-likelihoods are bucketed by absolute position, averaged across documents, and exponentiated to produce position-bucket perplexity curves.

Example:

```bash
python benchmark_msgla/eval_msgla_perplexity.py \
  --checkpoint_path /path/to/exp/ms_gla_340M \
  --step 106813 \
  --model_ref benchmark_msgla \
  --datasets fla-hub/pg19,fla-hub/slimpajama-test \
  --split test \
  --max_ctx 2048 \
  --stride 1024 \
  --bucket_size 128 \
  --results_json benchmark_msgla/results/7B/124/slimpg19/msgla_124_results.json \
  --plot_output plots/perplexity_vs_position/ms_gla_124.png \
  --device cuda
```

The combined plotting script discovers MS-GLA variants under `benchmark_msgla/results/7B/*/slimpg19/*.json`, includes the GLA baseline from `benchmark_gla/results/7B/slimpg19/gla_results.json`, and writes a combined PG19/SlimPajama plot:

```bash
python plot_perplexity_vs_position.py \
  --results-root benchmark_msgla/results/7B \
  --gla-json benchmark_gla/results/7B/slimpg19/gla_results.json \
  --output-dir plots/perplexity_vs_position/7B
```

### Saved Result Files

The repository includes saved results for the 7B-token runs:

```text
benchmark_gla/results/7B/ppl_acc/gla.csv
benchmark_gla/results/7B/sfs/gla.json
benchmark_gla/results/7B/slimpg19/gla_results.json

benchmark_msgla/results/7B/{12,14,24,124,1248}/ppl_acc/msgla.csv
benchmark_msgla/results/7B/{12,14,24,124,1248}/sfs/msgla.json
benchmark_msgla/results/7B/{12,14,24,124,1248}/slimpg19/*.json
```

In these directory names, `12` denotes scales `[1,2]`, `14` denotes `[1,4]`, `24` denotes `[2,4]`, `124` denotes `[1,2,4]`, and `1248` denotes `[1,2,4,8]`.

The main empirical pattern in the saved results is that the native-resolution branch is essential. Variants containing scale `1` improve over GLA in aggregate, while the coarse-only `[2,4]` variant performs poorly on perplexity, recall, and long-context extrapolation. The best overall configuration in the paper is `[1,2,4]` with head allocation `[2,1,1]`.

## Checkpoint Conversion

To convert a distributed checkpoint to a single `.pt` file:

```bash
python convert_dcp_to_pt.py \
  --checkpoint_path /path/to/exp/ms_gla_340M \
  --step 106813 \
  --output_path /tmp/msgla_step106813.pt
```

To convert a DCP directory or `.pt` checkpoint to a Hugging Face `save_pretrained` folder:

```bash
python convert_checkpoint_to_hf.py \
  --checkpoint_path /path/to/exp/ms_gla_340M \
  --step 106813 \
  --config flame/configs/ms_gla_340M.json \
  --tokenizer benchmark_msgla \
  --output_dir /tmp/msgla-340M-hf
```

The conversion script imports `custom_models` so that `MSGLAConfig` and `MSGLAForCausalLM` are registered before the model is constructed.
