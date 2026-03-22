#!/usr/bin/bash

set -euo pipefail

FLAME_ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESEARCH_ROOT=$(cd -- "${FLAME_ROOT}/.." && pwd)

params=""
if [ $# -ne 0 ]; then
    params="$*"
fi

# use envs as local params for convenience
# e.g.
# NNODE=1 NGPU=8 LOG_RANK=0 ./train.sh
NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

if [[ -z "${MASTER_ADDR:-}" ]]; then
  export MASTER_ADDR="localhost"
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
  export MASTER_PORT="0"
fi

: '
Usage:

bash train.sh -h

Training the GLA 340M baseline for 4.5B tokens:

NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/gla_340M-4p5B/batch32.seqlen2048.warmup3400.update1.steps68664.lr3e-4 \
  --model.config configs/gla_340M.json \
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
  --training.steps 68664 \
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

Training the MS-GLA 340M comparison model for the same 4.5B-token budget:

NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/ms_gla_340M-4p5B/batch32.seqlen2048.warmup3400.update1.steps68664.lr3e-4 \
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
  --training.steps 68664 \
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

If you want the faster two-scale MS-GLA variant instead, replace
`configs/ms_gla_340M.json` with `configs/ms_gla_340M_s12.json`.

'

echo "Launching training..."

set -x
path=$(grep -oP '(?<=--job.dump_folder )[^ ]+' <<< "$params")
steps=$(grep -oP '(?<=--training.steps )[^ ]+' <<< "$params")
config=$(grep -oP '(?<=--model.config )[^ ]+' <<< "$params")
tokenizer=$(grep -oP '(?<=--model.tokenizer_path )[^ ]+' <<< "$params")
model=$(
  PYTHONPATH="${FLAME_ROOT}:${PYTHONPATH:-}" python -c "import sys; import fla, custom_models; from transformers import AutoConfig; print(AutoConfig.from_pretrained(sys.argv[1]).to_json_string())" "$config" | jq -r '.model_type'
)

mkdir -p "$path"
cp "${FLAME_ROOT}/train.sh" "$path"
cp -r "${FLAME_ROOT}/configs" "$path"
cp -r "${FLAME_ROOT}/custom_models" "$path"
cp -r "${FLAME_ROOT}/flame" "$path"
mkdir -p "$path/3rd_party/flash-linear-attention" "$path/3rd_party/torchtitan"
cp -r "${RESEARCH_ROOT}/3rd_party/flash-linear-attention/fla" "$path/3rd_party/flash-linear-attention"
cp -r "${RESEARCH_ROOT}/3rd_party/torchtitan/torchtitan" "$path/3rd_party/torchtitan"

# for offline systems
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
if [ "${date:-}" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi
RUN_NAME="$model-$(basename $path)"
RUN_ID="$RUN_NAME-$date"

export WANDB_RESUME=allow
export WANDB_PROJECT=${WANDB_PROJECT:-fla}
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_ID}
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
PYTHONPATH="${FLAME_ROOT}:${PYTHONPATH:-}" \
torchrun --nnodes=${NNODE} \
  --nproc_per_node=${NGPU} \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank \
  --tee 3 \
  --log-dir "$path/logs" \
  -m flame.train \
  $params

echo "TRAINING DONE!"
echo "Converting the DCP checkpoints to HF format..."

PYTHONPATH="${FLAME_ROOT}:${PYTHONPATH:-}" \
python -m flame.utils.convert_dcp_to_hf \
  --path "$path" \
  --step "$steps" \
  --config "$config" \
  --tokenizer "$tokenizer"

echo "RUNNING DONE!"
