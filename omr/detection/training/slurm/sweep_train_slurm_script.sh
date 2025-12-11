#!/usr/bin/env bash
set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}
CURRENT_DIR=$(pwd)
CSV_PATH="${CURRENT_DIR}/parameterSweep/sweep_configs.csv"
DATA_YAML="${CURRENT_DIR}/datasets/deepscores/prepared_ds2_dense/deepscores.yaml"
MODEL="${CURRENT_DIR}/yolo11n.pt"
WORKDIR="${CURRENT_DIR}/parameterSweep/runs/results_${SLURM_ARRAY_JOB_ID}"
GPU_COUNT=${SLURM_GPUS_ON_NODE}
NODE_COUNT=${SLURM_NNODES}
SWEEP_CSV="${CURRENT_DIR}/parameterSweep/runs/results_${SLURM_ARRAY_JOB_ID}/results_${SLURM_ARRAY_JOB_ID}.csv"

python -m torch.distributed.run --nproc_per_node ${GPU_COUNT} --nnodes ${NODE_COUNT} parameterSweep/train_script.py \
  --csv "$CSV_PATH" \
  --idx "$IDX" \
  --data "$DATA_YAML" \
  --model "$MODEL" \
  --workdir "$WORKDIR" \
  --gpuCount "${GPU_COUNT}" \
  --results_csv "$SWEEP_CSV"
