#!/usr/bin/env bash
set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}
CSV_PATH="parameterSweep/sweep_configs.csv"
DATA_YAML="datasets\deepscores\ds2_dense_prepared\deep_scores.yaml"
MODEL="yolov8n.pt"gi
WORKDIR=$(pwd)

python parameterSweep/train_script.py \
  --csv "$WORKDIR/$CSV_PATH" \
  --idx "$IDX" \
  --data "$WORKDIR/$DATA_YAML" \
  --model "$MODEL" \
  --workdir "$WORKDIR" \
  --results_csv "$WORKDIR/parameterSweep/runs/results_${SLURM_JOB_ID}.csv"
