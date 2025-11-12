#!/usr/bin/env bash
set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}
CSV_PATH="sweep_configs.csv"
DATA_YAML="deep_scores.yaml"
MODEL="yolov8n.pt"
WORKDIR="parameterSweep"

#module load Python/3.11.3-GCCcore-12.3.0
#cd /lustre/pd01/hpc-katjar5048-1762193819/OMR
#source .venv/bin/activate

python train_script.py \
  --csv "$CSV_PATH" \
  --idx "$IDX" \
  --data "$DATA_YAML" \
  --model "$MODEL" \
  --workdir "$WORKDIR" \
  --results_csv "$WORKDIR/runs/results.csv"
