#!/bin/bash
#SBATCH --job-name=yolo_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-23%8   # adjust based on number of configs

module load anaconda
source ~/.bashrc
conda activate yolov8-env

./run_task.sh