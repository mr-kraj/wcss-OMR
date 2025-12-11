#!/usr/bin/env bash

echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "SLURM_NODEID=$SLURM_NODEID"

ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
GPU_COUNT=${SLURM_GPUS_ON_NODE}
NODE_COUNT=${SLURM_NNODES}
NODE_RANK=${SLURM_NODEID}             # Node rank (0,1,...)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # First node
MASTER_PORT=12345                     # Some free port

echo "ID=$ID, GPUs=$GPU_COUNT, Nodes=$NODE_COUNT, NodeRank=$NODE_RANK, Master=$MASTER_ADDR:$MASTER_PORT"

python -m torch.distributed.run \
    --nproc_per_node $GPU_COUNT \
    --nnodes $NODE_COUNT \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --id "$ID" \
    --gpu-count "$GPU_COUNT"