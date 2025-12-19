#!/bin/bash
#SBATCH --job-name=micro_train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=micro_%j.log
#SBATCH --error=micro_%j.err

set -euo pipefail

source /home/s/sri007/miniconda3/etc/profile.d/conda.sh
conda activate tox21_env

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

nvidia-smi

# --- DDP rendezvous (single node) ---
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

# how many GPUs on this node
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}

echo " MASTER_ADDR=$MASTER_ADDR"
echo " MASTER_PORT=$MASTER_PORT"
echo " Launching torchrun on $NUM_GPUS GPUs..."

srun torchrun \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  train.py \
    --data_dir hackathon_dataset_npz_final \
    --out_dir checkpoints_ddp \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 8 \
    --val_interval 1 \
    --no_amp \
    --save_samples \
    --model swin \
    --metadata_mode sincos7

echo "✅ Job Finished!"