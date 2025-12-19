#!/bin/bash
# --- SLURM Configuration for H100 ---
#SBATCH --job-name=micro     # Job name               
#SBATCH --gres=gpu:h100-96:2  
#SBATCH --partition=gpu-long          # Request 1 H100 GPU
#SBATCH --cpus-per-task=32                
#SBATCH --mem=128G                        
#SBATCH --time=20:00:00                   
#SBATCH --output=micro_%j.log      # Log file



source /home/s/sri007/miniconda3/etc/profile.d/conda.sh
conda activate tox21_env
nvidia-smi

export OMP_NUM_THREADS=8  # Reduced for 4 workers
export MKL_NUM_THREADS=8

# Configuration
TOTAL_SAMPLES=10000
COMMON_SEED=42
TOTAL_WORKERS=4  # CORRECT: 4 workers total

echo "🚀 Starting 4 workers on 2 GPUs..."
echo "Total samples: $TOTAL_SAMPLES"
echo "Workers: $TOTAL_WORKERS (2 per GPU)"
echo "Seed: $COMMON_SEED"

WORKER_PIDS=()

# GPU 0: Workers 0 and 1
echo "🔵 Launching workers on GPU 0..."
for i in 0 1; do
  LOG_FILE="worker_gpu0_${i}.log"
  echo "  Starting worker $i on GPU 0..."
  
  CUDA_VISIBLE_DEVICES=0 python dataset.py \
    --gpu \
    --worker_id $i \
    --workers $TOTAL_WORKERS \
    --seed $COMMON_SEED \
    --total_samples $TOTAL_SAMPLES \
    --save_png_every 50 \
    --separate_worker_dirs > "$LOG_FILE" 2>&1 &
  
  WORKER_PIDS+=($!)
  sleep 1
done

# GPU 1: Workers 2 and 3
echo "🟢 Launching workers on GPU 1..."
for i in 2 3; do
  LOG_FILE="worker_gpu1_${i}.log"
  echo "  Starting worker $i on GPU 1..."
  
  CUDA_VISIBLE_DEVICES=1 python dataset.py \
    --gpu \
    --worker_id $i \
    --workers $TOTAL_WORKERS \
    --seed $COMMON_SEED \
    --total_samples $TOTAL_SAMPLES \
    --save_png_every 50 \
    --separate_worker_dirs > "$LOG_FILE" 2>&1 &
  
  WORKER_PIDS+=($!)
  sleep 1
done

echo "✅ All workers launched. Waiting..."

# Wait for all
for pid in "${WORKER_PIDS[@]}"; do
    wait $pid || echo "Worker $pid failed"
done

echo "✅ All 4 workers finished!"