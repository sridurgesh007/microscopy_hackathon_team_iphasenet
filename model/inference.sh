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

python inference.py \
  --data_dir hackathon_dataset_npz_final \
  --split val \
  --ckpt checkpoints_ddp/best_model.pt \
  --out_dir infer_val \
  --model swin \
  --metadata_mode sincos7 \
  --scaling_factor 600000 \
  --save_examples 30
echo "inference done"