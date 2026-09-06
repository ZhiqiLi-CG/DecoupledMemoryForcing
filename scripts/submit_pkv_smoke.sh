#!/usr/bin/env bash
#SBATCH --job-name=pkv_smoke
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/slurm-%j.out

# Self Gradient Forcing framewise training on 4x H100 (hpc-mid).
# cpu-low launcher pattern: gs-srun creates the real GPU job itself
# (it wipes SLURM_* and does not forward host env vars into the container,
# so all exports live inside the container command).
# Effective batch = 4 (batch_size 1 x 4 GPUs; repo trainer has no grad accum;
# paper recipe total_batch_size is 8 on 8 GPUs).
set -eo pipefail
source /mnt/data/shared/config/env.sh

gs-srun --partition=hpc-mid --gres=gpu:4 --time=24:00:00 --job-name=pkv_smoke_gpu bash -c "
  export PYTHONPATH=/mnt/data/zhaofeng/pylibs/diffusers031:/mnt/data/zhaofeng/venvs/streamingvideo/lib/python3.10/site-packages
  export NUM_GPUS=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
  export TRITON_CACHE_DIR=/tmp/triton-cache
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  bash scripts/train_self_gradient_forcing_framewise.sh \
    configs/persistent_kv_smoke.yaml \
    /mnt/data/zhaofeng/runs/sgf/pkv_smoke_${SLURM_JOB_ID}
"
