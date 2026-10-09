#!/usr/bin/env bash
#SBATCH --job-name=vbench_smoke
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/slurm-%j.out

# End-to-end smoke test of the official VBench path on 3 prompts:
# sample with VBench naming, then score with upstream's own evaluator.
# cpu-low launcher; gs-srun creates the GPU job (it does not forward host env).
set -eo pipefail
source /mnt/data/shared/config/env.sh

gs-srun --partition=hpc-mid --gres=gpu:1 --time=04:00:00 --job-name=vbench_smoke_gpu bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  OUT=/mnt/data/zhaofeng/runs/sgf/vbench_smoke

  echo '### STEP 1: sample videos (SGF env) ###'
  PYTHONPATH=/mnt/data/zhaofeng/pylibs/diffusers031:/mnt/data/zhaofeng/venvs/streamingvideo/lib/python3.10/site-packages \
  TRITON_CACHE_DIR=/tmp/triton-cache \
  PROMPTS=/mnt/data/zhaofeng/runs/sgf/smoke_prompts.txt EXTENDED_PROMPTS=none OUTPUT_FOLDER=\$OUT/videos NUM_SAMPLES=5 \
    bash scripts/eval/sample_for_vbench.sh short \
      /mnt/data/zhaofeng/runs/sgf/framewise_624369/checkpoint_model_001500/model.pt

  echo '### STEP 2: official VBench evaluator (its own env) ###'
  bash scripts/eval/eval_vbench.sh short \$OUT/videos \$OUT/results \
    subject_consistency aesthetic_quality
"
