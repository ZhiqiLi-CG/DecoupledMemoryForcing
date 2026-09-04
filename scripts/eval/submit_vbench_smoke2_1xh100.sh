#!/usr/bin/env bash
#SBATCH --job-name=vbench_smoke2
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/slurm-%j.out

# Smoke test 2: score the 15 videos already sampled by smoke test 1 with the
# official evaluator in custom_input mode, which scores exactly the videos
# present (standard mode instead walks the full 946-prompt suite and divides by
# zero when a dimension has no matching videos, which is what smoke test 1 hit).
set -eo pipefail
source /mnt/data/shared/config/env.sh

gs-srun --partition=hpc-mid --gres=gpu:1 --time=04:00:00 --job-name=vbench_smoke2_gpu bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  OUT=/mnt/data/zhaofeng/runs/sgf/vbench_smoke
  bash scripts/eval/eval_vbench.sh custom \$OUT/videos \$OUT/results_custom
  echo '### RESULT FILES ###'
  for f in \$OUT/results_custom/*eval_results.json; do echo \"-- \$f\"; cat \"\$f\"; done
"
