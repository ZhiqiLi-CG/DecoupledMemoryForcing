#!/usr/bin/env bash
#SBATCH --job-name=vbl_eval
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/vbench/slurm-%j.out

# Score an already-sampled 60s suite with VBench-Long.
#
#   sbatch submit_vbench_long_eval.sh <tag>
#
# Split out from the sampling script because scoring failed on its own the first
# time -- VBench-Long slices each clip, which needs PyAV, and the vbench venv
# did not have it (the 5s standard path never touches that code, so this only
# surfaced here). The videos survive such a failure, so re-scoring is cheap;
# re-sampling would not be.
set -eo pipefail
source /mnt/data/shared/config/env.sh

TAG="${1:?tag, e.g. sgf_cd or mem_lora}"
ROOT=/mnt/data/zhaofeng/runs/sgf/vbench
VID="$ROOT/${TAG}_long60/videos"
RES="$ROOT/${TAG}_long60/results"

n=$(ls "$VID" 2>/dev/null | grep -c '\.mp4$')
[[ "$n" -eq 128 ]] || { echo "expected 128 videos, found $n" >&2; exit 1; }
echo "scoring $TAG: $n videos"

gs-srun --partition=hpc-mid --gres=gpu:1 --time=12:00:00 --job-name="vble_${TAG}" bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  for dim in subject_consistency background_consistency motion_smoothness \\
             dynamic_degree aesthetic_quality imaging_quality; do
    echo \"##### \$dim #####\"
    bash scripts/eval/eval_vbench.sh long '$VID' "$RES/\$dim" \"\$dim\" || echo \"FAILED_DIM \$dim\"
  done
  echo '##### LONG EVAL DONE #####'
"
