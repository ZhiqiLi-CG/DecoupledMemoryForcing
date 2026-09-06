#!/usr/bin/env bash
#SBATCH --job-name=vb_long
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/vbench/slurm-%j.out

# Sample AND score the 60s suite for one method.
#
#   sbatch submit_vbench_long.sh <tag> <config.yaml> <checkpoint.pt> [use_ema]
#
# 128 MovieGen prompts x 1 sample at 241 latent frames (~60s at 16fps), with the
# streaming KV-cache geometry the long-video inference uses. Far smaller than
# the 5s suite (128 clips, not 6210), but each clip is ~11x longer to generate.
#
# VBench-Long's long_custom_input mode scores only the dimensions that need no
# prompt-to-video correspondence -- consistency, motion and quality -- which is
# why the prompt names may be truncated here without harm, unlike the standard
# 5s mode that matches videos to prompts by filename.
#
# Scoring runs right after sampling in the same job: the long suite is small
# enough that splitting them buys nothing.
set -eo pipefail
source /mnt/data/shared/config/env.sh

TAG="${1:?tag, e.g. sgf_cd or mem_lora}"
CFG="${2:?path to config yaml}"
CKPT="${3:?path to checkpoint .pt}"
USE_EMA="${4:-1}"

ROOT=/mnt/data/zhaofeng/runs/sgf/vbench
VID="$ROOT/${TAG}_long60/videos"
RES="$ROOT/${TAG}_long60/results"
mkdir -p "$VID"

ATTEMPTS="${ATTEMPTS:-10}"
for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== $TAG long attempt $attempt/$ATTEMPTS ($(ls "$VID" 2>/dev/null | grep -c '\.mp4$')/128 videos) ==="
  set +e
  gs-srun --partition=hpc-mid --gres=gpu:8 --time=12:00:00 --job-name="vbl_${TAG}_gpu" bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  export PYTHONPATH=/mnt/data/zhaofeng/pylibs/diffusers031:/mnt/data/zhaofeng/venvs/streamingvideo/lib/python3.10/site-packages
  export TRITON_CACHE_DIR=/tmp/triton-cache
  OUTPUT_FOLDER=$VID USE_EMA=$USE_EMA CONFIG=$CFG \
    bash scripts/eval/sample_for_vbench.sh long60 $CKPT framewise 8
  echo \"### $TAG: \$(find $VID -maxdepth 1 -name '*.mp4' | wc -l)/128 videos ###\"
"
  status=$?
  set -e
  [[ $status -eq 0 ]] && break
  echo "=== attempt $attempt exited $status (preempted?); retrying in 60s ==="
  sleep 60
done

n=$(ls "$VID" 2>/dev/null | grep -c '\.mp4$')
[[ "$n" -eq 128 ]] || { echo "only $n/128 videos, not scoring" >&2; exit 1; }

echo "=== scoring $TAG with VBench-Long ==="
gs-srun --partition=hpc-mid --gres=gpu:1 --time=12:00:00 --job-name="vbl_${TAG}_eval" bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  for dim in subject_consistency background_consistency motion_smoothness \\
             dynamic_degree aesthetic_quality imaging_quality; do
    echo \"##### \$dim #####\"
    bash scripts/eval/eval_vbench.sh long '$VID' '$RES/\$dim' \"\$dim\" || echo \"FAILED_DIM \$dim\"
  done
  echo '##### LONG EVAL DONE #####'
"
