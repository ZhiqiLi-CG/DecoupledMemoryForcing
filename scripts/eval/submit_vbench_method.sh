#!/usr/bin/env bash
#SBATCH --job-name=vb_sample
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/vbench/slurm-%j.out

# Sample the official VBench standard suite for one method.
#
#   sbatch submit_vbench_method.sh <tag> <config.yaml> <checkpoint.pt> [use_ema]
#
# Pass 1: all 946 official prompts x 5 samples  = 4730 videos
# Pass 2: the 74 temporal_flickering prompts x 25, which the official protocol
#         requires so enough clips survive the static filter. Indices 0-4 exist
#         from pass 1 and are skipped, so this adds 1480.
# Both write into ONE folder: standard mode reads a single directory.
#
# hpc-mid is PreemptMode=REQUEUE with GraceTime=0, so a higher-priority job can
# SIGTERM this at any moment. Generation skips videos that already exist, so
# each attempt resumes where the last was killed; retry until it completes.
set -eo pipefail
source /mnt/data/shared/config/env.sh

TAG="${1:?tag, e.g. sgf_cd or mem_lora}"
CFG="${2:?path to config yaml}"
CKPT="${3:?path to checkpoint .pt}"
USE_EMA="${4:-1}"

ROOT=/mnt/data/zhaofeng/runs/sgf/vbench
VID="$ROOT/${TAG}_short/videos"
mkdir -p "$VID"

ATTEMPTS="${ATTEMPTS:-20}"
for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== $TAG attempt $attempt/$ATTEMPTS ($(ls "$VID" 2>/dev/null | grep -c '\.mp4$') videos so far) ==="
  set +e
  gs-srun --partition=hpc-mid --gres=gpu:8 --time=12:00:00 --job-name="vb_${TAG}_gpu" bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  export PYTHONPATH=/mnt/data/zhaofeng/pylibs/diffusers031:/mnt/data/zhaofeng/venvs/streamingvideo/lib/python3.10/site-packages
  export TRITON_CACHE_DIR=/tmp/triton-cache

  echo '### PASS 1: 944 VBench prompts x 5 samples ###'
  OUTPUT_FOLDER=$VID NUM_SAMPLES=5 USE_EMA=$USE_EMA CONFIG=$CFG \
    bash scripts/eval/sample_for_vbench.sh short $CKPT framewise 8

  echo '### PASS 2: temporal_flickering 75 prompts x 25 ###'
  OUTPUT_FOLDER=$VID USE_EMA=$USE_EMA CONFIG=$CFG \
    bash scripts/eval/sample_for_vbench.sh short_flicker $CKPT framewise 8

  echo \"### TOTAL: \$(find $VID -maxdepth 1 -name '*.mp4' | wc -l) videos ###\"
"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "=== $TAG complete after $attempt attempt(s): $(ls "$VID" | grep -c '\.mp4$') videos ==="
    exit 0
  fi
  echo "=== attempt $attempt exited $status (preempted?); retrying in 60s ==="
  sleep 60
done
echo "=== gave up after $ATTEMPTS attempts ===" >&2
exit 1
