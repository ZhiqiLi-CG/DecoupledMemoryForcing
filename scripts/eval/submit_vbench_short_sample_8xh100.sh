#!/usr/bin/env bash
#SBATCH --job-name=vb_short_gen
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/slurm-%j.out

# Sample the FULL official VBench standard suite for the 5s (short) setting.
#
# Pass 1: all 946 official prompts x 5 samples  = 4730 videos
# Pass 2: the 74 temporal_flickering prompts x 25 samples, which the official
#         protocol requires so enough clips survive the static filter. Indices
#         0-4 already exist from pass 1 and are skipped, so this adds 1480.
# Both passes write into ONE folder: VBench's standard mode reads a single dir.
#
# ~7.8 s/video on one H100, so ~1.7 h across 8 GPUs.
#
# hpc-mid is PreemptMode=REQUEUE with GraceTime=0, so a higher-priority job can
# SIGTERM this at any moment. Generation skips videos that already exist, so
# each attempt resumes where the last one was killed; retry until it completes.
set -eo pipefail
source /mnt/data/shared/config/env.sh

VID=/mnt/data/zhaofeng/runs/sgf/vbench/framewise_short/videos
ATTEMPTS="${ATTEMPTS:-20}"
for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== attempt $attempt/$ATTEMPTS ($(ls "$VID" 2>/dev/null | grep -c '\.mp4$') videos so far) ==="
  set +e
  gs-srun --partition=hpc-mid --gres=gpu:8 --time=12:00:00 --job-name=vb_short_gen_gpu bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  export PYTHONPATH=/mnt/data/zhaofeng/pylibs/diffusers031:/mnt/data/zhaofeng/venvs/streamingvideo/lib/python3.10/site-packages
  export TRITON_CACHE_DIR=/tmp/triton-cache
  CKPT=/mnt/data/zhaofeng/runs/sgf/framewise_624369/checkpoint_model_001500/model.pt
  OUT=/mnt/data/zhaofeng/runs/sgf/vbench/framewise_short/videos

  echo '### PASS 1: 946 prompts x 5 samples ###'
  OUTPUT_FOLDER=\$OUT NUM_SAMPLES=5 \
    bash scripts/eval/sample_for_vbench.sh short \$CKPT framewise 8

  echo '### PASS 2: temporal_flickering 74 prompts x 25 samples ###'
  OUTPUT_FOLDER=\$OUT NUM_SAMPLES=25 \
  PROMPTS=VBench/prompts/prompts_per_dimension/temporal_flickering.txt \
    bash scripts/eval/sample_for_vbench.sh short \$CKPT framewise 8

  echo \"### TOTAL: \$(find \$OUT -maxdepth 1 -name '*.mp4' | wc -l) videos ###\"
"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "=== generation complete after $attempt attempt(s): $(ls "$VID" | grep -c '\.mp4$') videos ==="
    exit 0
  fi
  echo "=== attempt $attempt exited $status (preempted?); retrying in 60s ==="
  sleep 60
done
echo "=== gave up after $ATTEMPTS attempts ===" >&2
exit 1
