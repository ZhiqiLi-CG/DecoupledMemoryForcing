#!/usr/bin/env bash
#SBATCH --job-name=vb240
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/vbench/slurm-%j.out

# Sample and score the 240s suite for one method.
#
#   sbatch submit_vbench_long240.sh <tag> <config.yaml> <checkpoint.pt> [use_ema]
#
# 128 MovieGen prompts x 1 sample at 963 latent frames (~240s at 16fps), the
# length the Self Forcing paper reports. Sampling goes through
# infer_long_chunked.py rather than inference.py: the latter decodes the whole
# clip at once, which needs ~18.4 GiB for the pixel tensor alone and OOMs well
# before this length.
#
# Scoring is the expensive half here. VBench-Long slices every clip before
# extracting features, and these clips are 3849 frames instead of 961, so each
# dimension costs roughly four times what it does at 60s.
set -eo pipefail
source /mnt/data/shared/config/env.sh

TAG="${1:?tag, e.g. sgf_cd or mem_lora}"
CFG="${2:?path to config yaml}"
CKPT="${3:?path to checkpoint .pt}"
USE_EMA="${4:-1}"

ROOT=/mnt/data/zhaofeng/runs/sgf/vbench
VID="$ROOT/${TAG}_long240/videos"
RES="$ROOT/${TAG}_long240/results"
mkdir -p "$VID"

EMA_ARG=""
[[ "$USE_EMA" == "1" ]] && EMA_ARG="--use_ema"

ATTEMPTS="${ATTEMPTS:-20}"
for attempt in $(seq 1 "$ATTEMPTS"); do
  have=$(ls "$VID" 2>/dev/null | grep -c '\.mp4$' || true)
  echo "=== $TAG 240s attempt $attempt/$ATTEMPTS ($have/128) ==="
  [[ "$have" -ge 128 ]] && break
  set +e
  gs-srun --partition=hpc-mid --gres=gpu:8 --time=12:00:00 --job-name="v240_${TAG}" bash -c "
  set -eo pipefail
  export PYTHONPATH=/mnt/data/zhaofeng/pylibs/diffusers031:/mnt/data/zhaofeng/venvs/streamingvideo/lib/python3.10/site-packages
  export TRITON_CACHE_DIR=/tmp/triton-cache
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK LOCAL_WORLD_SIZE GROUP_RANK
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/eval/infer_long_chunked.py \
    --config_path $CFG --checkpoint_path $CKPT \
    --data_path prompts/moviegen_128.txt \
    --output_folder $VID \
    --num_output_frames 963 --seed 0 $EMA_ARG
  echo \"### $TAG: \$(ls $VID/*.mp4 2>/dev/null | wc -l)/128 ###\"
"
  status=$?
  set -e
  [[ $status -eq 0 ]] && break
  echo "=== attempt $attempt exited $status (preempted?); retrying in 60s ==="
  sleep 60
done

n=$(ls "$VID" 2>/dev/null | grep -c '\.mp4$' || true)
[[ "$n" -eq 128 ]] || { echo "only $n/128 videos, not scoring" >&2; exit 1; }

echo "=== scoring $TAG 240s with VBench-Long ==="
gs-srun --partition=hpc-mid --gres=gpu:1 --time=48:00:00 --job-name="v240e_${TAG}" bash -c "
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  for dim in subject_consistency background_consistency motion_smoothness \\
             dynamic_degree aesthetic_quality imaging_quality; do
    echo \"##### \$dim #####\"
    bash scripts/eval/eval_vbench.sh long '$VID' "$RES/\$dim" \"\$dim\" || echo \"FAILED_DIM \$dim\"
  done
  echo '##### 240s EVAL DONE #####'
"
