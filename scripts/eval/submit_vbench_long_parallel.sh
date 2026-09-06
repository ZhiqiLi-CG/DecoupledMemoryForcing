#!/usr/bin/env bash
# Score a long suite with one Slurm job PER DIMENSION, run in parallel.
#
#   bash submit_vbench_long_parallel.sh <tag> <suite>
#   e.g. bash submit_vbench_long_parallel.sh sgf_cd long240
#
# The dimensions are independent, so running them sequentially in one job makes
# the wall clock the SUM of them -- which at 240s (each clip is 3849 frames, four
# times a 60s one) runs to days. One job each makes it the max instead.
#
# The seven dimensions are the ones Self Gradient Forcing reports for long video:
# aesthetic quality, background consistency, dynamic degree, imaging quality,
# motion smoothness, subject consistency and flickering.
set -eo pipefail
source /mnt/data/shared/config/env.sh

TAG="${1:?tag, e.g. sgf_cd}"
SUITE="${2:?suite, long60 or long240}"
ROOT=/mnt/data/zhaofeng/runs/sgf/vbench
VID="$ROOT/${TAG}_${SUITE}/videos"
RES="$ROOT/${TAG}_${SUITE}/results"

n=$(ls "$VID" 2>/dev/null | grep -c '\.mp4$' || true)
echo "$TAG $SUITE: $n videos"
[[ "$n" -gt 0 ]] || { echo "no videos in $VID" >&2; exit 1; }

DIMS="${DIMS:-subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality temporal_flickering}"

seed=""
SCRIPTS="$ROOT/${TAG}_${SUITE}/jobs"
mkdir -p "$SCRIPTS"

for dim in $DIMS; do
  # Skip a dimension that already has a result, so this can be re-run to fill
  # in whatever failed without redoing the expensive ones.
  if compgen -G "$RES/$dim/*eval_results.json" > /dev/null; then
    echo "  $dim: already scored, skipping"
    continue
  fi
  # A real script rather than --wrap: sbatch runs a wrap with /bin/sh, which is
  # dash here and has no `source`, so env.sh never loads and gs-srun is not
  # found -- every job exits 127 before doing anything.
  f="$SCRIPTS/$dim.sh"
  cat > "$f" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=vb_${TAG}_${dim}
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=$ROOT/slurm-%j.out
set -eo pipefail
source /mnt/data/shared/config/env.sh
gs-srun --partition=hpc-mid --gres=gpu:1 --time=47:00:00 --job-name=vbd_${TAG}_${dim} bash -c '
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  echo "##### $dim #####"
  bash scripts/eval/eval_vbench.sh long "$VID" "$RES/$dim" "$dim"
  echo "##### $dim DONE #####"
'
EOF
  chmod +x "$f"
  # VBench-Long splits every clip to disk under videos/split_clip and REUSES that
  # split if it is already there. Launching all dimensions at once means they all
  # find it missing and all write it at the same time, which produces truncated
  # mp4s and a decord "Error sending packet" much later. So the first dimension
  # runs alone to build the split, and the rest wait on it and then reuse it.
  if [[ -d "$VID/split_clip" || -n "$seed" ]]; then
    jid=$(sbatch --parsable ${seed:+--dependency=afterok:$seed} "$f")
    echo "  $dim -> job $jid${seed:+ (waits for $seed)}"
  else
    jid=$(sbatch --parsable "$f")
    seed=$jid
    echo "  $dim -> job $jid  [seed: builds split_clip, others wait on it]"
  fi
done
