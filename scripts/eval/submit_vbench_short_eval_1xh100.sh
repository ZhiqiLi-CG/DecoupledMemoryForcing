#!/usr/bin/env bash
#SBATCH --job-name=vb_short_eval
#SBATCH --partition=cpu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/zhaofeng/runs/sgf/slurm-%j.out

# Score the sampled 5s suite with the official VBench evaluator, all 16
# dimensions, standard mode. Dimensions run one per call so a failure in one
# (e.g. a missing submodule) does not lose the others; each writes its own
# results json into the same output dir.
set -eo pipefail
source /mnt/data/shared/config/env.sh

gs-srun --partition=hpc-mid --gres=gpu:1 --time=24:00:00 --job-name=vb_short_eval_gpu bash -c '
  set -eo pipefail
  cd /mnt/home/zhaofeng/tem_zhiqi/causal/SelfGradientForcing
  VID=/mnt/data/zhaofeng/runs/sgf/vbench/framewise_short/videos
  OUT=/mnt/data/zhaofeng/runs/sgf/vbench/framewise_short/results

  echo "videos: $(find $VID -maxdepth 1 -name "*.mp4" | wc -l)"
  for dim in subject_consistency background_consistency temporal_flickering \
             motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
             object_class multiple_objects human_action color \
             spatial_relationship scene appearance_style temporal_style \
             overall_consistency; do
    echo "##### $dim #####"
    bash scripts/eval/eval_vbench.sh short "$VID" "$OUT/$dim" "$dim" || echo "FAILED_DIM $dim"
  done
  echo "##### ALL DIMENSIONS DONE #####"
'
