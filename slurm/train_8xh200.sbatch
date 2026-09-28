#!/usr/bin/env bash
#SBATCH --job-name=recast-stage1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

: "${PROFILES:?Export PROFILES before submitting}"
: "${PROMPTS:?Export PROMPTS before submitting}"

export CONFIG=${CONFIG:-configs/h200_8gpu.toml}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/stage1-h200}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun --ntasks=1 --kill-on-bad-exit=1 scripts/launch_8xh200.sh "$@"
