# ReCAST training

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
pip install -e ".[h200,tracking]"
```

Authenticate with Hugging Face and accept the SD3.5-Medium model license.

## Inputs

Training uses one prompt file and one density-profile NPZ file per stage.

Stage 1 prompts are plain text, one prompt per line. Its profile contains:

```text
sigmas                 shape (26,)
ratio__clipscore       shape (N, 25)
ratio__hpsv2           shape (N, 25)
ratio__pickscore       shape (N, 25)
```

Stage 2 uses GenEval JSONL records containing `prompt`, `tag`, and `include`:

```json
{"tag":"counting","include":[{"class":"cat","count":2}],"prompt":"a photo of two cats"}
```

Its profile additionally contains `ratio__geneval` with shape `(N, 25)`.

`sigmas` runs from `1` (noise) to `0` (clean), and ratio column `j` describes
the destination `sigmas[j+1]`. The saved `weights` array is `25*W`, with shape
`(number of rewards, 25)` and column sums `1`; `paper_weights` stores Figure 1's
`W`. Both use clean-to-noise order, and training reverses the timestep axis.

## Stage 1: Pick-a-Pic

Validate the inputs and calculate the weights without loading the model:

```bash
python train_recast.py \
  --config configs/h200_8gpu.toml \
  --profiles /path/to/stage1_profiles.npz \
  --prompts /path/to/pickapic_prompts.txt \
  --dry-run
```

Train on one node with 8 H200 GPUs:

```bash
export CONFIG=configs/h200_8gpu.toml
export PROFILES=/path/to/stage1_profiles.npz
export PROMPTS=/path/to/pickapic_prompts.txt
export OUTPUT_DIR=/scratch/$USER/recast-stage1
scripts/launch_8xh200.sh
```

The Stage-1 recipe trains for 121 epochs and writes its final checkpoint to
`$OUTPUT_DIR/checkpoint-0121`.

## Stage 2: GenEval

Stage 2 starts a new 61-epoch schedule from the completed Stage-1 state. Use
`--initialize-from`; this preserves the LoRA, optimizer, old policy, and global
update count while resetting the stage-local epoch to 1.

```bash
export CONFIG=configs/h200_8gpu_stage2_geneval.toml
export PROFILES=/path/to/stage2_geneval_profiles.npz
export PROMPTS=/path/to/geneval_train_metadata.jsonl
export OUTPUT_DIR=/scratch/$USER/recast-stage2-geneval
scripts/launch_8xh200.sh \
  --initialize-from /scratch/$USER/recast-stage1/checkpoint-0121
```

The final Stage-2 checkpoint is `$OUTPUT_DIR/checkpoint-0061`.

## Resume an interrupted stage

Use `--resume-from` only for a checkpoint from the same stage:

```bash
scripts/launch_8xh200.sh \
  --resume-from /scratch/$USER/recast-stage2-geneval/checkpoint-0030
```

`--initialize-from` and `--resume-from` cannot be used together.

## Slurm

```bash
mkdir -p outputs/slurm
sbatch --export=ALL slurm/train_8xh200.sbatch
```

The launchers contain no retry or automatic resubmission logic. Runtime logs,
weights, checkpoints, profiles, and generated outputs are ignored by Git.
