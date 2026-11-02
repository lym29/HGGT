#!/usr/bin/env bash
#SBATCH --job-name=hggt-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=hggt-train-%j.log
#SBATCH --error=hggt-train-%j.err

# Submit from the repository root. Add your cluster's partition/account options
# to sbatch, and adjust GPU/CPU/time requests for your allocation.
# Example: sbatch training/slurm/train.slurm logging.output_dir=outputs/train
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"

# Use the active Python environment, or set both variables before submission:
#   CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV=hggt
if [[ -n "${CONDA_ENV:-}" ]]; then
    : "${CONDA_SH:?Set CONDA_SH to your conda.sh activation script}"
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

# These variables are read by training/configs/finetune.yaml:
#   HAMER_DATA_DIR, HGGT_SYNTHETIC_DIR, MANO_MODEL_DIR,
#   HGGT_INIT_CHECKPOINT, HGGT_OUTPUT_DIR.
# Paths can also be passed as OmegaConf dotlist arguments to this script.
TRAIN_CONFIG="${TRAIN_CONFIG:-training/configs/finetune.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"  # Match --gpus-per-node above.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# One Slurm task launches all local DDP workers through torchrun.
srun --ntasks=1 --kill-on-bad-exit=1 \
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
    -m hggt.training.launch --config "${TRAIN_CONFIG}" "$@"
