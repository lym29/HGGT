#!/bin/bash
#SBATCH -J eval_mv_image
#SBATCH -o eval_mv_image-%j.log
#SBATCH -e eval_mv_image-%j.err
#SBATCH -p GPU-8A100
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu_8a100
#SBATCH --cpus-per-task=16

# Multi-view image evaluation on Slurm.
# Edit the configuration block and SBATCH headers for your cluster, then:
#   sbatch eval/eval_mv_image.slurm

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"

# Optional cluster modules (uncomment / adjust as needed)
# source /etc/profile.d/modules.sh
# module load cuda/12.1

# Repository root (directory that contains .project-root / hggt/)
REPO_ROOT="${REPO_ROOT:-$HOME/workspace/HGGT/main}"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=1
export OPENCV_THREAD_COUNT=0
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ============================================================
# Configuration — edit these paths before submitting
# ============================================================

# Local checkpoint (.pt) or Hugging Face repo id
CHECKPOINT="${CHECKPOINT:-catmint123/HGGT}"

MANO_PATH="${MANO_PATH:-${REPO_ROOT}/assets/mano_v1_2/models}"

# Multi-view image WebDataset root (download from
# https://huggingface.co/datasets/JubSteven/POEM-v2 )
DATA_ROOT="${DATA_ROOT:-/path/to/mv_image_data}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval_mv_image/$(date +%Y%m%d_%H%M%S)}"

# Space-separated subset, or all six
DATASETS="${DATASETS:-HO3D DexYCB Arctic Interhand Oakink Freihand}"

# Must match --gres=gpu:N above
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

IMG_SIZE="${IMG_SIZE:-518}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

# Optional: set SAVE_CAMERAS="--save_cameras" to dump camera pickles
SAVE_CAMERAS="${SAVE_CAMERAS:---save_cameras}"

declare -A BATCH_SIZES
BATCH_SIZES["Freihand"]=64
BATCH_SIZES["Oakink"]=16
BATCH_SIZES["HO3D"]=12
BATCH_SIZES["DexYCB"]=24
BATCH_SIZES["Arctic"]=24
BATCH_SIZES["Interhand"]=24
DEFAULT_BATCH_SIZE=4

# ============================================================

echo "========================================================"
echo "HGGT multi-view image evaluation (Slurm)"
echo "========================================================"
echo "Repo root:   ${REPO_ROOT}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Data root:   ${DATA_ROOT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Datasets:    ${DATASETS}"
echo "GPUs:        ${NPROC_PER_NODE}"
echo "========================================================"

mkdir -p "${OUTPUT_DIR}"

for DATASET in ${DATASETS}; do
    BATCH_SIZE="${BATCH_SIZES[${DATASET}]:-${DEFAULT_BATCH_SIZE}}"
    MASTER_PORT="$(shuf -i 20000-65000 -n 1)"

    echo ""
    echo "========================================================"
    echo "Evaluating dataset: ${DATASET} (batch_size=${BATCH_SIZE})"
    echo "========================================================"

    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        eval/eval_mv_image.py \
            --checkpoint "${CHECKPOINT}" \
            --mano_model_path "${MANO_PATH}" \
            --data_root "${DATA_ROOT}" \
            --output_dir "${OUTPUT_DIR}" \
            --datasets "${DATASET}" \
            --img_size "${IMG_SIZE}" \
            --batch_size "${BATCH_SIZE}" \
            --max_samples "${MAX_SAMPLES}" \
            --log_file "${OUTPUT_DIR}/eval_${DATASET}.log" \
            ${SAVE_CAMERAS}

    echo "Finished evaluating ${DATASET}"
done

echo ""
echo "========================================================"
echo "Evaluation complete. Results in: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "========================================================"
