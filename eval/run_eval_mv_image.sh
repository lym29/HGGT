#!/usr/bin/env bash
# Multi-view image (mv-image) evaluation launcher for HGGT.
# Edit the configuration block below, then run from the repository root:
#   bash eval/run_eval_mv_image.sh

set -euo pipefail

# Run from repository root (directory that contains .project-root / hggt/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=1
export OPENCV_THREAD_COUNT=0
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ============================================================
# Configuration — edit these before running
# ============================================================

# Local checkpoint (.pt) or Hugging Face repo id
CHECKPOINT="${CHECKPOINT:-catmint123/HGGT}"

MANO_PATH="${MANO_PATH:-${REPO_ROOT}/assets/mano_v1_2/models}"

# Multi-view image WebDataset root (must contain HO3D_mv_test/, DexYCB_mv/, ...)
DATA_ROOT="${DATA_ROOT:-/path/to/mv_image_data}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval_mv_image/$(date +%Y%m%d_%H%M%S)}"

# Space-separated subset, or all six
DATASETS="${DATASETS:-HO3D DexYCB Arctic Interhand Oakink Freihand}"

# Must match available GPUs
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

IMG_SIZE="${IMG_SIZE:-518}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

# Optional: set SAVE_CAMERAS="--save_cameras" to dump camera pickles
SAVE_CAMERAS="${SAVE_CAMERAS:---save_cameras}"

# Batch size per dataset (tuned by number of views)
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
echo "HGGT multi-view image evaluation"
echo "========================================================"
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
echo "========================================================"
