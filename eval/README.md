# Multi-view image evaluation

Evaluate HGGT on multi-view (mv-image) WebDataset tars from standard hand datasets (HO3D, DexYCB, Arctic, Interhand, Oakink, Freihand). Reported metrics include MPJPE / MPVPE under several alignments (original, PA, RR, CA) plus AUC and PCK.

Metric helpers under [`metrics/`](metrics/) are borrowed from [POEM-v2](https://github.com/JubSteven/POEM).

## Data

Download the packaged multi-view WebDataset from Hugging Face:

- [JubSteven/POEM-v2](https://huggingface.co/datasets/JubSteven/POEM-v2/tree/main)

MANO is **required**. Place model files under `assets/mano_v1_2/models/MANO_RIGHT.pkl` (see the main README).

After downloading, `--data_root` should contain the expected tar layouts, for example:

```text
DATA_ROOT/
  HO3D_mv_test/HO3D_mv_test-000000.tar ...
  DexYCB_mv/DexYCB_mv_test-000000.tar ...
  Arctic_mv/Arctic_mv_val_p1-000000.tar ...
  Interhand_mv/Interhand_mv_val-000000.tar ...
  Oakink_mv/Oakink_mv_test-000000.tar ...
  Freihand_mv/Freihand_mv_test-000000.tar ...
```

See [`eval_mv_image.py`](eval_mv_image.py) (`DATASET_META`) for exact shard patterns.

## Quick start

```bash
# From repository root
# Single GPU (Hugging Face weights by default)
python eval/eval_mv_image.py \
  --data_root /path/to/mv_image_data \
  --output_dir outputs/eval_mv_image \
  --datasets HO3D

# Multi-GPU
torchrun --nproc_per_node=4 eval/eval_mv_image.py \
  --checkpoint /path/to/checkpoint.pt \
  --mano_model_path assets/mano_v1_2/models \
  --data_root /path/to/mv_image_data \
  --output_dir outputs/eval_mv_image \
  --datasets HO3D DexYCB Arctic Interhand Oakink Freihand \
  --batch_size 12 \
  --save_cameras
```

Or edit paths in [`run_eval_mv_image.sh`](run_eval_mv_image.sh) and run:

```bash
bash eval/run_eval_mv_image.sh
```

On a Slurm cluster, edit placeholders in [`eval_mv_image.slurm`](eval_mv_image.slurm) and submit:

```bash
sbatch eval/eval_mv_image.slurm
```

## Outputs

Typical files under `--output_dir`:

- `<Dataset>_results.txt` — full metric breakdown per dataset
- `all_results_summary.txt` — compact table across datasets
- `cameras/<Dataset>/cameras.pkl` — optional predicted / GT cameras (`--save_cameras`)
