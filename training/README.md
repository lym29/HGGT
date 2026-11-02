# HGGT training

Fine-tune the public HGGT model with alternating single-view hand data and synthetic multi-view data. This recipe starts from a compatible HGGT checkpoint and trains the backbone, camera head, and MANO head together.

Run the commands below from the repository root, in the environment described in the [installation guide](../README.md#⚙️-installation). Training requires CUDA GPUs with bf16 support and the additional dependencies in `training/requirements.txt`:

```bash
pip install -r training/requirements.txt
```

## Data and model setup

Prepare the single-view training data, the [HGGT synthetic dataset](https://huggingface.co/datasets/catmint123/HGGT-synthetic-data), and the licensed [MANO models](https://mano.is.tue.mpg.de/). Set the paths for your installation:

```bash
export HAMER_DATA_DIR=/path/to/hamer_data
export HGGT_SYNTHETIC_DIR=/path/to/hggt_synthetic
export MANO_MODEL_DIR=/path/to/mano_v1_2/models
export HGGT_INIT_CHECKPOINT=/path/to/hggt_checkpoint.pt
export HGGT_OUTPUT_DIR=outputs/train
```

See the [training data guide](../hggt/training/data/README.md) for download sources, shard names, synthetic sequence layout, and split files. `HAMER_DATA_DIR` contains `dataset_tars/`; `HGGT_SYNTHETIC_DIR` contains `splits/train.txt` and the extracted sequences; `MANO_MODEL_DIR` contains `MANO_RIGHT.pkl`.

The config maps these variables to `paths.hamer_data`, `paths.synthetic_data`, `paths.mano_models`, `checkpoint.init_path`, and `logging.output_dir`. `MANO_MODEL_DIR` defaults to `assets/mano_v1_2/models`, and `HGGT_OUTPUT_DIR` defaults to `outputs/train`.

You can initialize from the released [HGGT weights](https://huggingface.co/catmint123/HGGT):

```bash
export HGGT_INIT_CHECKPOINT="$(python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("catmint123/HGGT", "pytorch_model.bin"))')"
```

Initialization accepts a compatible HGGT state dict, a `.pt` checkpoint containing `model`, or a `.safetensors` file. It loads model weights and starts a new optimizer and training schedule. The model uses camera and MANO heads, with `mano_qk_norm=false`.

## Check the configuration and data

Inspect the resolved configuration without model weights, data, or GPUs:

```bash
python -m hggt.training.launch \
  --config training/configs/finetune.yaml --check-config
```

After preparing the data, check one batch from each training stream on the CPU:

```bash
python -m hggt.training.launch \
  --config training/configs/finetune.yaml --check-data
```

The data check verifies required tensor shapes and finite values before a training allocation is used.

## Train

The default [configuration](configs/finetune.yaml) uses:

| Setting | Value |
| --- | --- |
| GPUs | 4 |
| Input resolution | 518 pixels |
| Images per GPU per microbatch | Up to 32, shared across views |
| Data streams | Alternating single-view and synthetic multi-view |
| Synthetic views per sample | 2–10 |
| Gradient accumulation | 4 microbatches |
| Training length | 20 epochs, 800 microbatches per epoch |
| Learning rate | `1e-6`, 5% warmup from `1e-8`, cosine decay to `1e-8` |
| Weight decay | `0.05` |
| Precision | bf16 |
| Trainable modules | Backbone, camera head, and MANO head |

```bash
torchrun --standalone --nproc_per_node=4 \
  -m hggt.training.launch --config training/configs/finetune.yaml
```

Override configuration values with OmegaConf dotlist arguments:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m hggt.training.launch --config training/configs/finetune.yaml \
  max_img_per_gpu=16 num_workers=2 logging.output_dir=outputs/finetune
```

Changing GPU count, the image limit, or gradient accumulation changes the effective batch size. For a short GPU smoke run:

```bash
torchrun --standalone --nproc_per_node=1 \
  -m hggt.training.launch --config training/configs/finetune.yaml \
  max_img_per_gpu=2 num_workers=2 \
  'data.train.common_config2.img_nums=[2,2]' \
  'data.val.common_config2.img_nums=[2,2]' \
  train.epochs=1 train.batches_per_epoch=4 train.save_every=1 \
  val.batches=0 logging.output_dir=outputs/train_smoke
```

For Slurm, activate the environment, export the paths above, and submit the [launcher](slurm/train.slurm) from the repository root:

```bash
sbatch training/slurm/train.slurm
```

Add the partition, account, or QoS required by your cluster to the `sbatch` command. Adjust the resource directives for your allocation; `NPROC_PER_NODE` must match the requested GPUs. The script starts one Slurm task, which launches the local workers with `torchrun`. To activate conda inside the job, set both `CONDA_SH` to your `conda.sh` path and `CONDA_ENV` to your environment name. Dotlist overrides can also follow the script name.

## Checkpoints and resume

The output directory contains the resolved `config.yaml`, `metrics.jsonl`, TensorBoard logs, and `checkpoints/`. `checkpoints/checkpoint.pt` is updated after each completed epoch; numbered checkpoints are saved at the `train.save_every` interval.

Resume an epoch-boundary checkpoint produced by this training entry point:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m hggt.training.launch --config training/configs/finetune.yaml \
  checkpoint.init_path=null \
  checkpoint.resume_path=outputs/train/checkpoints/checkpoint.pt
```

Resume restores the training state as well as model weights. Set `checkpoint.init_path=null` when resuming, including when `HGGT_INIT_CHECKPOINT` is exported. Use `checkpoint.init_path` to start a new fine-tuning run from weights instead.

Keep the same GPU count and training recipe when resuming; the trainer checks the saved schedule, optimizer settings, and batch configuration.

## Validation and evaluation

The default validation alternates HO3D training data and synthetic test examples, stopping when a stream ends or the `val.batches` limit is reached. These losses are training diagnostics; use the separate [evaluation workflow](../eval/README.md) for held-out benchmark metrics. Evaluate the trained model by passing `checkpoints/checkpoint.pt` to the existing evaluation script's `--checkpoint` argument.

## Acknowledgements

The training components build on [VGGT](https://github.com/facebookresearch/vggt) and the hand data processing in [HaMeR](https://github.com/geopavlakos/hamer). Their applicable upstream licenses are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), including the CC BY-NC 4.0 terms for the VGGT-derived training components and the MIT terms for HaMeR. Follow each dataset's access and license terms.
