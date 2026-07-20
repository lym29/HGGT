<h1 align="center">HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from Uncalibrated Images</h1>

<p align="center">
  <a href="https://lym29.github.io/">Yumeng Liu</a><sup>1</sup>,
  <a href="https://www.xxlong.site/">Xiao-Xiao Long</a><sup>2</sup>,
  <a href="https://people.mpi-inf.mpg.de/~mhaberma/">Marc Habermann</a><sup>3</sup>,
  <a href="https://github.com/TheVaticanCameos">Xuanze Yang</a><sup>1</sup>,
  <a href="https://clinplayer.github.io/">Cheng Lin</a><sup>4</sup>,
  <br>
  <a href="https://liuyuan-pal.github.io/">Yuan Liu</a><sup>5</sup>,
  <a href="https://yuexinma.me/">Yuexin Ma</a><sup>6</sup>,
  <a href="http://staff.ustc.edu.cn/~lgliu/">Ligang Liu</a><sup>1*</sup>
</p>

<p align="center">
  <sup>1</sup>USTC &nbsp;
  <sup>2</sup>Nanjing University &nbsp;
  <sup>3</sup>MPI-INF &nbsp;
  <sup>4</sup>MUST Macau &nbsp;
  <sup>5</sup>HKUST &nbsp;
  <sup>6</sup>ShanghaiTech &nbsp;
  <br>
  <sup>*</sup>Corresponding author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.23997"><img src="https://img.shields.io/badge/arXiv-2603.23997-b31b1b.svg?logo=arxiv" alt="arXiv"></a>
  <a href="https://lym29.github.io/HGGT/"><img src="https://img.shields.io/badge/Project_Page-HGGT-blue?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/catmint123/HGGT-synthetic-data"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow" alt="Hugging Face Dataset"></a>
  <a href="https://huggingface.co/catmint123/HGGT"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green" alt="Model Weight"></a>
  <a href="https://github.com/lym29/HGGT"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="GitHub"></a>
</p>

---

<p align="center">
  <img src="media/teaser.png" alt="HGGT teaser: 3D hand mesh reconstruction from uncalibrated multi-view images" width="100%">
</p>

<p align="center">
  We introduce <b>H</b>and <b>G</b>eometry <b>G</b>rounding <b>T</b>ransformer (<b>HGGT</b>), a scalable and generalized solution for 3D hand mesh recovery. Our method unifies diverse data sources to achieve robust performance across varying camera viewpoints and environments.
</p>

---

## Table of Contents

- [⚙️ Installation](#⚙️-installation)
- [🚀 Demo (pre-cropped multi-view images)](#🚀-demo-pre-cropped-multi-view-images)
- [📊 Evaluation](#📊-evaluation)
- [📁 Dataset](#📁-dataset)
- [🙏 Acknowledgements](#🙏-acknowledgements)

---

## TL;DR

We present the **first feed-forward framework** that jointly estimates 3D hand meshes and camera poses from **uncalibrated** multi-view images.

## TODO

- [x] Release synthetic dataset on Hugging Face
- [x] Release dataset generation pipeline code (due July 17)
- [x] Release pretrained model checkpoints
- [x] Release model inference code (due July 17)
- [x] Release evaluation scripts (due July 20)

---

## ⚙️ Installation

### Create the conda environment

```bash
conda create -n hggt python=3.10 -y
conda activate hggt

pip install -r requirements.txt
```

### MANO models (required for mesh overlay and evaluation)

Download MANO from the [MANO website](http://mano.is.tue.mpg.de/), unzip, and place the model files under:

```text
assets/mano_v1_2/models/MANO_RIGHT.pkl
```

Mesh overlays and multi-view image evaluation require MANO; loading the network and writing `mano_params` to `result.npz` does not.

### Pretrained weights

Default demo loading uses Hugging Face:

```text
https://huggingface.co/catmint123/HGGT
```

via `HGGT.from_pretrained("catmint123/HGGT")`. A local training checkpoint (`.pt` with a `model` state dict) can be passed with `--checkpoint`.

---

## 🚀 Demo (pre-cropped multi-view images)

This demo reads **already hand-cropped** multi-view images (one square crop per view), runs HGGT, and writes a mosaic / optional mesh overlays / `result.npz`.

Hand detection from full-frame images and video demos will be added in a follow-up release.

```bash
# Example: Arctic sample (multi-view)
python demo/demo_multiview_images.py \
  --image_folder examples/multiview/Arctic/sample_0000 \
  --output_dir outputs/demo_arctic_0000

# Optional: local checkpoint and MANO path
python demo/demo_multiview_images.py \
  --image_folder examples/multiview/HO3D/sample_0000 \
  --checkpoint /path/to/checkpoint.pt \
  --mano_model_path assets/mano_v1_2/models \
  --output_dir outputs/demo_ho3d_0000
```

Bundled examples live under [`examples/multiview/`](examples/multiview/) (2 samples from each of HO3D, DexYCB, Arctic, Interhand, Oakink, Freihand). Freihand examples are single-view.

Typical outputs:

- `input_mosaic.jpg` — input views side-by-side
- `overlay_mosaic.jpg` — solid mesh overlays for all views side-by-side (pyrender)
- `result.npz` — `mano_params`, cameras, and vertices when MANO is available

---

## 📊 Evaluation

Evaluate HGGT on multi-view WebDataset tars from standard hand datasets (HO3D, DexYCB, Arctic, Interhand, Oakink, Freihand).

Download the evaluation data from [JubSteven/POEM-v2](https://huggingface.co/datasets/JubSteven/POEM-v2/tree/main). Full setup, launchers (`run_eval_mv_image.sh`, `eval_mv_image.slurm`), and metric details are in [`eval/README.md`](eval/README.md).

---

## 📁 Dataset

### Download

Our synthetic dataset is available on Hugging Face:

<a href="https://huggingface.co/datasets/catmint123/HGGT-synthetic-data">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow" alt="Hugging Face Dataset">
</a>

```bash
# Download via huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='catmint123/HGGT-synthetic-data',
    repo_type='dataset',
    local_dir='data/hggt_synthetic',
)
"
```

After downloading, extract the tar shards:

```bash
cd data/hggt_synthetic/small
for f in *.tar; do tar -xf "$f"; done
```

Please refer to the [dataset page](https://huggingface.co/datasets/catmint123/HGGT-synthetic-data) for details on the dataset structure.

---

## 🙏 Acknowledgements

We would like to express our gratitude to the authors and contributors of the following projects:

- [VGGT](https://github.com/facebookresearch/vggt)
- [GraspXL](https://github.com/zdchan/GraspXL)
- [POEM-v2](https://github.com/JubSteven/POEM-v2/tree/release)

---

## Citation

If you find our work useful, please consider citing us using the following BibTeX entry:

```bibtex
@article{liu2026hggt,
  title={HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from Uncalibrated Images},
  author={Liu, Yumeng and Long, Xiao-Xiao and Habermann, Marc and Yang, Xuanze and Lin, Cheng and Liu, Yuan and Ma, Yuexin and Liu, Ligang},
  journal={arXiv preprint arXiv:2603.23997},
  year={2026}
}
```
---

## License

This project is licensed under the [Apache License 2.0](LICENSE).


