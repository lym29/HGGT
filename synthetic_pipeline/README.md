# Synthetic Data Pipeline

## Installation

This pipeline requires three external data sources: the GraspXL dataset, Objaverse
objects, and hand textures from DART.

### 1. Download the GraspXL dataset

Download the GraspXL dataset following the official instructions:

- [GraspXL GitHub repository](https://github.com/zdchan/GraspXL)

The object meshes provided by GraspXL are texture-free. To render textured
objects, we download the original Objaverse assets using the Objaverse IDs
extracted from the GraspXL data. Download the prepared ID list from Hugging Face
and place it under the repository `assets` directory:

```bash
wget -O assets/graspxl_obj_ids.txt \
  https://huggingface.co/datasets/catmint123/HGGT-synthetic-data/resolve/main/metadata/graspxl_obj_ids.txt
```

### 2. Download Objaverse objects

After preparing `graspxl_obj_ids.txt`, download the Objaverse objects with:

```bash
cd synthetic_pipeline

OBJ_ID_FILE=../assets/graspxl_obj_ids.txt
OBJAVERSE_DIR=<path/to/objaverse_download_dir>

python download_objaverse.py \
  --uid_file ${OBJ_ID_FILE} \
  --download_path ${OBJAVERSE_DIR}
```

Both `--uid_file` and `--download_path` are required.

You can also change the number of parallel download processes:

```bash
python download_objaverse.py \
  --uid_file ${OBJ_ID_FILE} \
  --download_path ${OBJAVERSE_DIR} \
  --download_processes 16
```

### 3. Download hand textures

Download DART's raw hand textures and accessories:

- [DART project](https://github.com/DART2022/DART?tab=readme-ov-file)
- [Raw textures and accessories](https://drive.google.com/file/u/1/d/1_KPzMFjXLHagPhhos7NXvzdzMMN-b1bd/view)

After downloading, extract the files and place them in the texture directory used
by your rendering configuration.
