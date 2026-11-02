# Training data

The training recipe alternates real **single-view** hand images and synthetic **multi-view** hand-object sequences. Set the three paths through `HAMER_DATA_DIR`, `HGGT_SYNTHETIC_DIR`, and `MANO_MODEL_DIR`, or the corresponding `paths.*` configuration fields.

## MANO

Obtain MANO from the [official project](https://mano.is.tue.mpg.de/) and follow its license. `MANO_MODEL_DIR` must contain `MANO_RIGHT.pkl`. The loader uses the right-hand model with the MANO mean hand pose, full axis-angle pose parameters, and OpenPose's 21-joint order. The synthetic annotations use meters and world-to-camera OpenCV extrinsics. MANO files are not redistributed here.

## Real single-view images

Obtain the training data following [HaMeR's data preparation instructions](https://github.com/geopavlakos/hamer#training), including the source datasets' access requirements. Keep HaMeR's WebDataset tar shards:

```text
HAMER_DATA_DIR/
└── dataset_tars/
    ├── freihand-train/*.tar
    ├── ho3d-train/*.tar
    ├── interhand26m-train/*.tar
    ├── h2o3d-train/*.tar
    ├── dex-train/*.tar
    ├── rhd-train/*.tar
    ├── mtc-train/*.tar
    ├── cocow-train/*.tar
    ├── halpe-train/*.tar
    └── mpiinzsl-train/*.tar
```

Each image (`jpg`, `jpeg`, or `png`) shares a WebDataset key with `data.pyd`, a pickled list of hand annotations. An annotation includes `center`, `scale` (bounding-box dimensions divided by 200), and `keypoints_2d` shaped `[21,3]` with confidence in the last column. Optional supervision uses `keypoints_3d` `[21,4]`, `hand_pose` `[48]`, `betas` `[10]`, `has_hand_pose`, `has_betas`, and `right`. Both MANO availability flags must be positive to use pose/shape supervision. Samples with fewer than four 2D keypoints above confidence 0.3 are filtered. Left hands are mirrored into the right-hand convention.

These datasets stream samples. `epoch_size` is a **virtual sampling length**, not the number of unique annotations: each real dataset has length 50,000 in the provided recipe, so the ten datasets have equal sampling weight. An explicit `max_samples` caps that virtual length; it does not create a fixed subset of the tar contents. Workers cycle through their assigned shards, and report an error if no valid annotations remain. Shards are partitioned across ranks and workers; when there are fewer shards than workers, some workers repeat a shard and emit a warning. Worker count affects the sample order.

The validation recipe samples the HaMeR HO3D **training** shards with augmentation disabled. This is a training diagnostic, not held-out HO3D evaluation or the paper's reported evaluation protocol.

## Synthetic multi-view sequences

Download [HGGT-synthetic-data](https://huggingface.co/datasets/catmint123/HGGT-synthetic-data). The current release puts tar shards inside `data/small`, `data/medium`, and `data/large`, with `splits/` at the repository root. Extract every shard into its containing category directory and set `HGGT_SYNTHETIC_DIR` to the download root:

```text
HGGT_SYNTHETIC_DIR/
├── splits/
│   ├── train.txt
│   └── test.txt
└── data/
    └── small/                       # also medium/ and large/
        └── <object-id>/
            └── <motion-id>/
                └── sequence_<id>/
                    ├── info.json
                    ├── images/<view>.png
                    ├── camera/
                    │   ├── intrinsics.txt
                    │   └── extrinsics/<view>.txt
                    └── masks/
                        ├── <view>_hand_right_mask0001.png
                        └── <view>_object_mask0001.png
```

The loader also accepts category directories directly under the root. Split entries can identify `category/object/motion/sequence_<id>`, `category/object`, or an object ID. Object entries expand to their motion sequences. Full sequence paths avoid ambiguity; duplicate entries are removed while preserving order. Paths must be relative and the selected files must already be extracted.

`info.json` must embed `mano_params` with `global_orient` `[3]`, `hand_pose` `[45]`, `betas` `[10]`, and `transl` `[3]`. External motion-file references are not needed. Intrinsics are `[3,3]`, and extrinsics are `[3,4]` or `[4,4]`. At least eight RGB views are expected per sequence. RGB images, camera files, and matching hand/object masks are required. The loader computes world-space MANO vertices/joints on CPU and caches metadata for up to 128 sequences per worker.

Optional regular depths are read from `depths/<view>.npy` when present. Missing regular depth produces a zero placeholder. The hand-only recipe does not supervise depth, point maps, or the released `multi_depth_gt` layers. Do not enable a depth loss without supplying the required depth supervision.

## Batch interface

`AlternatingMultiDataLoader.get_loader(epoch)` yields `(batch, dataset_name, accumulation_index)`. A complete pair contains a single-view batch followed by a synthetic multi-view batch. All samples in each batch share the view count and aspect ratio. The synthetic sampler keeps `batch_size * views <= max_img_per_gpu`; sampling works in a single process and in distributed training. `images` has shape `[B,S,3,H,W]` in `[0,1]`, camera arrays retain the view dimension, and MANO pose/shape/vertices/3D joints are sequence-level fields. Synthetic targets remain in world coordinates until the trainer normalizes them into the first camera frame.
