# Multi-view example crops for HGGT

Pre-cropped hand images exported from POEM WebDataset shards
(`/data1/DATA/POEM-v2/`), matching the datasets used by
`demo_hand_poem.py` in HOIrecon.

Layout:

```text
examples/multiview/
  <Dataset>/sample_XXXX/view_YY.jpg
```

| Dataset   | Samples     | Notes                          |
|-----------|-------------|--------------------------------|
| HO3D      | 0000, 0100  | multi-view                     |
| DexYCB    | 0000, 0100  | multi-view                     |
| Arctic    | 0000, 0100  | multi-view                     |
| Interhand | 0000, 0100  | multi-view                     |
| Oakink    | 0000, 0100  | multi-view                     |
| Freihand  | 0000, 0100  | single-view (`view_00.jpg`)    |

Images are already square hand crops at 518×518 (training-style input).
No detector is required for the v1 demo.

Re-export (maintainers only):

```bash
python scripts/export_poem_examples.py \
  --hoirecon_root /path/to/HOIrecon-bare-repo/main \
  --data_root /data1/DATA/POEM-v2/ \
  --output_root examples/multiview \
  --samples 0 100
```
