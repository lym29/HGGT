# Multi-view examples

Pre-cropped hand images for the HGGT demo (`demo/demo_multiview_images.py`).

```text
examples/multiview/
  <Dataset>/sample_XXXX/view_YY.jpg
```

| Dataset   | Samples     | Notes                       |
|-----------|-------------|-----------------------------|
| HO3D      | 0000, 0100  | multi-view                  |
| DexYCB    | 0000, 0100  | multi-view                  |
| Arctic    | 0000, 0100  | multi-view                  |
| Interhand | 0000, 0100  | multi-view                  |
| Oakink    | 0000, 0100  | multi-view                  |
| Freihand  | 0000, 0100  | single-view (`view_00.jpg`) |

Images are square hand crops (518×518). No detector is required for this demo.
