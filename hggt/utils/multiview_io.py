"""Load pre-cropped multi-view hand images for HGGT."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_image_paths(image_folder: str | Path | None = None, image_paths: list[str] | None = None) -> list[Path]:
    """Collect sorted image paths from a folder or an explicit list."""
    if image_paths:
        paths = [Path(p) for p in image_paths]
    elif image_folder:
        folder = Path(image_folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Image folder not found: {folder}")
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        raise ValueError("Either image_folder or image_paths must be provided")

    if not paths:
        raise ValueError("No images found")
    return paths


def load_rgb_images(paths: list[Path]) -> list[np.ndarray]:
    """Load images as RGB uint8 arrays."""
    images: list[np.ndarray] = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return images


def resize_square(image: np.ndarray, img_size: int) -> np.ndarray:
    """Resize a (possibly non-square) crop to ``img_size x img_size``."""
    if image.shape[0] == img_size and image.shape[1] == img_size:
        return image
    return cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)


def images_to_multiview_tensor(
    images: list[np.ndarray],
    device: str | torch.device,
    img_size: int = 518,
) -> torch.Tensor:
    """Stack RGB uint8 crops into ``(1, S, 3, H, W)`` float tensor in ``[0, 1]``."""
    resized = [resize_square(img, img_size) for img in images]
    tensors = [torch.from_numpy(img).float().permute(2, 0, 1) / 255.0 for img in resized]
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)


def load_multiview_folder(
    image_folder: str | Path | None = None,
    image_paths: list[str] | None = None,
    device: str | torch.device = "cpu",
    img_size: int = 518,
) -> tuple[torch.Tensor, list[np.ndarray], list[Path]]:
    """Load a folder (or path list) of pre-cropped views for HGGT."""
    paths = list_image_paths(image_folder, image_paths)
    images = load_rgb_images(paths)
    tensor = images_to_multiview_tensor(images, device=device, img_size=img_size)
    resized = [resize_square(img, img_size) for img in images]
    return tensor, resized, paths


def build_mosaic(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    """Concatenate RGB images horizontally with a small gap."""
    if not images:
        raise ValueError("No images to mosaic")
    h = max(img.shape[0] for img in images)
    tiles = []
    for img in images:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            w = int(round(img.shape[1] * scale))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        tiles.append(img)
        tiles.append(np.full((h, gap, 3), 255, dtype=np.uint8))
    tiles = tiles[:-1]
    return np.concatenate(tiles, axis=1)
