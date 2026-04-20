from __future__ import annotations

from collections import deque
import io

import numpy as np
from PIL import Image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    rgb = image[..., :3].astype(np.float32)
    return (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        padded[:-2, :-2]
        + 2 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        - padded[2:, :-2]
        - 2 * padded[2:, 1:-1]
        - padded[2:, 2:]
    )
    return np.sqrt((gx * gx) + (gy * gy))


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_val) / (max_val - min_val)
    return (scaled * 255).clip(0, 255).astype(np.uint8)


def sharpen(gray: np.ndarray, strength: float) -> np.ndarray:
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    blur = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    return gray + (strength * (gray - blur))


def load_image(uploaded_file: io.BytesIO) -> np.ndarray:
    return np.array(Image.open(uploaded_file))


def threshold_mask(gray: np.ndarray, percentile: int) -> np.ndarray:
    return gray >= np.percentile(gray, percentile)


def connected_components(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict[str, float]] = []
    label_id = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue

            queue: deque[tuple[int, int]] = deque([(r, c)])
            pixels: list[tuple[int, int]] = []
            visited[r, c] = True

            while queue:
                rr, cc = queue.popleft()
                pixels.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            area = len(pixels)
            if area < min_pixels:
                continue

            label_id += 1
            rows = np.array([p[0] for p in pixels])
            cols = np.array([p[1] for p in pixels])
            labels[rows, cols] = label_id

            min_r, max_r = int(rows.min()), int(rows.max())
            min_c, max_c = int(cols.min()), int(cols.max())
            bbox_h = max_r - min_r + 1
            bbox_w = max_c - min_c + 1

            perimeter = 0
            for pr, pc in pixels:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = pr + dr, pc + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width or not mask[nr, nc]:
                        perimeter += 1

            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
            components.append(
                {
                    "label": float(label_id),
                    "area_px": float(area),
                    "perimeter_px": float(perimeter),
                    "bbox_width_px": float(bbox_w),
                    "bbox_height_px": float(bbox_h),
                    "aspect_ratio": float(bbox_w / bbox_h if bbox_h else 0.0),
                    "circularity": float(circularity),
                }
            )

    components.sort(key=lambda item: item["area_px"], reverse=True)
    return labels, components


def label_overlay(image: np.ndarray, labels: np.ndarray, highlight_label: int | None) -> np.ndarray:
    if image.ndim == 2:
        base = np.stack([image, image, image], axis=-1).astype(np.uint8)
    else:
        base = image[..., :3].copy().astype(np.uint8)

    if highlight_label is None:
        return base
    region = labels == highlight_label
    if not np.any(region):
        return base

    padded = np.pad(region, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    border = region & (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
    )
    base[border] = np.array([255, 0, 0], dtype=np.uint8)
    return base
