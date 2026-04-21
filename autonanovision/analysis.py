from __future__ import annotations

from collections import deque
import csv
import io
from typing import Any

import numpy as np
from PIL import Image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA image array to grayscale."""
    if image.ndim == 2:
        return image.astype(np.float32)
    rgb = image[..., :3].astype(np.float32)
    return (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Compute edge intensity with a Sobel operator using NumPy only."""
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
    """Normalize float image to uint8 display range."""
    image = np.asarray(image, dtype=np.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_val) / (max_val - min_val)
    return (scaled * 255).clip(0, 255).astype(np.uint8)


def sharpen(gray: np.ndarray, strength: float) -> np.ndarray:
    """Simple unsharp masking-style enhancement."""
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
    """Load uploaded image file into NumPy array."""
    return np.array(Image.open(uploaded_file))


def otsu_threshold(gray: np.ndarray) -> float:
    """Compute Otsu threshold on grayscale image."""
    values = normalize_uint8(gray).ravel()
    hist = np.bincount(values, minlength=256).astype(np.float64)
    total = values.size
    cumulative = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumulative_mean[-1]

    numerator = (global_mean * cumulative - cumulative_mean) ** 2
    denominator = cumulative * (total - cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_squared = np.where(denominator > 0, numerator / denominator, 0)

    threshold = int(np.argmax(sigma_b_squared))
    return float(threshold)


def threshold_mask(gray: np.ndarray, percentile: int | None = None, method: str = "percentile") -> np.ndarray:
    """Build binary foreground mask from grayscale array."""
    method = method.lower()
    if method == "otsu":
        threshold = otsu_threshold(gray)
        return normalize_uint8(gray) >= threshold

    if percentile is None:
        raise ValueError("percentile must be provided for percentile threshold method")
    threshold = np.percentile(gray, percentile)
    return gray >= threshold


def connected_components(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Label 8-connected components and return shape stats."""
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


def enrich_components(components: list[dict[str, float]], microns_per_pixel: float) -> list[dict[str, Any]]:
    """Return component rows with optional calibrated real-world units."""
    rows: list[dict[str, Any]] = []
    has_calibration = microns_per_pixel > 0

    for idx, comp in enumerate(components[:100], start=1):
        row: dict[str, Any] = {
            "rank": idx,
            "label": int(comp["label"]),
            "area_px": int(comp["area_px"]),
            "perimeter_px": int(comp["perimeter_px"]),
            "bbox_w_px": int(comp["bbox_width_px"]),
            "bbox_h_px": int(comp["bbox_height_px"]),
            "aspect_ratio": round(comp["aspect_ratio"], 3),
            "circularity": round(comp["circularity"], 3),
        }

        if has_calibration:
            row["area_um2"] = round(comp["area_px"] * (microns_per_pixel**2), 3)
            row["perimeter_um"] = round(comp["perimeter_px"] * microns_per_pixel, 3)
            row["bbox_w_um"] = round(comp["bbox_width_px"] * microns_per_pixel, 3)
            row["bbox_h_um"] = round(comp["bbox_height_px"] * microns_per_pixel, 3)

        rows.append(row)

    return rows


def components_csv(rows: list[dict[str, Any]]) -> str:
    """Serialize component table rows to CSV text."""
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def label_overlay(image: np.ndarray, labels: np.ndarray, highlight_label: int | None) -> np.ndarray:
    """Overlay selected component boundary in red on top of original image."""
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
