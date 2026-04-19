from __future__ import annotations

from collections import deque
import io

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Autonanovision", page_icon="🔬", layout="wide")


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
    """Simple unsharp masking style enhancement."""
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
    detail = gray - blur
    return gray + (strength * detail)


def load_image(uploaded_file: io.BytesIO) -> np.ndarray:
    image = Image.open(uploaded_file)
    return np.array(image)


def threshold_mask(gray: np.ndarray, percentile: int) -> np.ndarray:
    """Build a binary mask from grayscale intensity percentile."""
    threshold = np.percentile(gray, percentile)
    return gray >= threshold


def connected_components(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Label 8-connected components and return stats for each large enough component."""
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
            aspect_ratio = bbox_w / bbox_h if bbox_h else 0.0

            components.append(
                {
                    "label": float(label_id),
                    "area_px": float(area),
                    "perimeter_px": float(perimeter),
                    "bbox_width_px": float(bbox_w),
                    "bbox_height_px": float(bbox_h),
                    "aspect_ratio": float(aspect_ratio),
                    "circularity": float(circularity),
                }
            )

    components.sort(key=lambda item: item["area_px"], reverse=True)
    return labels, components


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


st.title("🔬 Autonanovision")
st.caption("Nano-imaging sandbox: upload an image and inspect edges plus size/shape statistics.")

with st.sidebar:
    st.header("Analysis controls")
    edge_percentile = st.slider(
        "Edge highlight percentile",
        min_value=50,
        max_value=99,
        value=85,
        help="Higher values isolate only the strongest edges.",
    )
    sharpen_strength = st.slider("Sharpen strength", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    mask_percentile = st.slider(
        "Bright region percentile",
        min_value=50,
        max_value=99,
        value=80,
        help="Pixels above this percentile are treated as foreground objects.",
    )
    min_component_pixels = st.slider("Minimum component size (px)", min_value=10, max_value=5000, value=200, step=10)

uploaded = st.file_uploader("Upload microscopy or camera image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an image to begin analysis.")
    st.stop()

image = load_image(uploaded)
gray = to_grayscale(image)
edges = sobel_edges(gray)
sharpened = sharpen(gray, sharpen_strength)

edge_threshold = np.percentile(edges, edge_percentile)
edge_mask = (edges >= edge_threshold).astype(np.uint8) * 255

binary_mask = threshold_mask(gray, mask_percentile)
labels, components = connected_components(binary_mask, min_component_pixels)

selected_label = int(components[0]["label"]) if components else None
overlay = label_overlay(image, labels, selected_label)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Grayscale + Sharpened")
    st.image(normalize_uint8(sharpened), use_container_width=True, clamp=True)
with col3:
    st.subheader("Edge map")
    st.image(edge_mask, use_container_width=True, clamp=True)

st.subheader("Object segmentation preview")
st.image(overlay, use_container_width=True, caption="Largest detected component outlined in red")

edge_density = float((edge_mask > 0).mean() * 100)
contrast = float(gray.std())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Image size", f"{image.shape[1]}×{image.shape[0]}")
k2.metric("Contrast (std)", f"{contrast:.2f}")
k3.metric("Edge density", f"{edge_density:.2f}%")
k4.metric("Components", f"{len(components)}")

st.subheader("Size and shape statistics")
if not components:
    st.warning("No components found with current threshold/settings. Try lowering the percentile or minimum size.")
else:
    table_rows = []
    for idx, comp in enumerate(components[:20], start=1):
        table_rows.append(
            {
                "rank": idx,
                "area_px": int(comp["area_px"]),
                "perimeter_px": int(comp["perimeter_px"]),
                "bbox_w_px": int(comp["bbox_width_px"]),
                "bbox_h_px": int(comp["bbox_height_px"]),
                "aspect_ratio": round(comp["aspect_ratio"], 3),
                "circularity": round(comp["circularity"], 3),
            }
        )
    st.dataframe(table_rows, use_container_width=True)

st.markdown("---")
st.write(
    "Tip: for real-world measurements (µm units), add your microscope pixel-to-micron calibration and "
    "scale `area`/`perimeter` values accordingly."
)
