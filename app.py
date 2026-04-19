from __future__ import annotations

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
    # Luma transform (ITU-R BT.601)
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

    magnitude = np.sqrt((gx * gx) + (gy * gy))
    return magnitude


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


st.title("🔬 Autonanovision")
st.caption("Nano-imaging sandbox: upload an image and inspect contrast, sharpness, and edge maps.")

with st.sidebar:
    st.header("Analysis controls")
    edge_percentile = st.slider(
        "Edge highlight percentile",
        min_value=50,
        max_value=99,
        value=85,
        help="Higher values isolate only the strongest edges.",
    )
    sharpen_strength = st.slider(
        "Sharpen strength",
        min_value=0.0,
        max_value=2.0,
        value=0.8,
        step=0.1,
    )

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

edge_density = float((edge_mask > 0).mean() * 100)
contrast = float(gray.std())

k1, k2, k3 = st.columns(3)
k1.metric("Image size", f"{image.shape[1]}×{image.shape[0]}")
k2.metric("Contrast (std)", f"{contrast:.2f}")
k3.metric("Edge density", f"{edge_density:.2f}%")

st.markdown("---")
st.write(
    "Tip: This starter app is intentionally lightweight and NumPy-based, "
    "so it deploys easily on Streamlit Community Cloud."
)
