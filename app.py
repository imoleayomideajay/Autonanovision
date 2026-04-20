from __future__ import annotations

import numpy as np
import streamlit as st

from autonanovision.analysis import (
    connected_components,
    label_overlay,
    load_image,
    normalize_uint8,
    sharpen,
    sobel_edges,
    threshold_mask,
    to_grayscale,
)

st.set_page_config(page_title="Autonanovision", page_icon="🔬", layout="wide")

st.title("🔬 Autonanovision")
st.caption("Nano-imaging sandbox: upload an image and inspect edges plus size/shape statistics.")

with st.sidebar:
    st.header("Analysis controls")
    edge_percentile = st.slider("Edge highlight percentile", min_value=50, max_value=99, value=85)
    sharpen_strength = st.slider("Sharpen strength", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    mask_percentile = st.slider("Bright region percentile", min_value=50, max_value=99, value=80)
    min_component_pixels = st.slider("Minimum component size (px)", min_value=10, max_value=5000, value=200, step=10)

uploaded = st.file_uploader("Upload microscopy or camera image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an image to begin analysis.")
    st.stop()

image = load_image(uploaded)
gray = to_grayscale(image)
sharpened = sharpen(gray, sharpen_strength)
edges = sobel_edges(gray)

edge_threshold = np.percentile(edges, edge_percentile)
edge_mask = (edges >= edge_threshold).astype(np.uint8) * 255

binary_mask = threshold_mask(sharpened, mask_percentile)
labels, components = connected_components(binary_mask, min_component_pixels)

component_ids = [int(component["label"]) for component in components]
selected_label = st.selectbox(
    "Highlight component",
    options=[None] + component_ids,
    format_func=lambda x: "None" if x is None else f"Label {x}",
)
overlay = label_overlay(image, labels, selected_label)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Sharpened grayscale")
    st.image(normalize_uint8(sharpened), use_container_width=True, clamp=True)
with col3:
    st.subheader("Edge map")
    st.image(edge_mask, use_container_width=True, clamp=True)

st.subheader("Segmentation preview")
st.image(overlay, use_container_width=True, caption="Selected component boundary shown in red")

edge_density = float((edge_mask > 0).mean() * 100)
contrast = float(gray.std())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Image size", f"{image.shape[1]}×{image.shape[0]}")
k2.metric("Contrast (std)", f"{contrast:.2f}")
k3.metric("Edge density", f"{edge_density:.2f}%")
k4.metric("Components", f"{len(components)}")

st.subheader("Size and shape statistics")
if not components:
    st.warning("No components found with current threshold/settings.")
else:
    rows = []
    for idx, comp in enumerate(components[:50], start=1):
        rows.append(
            {
                "rank": idx,
                "label": int(comp["label"]),
                "area_px": int(comp["area_px"]),
                "perimeter_px": int(comp["perimeter_px"]),
                "bbox_w_px": int(comp["bbox_width_px"]),
                "bbox_h_px": int(comp["bbox_height_px"]),
                "aspect_ratio": round(comp["aspect_ratio"], 3),
                "circularity": round(comp["circularity"], 3),
            }
        )

    st.dataframe(rows, use_container_width=True)

    st.subheader("Component graphs")
    top_n = min(20, len(components))
    top_components = components[:top_n]
    st.caption("Top components by area (descending).")
    st.bar_chart({"area_px": [c["area_px"] for c in top_components]})

    st.caption("Circularity trend for top components.")
    st.line_chart({"circularity": [c["circularity"] for c in top_components]})

# Global intensity graph
hist_counts, _ = np.histogram(gray, bins=32)
st.subheader("Grayscale intensity distribution")
st.area_chart({"pixels": hist_counts.tolist()})

st.markdown("---")
st.write("Tip: set your microscope calibration to convert pixel statistics to real-world units.")
