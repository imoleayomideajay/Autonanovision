from __future__ import annotations

import numpy as np
import streamlit as st

import autonanovision.analysis as analysis
from autonanovision.analysis import (
    auto_select_grayscale,
    components_csv,
    connected_components,
    filter_component_rows_by_size,
    enrich_components,
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
st.caption("Production-ready nano/micro image QC: edges, segmentation, calibrated size/shape stats, and exports.")

with st.sidebar:
    st.header("Analysis controls")
    known_particle_size_um = st.slider("Known particle size (µm)", min_value=0.1, max_value=200.0, value=10.0, step=0.1)
    edge_percentile = st.slider("Edge highlight percentile", min_value=50, max_value=99, value=85)
    sharpen_strength = st.slider("Sharpen strength", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    threshold_method = st.selectbox(
        "Threshold method",
        options=["auto", "percentile", "otsu", "mean", "triangle", "li", "adaptive_mean"],
        index=0,
        help=(
            "auto: pick percentile vs otsu by contrast • "
            "percentile: bright-region cutoff • "
            "otsu: between-class variance • "
            "mean: global mean intensity • "
            "triangle: Zack's triangle method (skewed histograms) • "
            "li: iterative cross-entropy • "
            "adaptive_mean: per-pixel local mean (handles uneven lighting)"
        ),
    )
    mask_percentile = st.slider(
        "Bright region percentile",
        min_value=50,
        max_value=99,
        value=80,
        disabled=threshold_method not in ("auto", "percentile"),
    )
    adaptive_window = st.slider(
        "Adaptive window (px)",
        min_value=3,
        max_value=151,
        value=31,
        step=2,
        disabled=threshold_method != "adaptive_mean",
    )
    adaptive_offset = st.slider(
        "Adaptive offset",
        min_value=-30.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        disabled=threshold_method != "adaptive_mean",
    )
    min_component_pixels = st.slider("Minimum component size (px)", min_value=10, max_value=10000, value=200, step=10)
    st.header("Manufacturer size filter (optional)")
    manufacturer_min_um = st.number_input("Min particle size (µm)", min_value=0.0, value=0.0, step=0.1)
    manufacturer_max_um = st.number_input("Max particle size (µm)", min_value=0.0, value=0.0, step=0.1)

    st.header("Calibration (optional)")
    microns_per_pixel = st.number_input(
        "Microns per pixel (µm/px)", min_value=0.0, value=0.0, step=0.01, help="Set > 0 to output calibrated units."
    )

uploaded = st.file_uploader("Upload microscopy or camera image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an image to begin analysis.")
    st.stop()

image = load_image(uploaded)
gray_default = to_grayscale(image)
gray = auto_select_grayscale(image)
sharpened = sharpen(gray, sharpen_strength)
edges = sobel_edges(gray)

edge_percentile_dynamic = edge_percentile
if microns_per_pixel > 0 and known_particle_size_um > 0:
    particle_px = known_particle_size_um / microns_per_pixel
    edge_percentile_dynamic = int(np.clip(92 - (0.35 * particle_px), 60, 98))

edge_threshold = np.percentile(edges, edge_percentile_dynamic)
edge_mask = (edges >= edge_threshold).astype(np.uint8) * 255

effective_method = threshold_method
effective_percentile = mask_percentile
if threshold_method == "auto":
    contrast = float(gray.std())
    effective_method = "otsu" if contrast > 18 else "percentile"
    if effective_method == "percentile":
        effective_percentile = int(np.clip(75 + (contrast * 0.5), 70, 95))

binary_mask = threshold_mask(
    sharpened,
    percentile=effective_percentile,
    method=effective_method,
    adaptive_window=adaptive_window,
    adaptive_offset=adaptive_offset,
)
labels, components = connected_components(binary_mask, min_component_pixels)
try:
    rows = enrich_components(components, microns_per_pixel)
    rows = filter_component_rows_by_size(rows, manufacturer_min_um, manufacturer_max_um, microns_per_pixel)
except Exception as exc:
    st.error(f"Failed to build calibrated component table: {exc}")
    rows = enrich_components(components, 0.0)

component_ids = [int(component["label"]) for component in components]
default_selected = component_ids[0] if component_ids else None
selected_label = st.selectbox(
    "Highlight component",
    options=[None] + component_ids,
    index=([None] + component_ids).index(default_selected) if default_selected is not None else 0,
    format_func=lambda x: "None" if x is None else f"Label {x}",
)
overlay = label_overlay(image, labels, selected_label)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Sharpened grayscale (auto-selected)")
    st.image(normalize_uint8(sharpened), use_container_width=True, clamp=True)
with col3:
    st.subheader("Edge map")
    st.image(edge_mask, use_container_width=True, clamp=True)

st.subheader("Segmentation preview")
st.image(overlay, use_container_width=True, caption="Selected component boundary shown in red")
if gray_default.shape == gray.shape and np.any(gray_default != gray):
    st.caption("Auto grayscale selected a channel/projection different from default RGB weights.")

edge_density = float((edge_mask > 0).mean() * 100)
contrast = float(gray.std())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Image size", f"{image.shape[1]}×{image.shape[0]}")
k2.metric("Contrast (std)", f"{contrast:.2f}")
k3.metric("Edge density", f"{edge_density:.2f}%")
k4.metric("Components", f"{len(components)}")
st.caption(f"Segmentation mode: **{effective_method}**. Edge percentile in use: **{edge_percentile_dynamic}**.")

st.subheader("Size and shape statistics")
if not rows:
    st.warning("No components found with current threshold/settings.")
else:
    mean_area = np.mean([r["area_px"] for r in rows])
    median_area = np.median([r["area_px"] for r in rows])
    mean_circularity = np.mean([r["circularity"] for r in rows])

    s1, s2, s3 = st.columns(3)
    s1.metric("Mean area (px²)", f"{mean_area:.1f}")
    s2.metric("Median area (px²)", f"{median_area:.1f}")
    s3.metric("Mean circularity", f"{mean_circularity:.3f}")

    st.dataframe(rows, use_container_width=True)

    st.download_button(
        "Download component statistics (CSV)",
        data=components_csv(rows),
        file_name="autonanovision_components.csv",
        mime="text/csv",
    )

    st.subheader("Component graphs")
    top_n = min(20, len(rows))
    st.caption("Top components by area (descending).")
    st.bar_chart({"area_px": [r["area_px"] for r in rows[:top_n]]})

    st.caption("Circularity trend for top components.")
    st.line_chart({"circularity": [r["circularity"] for r in rows[:top_n]]})

# Global intensity graph
hist_counts, _ = np.histogram(gray, bins=32)
st.subheader("Grayscale intensity distribution")
st.area_chart({"pixels": hist_counts.tolist()})

st.markdown("---")
if microns_per_pixel > 0:
    st.success(f"Calibration active: 1 px = {microns_per_pixel:.4f} µm.")
else:
    st.info("Tip: set microscope calibration to get physical units (µm, µm²) in the table export.")
