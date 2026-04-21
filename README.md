# Autonanovision

A Streamlit app for nano/micro image quality control with interactive segmentation and calibrated morphology analytics.

## Features

- Upload microscopy/camera images.
- NumPy-based grayscale, sharpening, and Sobel edge maps.
- Segmentation using either percentile thresholding or Otsu auto-thresholding.
- Per-object statistics: area, perimeter, bounding box, aspect ratio, circularity.
- Optional microscope calibration (`µm/px`) to derive physical units (`µm`, `µm²`).
- Built-in graphs for component area/circularity and grayscale intensity distribution.
- One-click CSV export of component statistics.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Set **Main file path** to `app.py`.
4. Deploy.

## Layout

- `app.py` – Streamlit user interface and dashboards.
- `autonanovision/analysis.py` – reusable image-processing/statistics core.
- `tests/test_analysis.py` – unit tests for analysis logic.
