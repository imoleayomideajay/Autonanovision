# Autonanovision

A lightweight Streamlit app for nano/micro image inspection with interactive size and shape statistics.

## Features

- Upload microscopy/camera images.
- NumPy-based grayscale, sharpening, and Sobel edge maps.
- Bright-region segmentation with connected-component extraction.
- Per-object statistics: area, perimeter, bounding box, aspect ratio, circularity.
- Optional component highlighting directly on the original image.

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

- `app.py` – Streamlit user interface.
- `autonanovision/analysis.py` – pure processing/statistics functions.
- `tests/test_analysis.py` – basic unit tests for analysis utilities.
