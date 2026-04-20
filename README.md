# Autonanovision

A lightweight Streamlit app for nano/micro image inspection with interactive size and shape statistics.

## Features

- Upload microscopy/camera images.
- NumPy-based grayscale, sharpening, and Sobel edge maps.
- Bright-region segmentation with connected-component extraction.
- Per-object statistics: area, perimeter, bounding box, aspect ratio, circularity.
- Optional component highlighting directly on the original image.
- Built-in graphs for component area/circularity and grayscale intensity distribution.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Streamlit Community Cloud deployment

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from this repository.
3. Use `app.py` as the **Main file path**.
4. Deploy.

## Project files

- `app.py` – Streamlit entrypoint and image-analysis UI.
- `requirements.txt` – deployment/runtime dependencies.
- `.streamlit/config.toml` – Streamlit runtime/theme config.

## Notes

- Shape stats are pixel-based unless you apply instrument calibration.
- You can tune percentile and minimum-component filters from the sidebar to stabilize segmentation.
