# Autonanovision

A lightweight Streamlit app for quick nano/micro image inspection.

## What it does

- Upload a microscopy or camera image.
- Generates a grayscale view and applies adjustable sharpening.
- Computes a Sobel-based edge map (NumPy only).
- Reports quick quality metrics (contrast and edge density).

## Local development

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
