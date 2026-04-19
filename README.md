# Autonanovision (Streamlit-ready)

This repository is configured to deploy on Streamlit.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set **Main file path** to `app.py`.
4. Deploy.

## Files added for deployment

- `app.py` — Streamlit entrypoint.
- `requirements.txt` — Python dependencies for deployment.
