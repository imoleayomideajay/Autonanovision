import streamlit as st

st.set_page_config(page_title="Autonanovision", page_icon="🔬", layout="centered")

st.title("🔬 Autonanovision")
st.write(
    "This repository is now configured for Streamlit deployment. "
    "Replace this starter UI with your app logic."
)

st.subheader("Quick checks")
st.success("Streamlit is running correctly.")

name = st.text_input("Your name", placeholder="Ada")
if name:
    st.write(f"Welcome, {name}!")

st.markdown("---")
st.caption("Deploy on Streamlit Community Cloud by pointing to `app.py` in this repo.")
