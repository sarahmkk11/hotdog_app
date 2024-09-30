from transformers import pipeline
import streamlit as st
from PIL import Image


pipe = pipeline("image-classification", model="julien-c/hotdog-not-hotdog")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)
    predictions = pipe(img)
    st.write(predictions)
