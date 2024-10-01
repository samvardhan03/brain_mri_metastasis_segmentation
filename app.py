import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Send image to FastAPI backend
    files = {"file": ("filename", img_bytes, "image/png")}
    response = requests.post("http://localhost:8000/predict/", files=files)
    
    prediction = np.array(response.json()['prediction'])
    
    st.image(prediction, caption="Metastasis Segmentation Result", use_column_width=True)
