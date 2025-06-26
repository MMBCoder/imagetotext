import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image, ImageOps

# Initialize the OCR reader once
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)  # Use GPU=True if available

reader = load_reader()

# Title
st.title("📟 Advanced Image to Text Converter")

# Upload section
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)  # Remove noise
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Perform OCR
    with st.spinner("🔍 Extracting text..."):
        result = reader.readtext(thresh, detail=0, paragraph=True)

    # Display results
    st.subheader("📄 Extracted Text:")
    if result:
        full_text = "\n".join(result)
        st.text_area("Output", full_text, height=400)
    else:
        st.warning("No text found in the image.")
