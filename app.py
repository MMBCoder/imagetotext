import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2

# Initialize EasyOCR reader (English as default language)
reader = easyocr.Reader(['en'])

# Streamlit UI
st.title("Image to Text Extraction")

uploaded_image = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    img_array = np.array(image)

    # Convert RGB to BGR format for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform OCR to extract text
    with st.spinner("Extracting text..."):
        results = reader.readtext(img_array, detail=0, paragraph=True)

    # Display extracted text
    st.subheader("Extracted Text")
    if results:
        extracted_text = "\n".join(results)
        st.write(extracted_text)
    else:
        st.write("No text detected!")