import streamlit as st
import easyocr
import cv2
from PIL import Image
import numpy as np
import os

os.environ['SSL_CERT_FILE'] = ''

# Your Streamlit app code here


# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to preprocess the image
def preprocess_image(image):
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Resize the image by a factor of 2
    resized_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    return gray_img

# Function to perform OCR on the image
def ocr_image(image):
    processed_img = preprocess_image(image)
    results = reader.readtext(processed_img)
    return results

# Streamlit app layout
st.title('License Plate Recognition')
st.write("Upload an image of a license plate and the app will detect and display the license plate number.")

# File uploader for user to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform OCR and display the result
    st.write("Processing...")
    results = ocr_image(image)
    for (bbox, text, prob) in results:
        st.write(f"Detected License Plate Number: **{text}** with confidence: **{prob:.2f}**")
