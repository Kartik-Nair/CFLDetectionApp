import streamlit as st
import requests
import json
from PIL import Image
import os
import numpy as np
import cv2

# Inject custom CSS with st.markdown to change the button color
st.markdown("""
<style>
           
button {
    background-color: #6A0DAD !important;
    color: white !important;
    border: 1px solid #6A0DAD;
}
button:hover {
    background-color: #5e0ca5 !important;
    border: 1px solid #5e0ca5;
}
</style>
""", unsafe_allow_html=True)

# Set the title of the app
st.header('CFL Detection App')



# Buttons for selecting input image/video
col1, col2 = st.columns(2)
with col1:
    input_type = st.radio("Select Input Type:", ("Single Image", "Batch Images", "Video"))

with col2:
    if input_type in ["Single Image", "Batch Images"]:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'mov', 'avi', 'mkv'])


# Buttons for "Detect CFL" and "Save Output" side by side
col3, col4 = st.columns(2)
with col3:
    if st.button("Detect CFL"):
        st.write("CFL Detection Initiated...")

with col4:
    if st.button("Save Output"):
        st.write("Output Saved Successfully.")

# Displaying images in a row (input + 2 dummy outputs)
col5, col6, col7 = st.columns(3)
with col5:
    st.write("INPUT IMAGE")
    # Display uploaded image if available, otherwise display a message
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).resize((256,256))
        st.image(input_image, use_column_width=True)
    else:
        st.write("Please upload an image.")

with col6:
    st.write("CORE SEGMENTATION")
    # Display uploaded image if available, otherwise display a message
    if uploaded_file is not None:
       
        image_path = "C:/Users/kgrna/Downloads/Kartik/Kartik/ELG5902/CFL_training_data-20240207T044725Z-001/CFL_training_data/input/VesselD20H15_0.jpg"

        output_image1 = Image.open(image_path).convert("RGB").resize((256, 256))

        image_array1 = np.array(output_image1)
        image_array=np.expand_dims(image_array1,axis=0)
        print(image_array1.shape)

        # Creating JSON data for the request
        data = json.dumps({"signature_name": "serving_default", "instances": image_array.tolist()})

        # # Sending POST request to TensorFlow Serving API
        url = "http://localhost:8501/v1/models/saved_model:predict"
        headers = {"content-type": "application/json"}
        response = requests.post(url, data=data, headers=headers)
        predictions = response.json()['predictions']
        predictions_array = np.array(predictions)
        threshold = 0.7
        predictions_array[predictions_array>=threshold] = 1
        predictions_array[predictions_array<threshold] = 0
        predictions_array = predictions_array.reshape(256,256)

        output_image1 = Image.fromarray((predictions_array * 255).astype(np.uint8))
        st.image(output_image1, use_column_width=True)
    else:
       #st.write("Please upload an image.")
        None
with col7:
    st.write("WALL EDGE DETECTION")
    # Display uploaded image if available, otherwise display a message
    if uploaded_file is not None:
        output_image2 = Image.open(uploaded_file).resize((256,256))
        st.image(output_image2, use_column_width=True)
    else:
        None
