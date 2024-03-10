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

core_seg_output_image = None
wall_edge_output_image = None
wall_edge_output_image = []
uploaded_files = []

# Buttons for selecting input image/video
col1, col2 = st.columns(2)
with col1:
    input_type = st.radio("Select Input Type:", ("Single Image", "Batch Images", "Video"))

with col2:
    if input_type in "Single Image":
        file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if file:
            uploaded_files.append(file)        
    elif input_type == "Batch Images":
        uploaded_files = st.file_uploader("Choose input folder: ", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    elif input_type == "Video":
        uploaded_files.append(st.file_uploader("Choose a video...", type=['mp4', 'mov', 'avi', 'mkv']))


# Buttons for "Detect CFL" and "Save Output" side by side
col3, col4 = st.columns(2)
with col3:
    if st.button("Detect CFL"):
        st.write("CFL Detection Initiated...")
                
        for file in uploaded_files:
            input_image = Image.open(file).convert("RGB").resize((256, 256))

            image_array1 = np.array(input_image)
            image_array=np.expand_dims(image_array1,axis=0)

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

            core_seg_output_image = Image.fromarray((predictions_array * 255).astype(np.uint8))
            core_seg_output_image.save(os.path.join(os.curdir+"/output/",file.name))

            # TODO: Wall Edge Detection
            wall_edge_output_image = Image.open(file).resize((256,256))





# Displaying images in a row (input + 2 dummy outputs)
col5, col6, col7 = st.columns(3)
with col5:
    st.write("INPUT IMAGE")
    # Display uploaded image if available, otherwise display a message
    if len(uploaded_files) > 0:
        input_image = Image.open(uploaded_files[0]).resize((256,256))
        st.image(input_image, use_column_width=True)
    else:
        st.write("Please upload an image.")

with col6:
    st.write("CORE SEGMENTATION")
    # Display uploaded image if available, otherwise display a message
    if core_seg_output_image is not None:     
        st.image(core_seg_output_image, use_column_width=True)
    else:
        None
with col7:
    st.write("WALL EDGE DETECTION")
    # Display uploaded image if available, otherwise display a message
    if wall_edge_output_image is not None:
        st.image(wall_edge_output_image, use_column_width=True)
    else:
        None
