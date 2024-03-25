import streamlit as st
import requests
import json
from PIL import Image
import os
import numpy as np

from wall_detector import detect_wall_edge

# Inject custom CSS with st.markdown to change the button color
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Set the title of the app
st.header("CFL Detection App")

core_seg_output_image = None
wall_edge_output_image = None
uploaded_files = []


# Function to perform CFL detection
def detect_cfl(selected_image):
    # Perform CFL detection on the provided image

    input_image = Image.open(selected_image).convert("RGB").resize((256, 256))

    image_array1 = np.array(input_image)
    image_array = np.expand_dims(image_array1, axis=0)

    # Creating JSON data for the request
    data = json.dumps(
        {"signature_name": "serving_default", "instances": image_array.tolist()}
    )

    # # Sending POST request to TensorFlow Serving API
    url = "http://localhost:8501/v1/models/core_model:predict"
    headers = {"content-type": "application/json"}
    response = requests.post(url, data=data, headers=headers)
    predictions = response.json()["predictions"]
    predictions_array = np.array(predictions)
    threshold = 0.9
    predictions_array[predictions_array >= threshold] = 1
    predictions_array[predictions_array < threshold] = 0
    predictions_array = predictions_array.reshape(256, 256)

    core_seg_output_image = Image.fromarray((predictions_array * 255).astype(np.uint8))
    return core_seg_output_image


# Function to display images based on dropdown selection
def display_images(selected_image):

    with col5:
        # Display input image
        # input_image_path = os.path.join(os.curdir, "input", selected_image)
        st.write("INPUT IMAGE")
        input_image = Image.open(selected_image).convert("RGB").resize((256, 256))
        st.image(input_image, use_column_width=True)

    with col6:
        # Perform CFL detection
        st.write("CORE SEGMENTATION")

        core_seg_output_image = detect_cfl(selected_image)
        st.image(core_seg_output_image, use_column_width=True)

    with col7:
        # Perform wall edge detection
        st.write("WALL EDGE DETECTION")
        wall_edge_output_image = detect_wall_edge(selected_image)[0]
        st.image(wall_edge_output_image, use_column_width=True)


# Buttons for selecting input image
col1, col2 = st.columns(2)
with col1:
    input_type = st.radio("Select Input Type:", ("Single Image", "Batch Images"))

with col2:
    if input_type in "Single Image":
        file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if file:
            uploaded_files.append(file)
    elif input_type == "Batch Images":
        uploaded_files = st.file_uploader(
            "Choose input folder: ",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
        )


# Buttons for "Detect CFL" and "Save Output" side by side
col3, col4 = st.columns(2)
with col3:
    # if st.button("Detect CFL"):
    #     detect_cfl_clicked = True
    #     st.write("CFL Detection Initiated...")

    for file in uploaded_files:
        core_seg_output_image = detect_cfl(file)
        wall_edge_output_image = detect_wall_edge(file)[0]
        output_dir = os.path.join(os.curdir + "/output/")
        if not os.path.exists(output_dir + "wall_"):
            os.makedirs(output_dir + "wall_", mode=0o077)
        core_seg_output_image.save(output_dir + "wall_" + file.name)
        output_dir_wall = os.path.join(os.curdir + "/output/", "wall_")
        wall_edge_output_image.save(output_dir + "wall_" + file.name)


col5, col6, col7 = st.columns(3)


# Get list of uploaded image names
if isinstance(uploaded_files, list):
    uploaded_images = [file for file in uploaded_files]
    uploaded_images_dict = {file.name: file for file in uploaded_files}
else:
    uploaded_images = []

# Dropdown to select image
selected_image_name = st.selectbox("Select Image", list(uploaded_images_dict.keys()))
if len(uploaded_images_dict) > 0:
    selected_image = uploaded_images_dict[selected_image_name]
else:
    selected_image = None

# Display images based on selected image
if selected_image:
    display_images(selected_image)
