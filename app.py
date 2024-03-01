import streamlit as st
from PIL import Image

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



# Sliders for timestamp and image scaling side by side
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
        input_image = Image.open(uploaded_file)
        st.image(input_image, use_column_width=True)
    else:
        st.write("Please upload an image.")

with col6:
    st.write("CORE SEGMENTATION")
    # Display uploaded image if available, otherwise display a message
    if uploaded_file is not None:
        output_image1 = Image.open(uploaded_file)
        st.image(output_image1, use_column_width=True)
    else:
       #st.write("Please upload an image.")
        None
with col7:
    st.write("WALL EDGE DETECTION")
    # Display uploaded image if available, otherwise display a message
    if uploaded_file is not None:
        output_image2 = Image.open(uploaded_file)
        st.image(output_image2, use_column_width=True)
    else:
        None
