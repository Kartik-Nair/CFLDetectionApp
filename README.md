# CFLDetectionApp

## Save Tensorflow Model
import os
import tensorflow as tf
hdf5_model = tf.keras.models.load_model(os.path.join(data_path, 'put_your_model.hdf5'), custom_objects={"dice_coef": dice_coef, "true_positive_rate":true_positive_rate})
tf.saved_model.save(hdf5_model, 'your_model_name')

!zip -r model.zip *

from google.colab import files
files.download('model.zip')

## Extract model.zip and create a new folder named "1" inside and put all the contents in that folder

## Install Docker and run below commands

docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving

docker run -p 8501:8501 --name=tf_serving_container --mount type=bind,source=C:\Users\kgrna\Downloads\model,target=/models/saved_model -e MODEL_NAME=saved_model -t tensorflow/serving

## Run streamlit application on 8080 port

streamlit run app.py --server.port 8080
