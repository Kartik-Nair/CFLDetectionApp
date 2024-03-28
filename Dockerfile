FROM tensorflow/serving:latest
ENV MODEL_BASE_PATH="/models"
ENV MODEL_NAME=core_model

WORKDIR /models/$MODEL_NAME/1
COPY ./core_model/1 .

EXPOSE 8501

CMD ["tensorflow_model_server", "--port=8501", "--model_name=$MODEL_NAME", "--model_base_path=/models/$MODEL_NAME"]
