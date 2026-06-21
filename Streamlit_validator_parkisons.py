import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

# Disable TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load models (skip optimizer to avoid pop error)
models = {
    'VGG16': load_model('VGG16_model.keras', compile=False),
    'DenseNet121': load_model('DenseNet121_model.keras', compile=False),
    'ResNet50': load_model('ResNet50_model.keras', compile=False),
    'CNN': load_model('CNN_model.keras', compile=False)
}

# Preprocessors
preprocessors = {
    'VGG16': vgg_preprocess,
    'DenseNet121': densenet_preprocess,
    'ResNet50': resnet_preprocess,
    'CNN': lambda x: x/255.0  # no preprocessing
}

# Optional: Add your best accuracy scores
model_metrics = {
    "VGG16": {"Accuracy": 0.92},
    "DenseNet121": {"Accuracy": 0.87},
    "ResNet50": {"Accuracy": 0.78},
    "CNN": {"Accuracy": 0.60}
}

# Streamlit UI
st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("Parkinson's Disease Detection from Drawing")

# File upload
uploaded_file = st.file_uploader("Upload a spiral or wave drawing (PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (128, 128))
    image_array = np.expand_dims(image_resized.astype("float32"), axis=0)

    st.image(image_resized, caption="Uploaded Image", use_column_width=True)

    # Display the accuracy of all models
    st.write("Model Accuracies:")
    for model_name, metrics in model_metrics.items():
        st.write(f"{model_name}: {metrics['Accuracy'] * 100:.2f}%")

    if st.button("Predict All Models"):
        predictions = {}

        for model_name in models:
            model = models[model_name]
            preprocess = preprocessors[model_name]

            # Preprocess and predict
            img_prep = preprocess(image_array.copy())
            prediction = model.predict(img_prep)
            class_index = np.argmax(prediction)
            confidence = prediction[0][class_index]

            label = "Parkinson's" if class_index == 1 else "Healthy"
            predictions[model_name] = (label, confidence)

        # Display the predictions for all models
        st.write("\nPredictions from all models:")
        for model_name, (label, confidence) in predictions.items():
            st.success(f"{model_name} Prediction: **{label}** with **{confidence * 100:.2f}%** confidence")
