import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('models/your_model.h5')

st.title("🩺 Pneumonia Detection from X-ray")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    confidence = prediction[0][0]  # value between 0 and 1

    if confidence > 0.5:
        st.success("🩻 Prediction: Pneumonia")
        st.write(f"🔬 Confidence Score: {confidence:.2%}")
    else:
        st.info("🫁 Prediction: Normal")
        st.write(f"🔬 Confidence Score: {(1 - confidence):.2%}")
