import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model once when this file is imported
model = tf.keras.models.load_model('models/your_model.h5')

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalize the image.
    """
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image: Image.Image) -> str:
    """
    Predict if the image is Pneumonia or Normal.
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
