import tensorflow as tf
import os

from utils.preprocess import preprocess_image

# Load the trained model only once
MODEL_PATH = os.path.join("model", "pneumonia_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)


def predict_pneumonia(image_path):
    """
    Predict whether the uploaded chest X-ray image
    indicates NORMAL or PNEUMONIA.
    """

    # Preprocess image
    processed_image = preprocess_image(image_path)

    # Predict
    prediction = model.predict(processed_image)[0][0]

    # Convert prediction to label
    if prediction >= 0.5:
        label = "PNEUMONIA"
        confidence = prediction * 100

        suggestion = """
The uploaded chest X-ray shows patterns that may be be associated with pneumonia. Please consult a healthcare professional for further evaluation. This AI prediction is intended to support preliminary screening and is not a substitute for a medical diagnosis.
"""

    else:
        label = "NORMAL"
        confidence = (1 - prediction) * 100

        suggestion = """
No visible signs of pneumonia were detected in the uploaded chest X-ray. If you continue to experience symptoms, please consult a healthcare professional. This AI prediction is intended for educational purposes only.
"""

    return label, round(confidence, 2), suggestion