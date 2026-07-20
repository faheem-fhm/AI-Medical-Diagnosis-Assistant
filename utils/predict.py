import tensorflow as tf
import numpy as np
import os

from utils.preprocess import preprocess_image

# Load TensorFlow Lite model
MODEL_PATH = os.path.join("model", "model.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict_pneumonia(image_path):
    """
    Predict whether the uploaded chest X-ray image
    indicates NORMAL or PNEUMONIA using TensorFlow Lite.
    """

    # Preprocess image
    processed_image = preprocess_image(image_path)

    # Convert to float32 (required for most TFLite models)
    processed_image = processed_image.astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], processed_image)

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    if prediction >= 0.5:
        label = "PNEUMONIA"
        confidence = prediction * 100

        suggestion = (
            "The uploaded chest X-ray shows patterns that may be associated "
            "with pneumonia. Please consult a healthcare professional for "
            "further evaluation. This AI prediction is intended to support "
            "preliminary screening and is not a substitute for a medical diagnosis."
        )

    else:
        label = "NORMAL"
        confidence = (1 - prediction) * 100

        suggestion = (
            "No visible signs of pneumonia were detected in the uploaded chest "
            "X-ray. If you continue to experience symptoms, please consult a "
            "healthcare professional. This AI prediction is intended for "
            "educational purposes only."
        )

    return label, round(float(confidence), 2), suggestion
