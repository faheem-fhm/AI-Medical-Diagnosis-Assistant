import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


IMG_SIZE = (224, 224)


def preprocess_image(image_path):
    """
    Load an image, resize it, preprocess it,
    and return it in the format required by EfficientNet.
    """

    # Load image
    img = image.load_img(image_path, target_size=IMG_SIZE)

    # Convert image to NumPy array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    return img_array