#  AI Medical Diagnosis Assistant

An AI-powered web application that analyzes chest X-ray images and predicts whether the patient is likely to have **Pneumonia** or **Normal** lungs using a deep learning model based on EfficientNetB0.

---

##  Overview

The AI Medical Diagnosis Assistant is a medical image classification project developed using Flask and TensorFlow. Users can upload a chest X-ray image through a simple web interface, and the application processes the image using a trained EfficientNetB0 model to generate a prediction along with a confidence score.

The project demonstrates how transfer learning can be applied to healthcare problems and provides a practical example of integrating deep learning models into a web application.

---

##  Features

- Upload chest X-ray images
- Automatic image preprocessing
- Deep learning prediction using EfficientNetB0
- Detects:
  - Normal
  - Pneumonia
- Displays prediction confidence
- Simple and responsive user interface
- Fast real-time inference
- Built with Flask for easy deployment

---

##  Technologies Used

### Programming Language
- Python

### Framework
- Flask

### Deep Learning
- TensorFlow
- Keras
- EfficientNetB0 (Transfer Learning)

### Libraries
- NumPy
- OpenCV
- Pillow

### Frontend
- HTML
- CSS
- JavaScript

---

##  Project Structure

AI_Medical_Diagnosis_Assistant/

├── app.py

├── train_model.py

├── model/

│ └── pneumonia_model.keras

├── uploads/

├── utils/

│ ├── preprocess.py

│ └── predict.py

├── templates/

│ ├── index.html

│ └── result.html

├── static/

│ ├── css/

│ └── images/

├── requirements.txt

└── README.md

---

##  Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI_Medical_Diagnosis_Assistant.git
```

Move into the project folder

```bash
cd AI_Medical_Diagnosis_Assistant
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

##  How It Works

1. Upload a chest X-ray image.
2. The image is resized and preprocessed.
3. The EfficientNetB0 model analyzes the image.
4. The model predicts whether the X-ray is **Normal** or **Pneumonia**.
5. The application displays the prediction along with its confidence score.

---

##  Model Information

- Model Architecture: EfficientNetB0
- Transfer Learning: Yes
- Input Size: 224 × 224
- Framework: TensorFlow/Keras

---

##  Future Improvements

- Support for additional lung diseases
- Heatmap visualization using Grad-CAM
- PDF report generation
- User authentication
- Prediction history dashboard
- Cloud deployment
- Multi-language support

---

##  Disclaimer

This application is developed for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Clinical decisions should always be made by qualified healthcare professionals.

---

##  Author

**Mohamed Faheem**

Artificial Intelligence & Data Science Student

Passionate about Artificial Intelligence, Deep Learning, Computer Vision, and Healthcare AI applications.

---