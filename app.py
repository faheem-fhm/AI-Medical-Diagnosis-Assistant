import os
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
    flash
)
from werkzeug.utils import secure_filename

# AI Prediction
from utils.predict import predict_pneumonia

app = Flask(__name__)

app.secret_key = "medical_ai_secret_key"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# Check Allowed File Extensions
# -----------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # No file uploaded
    if "image" not in request.files:
        flash("Please select a Chest X-ray image.")
        return redirect(url_for("home"))

    file = request.files["image"]

    # Empty filename
    if file.filename == "":
        flash("Please select a Chest X-ray image.")
        return redirect(url_for("home"))

    # Invalid file format
    if not allowed_file(file.filename):
        flash("Only JPG, JPEG and PNG images are allowed.")
        return redirect(url_for("home"))

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # AI Prediction
    prediction, confidence, suggestion = predict_pneumonia(filepath)

    return render_template(
        "result.html",
        image=filename,
        prediction=prediction,
        confidence=f"{confidence:.2f}%",
        suggestion=suggestion
    )


# -----------------------------
# Display Uploaded Image
# -----------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)