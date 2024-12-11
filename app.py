from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("digit_recog.h5")

# Preprocess input image
def preprocess_image(image):
    # Convert the image to grayscale (if not already)
    image = image.convert("L")
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a NumPy array and normalize pixel values
    image_array = img_to_array(image) / 255.0
    # Reshape the array to include batch and channel dimensions
    return np.reshape(image_array, (1, 28, 28, 1))

@app.route("/")
def index():
    return render_template("index.html")  # HTML form to upload an image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    try:
        # Open the image file
        image = Image.open(file)
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        # Get predictions
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)  # Get the predicted digit
        confidence = predictions[0][predicted_class]  # Confidence score

        # Return prediction as JSON response
        return jsonify({
            "predicted_digit": int(predicted_class),
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
