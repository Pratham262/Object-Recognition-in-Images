from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from tensorflow.keras.models import load_model
import json

# Initialize Flask App
app = Flask(__name__, static_folder="static")

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO)

# CIFAR-10 class labels
CLASS_LABELS = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

# Load the model with error handling
MODEL_PATH = "CIFAR-10model.h5"
model = None

try:
    model = load_model(MODEL_PATH, compile=False)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Save model structure as JSON
if model:
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json.dump(model_json, json_file)


# Flask Routes
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/classify', methods=['POST'])
def classify_image():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (32, 32))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = CLASS_LABELS[class_index]

    return jsonify({"prediction": class_label})


# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to port 5000
    app.run(host='0.0.0.0', port=port, debug=True)
