from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from utils.logger import get_logger  # Import the logger

app = Flask(__name__)

# Initialize Logger
logger = get_logger("FlaskApp")

try:
    # Load the trained model
    MODEL_PATH = "models/best_crop_model.pkl"
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully.")

    # Load the Label Encoder dynamically
    ENCODER_PATH = "models/label_encoder.pkl"
    if os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("LabelEncoder loaded successfully.")
    else:
        logger.error("LabelEncoder file not found!")
        label_encoder = None

except Exception as e:
    logger.error(f"Error loading model or encoder: {str(e)}", exc_info=True)
    model, label_encoder = None, None  # Ensure app doesn't break

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model and label encoder are loaded
        if model is None or label_encoder is None:
            return render_template('index.html', prediction_text="Error: Model or Encoder not loaded.")

        # Get input data from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare input for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Make prediction
        prediction = model.predict(input_data)

        # Convert to actual label
        crop = label_encoder.inverse_transform(prediction)[0]

        logger.info(f"Prediction made successfully: {crop}")
        return render_template('index.html', prediction_text=f"Recommended Crop: {crop}")

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return render_template('index.html', prediction_text="Error in processing your request.")

if __name__ == "__main__":
    app.run(debug=True)
