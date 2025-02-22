import pickle
import numpy as np
import pandas as pd
from utils.logger import get_logger  # Import the logger

# Initialize Logger
logger = get_logger("prediction")

try:
    # Load the trained model
    MODEL_PATH = "models/best_crop_model.pkl"
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")

    # Load the Label Encoder
    ENCODER_PATH = "models/label_encoder.pkl"
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    logger.info("LabelEncoder loaded successfully.")

    def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
        """
        Predict the crop based on input features.
        :param N: Nitrogen value
        :param P: Phosphorus value
        :param K: Potassium value
        :param temperature: Temperature in Celsius
        :param humidity: Humidity percentage
        :param ph: pH value of soil
        :param rainfall: Rainfall in mm
        :return: Predicted crop name
        """
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction_encoded = model.predict(input_data)[0]  # Get encoded class
        prediction = le.inverse_transform([prediction_encoded])[0]  # Convert back to crop name
        
        logger.info(f"Prediction: {prediction}")
        return prediction

except Exception as e:
    logger.error(f"Error in loading model or making predictions: {str(e)}", exc_info=True)
