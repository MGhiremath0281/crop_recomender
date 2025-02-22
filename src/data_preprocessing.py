import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger

# Initialize logger
logger = get_logger()

# Paths
RAW_DATA_PATH = "data/crop_prediction.csv"
PREPROCESSED_DATA_PATH = "data/Preprocss.csv"

# Load dataset
logger.info("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

# Check for missing values
if df.isnull().sum().any():
    df = df.dropna()
    logger.info("Missing values detected and removed.")

# Encode target variable
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Save preprocessed data
df.to_csv(PREPROCESSED_DATA_PATH, index=False)
logger.info(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
