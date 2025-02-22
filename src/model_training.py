import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from utils.logger import get_logger  # Import the logger

# Initialize Logger
logger = get_logger("model_training")

try:
    logger.info("Loading dataset...")

    # Load preprocessed dataset
    DATA_PATH = "data/Preprocss.csv"  
    df = pd.read_csv(DATA_PATH)

    logger.info("Dataset loaded successfully.")

    # Define features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Save the Label Encoder
    ENCODER_PATH = "models/label_encoder.pkl"
    os.makedirs("models", exist_ok=True)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    
    logger.info(f"LabelEncoder saved successfully. Classes: {le.classes_}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    logger.info("Training started for all models.")

    best_model = None
    best_accuracy = 0

    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        logger.info(f"{name} Accuracy: {accuracy:.4f}")

        print(f"Results for {name}:\n")
        print(classification_report(y_test, y_pred))
        print("-" * 60)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_model_name = name

    if best_model:
        # Save best model
        MODEL_PATH = "models/best_crop_model.pkl"
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)

        logger.info(f"Best model ({best_model_name}) saved with accuracy: {best_accuracy:.4f}")
    else:
        logger.warning("No model was selected as the best.")

    logger.info("Model training completed successfully.")

except Exception as e:
    logger.error(f"Error during model training: {str(e)}", exc_info=True)
