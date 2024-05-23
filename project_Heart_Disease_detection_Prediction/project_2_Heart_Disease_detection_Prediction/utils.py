import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Load the saved pipeline
pipeline_filename = 'random_forest_model.pkl'
pipeline = load(pipeline_filename)

def preprocess_data_single(data):
    """Preprocess single data point for prediction."""
    # Assuming data is a dictionary with keys as feature names
    df = pd.DataFrame([data])
    return df

def preprocess_data_batch(file_path):
    """Preprocess batch data for prediction."""
    df = pd.read_csv(file_path)
    return df

def predict_single(data):
    """Predict using the loaded pipeline for a single instance."""
    df = preprocess_data_single(data)
    prediction = pipeline.predict(df)
    return prediction[0]

def predict_batch(file_path):
    """Predict using the loaded pipeline for batch data."""
    df = preprocess_data_batch(file_path)
    predictions = pipeline.predict(df)
    return predictions.tolist()
