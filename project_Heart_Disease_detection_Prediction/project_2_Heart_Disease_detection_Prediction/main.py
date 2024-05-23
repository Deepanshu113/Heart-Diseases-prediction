import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier from scikit-learn

# Utility function for processing input data
def process_input_data(data):
    """
    Process input data to ensure it's in the correct format for prediction.
    
    Args:
    - data (dict): Dictionary containing input data.
    
    Returns:
    - pd.DataFrame: Processed DataFrame ready for prediction.
    """
    processed_data = pd.DataFrame([data])
    return processed_data

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for your API

# Load the trained model from pickle file
with open('model\\random_forest_pipeline_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")
@app.route('/model_info')
def model_info():
    return render_template('model_info.html')
@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        # Extract data from JSON request
        data = request.json["data"]

        # Process input data using utility function
        input_df = process_input_data(data)

        # Make predictions using the model
        predictions = model.predict(input_df)

        # Convert prediction result to a single output value
        output = predictions[0]

        return jsonify(output)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
