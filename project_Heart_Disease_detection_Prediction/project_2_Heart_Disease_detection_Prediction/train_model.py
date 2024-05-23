# importing required libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn import metrics
import time


# Function to evaluate classification performance
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

    train_precision = metrics.precision_score(y_train, model.predict(X_train), average='weighted', zero_division=1)
    test_precision = metrics.precision_score(y_test, model.predict(X_test), average='weighted', zero_division=1)

    train_recall = metrics.recall_score(y_train, model.predict(X_train), average='weighted', zero_division=1)
    test_recall = metrics.recall_score(y_test, model.predict(X_test), average='weighted', zero_division=1)

    end_time = time.time()
    print(f"Evaluation took {end_time - start_time:.2f} seconds")

    print("Training Set Metrics:")
    print("Training Accuracy {}: {:.2f}%".format(name, train_accuracy * 100))
    print("Training Precision {}: {:.2f}%".format(name, train_precision * 100))
    print("Training Recall {}: {:.2f}%".format(name, train_recall * 100))

    print("\nTest Set Metrics:")
    print("Test Accuracy {}: {:.2f}%".format(name, test_accuracy * 100))
    print("Test Precision {}: {:.2f}%".format(name, test_precision * 100))
    print("Test Recall {}: {:.2f}%".format(name, test_recall * 100))
    
# Function to train the model and save it
def train_and_save_model(num_rows=None):
    start_time = time.time()
    print("Loading the dataset...")
    
    df = pd.read_csv("data\heart.csv")

    # Separate features and target
    X = df.drop('output', axis=1)
    y = df['output']

    # # Define numerical and categorical features
    # numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing 
    steps = [
        ('scaler', StandardScaler()),  # Step 1: Feature scaling
        ('classifier', RandomForestClassifier(n_estimators=150,
                            min_samples_split=10,
                            min_samples_leaf=4,
                            random_state=7))  # Step 2: Logistic Regression
    ]
    # Build the pipeline
    # Create the pipeline
    
    pipeline = Pipeline(steps)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Fit the model using the pipeline
    pipeline.fit(X_train, y_train)

    # Save the model
    dump(pipeline, 'model_pipeline.joblib')

    end_time = time.time()
    print(f"Model training and saving took {end_time - start_time:.2f} seconds")
    print("model evaluation...\n")
    # Evaluate the model
    evaluate_classification(pipeline, "RandomForest", X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    num_rows = None  # Set the number of rows for training (e.g., num_rows = 1000000)
    train_and_save_model(num_rows)
