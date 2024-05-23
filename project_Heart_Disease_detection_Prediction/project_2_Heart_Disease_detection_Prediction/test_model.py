import pandas as pd
from sklearn.metrics import classification_report
from joblib import load
from sklearn import metrics
import numpy as np


def print_classification_report(y_true, y_pred):
    # Calculate accuracy, precision, recall, and F1 measure for each class
    class_labels = sorted(set(y_true))  # Replace with your actual class labels
    print("\n\n------------------result----------------\n\n")

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=1)
    class_accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"Accuracy : {class_accuracy * 100:.2f}%")
    # Convert precision, recall, and f1 to scalars
    precision_scalar = precision.item() if isinstance(precision, np.ndarray) else precision
    recall_scalar = recall.item() if isinstance(recall, np.ndarray) else recall
    f1_scalar = f1.item() if isinstance(f1, np.ndarray) else f1

    print(f"Precision: {precision_scalar * 100:.2f}%")
    print(f"Recall: {recall_scalar * 100:.2f}%")
    print(f"F1 Score: {f1_scalar * 100:.2f}%\n")

    for label in class_labels:
        mask = (y_true == label)
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]

        # Accuracy
        class_accuracy = metrics.accuracy_score(y_true_masked, y_pred_masked)
        print(f"Accuracy for class {label}: {class_accuracy * 100:.2f}%")

        # Number of correctly classified instances
        correct_predictions = sum(y_true_masked == y_pred_masked)
        print(f"Number of correctly classified instances for class {label}: {correct_predictions}")

        # Precision, Recall, and F1 Score
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_true_masked, y_pred_masked, labels=[label], average='micro', zero_division=1
        )
        print(f"Precision for class {label}: {precision * 100:.2f}%")
        print(f"Recall for class {label}: {recall * 100:.2f}%")
        print(f"F1 Score for class {label}: {f1 * 100:.2f}%")

# Load the trained pipeline model
model = load('model_pipeline.joblib')

# Load the test data
print("Loading dataset...")

df = pd.read_csv("data\heart.csv")

num_rows_to_test = None  # Set this to the number of rows you want to test, or leave it as None to use the entire dataset
if num_rows_to_test:
    df = df.sample(num_rows_to_test)

# Separate features and target
X_test = df.drop('output', axis=1)
y_test = df['output']

# Make predictions on the test data
y_pred = model.predict(X_test)

# Generate and print the classification report
print("\nClassification Report:")
print_classification_report(y_test, y_pred)