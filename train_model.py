import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Set the project path
project_path = "/home/masonl/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"

data_path = 'data/census.csv'
print(data_path)

# Load the census.csv data
data = pd.read_csv(data_path)

# Split the provided data to have a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Ensure the model directory exists
model_dir = os.path.join(project_path, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model and the encoder
model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)

# Load the model
model = load_model(model_path)

# Run inference on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices using the performance_on_categorical_slice function
for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            test, col, slicevalue, cat_features, "salary", encoder, lb, model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)


