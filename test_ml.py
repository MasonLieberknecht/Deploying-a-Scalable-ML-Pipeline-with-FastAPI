import pytest
import numpy as np
import pandas as pd
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test if the trained model is a RandomForestClassifier.
    """
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(2, size=100)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test if compute_model_metrics returns the expected precision, recall, and F1 score.
    """
    y_true = [1, 0, 1, 1, 0, 1, 0, 1]
    y_preds = [1, 0, 1, 0, 0, 1, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert precision == pytest.approx(1.0, rel=1e-9)
    assert recall == pytest.approx(0.8, rel=1e-9)
    assert fbeta == pytest.approx(0.888888888888889, rel=1e-9)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test if process_data returns the correct shape for X and y.
    """
    data = pd.DataFrame({
        "workclass": ["Private", "Self-emp-not-inc", "Private"],
        "education": ["Bachelors", "HS-grad", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
        "occupation": ["Exec-managerial", "Craft-repair", "Sales"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family"],
        "race": ["White", "Black", "White"],
        "sex": ["Male", "Female", "Male"],
        "native-country": ["United-States", "United-States", "Mexico"],
        "salary": [">50K", "<=50K", ">50K"]
    })
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert X.shape[0] == data.shape[0]
    assert len(y) == data.shape[0]


