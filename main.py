import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# TODO: Enter the paths for your saved encoder and model
path = "/home/masonl/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/model/encoder.pkl"
encoder = load_model(path)

path = "/home/masonl/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/model/model.pkl"
model = load_model(path)

# TODO: Create a RESTful API using FastAPI
app = FastAPI()

# TODO: Create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message": "Hello from the API!"}

# TODO: Create a POST on a different path that does model inference
@app.post("/inference/")
async def post_inference(data: Data):
    # DO NOT MODIFY: Turn the Pydantic model into a dict.
    data_dict = data.dict()
    
    # Clean up the dict and create a DataFrame
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    # Process the data
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
    data_processed, _, _, _ = process_data(
        data, categorical_features=cat_features, label=None, training=False, encoder=encoder
    )

    # Predict using the model
    _inference = inference(model, data_processed)

    # Return the result
    return {"result": apply_label(_inference)}

