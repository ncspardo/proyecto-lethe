import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Lethe.main import mpredict

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def predict(
       input_data: dict   # File with frequencies
    ):
    
    data = input_data['data']
    column_list = list(data[0].keys())
    values_list = list(data[0].values())
    values_list = np.array(values_list).reshape(1, -1)
    df = pd.DataFrame(values_list, columns=column_list)

    diagnosis = mpredict(df)
    index = np.argmax(diagnosis)
    return  {'diagnosis': str(index)}

@app.get("/")
def root():
    pass  # YOUR CODE HERE
