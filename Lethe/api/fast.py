import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Lethe.main import predict

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(input):
    diagnosis = predict(input)

    return  {'diagnosis': diagnosis}

@app.get("/")
def root():
    pass  # YOUR CODE HERE
