from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title = 'Laptop price Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

model = load(pathlib.Path('model/laptop-v1.joblib'))

class InputData(BaseModel):
    Ram:int=8
    Weight:float=1.86
    TouchScreen:int=0
    Ips:int=1
    Ppi:float=154.875632
    HDD:int=0
    SSD:int=256

class OutputData(BaseModel):
    score:float=0.22275930091861187

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)

    return {'score':result}
