import uvicorn #to intract with server and request
import joblib
import numpy as np
import pandas as pd

from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

tfidf=joblib.load('./mymodel/tfidf_model.joblib')
logistic =joblib.load('./mymodel/logistic_model.joblib')
naive =joblib.load('./mymodel/naive_bayes_model.joblib')


class Statment(BaseModel):
        option:str
        input_feature:str


@app.get('/')
def index():
    return{'key' : "Working fine."}


@app.post('/predict')
def predict(statment : Statment):
    check_my_emotion = tfidf.transform([statment.input_feature])

    if(statment.option == "Naive Bias"):        
        y = naive.predict(check_my_emotion)
        emotions = naive.predict_proba(check_my_emotion)
        detected_emotion = naive.classes_
    else:
        y = logistic.predict(check_my_emotion)
        emotions = logistic.predict_proba(check_my_emotion)
        detected_emotion = logistic.classes_

    return{'prediction': y[0] , "emotion" : emotions.tolist() , 'data' : detected_emotion.tolist()}

if __name__=="__main__":
    uvicorn.run("server:app", reload = "True")