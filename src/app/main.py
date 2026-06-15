import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.pipelines.inference_pipeline import inference_pipeline

app = FastAPI()



class LoanFeatures(BaseModel):
    id: int | None = None
    person_age: int | None = None
    person_income: int | None = None
    person_home_ownership: str | None = None
    person_emp_length: float | None = None
    loan_intent: str | None = None
    loan_grade: str | None = None
    loan_amnt: int | None = None
    loan_int_rate: float | None = None
    loan_percent_income: float | None = None
    cb_person_default_on_file: str | None = None
    cb_person_cred_hist_length: int | None = None


@app.get("/")
def read_root():
    return {"message":"API working"}


@app.post("/predict")
def predict(data: LoanFeatures):

    input_dict = data.model_dump(by_alias=True)

    print(input_dict.keys())

    df = pd.DataFrame([input_dict])

    print(df.columns.tolist())
    print(df.shape)
    print(df.head())

    prediction = inference_pipeline(df)

    return {
        "prediction": prediction
    }


@app.post("/multipredict")
def predict(data: list[LoanFeatures]):

    df = pd.DataFrame([item.model_dump(by_alias=True) for item in data])

    prediction = inference_pipeline(df)

    return {
        "prediction": prediction
    }