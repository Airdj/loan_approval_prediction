import requests
import pandas as pd
import numpy as np

test_api_df = pd.read_csv('data/raw/train.csv')
test_api_df = test_api_df.iloc[1:100,:]
test_api_df = test_api_df.drop('loan_status', axis=1)


def clean_payload(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")

url = "http://127.0.0.1:8000/multipredict"


payload = clean_payload(test_api_df)
response = requests.post(url, json=payload)

print(response.status_code)
print(response.text)