import pandas as pd
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_loan_data
from src.serving.inference import ModelService

model_path = 'src/serving/tuned_model.pkl'
model_service = ModelService(model_path)

cols_path = 'src/utils/columns.csv'
columns = pd.read_csv(cols_path, header=None)[0].tolist()

def inference_pipeline(df_inference):
    validation_result, failed_expectations = validate_loan_data(df_inference, columns)
    if failed_expectations:
        print('failed data entrance')


    # inference data(preprocess)
    df_inference = preprocess_data(inference=True, df_inference=df_inference)

    # inference data(build features)
    final_inference_df = build_features(inference=True, df_inference=df_inference)
    final_inference_df = final_inference_df.drop(columns=['id'],axis=1)
    prediction = model_service.predict(final_inference_df)
    prediction = (prediction >= 0.7271455223880597).astype(int) #EXTREME CONSERVATISM
    return prediction.tolist()
