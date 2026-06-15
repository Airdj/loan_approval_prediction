from src.serving.inference import ModelService
from src.features.build_features import build_features
from src.data.preprocess_data import preprocess_data
from src.utils.validate_data import validate_loan_data
from src.data.load_data import load_data


if __name__ == '__main__':
    # load data
    df_train, df_test, df_eval = load_data()
    df_inference = df_test.loc[[100]]
    columns = list(df_test.columns)
    # inference data(validation)
    validation_result, failed_expectations = validate_loan_data(df_inference, columns)
    if failed_expectations:
        print('failed data entrance')
    else:
        # inference data(preprocess)
        df_inference = preprocess_data(inference=True, df_inference=df_inference)
        print(df_inference.columns)
        # inference data(build features)
        final_inference_df = build_features(inference=True, df_inference=df_inference)
        print(final_inference_df.columns)
        final_inference_df.to_csv('data/processed/processed_final_inference_df.csv', index=False)

        # load model
        final_model_path = 'src/serving/tuned_model.pkl'
        final_model = ModelService(final_model_path)
        final_inference_df = final_inference_df.drop(columns=['id'], axis=1)
        prediction = final_model.predict(final_inference_df)
        prediction = (prediction >= 0.7271455223880597).astype(int)
        print(prediction)