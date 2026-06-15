import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, precision_recall_curve)

final_eval_df = pd.read_csv('data/processed/processed_final_eval_df.csv').drop('id', axis=1)
test_df = pd.read_csv('data/processed/processed_final_test_df.csv')
test_id = pd.read_csv('data/raw/test.csv')

model_path = 'src/serving/base_model.pkl'
final_model = joblib.load(model_path)

eval_target_feature = final_eval_df['loan_status']
final_eval_df = final_eval_df.drop('loan_status', axis=1)


def evaluate_model(model, df, target_feature):
    y_val = target_feature

    y_pred = model.predict_proba(df)[:,1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred)

    print(precisions,recalls,thresholds)

    idx = np.where(precisions >= 0.9)[0][0]
    print(thresholds[idx], precisions[idx], recalls[idx])


    #what is the main goal? EXTREME CONSERVATISM
    y_pred = (y_pred >= 0.7271455223880597).astype(int)

    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_pred))


if __name__ == '__main__':

    # predictions = final_model.predict_proba(test_df.drop('id',axis=1))[:,1]
    # output = pd.DataFrame({'id': test_id.id,
    #                             'loan_status': predictions})
    # output.to_csv('src/utils/predictions.csv', index=False)
    # print('prediction done')
    evaluate_model(final_model, final_eval_df, eval_target_feature)



