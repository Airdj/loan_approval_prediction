import joblib
from sklearn.linear_model import LinearRegression


def outliers_handler(df):
    df = df[(df['person_emp_length'].isna()) |(df['person_emp_length'] <= 100)]
    return df


def handle_emp_length(df, artifact=None):
    person_emp_length_missing = df[df['person_emp_length'].isna()]
    if not person_emp_length_missing.empty:
        if artifact:
            person_emp_length_lin_reg = artifact
        else:
            person_emp_length_mask = ~df.isin(person_emp_length_missing.to_dict(orient='list')).all(axis=1)
            person_emp_length_result = df[person_emp_length_mask]
            person_emp_length_features_train = person_emp_length_result[['person_age']]
            person_emp_length_target_train = person_emp_length_result[['person_emp_length']]

            person_emp_length_lin_reg = LinearRegression()
            person_emp_length_lin_reg.fit(person_emp_length_features_train, person_emp_length_target_train)

        person_emp_length_predictions = person_emp_length_lin_reg.predict(person_emp_length_missing[['person_age']])
        person_emp_length_predicted_data = person_emp_length_missing.copy()
        person_emp_length_predicted_data['person_emp_length'] = person_emp_length_predictions.astype('int')

        df.loc[df['person_emp_length'].isna(), 'person_emp_length'] = person_emp_length_predicted_data
        return df, person_emp_length_lin_reg

    else:
        return df, None


def handle_loan_int_rate(df, artifact=None):
    loan_int_rate_missing = df[df['loan_int_rate'].isna()]
    if not loan_int_rate_missing.empty:
        if artifact:
            loan_int_rate_lin_reg = artifact
        else:
            loan_int_rate_mask = ~df.isin(loan_int_rate_missing.to_dict(orient='list')).all(axis=1)
            loan_int_rate_result = df[loan_int_rate_mask]
            loan_int_rate_features_train = loan_int_rate_result[['loan_amnt']]
            loan_int_rate_target_train = loan_int_rate_result[['loan_int_rate']]

            loan_int_rate_lin_reg = LinearRegression()
            loan_int_rate_lin_reg.fit(loan_int_rate_features_train, loan_int_rate_target_train)

        loan_int_rate_predictions = loan_int_rate_lin_reg.predict(loan_int_rate_missing[['loan_amnt']])
        loan_int_rate_predicted_data = loan_int_rate_missing.copy()
        loan_int_rate_predicted_data['loan_int_rate'] = loan_int_rate_predictions
        df.loc[df['loan_int_rate'].isna(), 'loan_int_rate'] = loan_int_rate_predicted_data
        return df, loan_int_rate_lin_reg

    else:
        return df, None


def missing_others(df):
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object']).columns

    num_with_na = df[num_cols].columns[df[num_cols].isna().any()]
    cat_with_na = df[cat_cols].columns[df[cat_cols].isna().any()]

    for feature in list(num_with_na):
        df[feature] = df[feature].fillna(df[feature].mean())

    for feature in list(cat_with_na):
        df[feature] = df[feature].fillna(df[feature].mode()[0])

    return df


def preprocess_pipeline(df, lin_reg_int_rate=None, lin_reg_emp_length=None):
    df = outliers_handler(df)
    df, lr_int_rate = handle_loan_int_rate(df, lin_reg_int_rate)
    df, lr_emp_length = handle_emp_length(df, lin_reg_emp_length)
    df = missing_others(df)

    state = {'lin_reg_int_rate': lr_int_rate, 'lin_reg_emp_length': lr_emp_length}

    return df, state

def preprocess_data(df_train=None, df_test=None, df_eval=None, inference=False, df_inference=None):
    if inference:
        try:
            state = joblib.load("src/utils/preprocess_pipeline_artifacts.pkl")
            df_inference, _ = preprocess_pipeline(df_inference, state['lin_reg_int_rate'], state['lin_reg_emp_length'])
            return df_inference

        except Exception as e:
            print(f'Error: {e}')

    else:
        df_train, state = preprocess_pipeline(df_train)
        df_test, _ = preprocess_pipeline(df_test)
        df_eval, _ = preprocess_pipeline(df_eval)
        joblib.dump(state, "src/utils/preprocess_pipeline_artifacts.pkl")
        return df_train, df_test, df_eval
