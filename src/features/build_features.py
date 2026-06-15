import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

def feature_adder(df):
    # new features
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"]
    df["income_per_age"] = df["person_income"] / df["person_age"]
    df["high_loan_burden"] = (df["loan_percent_income"] > 0.3).astype(int)
    df["emp_to_age_ratio"] = df["person_emp_length"] / df["person_age"]
    df["young_high_loan"] = ((df["person_age"] < 25) & (df["loan_percent_income"] > 0.4)).astype(int)
    df["cred_hist_ratio"] = (df["cb_person_cred_hist_length"] / df["person_age"])
    df["thin_credit_history"] = (df["cb_person_cred_hist_length"] < 3).astype(int)
    df["had_default"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    home_map = {"OWN": 3, "MORTGAGE": 2, "RENT": 1, "OTHER": 0}
    df["home_stability"] = df["person_home_ownership"].map(home_map)

    grade_map = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}
    df["loan_grade_num"] = df["loan_grade"].map(grade_map)

    df["grade_interest_interaction"] = (df["loan_grade_num"] * df["loan_int_rate"])
    df["income_grade"] = (df["person_income"] * df["loan_grade_num"])

    df["log_income"] = np.log1p(df["person_income"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])

    return df


def type_transformer(df):
    for col in df.select_dtypes(include='number'):
        if df[col].nunique() <= 25:
            df[col] = df[col].astype('category')

    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('category')

    return df


def mapper(df):
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1})
    return df


class AutoOneHotEncoder:
    def __init__(self):
        self.encoder = None
        self.cat_cols = []
        self.num_cols = []

    def fit(self, df):
        df_copy = df.copy()

        self.has_target = 'loan_status' in df_copy.columns
        if self.has_target:
            df_copy = df_copy.drop('loan_status', axis=1)

        self.cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        self.num_cols = df_copy.select_dtypes(include=['number']).columns.tolist()

        if self.cat_cols:
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(df_copy[self.cat_cols])

        return self

    def transform(self, df):
        df_copy = df.copy()

        target = None
        if self.has_target and 'loan_status' in df_copy.columns:
            target = df_copy['loan_status']
            df_copy = df_copy.drop('loan_status', axis=1)

        if not self.cat_cols:
            df_final = df_copy[self.num_cols].copy()
        else:
            encoded = self.encoder.transform(df_copy[self.cat_cols])
            encoded_cols = self.encoder.get_feature_names_out(self.cat_cols)
            df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_copy.index)

            df_final = pd.concat([df_copy[self.num_cols], df_encoded], axis=1)

        if target is not None:
            df_final['loan_status'] = target

        return df_final

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

def features_scaler(df_train=None, df_test=None, df_eval=None, scalers=None, df_inference=None):
    cols_to_robust = ['income_grade', 'income_per_age', 'loan_to_income']
    cols_to_standard = ['person_age', 'cb_person_cred_hist_length', 'person_emp_length', 'loan_percent_income',
                        'cred_hist_ratio', 'emp_to_age_ratio', 'loan_int_rate', 'log_income', 'log_loan_amnt',
                        'grade_interest_interaction']

    if scalers:
        robust = scalers['robust']
        df_inference[cols_to_robust] = robust.transform(df_inference[cols_to_robust])

        standard = scalers['standard']
        df_inference[cols_to_standard] = standard.transform(df_inference[cols_to_standard])
        return df_inference

    else:
        robust = RobustScaler()
        robust.fit(df_train[cols_to_robust])
        df_train[cols_to_robust] = robust.transform(df_train[cols_to_robust])
        df_test[cols_to_robust] = robust.transform(df_test[cols_to_robust])
        df_eval[cols_to_robust] = robust.transform(df_eval[cols_to_robust])

        standard = StandardScaler()
        standard.fit(df_train[cols_to_standard])
        df_train[cols_to_standard] = standard.transform(df_train[cols_to_standard])
        df_test[cols_to_standard] = standard.transform(df_test[cols_to_standard])
        df_eval[cols_to_standard] = standard.transform(df_eval[cols_to_standard])

        scalers = {'robust': robust, 'standard': standard}
        return df_train, df_test, df_eval, scalers


def multicol_dropper(df):
    df_dropped = df.drop(['loan_percent_income','cb_person_cred_hist_length','person_emp_length','income_grade',
                          'loan_grade','person_home_ownership', 'person_income', 'loan_amnt'], axis=1)
    return df_dropped


def build_pipeline(df):
    steps = [
        feature_adder,
        type_transformer,
        mapper
    ]

    for step in steps:
        df = step(df)

    return df

def build_features(df_train=None, df_test=None, df_eval=None, inference=False, df_inference=None):
    if inference:
        try:
            state = joblib.load('src/utils/build_features_pipeline_artifacts.pkl')
            df_inference = build_pipeline(df_inference)
            df_inference = features_scaler(df_inference=df_inference, scalers=state['scalers'])
            df_inference = multicol_dropper(df_inference)
            ohe = state['ohe']
            df_inference = ohe.transform(df_inference)
            return df_inference

        except Exception as e:
            print(f'Error: {e}')

    else:
        df_train = build_pipeline(df_train)
        df_test = build_pipeline(df_test)
        df_eval = build_pipeline(df_eval)



        df_train, df_test, df_eval, scalers = features_scaler(df_train, df_test, df_eval)

        df_train = multicol_dropper(df_train)
        df_test = multicol_dropper(df_test)
        df_eval = multicol_dropper(df_eval)

        ohe = AutoOneHotEncoder()
        df_train = ohe.fit_transform(df_train)
        df_test = ohe.transform(df_test)
        df_eval = ohe.transform(df_eval)

        state = {'ohe': ohe, 'scalers': scalers}
        joblib.dump(state, 'src/utils/build_features_pipeline_artifacts.pkl')

        return df_train, df_test, df_eval, scalers

