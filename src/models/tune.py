import optuna
import mlflow
import pandas as pd

from sklearn.model_selection import cross_val_score,KFold
from ..utils.helpers import save_config_yaml
from lightgbm import LGBMClassifier

cv = KFold(n_splits=5, shuffle=True, random_state=12)

models = {
    'lgbm': LGBMClassifier,
}

hyperparams = {
    'lgbm': {
    'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1},
    'max_depth': {'type': 'int', 'low': 3, 'high': 15},
    'num_leaves': {'type': 'int', 'low': 31, 'high': 255},
    'min_child_samples': {'type': 'int', 'low': 10, 'high': 100},
    'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
    'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
    'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 10.0},
    'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 10.0},
    'min_split_gain': {'type': 'float', 'low': 0.0, 'high': 1.0},
    'boosting_type': {
        'type': 'categorical',
        'choices': ['gbdt', 'dart', 'goss']
    },
    'verbose': {'type': 'fixed', 'value': -1}
    }
}


def evaluate_model(model, X, y):
    return cross_val_score(model, X, y, scoring='accuracy', cv=cv).mean()

def run_optimization_mlflow(model_name, X, y, n_trials=50):
    model_class = models[model_name]
    param_config = hyperparams.get(model_name, {})

    #mlflow.set_experiment(model_name)

    def objective(trial):
        trial_params = {}
        for param_name, info in param_config.items():
            if info['type'] == 'float':
                trial_params[param_name] = trial.suggest_float(
                    param_name, info['low'], info['high'], log=info.get('log', False)
                )
            elif info['type'] == 'int':
                trial_params[param_name] = trial.suggest_int(param_name, info['low'], info['high'])
            elif info['type'] == 'categorical':
                trial_params[param_name] = trial.suggest_categorical(param_name, info['choices'])
            elif info['type'] == 'fixed':
                trial_params[param_name] = info['value']

        model = model_class(**trial_params)
        score = evaluate_model(model, X, y)

        with mlflow.start_run(nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_metric('accuracy', score)
            mlflow.log_param("model_name", model_name)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with mlflow.start_run(run_name=model_name):
        best_model = model_class(**study.best_params)
        best_score = evaluate_model(best_model, X, y)
        best_model.fit(X, y)
        mlflow.log_params(study.best_params)
        mlflow.log_metric('accuracy', best_score)
        mlflow.sklearn.log_model(best_model, name='model')
        mlflow.log_param("model_name", model_name)

    return model_name, study.best_params, study.best_value

if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('E2E_loan_prediction/tuned_models')

    best_params_dict = {}
    my_df = pd.read_csv('data/processed/processed_final_train_df.csv')
    X_train = my_df.drop(['loan_status'], axis=1)
    y_train= my_df['loan_status']
    for model_name in models:
        name, params, score = run_optimization_mlflow(model_name, X_train.drop('id', axis=1), y_train, n_trials=50)
        best_params_dict[name+'_params'] = {"params": params, "score": score}

    save_config_yaml(best_params_dict, 'configs/tuned_models_params.yaml')