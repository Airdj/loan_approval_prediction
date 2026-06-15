from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_base_models(config):
    models = {
        'lgbm': LGBMClassifier(**config['lgbm_params']),
        'xgb': XGBClassifier(**config['xgb_params']),
    }
    return models


def get_final_models(config):
    models = {
        'lgbm': LGBMClassifier(**config['lgbm_params']['params']),
        'xgb': XGBClassifier(**config['xgb_params']['params']),
    }
    return models