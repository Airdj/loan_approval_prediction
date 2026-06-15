import joblib

def load_final_model(final_model_path):
    final_model = joblib.load(final_model_path)
    return final_model


class ModelService:
    def __init__(self, model_path):
        self.model = load_final_model(model_path)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
