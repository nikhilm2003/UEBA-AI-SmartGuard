"""Inference helpers: load model, compute anomaly score and risk decision."""
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path('model/model.pkl')

def load_model(model_path=MODEL_PATH):
    data = joblib.load(model_path)
    return data['model'], data['scaler']

def score_session(session_row, model=None, scaler=None):
    # session_row: dict-like or pandas Series
    from src.feature_engineering import session_row_to_features
    x = session_row_to_features(session_row).reshape(1, -1)
    Xs = scaler.transform(x)
    # model.decision_function -> anomaly score; lower -> more anomalous
    score = model.decision_function(Xs)[0]
    is_anomaly = model.predict(Xs)[0] == -1
    return {'score': float(score), 'is_anomaly': bool(is_anomaly)}
