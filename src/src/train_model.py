"""Train a simple IsolationForest anomaly detector on synthetic data and save the model."""
import os
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest
from src.feature_engineering import generate_synthetic_session
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_and_save_model(out_path='model/model.pkl', random_state=42):
    Path('model').mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_session(n=5000, random_state=random_state)
    X = df.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=random_state)
    model.fit(Xs)
    # Save both scaler and model as a dict
    joblib.dump({'model': model, 'scaler': scaler}, out_path)
    print(f"Saved model to {out_path}")

if __name__ == '__main__':
    train_and_save_model()
