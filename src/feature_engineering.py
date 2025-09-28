"""
Feature engineering utilities for UEBA demo.

Generates features from a session dictionary and converts sample CSV rows into model-ready features.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hour_of_day",        # 0-23
    "is_weekend",         # 0/1
    "distance_km_from_home",  # geographic distance proxy
    "device_change",      # 0/1 (new device)
    "transaction_amount", # absolute amount (scaled later)
    "num_transactions",   # count in session
    "mouse_jitter",       # proxy for behavioral biometric
]

def generate_synthetic_session(n=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    hours = rng.randint(0,24,size=n)
    is_weekend = rng.choice([0,1], size=n, p=[0.8,0.2])
    dist = np.abs(rng.normal(loc=5, scale=20, size=n))  # km
    device_change = rng.choice([0,1], size=n, p=[0.95,0.05])
    txn_amount = np.abs(rng.exponential(scale=2000, size=n))  # in INR
    num_txn = rng.poisson(lam=1.2, size=n)
    mouse_jitter = np.abs(rng.normal(loc=0.3, scale=0.5, size=n))
    df = pd.DataFrame({
        "hour_of_day": hours,
        "is_weekend": is_weekend,
        "distance_km_from_home": dist,
        "device_change": device_change,
        "transaction_amount": txn_amount,
        "num_transactions": num_txn,
        "mouse_jitter": mouse_jitter,
    })
    return df

def session_row_to_features(row):
    # Accepts a pandas Series or dict-like; returns numpy array
    return np.array([
        row.get("hour_of_day", 12),
        row.get("is_weekend", 0),
        row.get("distance_km_from_home", 0.0),
        row.get("device_change", 0),
        row.get("transaction_amount", 0.0),
        row.get("num_transactions", 0),
        row.get("mouse_jitter", 0.0),
    ], dtype=float)
