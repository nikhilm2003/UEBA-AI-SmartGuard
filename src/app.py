import streamlit as st
from src.inference import load_model, score_session
import pandas as pd
import numpy as np

st.set_page_config(page_title='UEBA Demo', layout='centered')

st.title('AI-Powered UEBA — Demo')

st.sidebar.title('Session Controls')
hour = st.sidebar.slider('Hour of day', 0, 23, 10)
is_weekend = st.sidebar.selectbox('Is weekend?', [0,1], index=0)
distance = st.sidebar.number_input('Distance from home (km)', value=5.0, step=1.0)
device_change = st.sidebar.selectbox('Device change?', [0,1], index=0)
txn_amount = st.sidebar.number_input('Transaction amount (INR)', value=1000.0, step=100.0)
num_txn = st.sidebar.number_input('Number of txns in session', min_value=0, value=1)
mouse_jitter = st.sidebar.slider('Mouse jitter', 0.0, 5.0, 0.3, step=0.1)

session = {
    'hour_of_day': hour,
    'is_weekend': is_weekend,
    'distance_km_from_home': distance,
    'device_change': device_change,
    'transaction_amount': txn_amount,
    'num_transactions': num_txn,
    'mouse_jitter': mouse_jitter
}

st.subheader('Session Preview')
st.json(session)

try:
    model, scaler = load_model()
    result = score_session(session, model=model, scaler=scaler)
    st.subheader('Anomaly Result')
    st.write(result)
    if result['is_anomaly']:
        st.error('⚠️ Session flagged as anomalous — prompt for additional authentication.')
    else:
        st.success('✅ Session looks normal.')
except Exception as e:
    st.warning('Model not loaded. Make sure you have trained the model and that model/model.pkl exists.')
    st.write(str(e))
