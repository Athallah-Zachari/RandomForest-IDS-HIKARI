import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_tools():
    """Memuat semua model dan scaler. Dijalankan sekali saja."""
    try:
        model =  joblib.load('./src/ult_model.joblib')
        scaler = joblib.load('./src/scaler.joblib')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error memuat model/scaler: Pastikan file .joblib ada di folder 'src'. Detail: {e}")
        return None, None

@st.cache_data
def load_data():
    """
    Memuat data tes yang fiturnya sudah diseleksi (untuk prediksi)
    dan DataFrame asli yang lengkap (untuk mencari nama serangan).
    """
    try:
        test_data = pd.read_csv('./src/xTest.csv')
        category = pd.read_csv('./src/yTest_category.csv')
        return test_data, category
    except FileNotFoundError as e:
        st.error(f"Error memuat data CSV: Pastikan file .csv ada di folder 'src'. Detail: {e}")
        return None, None
    