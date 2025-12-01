import streamlit as st
import pandas as pd
# Impor fungsi yang relevan dari uttils.py
from uttils import load_tools, load_data

# --- Muat Aset yang Diperlukan ---
model, scaler = load_tools()
test_data, category = load_data() 

st.title("🎲 Demo Prediksi Model")
st.caption("Pilih Model yang diinginkan!")

if model and scaler is not None and test_data is not None and category is not None:
    
    st.markdown("---")
    st.subheader("Prediksi Sampel Acak")

    if st.button("Ambil Sampel & Mulai Prediksi"):
        # Ambil satu baris acak dari data tes
        sample_features = test_data.sample(1)
        
        st.write("🔍 **Data Sampel (Fitur):**")
        st.dataframe(sample_features)

        # --- CARA YANG LEBIH BAIK & LEBIH SEDERHANA ---
        # 1. Dapatkan indeks asli dari sampel 
        sample_index = sample_features.index[0]

        # 2. Gunakan indeks ini untuk mencari 'traffic_category' langsung di DataFrame original
        true_category = category.loc[sample_index, 'traffic_category']
        # --- SELESAI ---

        # Lakukan scaling dan prediksi (tidak berubah)
        sample_scaled = scaler.transform(sample_features)
        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled)[0]

        # --- Tampilkan Hasil Lengkap (tidak berubah) ---
        st.markdown("---")
        st.subheader("Hasil Analisis")

        prediction_text = "ANOMALI (1)" if prediction == 1 else "NORMAL (0)"
        if prediction == 1:
            st.error(f"🚨 **Prediksi Model:** {prediction_text}")
        else:
            st.success(f"✅ **Prediksi Model:** {prediction_text}")
            
        st.info(f"**Jenis Lalu Lintas Sebenarnya:** {true_category}")

        st.write("**Keyakinan Model:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Peluang Normal (0)", f"{probability[0]*100:.2f}%")
        with col2:
            st.metric("Peluang Anomali (1)", f"{probability[1]*100:.2f}%")
else:
    st.warning("Gagal memuat satu atau lebih file aset.")