import streamlit as st
import pandas as pd
import pickle
import numpy as np
import folium
from streamlit_folium import st_folium

# ================================
# Load model
# ================================
with open("XGBM_model.pkl", "rb") as file:
    model = pickle.load(file)

# ================================
# Konfigurasi halaman
# ================================
st.set_page_config(page_title="Prediksi Diabetes", page_icon="💉", layout="centered")

st.title("💉 Aplikasi Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi kemungkinan terkena diabetes menggunakan model XGBoost.")

# ================================
# Input user
# ================================
st.header("Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, step=1)
    glucose = st.number_input("Glukosa (mg/dL)", min_value=0)
    blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0)
    skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0)

with col2:
    insulin = st.number_input("Insulin (µU/mL)", min_value=0)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Usia (tahun)", min_value=0, step=1)

# ================================
# Prediksi
# ================================
if st.button("Prediksi"):
    # Membuat DataFrame input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_df = pd.DataFrame(input_data, columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])
    
    # Prediksi
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    # ================================
    # Hasil Prediksi
    # ================================
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error(f"⚠️ Pasien **berpotensi diabetes** (Probabilitas: {proba:.2%})")
    else:
        st.success(f"✅ Pasien **tidak berpotensi diabetes** (Probabilitas: {proba:.2%})")

# ================================
# Maps Section
# ================================
st.header("🗺️ Peta Lokasi Rumah Sakit Diabetes")

# Create a folium map centered on Indonesia
m = folium.Map(location=[-6.2, 106.816666], zoom_start=10)

# Add some sample hospital markers (you can replace with real data)
hospitals = [
    {"name": "RSCM Jakarta", "lat": -6.2088, "lon": 106.8456},
    {"name": "RSUP Dr. Sardjito Yogyakarta", "lat": -7.7829, "lon": 110.3671},
    {"name": "RSUP Dr. Hasan Sadikin Bandung", "lat": -6.9175, "lon": 107.6191},
    {"name": "RSUP Sanglah Denpasar", "lat": -8.6731, "lon": 115.2126},
    {"name": "RSUP Dr. Kariadi Semarang", "lat": -6.9826, "lon": 110.4091}
]

for hospital in hospitals:
    folium.Marker(
        location=[hospital["lat"], hospital["lon"]],
        popup=hospital["name"],
        icon=folium.Icon(color="red", icon="plus-sign")
    ).add_to(m)

# Display the map
st_folium(m, width=700, height=500)

# ================================
# Footer
# ================================
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit dan XGBoost")
