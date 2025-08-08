import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import joblib
import json

# ===== Load Model & Mapping =====
model = joblib.load('best_model_LightGBM.pkl')

# Mapping Model_encoded dari file JSON
with open("model_price_mean.json", "r") as f:
    model_price_mean = pd.Series(json.load(f))

# Cek preprocessing
if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
    preprocessor = model.named_steps['preprocessor']
    expected_cols = preprocessor.feature_names_in_
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
else:
    st.error("Model pipeline tidak mengandung preprocessing. Simpan ulang model dengan preprocessing.")
    st.stop()

# ===== Halaman Home =====
html_home = """
<div style="
    background: linear-gradient(135deg, #4b6cb7, #182848);
    padding: 20px 40px; /* Tambah padding samping */
    border-radius: 15px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
    min-width: 450px; /* Lebar minimum biar nggak kepotong */
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
">
    <span style="font-size: 2.5em;">🚗</span>
    <h1 style="
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 2.2em;
        margin: 0;
        white-space: nowrap;
    ">
        Car Price Prediction
    </h1>
</div>
"""

desc_home = """
### Tentang Aplikasi  
Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan fitur-fitur yang Anda masukkan.  
Model yang digunakan adalah **LightGBM** dengan preprocessing otomatis yang sudah dilatih pada dataset Kaggle.  

#### Data Source  
[Kaggle - Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge)

---
**Cara Menggunakan:**
1. Buka menu *Car Price Prediction* di sidebar.  
2. Isi semua form sesuai detail mobil Anda.  
3. Klik tombol **Predict Price** untuk melihat estimasi harga mobil.
"""

# ===== Form Car Price Prediction =====
def run_car_price_app():
    st.subheader("Masukkan Detail Mobil")
    user_input = {}
    yes_no_map = {"Yes": 1, "No": 0}

    manufacturer_options = ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ',
                            'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN']
    model_options = list(model_price_mean.index)
    category_options = ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan']
    drive_wheels_options = ['4x4', 'Front', 'Rear']
    fuel_gear_options = ['Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
                         'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
                         'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual']
    doors_category_options = ['4-5', '2-3']

    # Numeric input
    for col in num_cols:
        if col == "Levy":
            user_input[col] = st.number_input(col, min_value=0, value=1000, step=100)
        elif col == "Mileage":
            user_input[col] = st.number_input(col, min_value=1, value=50000, step=1000)
        elif col == "Doors":
            user_input[col] = st.number_input(col, min_value=2, max_value=5, value=4, step=1)
        elif col == "Airbags":
            user_input[col] = st.number_input(col, min_value=1, value=2, step=1)
        elif col == "volume_per_cylinder":
            user_input[col] = st.number_input(col, min_value=0.5, value=2.0, step=0.1, format="%.1f")
        elif col == "car_age":
            user_input[col] = st.number_input(col, min_value=0, value=5, step=1)
        elif col == "Model_encoded":
            selected_model = st.selectbox("Model", model_options)
            user_input[col] = model_price_mean.get(selected_model, 0)
        elif col in ["Right_hand_drive", "Leather interior", "is_premium"]:
            user_input[col] = st.selectbox(col, ["Yes", "No"])
        else:
            user_input[col] = st.number_input(col, min_value=0, value=1, step=1)

    # Categorical input
    for col in cat_cols:
        if col == "Manufacturer":
            user_input[col] = st.selectbox(col, manufacturer_options)
        elif col == "Category":
            user_input[col] = st.selectbox(col, category_options)
        elif col == "Drive wheels":
            user_input[col] = st.selectbox(col, drive_wheels_options)
        elif col == "fuel_gear":
            user_input[col] = st.selectbox(col, fuel_gear_options)
        elif col == "Doors_category":
            user_input[col] = st.selectbox(col, doors_category_options)
        elif col not in user_input:
            user_input[col] = st.text_input(col, value="Unknown")

    # Convert ke DataFrame
    input_df = pd.DataFrame([user_input])

    # Map Yes/No ke 1/0
    for col in ["Right_hand_drive", "Leather interior", "is_premium"]:
        if col in input_df.columns:
            input_df[col] = input_df[col].map(yes_no_map)

    # Pastikan tipe data sesuai
    for col in num_cols:
        if col in input_df.columns and col != "volume_per_cylinder":
            input_df[col] = input_df[col].astype(float)
        elif col == "volume_per_cylinder":
            input_df[col] = input_df[col].astype(float)

    input_df = input_df.reindex(columns=expected_cols)

    # Predict
    if st.button("🔮 Predict Price"):
        try:
            log_prediction = model.predict(input_df)[0]
            prediction = np.expm1(log_prediction)
            st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===== Main App =====
def main():
    stc.html(html_home)
    menu = ["Home", "Car Price Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown(desc_home, unsafe_allow_html=True)
    elif choice == "Car Price Prediction":
        run_car_price_app()

if __name__ == "__main__":
    main()
