import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ======================
# Load model pipeline
# ======================
model = joblib.load('best_model_LightGBM.pkl')

# Cek pipeline punya preprocessing
if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
    preprocessor = model.named_steps['preprocessor']
    expected_cols = preprocessor.feature_names_in_
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
else:
    st.error("Model pipeline tidak mengandung preprocessing. Simpan ulang model dengan preprocessing.")
    st.stop()

# ======================
# Load mapping Model_encoded
# ======================
with open("model_price_mean.json", "r") as f:
    model_price_mean = json.load(f)
default_model_encoded = np.mean(list(model_price_mean.values()))

# ======================
# Mapping Yes/No → 1/0
# ======================
yes_no_map = {"Yes": 1, "No": 0}

# ======================
# Dropdown options
# ======================
manufacturer_options = [
    'LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ',
    'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN', 'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'
]
model_options = list(model_price_mean.keys())
category_options = ['Jeep', 'Hatchback', 'Sedan', 'Rare', 'Universal', 'Coupe', 'Minivan']
drive_wheels_options = ['4x4', 'Front', 'Rear']
fuel_gear_options = [
    'Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
    'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
    'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'
]
doors_category_options = ['4-5', '2-3']

# ======================
# Streamlit UI
# ======================
st.title("🚗 Car Price Prediction")
st.markdown("Masukkan detail mobil di bawah ini untuk memprediksi harga jual.")

user_input = {}

# Form input: numeric columns
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
        # User pilih model → ubah jadi encoded
        selected_model = st.selectbox("Model", model_options)
        encoded_value = model_price_mean.get(selected_model, default_model_encoded)
        user_input[col] = encoded_value
    elif col in ["Right_hand_drive", "Leather interior", "is_premium"]:
        yn_value = st.selectbox(col, ["Yes", "No"])
        user_input[col] = yes_no_map[yn_value]
    else:
        user_input[col] = st.number_input(col, min_value=0, value=1, step=1)

# Form input: categorical columns (tanpa kolom 'Model' karena sudah encoded)
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

# ======================
# Convert to DataFrame
# ======================
input_df = pd.DataFrame([user_input])

# Pastikan urutan kolom sesuai expected
input_df = input_df.reindex(columns=expected_cols)

# ======================
# Predict
# ======================
if st.button("🔮 Predict Price"):
    try:
        log_prediction = model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)  # Reverse log1p
        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
