import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import joblib
import json

# ====== Konstanta Kategori (bisa diubah gampang di sini) ======
MANUFACTURER_OPTIONS = [
    'LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA',
    'MERCEDES-BENZ', 'OPEL', 'Rare', 'BMW', 'AUDI', 'NISSAN',
    'SUBARU', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'VOLKSWAGEN'
]

CATEGORY_OPTIONS = [
    'Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe',
 'Minivan', 'Cabriolet', 'Limousine', 'Pickup'
]

DRIVE_WHEELS_OPTIONS = [
    '4x4', 'Front', 'Rear'
]

FUEL_GEAR_OPTIONS = [
    'Hybrid_Automatic', 'Petrol_Tiptronic', 'Petrol_Variator', 'Petrol_Automatic',
    'Diesel_Automatic', 'CNG_Manual', 'Rare', 'CNG_Automatic', 'Hybrid_Tiptronic',
    'Hybrid_Variator', 'Petrol_Manual', 'LPG_Automatic', 'Diesel_Manual'
]

DOORS_CATEGORY_OPTIONS = [
    '4-5', '2-3', '>5'
]

PREMIUM_BRANDS = ['BMW', 'MERCEDES-BENZ', 'AUDI', 'LEXUS']

# ===== Load Model & Mapping =====
model = joblib.load('best_model_LightGBM.pkl')

with open("manufacturer_category_model-2.json", "r") as f:
    manu_cat_model_map = json.load(f)

with open("model_price_mean.json", "r") as f:
    model_price_mean = pd.Series(json.load(f))

if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
    preprocessor = model.named_steps['preprocessor']
    expected_cols = preprocessor.feature_names_in_
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
else:
    st.error("Model pipeline tidak mengandung preprocessing. Simpan ulang model dengan preprocessing.")
    st.stop()

# ===== Load CSV premium & non-premium =====
df_premium = pd.read_csv('premium_models_summary.csv')      # is_premium harus 1
df_non_premium = pd.read_csv('non_premium_models_summary.csv')  # is_premium harus 0
df_all = pd.concat([df_premium, df_non_premium], ignore_index=True)

premium_map = dict(zip(df_all['Model'], df_all['is_premium']))

# ===== Halaman Home =====
html_home = """
<div style="
    background: linear-gradient(135deg, #4b6cb7, #182848);
    padding: 20px 30px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
">
    <div style="font-size: 2.5em;">🚗</div>
    <h1 style="
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 2em;
        margin: 0;
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

### 📂 Link Data Model Premium & Non-Premium  
Untuk menghindari kekacauan input data, silakan cek file **Model Premium** dan **Model Non-Premium** di Google Drive ini:  
[Google Drive Link](https://drive.google.com/drive/folders/1GhgCh04aUWoTmnR9XIF7AppsPxGN7tyN?usp=sharing)  

Gunakan file tersebut sebagai referensi agar input Model bisa tepat dan prediksi lebih akurat.

---

**Cara Menggunakan:**
1. Buka menu *Car Price Prediction* di sidebar.  
2. Isi semua form sesuai detail mobil Anda.  
3. Klik tombol **Predict Price** untuk melihat estimasi harga mobil.
"""


def run_car_price_app():
    st.subheader("Masukkan Detail Mobil")
    user_input = {}

    # ===== Dropdown Berjenjang =====
    # Manufacturer
    manufacturers = sorted(list(manu_cat_model_map.keys()))
    chosen_manufacturer = st.selectbox("Manufacturer", manufacturers)
    user_input["Manufacturer"] = chosen_manufacturer
    
    # Category terfilter dari JSON
    categories = sorted(list(manu_cat_model_map[chosen_manufacturer].keys()))
    chosen_category = st.selectbox("Category", categories)
    user_input["Category"] = chosen_category
    
    # Model terfilter dari JSON
    models = sorted(manu_cat_model_map[chosen_manufacturer][chosen_category])
    selected_model = st.selectbox("Model", models)
    user_input["Model"] = selected_model
    
    # Ambil encoding model
    user_input["Model_encoded"] = model_price_mean.get(selected_model, 0)
                                                                                                        
    # ===== Premium Detection =====
    is_premium_manufacturer = int(chosen_manufacturer in PREMIUM_BRANDS)
    is_premium_model = premium_map.get(selected_model, None)

    if is_premium_model is not None:
        user_input["is_premium"] = is_premium_model
    else:
        user_input["is_premium"] = is_premium_manufacturer

    if user_input["is_premium"] == 1:
        st.success("✅ Mobil terdeteksi sebagai premium (berdasarkan Model atau Manufacturer).")
    else:
        st.info("ℹ️ Mobil non-premium.")

    # ===== Numeric inputs =====
    for col in num_cols:
        if col in ["Model_encoded", "is_premium"]:
            continue
        if col == "Levy":
            user_input[col] = st.number_input(col, min_value=0, value=1000, step=100)
        elif col == "Mileage":
            user_input[col] = st.number_input(col, min_value=1, value=50000, step=1000)
        elif col == "Doors":
            user_input[col] = st.number_input(col, min_value=2, max_value=5, value=4, step=1)
        elif col == "Airbags":
            user_input[col] = st.selectbox(col, list(range(0, 21)), index=2)  # 0–16 pilihan
        elif col == "volume_per_cylinder":
            user_input[col] = st.number_input(col, min_value=0.5, value=2.0, step=0.1, format="%.1f")
        elif col == "car_age":
            user_input[col] = st.number_input(col, min_value=0, value=5, step=1)
        elif col in ["Right_hand_drive", "Leather interior"]:
            pass
        else:
            user_input[col] = st.number_input(col, min_value=0, value=1, step=1)

    # Categorical inputs
    for col in cat_cols:
        if col in ["Manufacturer", "Model", "Category", "is_premium"]:
            continue
        if col in ["Right_hand_drive", "Leather interior"]:
            user_input[col] = st.selectbox(col, ["Yes", "No"])
        elif col == "Drive wheels":
            user_input[col] = st.selectbox(col, DRIVE_WHEELS_OPTIONS)
        elif col == "fuel_gear":
            user_input[col] = st.selectbox(col, FUEL_GEAR_OPTIONS)
        elif col == "Doors_category":
            user_input[col] = st.selectbox(col, DOORS_CATEGORY_OPTIONS)
        else:
            user_input[col] = st.text_input(col, value="Unknown")

    # Convert Yes/No to binary
    for col in ["Right_hand_drive", "Leather interior"]:
        if col in user_input:
            user_input[col] = 1 if user_input[col] == "Yes" else 0

    # Convert numerics to float
    for col in num_cols:
        if col in user_input and col not in ["Model_encoded", "is_premium"]:
            user_input[col] = float(user_input[col])

    # ===== Prepare DataFrame =====
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=expected_cols)

    # ===== Prediction =====
    if st.button("🔮 Predict Price"):
        try:
            log_prediction = model.predict(input_df)[0]
            prediction = np.expm1(log_prediction)
            st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

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
