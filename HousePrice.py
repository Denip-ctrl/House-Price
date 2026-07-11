import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==========================================
# 1. STREAMLIT UI: SETUP SIDEBAR & CHECKBOX
# ==========================================
st.title("🏡 House Price Predictor – Ames Iowa Dataset")
st.sidebar.header("Konfigurasi & Input Fitur")

st.sidebar.subheader("⚙️ Pilih Fitur Tambahan (Eksperimen Akurasi)")
# Fitur wajib (selaluTrue)
selected_features = ['GrLivArea', 'LotArea', 'BedroomAbvGr', 'GarageCars']

# Checkbox untuk fitur tambahan yang bisa dibongkar-pasang
add_qual = st.sidebar.checkbox("Kualitas Bangunan (OverallQual)", value=False)
add_year = st.sidebar.checkbox("Tahun Dibangun (YearBuilt)", value=False)
add_bsmt = st.sidebar.checkbox("Total Area Bawah Tanah (TotalBsmtSF)", value=False)

# Masukkan fitur tambahan ke list jika dicentang
if add_qual:
    selected_features.append('OverallQual')
if add_year:
    selected_features.append('YearBuilt')
if add_bsmt:
    selected_features.append('TotalBsmtSF')


# ==========================================
# 2. FUNGSI LOAD DATA & TRAINING (Dinamis berdasarkan list fitur)
# ==========================================
# Kita gunakan argumen `features` (dalam bentuk tuple agar bisa di-cache oleh Streamlit)
@st.cache_data
def load_data(features_tuple):
    features_list = list(features_tuple)
    df = pd.read_csv("ames iowa housing.csv")
    
    # Menghapus baris yang memiliki nilai kosong (missing values) pada fitur yang dipilih
    df_clean = df[features_list + ['SalePrice']].dropna()
    
    X = df_clean[features_list]
    y = df_clean['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_lr(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Eksekusi Load Data menggunakan fitur yang sedang terpilih
X_train, X_test, y_train, y_test = load_data(tuple(selected_features))
feature_names = X_train.columns

# Latih ulang model secara otomatis jika kombinasi fiturnya berubah
xgb_model = train_xgb(X_train, y_train)
lr_model = train_lr(X_train, y_train)


# ==========================================
# 3. INPUT SLIDER DARI USER (Dinamis)
# ==========================================
st.sidebar.subheader("📋 Nilai Fitur Rumah")

# Slider Wajib
gr_liv_area = st.sidebar.slider("Above Ground Living Area (sqft)", 500, 5000, 1500)
lot_area = st.sidebar.slider("Lot Size (sqft)", 1000, 50000, 10000)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
garage = st.sidebar.slider("Garage Capacity (Cars)", 0, 4, 2)

# List untuk menampung nilai input user
user_inputs = [gr_liv_area, lot_area, bedrooms, garage]

# Tampilkan slider tambahan HANYA jika dicentang
if add_qual:
    overall_qual = st.sidebar.slider("Kualitas Bangunan (1-10)", 1, 10, 6)
    user_inputs.append(overall_qual)
if add_year:
    year_built = st.sidebar.slider("Tahun Bangun Rumah", 1872, 2010, 1970)
    user_inputs.append(year_built)
if add_bsmt:
    total_bsmt = st.sidebar.slider("Luas Bawah Tanah / Basement (sqft)", 0, 3000, 1000)
    user_inputs.append(total_bsmt)


# ==========================================
# 4. PREDIKSI USER
# ==========================================
# Membuat DataFrame Input dengan kolom yang dinamis sesuai fitur terpilih
input_data = pd.DataFrame([user_inputs], columns=feature_names)

st.subheader("Hasil Prediksi Harga Rumah")
xgb_pred = xgb_model.predict(input_data)[0]
lr_pred = lr_model.predict(input_data)[0]

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Prediksi XGBoost", value=f"${xgb_pred:,.2f}")
with col2:
    st.metric(label="Prediksi Linear Regression", value=f"${lr_pred:,.2f}")


# ==========================================
# 5. EVALUASI AKURASI & GRAFIK (Dinamis)
# ==========================================
st.write("---")
st.subheader("📊 Evaluasi & Analisis Kesalahan Model")
st.caption(f"Fitur aktif saat ini: {', '.join(selected_features)}")

# Prediksi data uji
xgb_test_pred = xgb_model.predict(X_test)
lr_test_pred = lr_model.predict(X_test)

# Hitung Metrik Evaluasi
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
rmse_lr = np.sqrt(mean_squared_error(y_test, lr_test_pred))

r2_xgb = r2_score(y_test, xgb_test_pred)
r2_lr = r2_score(y_test, lr_test_pred)

# Tampilkan Angka Kesalahan
eval_col1, eval_col2 = st.columns(2)
with eval_col1:
    st.markdown("### **Model XGBoost**")
    st.metric(label="Rata-rata Kesalahan (RMSE)", value=f"${rmse_xgb:,.2f}")
    st.metric(label="Akurasi ($R^2$ Score)", value=f"{r2_xgb*100:.2f}%")

with eval_col2:
    st.markdown("### **Linear Regression**")
    st.metric(label="Rata-rata Kesalahan (RMSE)", value=f"${rmse_lr:,.2f}")
    st.metric(label="Akurasi ($R^2$ Score)", value=f"{r2_lr*100:.2f}%")

# Tampilkan Grafik
st.write("---")
st.markdown("### 📈 Grafik Perbandingan: Harga Asli vs Prediksi")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot XGBoost
ax0_min = min(y_test.min(), xgb_test_pred.min())
ax0_max = max(y_test.max(), xgb_test_pred.max())
ax1.scatter(y_test, xgb_test_pred, alpha=0.5, color='blue')
ax1.plot([ax0_min, ax0_max], [ax0_min, ax0_max], 'r--', lw=2)
ax1.set_title(f"XGBoost (RMSE: ${rmse_xgb:,.0f})")
ax1.set_xlabel("Harga Asli")
ax1.set_ylabel("Harga Prediksi")

# Plot Linear Regression
ax1_min = min(y_test.min(), lr_test_pred.min())
ax1_max = max(y_test.max(), lr_test_pred.max())
ax2.scatter(y_test, lr_test_pred, alpha=0.5, color='green')
ax2.plot([ax1_min, ax1_max], [ax1_min, ax1_max], 'r--', lw=2)
ax2.set_title(f"Linear Regression (RMSE: ${rmse_lr:,.0f})")
ax2.set_xlabel("Harga Asli")
ax2.set_ylabel("Harga Prediksi")

plt.tight_layout()
st.pyplot(fig)
