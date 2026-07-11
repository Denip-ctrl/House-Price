import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Fungsi Load Data (Di-cache agar efisien)
@st.cache_data
def load_data():
    df = pd.read_csv("ames iowa housing.csv")
    X = df[['GrLivArea', 'LotArea', 'BedroomAbvGr', 'GarageCars']]
    y = df['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Fungsi Training Model (Di-cache agar tidak melatih ulang setiap user geser slider)
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

# 3. Eksekusi Load Data & Training
X_train, X_test, y_train, y_test = load_data()
feature_names = X_train.columns

xgb_model = train_xgb(X_train, y_train)
lr_model = train_lr(X_train, y_train)

# 4. Streamlit UI
st.title("🏡 House Price Predictor – Ames Iowa Dataset")

st.sidebar.header("Input Fitur Rumah")
gr_liv_area = st.sidebar.slider("Above Ground Living Area (sqft)", 500, 5000, 1500)
lot_area = st.sidebar.slider("Lot Size (sqft)", 1000, 50000, 10000)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
garage = st.sidebar.slider("Garage Capacity (Cars)", 0, 4, 2)

# 5. Membuat DataFrame Input dari User
input_data = pd.DataFrame(
    [[gr_liv_area, lot_area, bedrooms, garage]], 
    columns=feature_names
)

# 6. Prediksi dan Menampilkan Hasil
st.subheader("Hasil Prediksi Harga Rumah")

xgb_pred = xgb_model.predict(input_data)[0]
lr_pred = lr_model.predict(input_data)[0]

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Prediksi XGBoost", value=f"${xgb_pred:,.2f}")
with col2:
    st.metric(label="Prediksi Linear Regression", value=f"${lr_pred:,.2f}")

# --- SEKSI BARU: EVALUASI MODEL ---
st.write("---")
st.subheader("📊 Evaluasi & Analisis Kesalahan Model")

# 1. Lakukan prediksi pada data uji (X_test)
xgb_test_pred = xgb_model.predict(X_test)
lr_test_pred = lr_model.predict(X_test)

# 2. Hitung Metrik Evaluasi
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
rmse_lr = np.sqrt(mean_squared_error(y_test, lr_test_pred))

r2_xgb = r2_score(y_test, xgb_test_pred)
r2_lr = r2_score(y_test, lr_test_pred)

# 3. Tampilkan Angka Kesalahan dalam bentuk Tabel/Metrik
eval_col1, eval_col2 = st.columns(2)

with eval_col1:
    st.markdown("### **Model XGBoost**")
    st.metric(label="Rata-rata Kesalahan (RMSE)", value=f"${rmse_xgb:,.2f}")
    st.metric(label="Akurasi ($R^2$ Score)", value=f"{r2_xgb*100:.2f}%")

with eval_col2:
    st.markdown("### **Linear Regression**")
    st.metric(label="Rata-rata Kesalahan (RMSE)", value=f"${rmse_lr:,.2f}")
    st.metric(label="Akurasi ($R^2$ Score)", value=f"{r2_lr*100:.2f}%")

st.write("---")
st.markdown("### 📈 Grafik Perbandingan: Harga Asli vs Prediksi")
st.write("Jika model 100% akurat, titik-titik akan mengikuti garis diagonal merah.")

# 4. Membuat Grafik Menggunakan Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot XGBoost
ax1.scatter(y_test, xgb_test_pred, alpha=0.5, color='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_title(f"XGBoost (RMSE: ${rmse_xgb:,.0f})")
ax1.set_xlabel("Harga Asli (Actual Price)")
ax1.set_ylabel("Harga Prediksi (Predicted Price)")

# Plot Linear Regression
ax2.scatter(y_test, lr_test_pred, alpha=0.5, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_title(f"Linear Regression (RMSE: ${rmse_lr:,.0f})")
ax2.set_xlabel("Harga Asli (Actual Price)")
ax2.set_ylabel("Harga Prediksi (Predicted Price)")

plt.tight_layout()

# 5. Tampilkan Grafik di Streamlit
st.pyplot(fig)
