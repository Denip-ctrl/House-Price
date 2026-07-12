import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

# ==========================================
# 1. STREAMLIT UI: SETUP SIDEBAR
# ==========================================
st.set_page_config(layout="wide")
st.title("🏡 House Price Predictor – Ames Iowa Dataset")
st.sidebar.header("Konfigurasi & Input Fitur")

# Mengambil opsi daerah unik
@st.cache_data
def get_neighborhood_options():
    df = pd.read_csv("ames iowa housing.csv")
    return sorted(df['Neighborhood'].unique().tolist())

daftar_daerah = get_neighborhood_options()

st.sidebar.subheader("📍 Pilih Lokasi Properti")
daerah_terpilih = st.sidebar.selectbox("Neighborhood (Lingkungan)", daftar_daerah)

# --- FITUR EKSPERIMEN BARU ---
st.sidebar.subheader("🔬 Eksperimen Data Sciene")
use_log = st.sidebar.checkbox("Gunakan Transformasi Log (Perbaiki Skew)", value=False)

st.sidebar.subheader("⚙️ Pilih Fitur Tambahan (Akurasi)")
selected_features = ['GrLivArea', 'LotArea', 'BedroomAbvGr', 'GarageCars', 'Neighborhood']

add_qual = st.sidebar.checkbox("Kualitas Bangunan (OverallQual)", value=False)
add_year = st.sidebar.checkbox("Tahun Dibangun (YearBuilt)", value=False)
add_bsmt = st.sidebar.checkbox("Total Area Bawah Tanah (TotalBsmtSF)", value=False)

if add_qual:
    selected_features.append('OverallQual')
if add_year:
    selected_features.append('YearBuilt')
if add_bsmt:
    selected_features.append('TotalBsmtSF')


# ==========================================
# 2. FUNGSI LOAD DATA & PREPROCESSING (Dinamis dengan Opsi Log)
# ==========================================
@st.cache_data
def load_and_encode_data(features_tuple, use_log_transformation):
    features_list = list(features_tuple)
    df = pd.read_csv("ames iowa housing.csv")
    
    df_clean = df[features_list + ['SalePrice']].dropna()
    
    X = df_clean[features_list]
    
    # Kondisional: Gunakan Log jika di-centang user
    if use_log_transformation:
        y = np.log1p(df_clean['SalePrice'])
    else:
        y = df_clean['SalePrice']
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    encoder = TargetEncoder(cols=['Neighborhood'], smoothing=10.0)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    
    return X_train_encoded, X_test_encoded, y_train, y_test, encoder

# Eksekusi load data dengan mendeteksi status checkbox Log
X_train, X_test, y_train, y_test, target_encoder = load_and_encode_data(tuple(selected_features), use_log)
feature_names = X_train.columns


# ==========================================
# 3. FUNGSI TRAINING MODEL
# ==========================================
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

xgb_model = train_xgb(X_train, y_train)
lr_model = train_lr(X_train, y_train)


# ==========================================
# 4. INPUT SLIDER DARI USER
# ==========================================
st.sidebar.subheader("📋 Nilai Fitur Rumah")
gr_liv_area = st.sidebar.slider("Above Ground Living Area (sqft)", 500, 5000, 1500)
lot_area = st.sidebar.slider("Lot Size (sqft)", 1000, 50000, 10000)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
garage = st.sidebar.slider("Garage Capacity (Cars)", 0, 4, 2)

user_raw_dict = {
    'GrLivArea': gr_liv_area, 'LotArea': lot_area, 
    'BedroomAbvGr': bedrooms, 'GarageCars': garage, 'Neighborhood': daerah_terpilih
}

if add_qual: user_raw_dict['OverallQual'] = st.sidebar.slider("Kualitas Bangunan (1-10)", 1, 10, 6)
if add_year: user_raw_dict['YearBuilt'] = st.sidebar.slider("Tahun Bangun Rumah", 1872, 2010, 1970)
if add_bsmt: user_raw_dict['TotalBsmtSF'] = st.sidebar.slider("Luas Bawah Tanah (sqft)", 0, 3000, 1000)

input_raw_df = pd.DataFrame([user_raw_dict])
input_data = target_encoder.transform(input_raw_df)
input_data = input_data[feature_names]


# ==========================================
# 5. PREDIKSI USER (Dengan Inverse Log jika aktif)
# ==========================================
st.subheader("Hasil Prediksi Harga Rumah")

if use_log:
    xgb_pred = np.expm1(xgb_model.predict(input_data)[0])
    lr_pred = np.expm1(lr_model.predict(input_data)[0])
else:
    xgb_pred = xgb_model.predict(input_data)[0]
    lr_pred = lr_model.predict(input_data)[0]

col1, col2 = st.columns(2)
with col1: st.metric(label="Prediksi XGBoost", value=f"${xgb_pred:,.2f}")
with col2: st.metric(label="Prediksi Linear Regression", value=f"${lr_pred:,.2f}")


# ==========================================
# 6. EVALUASI AKURASI APEL-KE-APEL
# ==========================================
st.write("---")
st.subheader("📊 Evaluasi & Analisis Kesalahan Model")
st.caption(f"Status Log Transformation: **{use_log}** | Fitur Aktif: {', '.join(selected_features)}")

xgb_test_pred = xgb_model.predict(X_test)
lr_test_pred = lr_model.predict(X_test)

# Jika data dalam bentuk log, kembalikan ke skala asli untuk hitung performa nyata
if use_log:
    y_test_eval = np.expm1(y_test)
    xgb_test_pred_eval = np.expm1(xgb_test_pred)
    lr_test_pred_eval = np.expm1(lr_test_pred)
else:
    y_test_eval = y_test
    xgb_test_pred_eval = xgb_test_pred
    lr_test_pred_eval = lr_test_pred

rmse_xgb = np.sqrt(mean_squared_error(y_test_eval, xgb_test_pred_eval))
rmse_lr = np.sqrt(mean_squared_error(y_test_eval, lr_test_pred_eval))
r2_xgb = r2_score(y_test_eval, xgb_test_pred_eval)
r2_lr = r2_score(y_test_eval, lr_test_pred_eval)

eval_col1, eval_col2 = st.columns(2)
with eval_col1:
    st.markdown("### **Model XGBoost**")
    st.metric(label="Rata-rata Kesalahan (RMSE)", value=f"${rmse_xgb:,.2f}")
    st.metric(label="Akurasi ($R^2$ Score)", value=f"{r2_xgb*100:.2f}%")
with eval_col2:
    st.markdown("### **Linear Regression**")
    st.metric(label="Rata-rata Kesalahan (RMSE)", value=f"${rmse_lr:,.2f}")
    st.metric(label="Akurasi ($R^2$ Score)", value=f"{r2_lr*100:.2f}%")


# ==========================================
# 7. GRAFIK EVALUASI & FEATURE IMPORTANCE
# ==========================================
st.write("---")
grafik_col1, grafik_col2 = st.columns([2, 1])

with grafik_col1:
    st.markdown("### 📈 Grafik Perbandingan: Harga Asli vs Prediksi")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax0_min = min(y_test_eval.min(), xgb_test_pred_eval.min())
    ax0_max = max(y_test_eval.max(), xgb_test_pred_eval.max())
    ax1.scatter(y_test_eval, xgb_test_pred_eval, alpha=0.5, color='blue')
    ax1.plot([ax0_min, ax0_max], [ax0_min, ax0_max], 'r--', lw=2)
    ax1.set_title("XGBoost")
    
    ax1_min = min(y_test_eval.min(), lr_test_pred_eval.min())
    ax1_max = max(y_test_eval.max(), lr_test_pred_eval.max())
    ax2.scatter(y_test_eval, lr_test_pred_eval, alpha=0.5, color='green')
    ax2.plot([ax1_min, ax1_max], [ax1_min, ax1_max], 'r--', lw=2)
    ax2.set_title("Linear Regression")
    
    plt.tight_layout()
    st.pyplot(fig)

with grafik_col2:
    st.markdown("### 🔑 Variabel Paling Berpengaruh (XGBoost)")
    # Mengambil tingkat kepentingan fitur dari model XGBoost
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)
    
    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    ax_imp.barh(range(len(indices)), importances[indices], color='orange', align='center')
    ax_imp.set_yticks(range(len(indices)))
    ax_imp.set_yticklabels([feature_names[i] for i in indices])
    ax_imp.set_xlabel('Tingkat Kepentingan Relative')
    plt.tight_layout()
    st.pyplot(fig_imp)

# ==========================================
# 8. KESIMPULAN FITUR
# ==========================================
st.write("---")
st.subheader("💡 Kesimpulan Analisis Data: Fitur Apa yang Paling Berpengaruh?")
st.markdown("""
Berdasarkan grafik **Feature Importance** dari algoritma XGBoost di atas, kita dapat menyimpulkan urutan kekuatan pengaruh variabel terhadap harga jual rumah:

1. **Kualitas Bangunan (`OverallQual`) & Luas Bangunan Utama (`GrLivArea`):** Dua variabel ini secara konsisten memperebutkan posisi nomor 1 dan 2. Jika Anda mencentang `OverallQual`, Anda akan melihat kontribusinya melesat tajam. Penilaian subjektif atas material dan luas area di atas tanah adalah jangkar utama pembentukan harga properti.
2. **Konteks Wilayah (`Neighborhood` yang di-encode):** Wilayah terbukti menempati posisi papan atas (3 besar). Ini memvalidasi pepatah properti kuno: *"Lokasi, Lokasi, Lokasi"*. Bobot ekonomi wilayah memberikan nilai dasar yang kokoh sebelum ditambah variabel lain.
3. **Kapasitas Garasi (`GarageCars`) & Basement (`TotalBsmtSF`):** Menjadi fitur pendukung bernilai tinggi di Amerika Serikat. Rumah dengan garasi luas dan ruang bawah tanah fungsional memiliki daya tawar harga yang jauh lebih kuat.
4. **Ukuran Kavling Tanah (`LotArea`) & Jumlah Kamar (`BedroomAbvGr`):** Menariknya, jumlah kamar tidur memiliki pengaruh relatif lebih rendah. Hal ini dikarenakan penambahan kamar tanpa disertai perluasan total bangunan (`GrLivArea`) justru mempersempit ruang rumah dan tidak menaikkan harga secara signifikan.
""")
