import streamlit as st
import pandas as pd
import xgboost as xgb
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
